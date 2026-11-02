"""Score the full detection pipeline (preprocess -> model -> post-process) on the India val/test split.

Uses the same maths as Ultralytics model.val(): fp32 inference, NMS with conf 0.001 / IoU 0.7 /
multi-label, greedy IoU matching (BaseValidator.match_predictions) and ap_per_class. With
post-processing off it must reproduce model.val() (sanity gate).

Raw predictions for each TTA view are cached, so post-processing variants are cheap to compare.

Examples:
    python src/eval_pipeline.py --weights runs/yolo11l_v2/weights/best.pt --split val              # sanity gate
    python src/eval_pipeline.py --weights runs/yolo11l_v2/weights/best.pt --split val --tta
    python src/eval_pipeline.py --weights runs/yolo11l_v2/weights/best.pt --split val --merge
    python src/eval_pipeline.py --weights ... --split val --save-thresholds configs/thresholds.json
"""
import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import nms, ops
from ultralytics.utils.metrics import ap_per_class, box_iou

import bsw_yolo  # noqa: F401  (registers BSW-YOLO11 classes so their checkpoints can be loaded)
import gc_yolo  # noqa: F401  (same for the YOLO11 + GIA + CDSA classes)
import postprocess as pp
from preprocess import enhance

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["D00_Longitudinal_Crack", "D10_Transverse_Crack", "D20_Alligator_Crack", "D40_Pothole"]
IOUV = np.linspace(0.5, 0.95, 10)
LOOSE_IOUV = np.array([0.1, 0.3])


class Predictor:
    """Single-image inference that mirrors the Ultralytics validator."""

    def __init__(self, weights: str, imgsz: int = 640, device: str = "cuda:0"):
        self.model = AutoBackend(model=weights, device=torch.device(device), fp16=False, verbose=False)
        self.model.eval()
        self.device = torch.device(device)
        self.imgsz = imgsz
        self.stride = int(max(self.model.stride)) if hasattr(self.model.stride, "__iter__") else int(self.model.stride)

    @torch.no_grad()
    def __call__(self, img: np.ndarray, size_mult: float = 1.0):
        size = int(math.ceil(self.imgsz * size_mult / self.stride) * self.stride)
        h0, w0 = img.shape[:2]
        r = size / max(h0, w0)
        im = img
        if r != 1:
            im = cv2.resize(img, (min(math.ceil(w0 * r), size), min(math.ceil(h0 * r), size)),
                            interpolation=cv2.INTER_LINEAR)
        # Ultralytics val pads by half a stride (pad=0.5): a 640 image is fed as 672x672
        shape = int(math.ceil(size / self.stride + 0.5) * self.stride)
        h1, w1 = im.shape[:2]
        im = LetterBox(new_shape=(shape, shape), auto=False, scaleup=False)(image=im)
        left, top = round((shape - w1) / 2 - 0.1), round((shape - h1) / 2 - 0.1)  # LetterBox's centring
        x = torch.from_numpy(np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))).to(self.device)
        x = (x.float() / 255.0)[None]
        det = nms.non_max_suppression(self.model(x), 0.001, 0.7, multi_label=True, max_det=300)[0]
        if det.shape[0]:
            det[:, :4] = ops.scale_boxes(x.shape[2:], det[:, :4], (h0, w0),
                                         ratio_pad=((h1 / h0, w1 / w0), (left, top)))
        det = det.cpu().numpy()
        return det[:, :4], det[:, 4], det[:, 5]


def load_gt(lbl: Path, w: int, h: int):
    rows = [l.split() for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
    if not rows:
        return np.zeros(0), np.zeros((0, 4))
    a = np.array(rows, dtype=np.float64)
    xywh = a[:, 1:] * [w, h, w, h]
    return a[:, 0], np.column_stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                                     xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2])


def match(pred_cls, gt_cls, iou, iouv):
    """Greedy one-to-one matching per IoU threshold (same as BaseValidator.match_predictions)."""
    correct = np.zeros((len(pred_cls), len(iouv)), dtype=bool)
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])
    for i, t in enumerate(iouv):
        m = np.array(np.nonzero(iou >= t)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def raw_predictions(args, files):
    """Per image: list of (boxes, scores, classes) per TTA view, in original pixels. Cached on disk."""
    key = json.dumps([str(Path(args.weights).resolve()), args.data_root, args.split, args.pre, args.imgsz,
                      args.tta, args.tta_scale, Path(args.weights).stat().st_mtime])
    cache = ROOT / "runs" / "eval_pipeline" / f"cache_{hashlib.md5(key.encode()).hexdigest()[:12]}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    pred = Predictor(args.weights, args.imgsz, args.device)
    out = {}
    for f in files:
        img = cv2.imread(str(f))
        if args.pre != "none":
            img = enhance(img, clahe=args.pre == "clahe_gde")
        views = pp.tta_views(img, scales=(args.tta_scale,)) if args.tta else [(img, lambda b: b, 1.0)]
        per_view = []
        for v, unmap, mult in views:
            b, s, c = pred(v, mult)
            per_view.append((unmap(b), s, c))
        out[f.stem] = {"hw": img.shape[:2], "views": per_view}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default=str(ROOT / "data" / "rdd4"), help="dataset with <split>/images, labels")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--pre", default="none", choices=["none", "gde", "clahe_gde"], help="on-the-fly preprocessing")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda:0", help="'cpu' to avoid sharing the GPU with a training run")
    ap.add_argument("--tta", action="store_true", help="original + h-flip + larger input, fused with WBF")
    ap.add_argument("--tta-scale", type=float, default=1.25)
    ap.add_argument("--wbf-iou", type=float, default=0.45, help="tuned on India val (0.35-0.65 tried)")
    ap.add_argument("--merge", action="store_true", help="merge collinear D00/D10 crack fragments")
    ap.add_argument("--merge-gap", type=float, default=0.04, help="max gap as a fraction of image size")
    ap.add_argument("--image-conf", type=float, default=0.23, help="confidence for per-image metrics")
    ap.add_argument("--save-thresholds", default=None, help="write per-class best-F1 thresholds (JSON)")
    ap.add_argument("--tag", default=None, help="name for the results JSON")
    args = ap.parse_args()

    root = Path(args.data_root) / args.split
    files = sorted((root / "images").glob("*.jpg"))
    raw = raw_predictions(args, files)

    stats = {"tp": [], "tp_loose": [], "conf": [], "pred_cls": [], "target_cls": []}
    img_flags = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for f in files:
        (h, w), views = raw[f.stem]["hw"], raw[f.stem]["views"]
        if args.tta:
            b, s, c = pp.weighted_boxes_fusion(views, n_views=len(views), iou_thr=args.wbf_iou)
        else:
            b, s, c = views[0]
        if args.merge:
            b, s, c = pp.merge_crack_fragments(b, s, c, (h, w), gap_frac=args.merge_gap)
        order = np.argsort(-s)[:300]
        b, s, c = b[order], s[order], c[order]
        gt_cls, gt_box = load_gt(root / "labels" / f"{f.stem}.txt", w, h)
        if len(gt_cls) and len(c):
            iou = box_iou(torch.from_numpy(gt_box).float(), torch.from_numpy(b).float()).numpy()
            tp, tpl = match(c, gt_cls, iou, IOUV), match(c, gt_cls, iou, LOOSE_IOUV)
        else:
            tp, tpl = np.zeros((len(c), len(IOUV)), bool), np.zeros((len(c), len(LOOSE_IOUV)), bool)
        stats["tp"].append(tp); stats["tp_loose"].append(tpl); stats["conf"].append(s)
        stats["pred_cls"].append(c); stats["target_cls"].append(gt_cls)
        flagged = bool((s >= args.image_conf).any())
        img_flags[("tp" if flagged else "fn") if len(gt_cls) else ("fp" if flagged else "tn")] += 1

    st = {k: np.concatenate(v, 0) for k, v in stats.items()}
    _, _, p, r, _, ap, cls_ids, _, _, f1_curve, conf_axis, _ = ap_per_class(
        st["tp"], st["conf"], st["pred_cls"], st["target_cls"])
    ap_loose = ap_per_class(st["tp_loose"], st["conf"], st["pred_cls"], st["target_cls"])[5]

    per_class = [{"class": NAMES[int(c)], "precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                  "mAP50": round(float(ap[i, 0]), 4), "mAP50-95": round(float(ap[i].mean()), 4),
                  "AP@0.1": round(float(ap_loose[i, 0]), 4), "AP@0.3": round(float(ap_loose[i, 1]), 4)}
                 for i, c in enumerate(cls_ids)]
    t = img_flags
    report = {
        "weights": args.weights, "split": args.split, "pre": args.pre, "tta": args.tta, "merge": args.merge,
        "overall": {"precision": round(float(p.mean()), 4), "recall": round(float(r.mean()), 4),
                    "mAP50": round(float(ap[:, 0].mean()), 4), "mAP50-95": round(float(ap.mean()), 4),
                    "AP@0.1": round(float(ap_loose[:, 0].mean()), 4), "AP@0.3": round(float(ap_loose[:, 1].mean()), 4)},
        "per_class": per_class,
        "image_level": {"conf": args.image_conf,
                        "damaged_flagged": round(t["tp"] / max(1, t["tp"] + t["fn"]), 4),
                        "clean_false_alarm": round(t["fp"] / max(1, t["fp"] + t["tn"]), 4),
                        "flag_precision": round(t["tp"] / max(1, t["tp"] + t["fp"]), 4)},
    }

    print(f"\n{'class':24s} {'P':>6s} {'R':>6s} {'mAP50':>6s} {'m50-95':>7s} {'AP@.1':>6s} {'AP@.3':>6s}")
    for row in per_class:
        print(f"{row['class']:24s} {row['precision']:6.3f} {row['recall']:6.3f} {row['mAP50']:6.3f} "
              f"{row['mAP50-95']:7.3f} {row['AP@0.1']:6.3f} {row['AP@0.3']:6.3f}")
    o = report["overall"]
    print(f"{'all':24s} {o['precision']:6.3f} {o['recall']:6.3f} {o['mAP50']:6.3f} {o['mAP50-95']:7.3f} "
          f"{o['AP@0.1']:6.3f} {o['AP@0.3']:6.3f}")
    il = report["image_level"]
    print(f"Per image @conf {il['conf']}: damaged flagged {il['damaged_flagged']:.1%}, "
          f"clean false alarms {il['clean_false_alarm']:.1%}, flag precision {il['flag_precision']:.1%}")

    if args.save_thresholds:
        th = pp.best_f1_thresholds(f1_curve, conf_axis, cls_ids)
        Path(args.save_thresholds).write_text(json.dumps({NAMES[c]: v for c, v in th.items()}, indent=2))
        print(f"Per-class best-F1 thresholds: {th} -> {args.save_thresholds}")

    tag = args.tag or f"{Path(args.weights).parents[1].name}_{args.split}_{args.pre}" \
                      f"{'_tta' if args.tta else ''}{'_merge' if args.merge else ''}"
    out = ROOT / "runs" / "eval_pipeline" / f"{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

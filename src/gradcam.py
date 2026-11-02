"""Module 6 - Grad-CAM explanations for road-damage detections.

For each detection, back-propagates that detection's class score (the raw prediction behind the box,
found via NMS indices) to the three feature maps the detection head reads (P3/P4/P5, after the
SnakeConv blocks in BSW-YOLO11), and maps the result back to the original image.

Two methods (--method):
  hires    (default) element-wise Grad-CAM, HiResCAM (Draelos & Carin 2020): ReLU(sum_c dS/dA_c * A_c)
           per location. Instance-level: shows where THIS box's score came from.
  gradcam  classic Grad-CAM (Selvaraju et al. 2017): ReLU(sum_c mean(dS/dA_c) * A_c). The channel
           weights are averaged over the whole map, so for a detector it gives a class-level map that
           lights up every damage-like texture in the image, not just the explained box.
Layer maps are summed at input resolution (so layers with little gradient contribute little), then
normalised.

Uses its own copy of the network loaded straight from the checkpoint (the Detector's copy has been
run in inference mode, whose tensors cannot take gradients).

Examples:
    python src/gradcam.py --source photo.jpg
    python src/gradcam.py --source data/india_japan/test/images --limit 20 --top 3 --name ij_cams
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import nms, ops

import bsw_yolo  # noqa: F401  (BSW classes for the checkpoint)
import gc_yolo  # noqa: F401  (GIA/CDSA classes for the checkpoint)
from detector import IMG_EXT, MODELS, Detector, PRETTY, draw, read_image
from preprocess import enhance

ROOT = Path(__file__).resolve().parents[1]


class GradCAM:
    def __init__(self, detector: Detector, layers=None, method: str = "hires"):
        assert method in ("hires", "gradcam")
        if detector.cfg.get("backend", "ultralytics") != "ultralytics":
            raise ValueError(f"HiResCAM explanations are available only for the Ultralytics YOLO11 models, "
                             f"not for {detector.cfg['short']}.")
        self.method = method
        self.det, self.cfg = detector, detector.cfg
        self.device = torch.device("cpu" if detector.device == "cpu" else f"cuda:{detector.device}")
        self.net = YOLO(str(self.cfg["weights"])).model.float().eval().to(self.device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.layers = list(layers or self.net.model[-1].f)
        self.acts, self.grads = {}, {}
        for i in self.layers:
            self.net.model[i].register_forward_hook(self._hook(i))

    def _hook(self, i):
        def fn(module, inputs, output):
            self.acts[i] = output
            if output.requires_grad:   # plain no-grad forwards (e.g. re-scoring) must still work
                output.register_hook(lambda g: self.grads.__setitem__(i, g))
        return fn

    def _prepare(self, bgr: np.ndarray):
        """Same preprocessing as Detector, then square letterbox to imgsz."""
        h0, w0 = bgr.shape[:2]
        s = min(1.0, self.cfg["proc_size"] / max(h0, w0))
        small = cv2.resize(bgr, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        inp = enhance(small) if self.cfg["enhance"] else small
        sz = self.cfg["imgsz"]
        h1, w1 = inp.shape[:2]
        r = min(sz / h1, sz / w1)
        nw, nh = round(w1 * r), round(h1 * r)
        left, top = round((sz - nw) / 2 - 0.1), round((sz - nh) / 2 - 0.1)
        canvas = np.full((sz, sz, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(inp, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1))).float().div(255)[None]
        return x.to(self.device), {"orig": (h0, w0), "lb": (top, left, nh, nw)}

    def explain(self, bgr: np.ndarray, top_k: int = 3, all_above_threshold: bool = False) -> dict:
        """Returns {'detections': [...], 'cams': [HxW float maps in original pixels], 'combined': HxW}."""
        x, geo = self._prepare(bgr)
        h0, w0 = geo["orig"]
        top, left, nh, nw = geo["lb"]
        self.acts.clear()
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.net(x)
            y = y[0] if isinstance(y, (list, tuple)) else y                 # (1, 4 + nc, anchors)
            kept, idx = nms.non_max_suppression(y.detach(), 0.05, 0.7, max_det=300, return_idxs=True)
            kept, idx = kept[0], idx[0].view(-1)
            order = [k for k in range(len(kept)) if float(kept[k, 4]) >= self.det.threshold(int(kept[k, 5]))]
            order = order if all_above_threshold else order[:top_k]
            boxes = ops.scale_boxes(x.shape[2:], kept[:, :4].clone(), (h0, w0)) if len(kept) else kept[:, :4]
            dets, cams = [], []
            for k in order:
                cls, anchor = int(kept[k, 5]), int(idx[k])
                self.grads.clear()
                self.net.zero_grad(set_to_none=True)
                y[0, 4 + cls, anchor].backward(retain_graph=True)
                cam = torch.zeros((1, 1, x.shape[2], x.shape[3]), device=self.device)
                for i in self.layers:
                    a, g = self.acts[i], self.grads.get(i)
                    if g is None:
                        continue
                    w = g if self.method == "hires" else g.mean(dim=(2, 3), keepdim=True)
                    c = torch.relu((w * a).sum(1, keepdim=True))
                    cam += torch.nn.functional.interpolate(c, size=x.shape[2:], mode="bilinear", align_corners=False)
                cam = cam[0, 0, top:top + nh, left:left + nw]
                cam = cam / cam.max() if cam.max() > 0 else cam
                cam = cv2.resize(cam.detach().cpu().numpy(), (w0, h0), interpolation=cv2.INTER_LINEAR)
                b = [round(float(v), 1) for v in boxes[k]]
                dets.append({"cls": cls, "name": self.det.names[cls], "conf": round(float(kept[k, 4]), 4),
                             "box_xyxy": b, "anchor": anchor, **_localisation(cam, b)})
                cams.append(cam)
        combined = np.max(np.stack(cams), 0) if cams else np.zeros((h0, w0), np.float32)
        return {"detections": dets, "cams": cams, "combined": combined}


def _localisation(cam: np.ndarray, box) -> dict:
    """How well the explanation points at the detected damage.

    peak_in_box    the hottest Grad-CAM pixel lies inside the box ("pointing game")
    cam_in_box     share of Grad-CAM mass inside the box
    concentration  cam_in_box / (box area / image area); 1 = no better than a uniform map
    """
    h, w = cam.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, max(x2, x1 + 1)), min(h, max(y2, y1 + 1))
    tot = float(cam.sum())
    inside = float(cam[y1:y2, x1:x2].sum()) / tot if tot > 0 else 0.0
    area = (x2 - x1) * (y2 - y1) / (w * h)
    py, px = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return {"peak_in_box": bool(x1 <= px < x2 and y1 <= py < y2), "cam_in_box": round(inside, 3),
            "concentration": round(inside / area, 2) if area > 0 else 0.0}


def overlay(bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)


def explanation_sheet(bgr: np.ndarray, res: dict, panel: int = 360, method: str = "hires") -> np.ndarray:
    """[detections | combined Grad-CAM | one panel per explained detection], each scaled to `panel` px high."""
    from detector import Detection
    def fit(im):
        s = panel / im.shape[0]
        return cv2.resize(im, (round(im.shape[1] * s), panel))
    def label(im, text):
        cv2.putText(im, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(im, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return im
    h, w = bgr.shape[:2]
    dets = [Detection(d["cls"], d["name"], d["conf"], d["box_xyxy"], [0, 0, 0, 0], 0.0) for d in res["detections"]]
    panels = [label(fit(draw(bgr, dets)), "detections"), label(fit(overlay(bgr, res["combined"])), ("HiResCAM" if method == "hires" else "Grad-CAM") + " (all)")]
    for d, cam in zip(dets, res["cams"]):
        panels.append(label(fit(draw(overlay(bgr, cam), [d])), f"{PRETTY.get(d.name, d.name).split(' ', 1)[-1]} {d.conf:.2f}"))
    return np.hstack(panels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="image or folder of images")
    ap.add_argument("--model", default="india_japan", choices=list(MODELS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--top", type=int, default=3, help="explain the N most confident detections per image")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images of a folder (0 = all)")
    ap.add_argument("--method", default="hires", choices=["hires", "gradcam"],
                    help="hires = element-wise (instance-level, default); gradcam = classic (class-level)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    src = Path(args.source)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    files = files[:args.limit] if args.limit else files
    det = Detector(args.model, device=args.device)
    cam = GradCAM(det, method=args.method)
    base = ROOT / "runs" / "gradcam"
    out, i = base / (args.name or src.stem), 2
    while out.exists():
        out, i = base / f"{args.name or src.stem}{i}", i + 1
    out.mkdir(parents=True)
    explained = []
    for f in files:
        try:
            img = read_image(f)
        except Exception as e:
            print(f"  skipped {f.name}: not a readable image ({e.__class__.__name__})")
            continue
        res = cam.explain(img, top_k=args.top)
        explained += res["detections"]
        cv2.imwrite(str(out / f"{f.stem}_gradcam.jpg"), explanation_sheet(img, res, method=args.method), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {f.name}: " + (", ".join(
            f"{d['name'][:3]} {d['conf']:.2f} (peak in box: {'yes' if d['peak_in_box'] else 'no'}, "
            f"concentration x{d['concentration']:.1f})" for d in res["detections"]) or "no detections above threshold"))
    if explained:
        n = len(explained)
        print(f"\n{n} detections explained | peak inside its box: {sum(d['peak_in_box'] for d in explained) / n:.0%} | "
              f"median concentration x{float(np.median([d['concentration'] for d in explained])):.1f} "
              f"(1 = no better than uniform)")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

"""Tune the RoadSense ensemble on India + Japan VAL, then score it once on TEST.

Runs the real engine code path (src/ensemble.py: resize -> guided filter -> 3 members -> WBF) over the split.
Member outputs are cached in runs/ensemble/views_<split>.pkl, so fusion settings are cheap to compare.

  --split val   picks the fusion IoU (best mean F1), per-class thresholds (best F1 per class), the confidence
                calibration (isotonic fit of "was this box real damage" at IoU 0.5) and the photo-quality limits,
                and writes configs/ensemble.json. BSW-YOLO11s alone is tuned the same way as the reference.
  --split test  scores the saved settings (no tuning) -> runs/ensemble/test.json, and adds the test numbers to
                configs/ensemble.json for the website's About page.

Examples:
    python scripts/tune_ensemble.py --split val
    python scripts/tune_ensemble.py --split test
"""
import argparse
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import eval_pipeline as ep  # noqa: E402
import postprocess as pp  # noqa: E402
from detector import read_image  # noqa: E402
from ensemble import CONFIG, EnsembleDetector, photo_quality  # noqa: E402
from ultralytics.utils.metrics import ap_per_class, box_iou  # noqa: E402

NAMES = ep.NAMES
OUT = ROOT / "runs" / "ensemble"
IOU_CANDIDATES = (0.45, 0.5, 0.55, 0.6, 0.65)


def collect(det: EnsembleDetector, split: str):
    """Per image: member views, image size, ground truth and photo-quality stats (cached)."""
    cache = OUT / f"views_{split}.pkl"
    root = ROOT / "data" / "india_japan" / split
    files = sorted((root / "images").glob("*.jpg"))
    if cache.exists():
        data = pickle.loads(cache.read_bytes())
        if data["members"] == det.settings["members"] and data["member_min_conf"] == det.settings["member_min_conf"] \
                and len(data["images"]) == len(files):
            return data
    images = {}
    for n, f in enumerate(files, 1):
        bgr = read_image(f)
        views, _, _ = det.member_views(bgr)
        h, w = bgr.shape[:2]
        q = photo_quality(bgr, None)
        images[f.stem] = {"hw": (h, w), "views": views, "gt": ep.load_gt(root / "labels" / f"{f.stem}.txt", w, h),
                          "brightness": q["brightness"], "sharpness": q["sharpness"]}
        if n % 250 == 0 or n == len(files):
            print(f"  {n}/{len(files)} images")
    data = {"members": det.settings["members"], "member_min_conf": det.settings["member_min_conf"], "images": images}
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(data))
    return data


def fused(data, iou: float | None, member: int | None = None):
    """stem -> (boxes, scores, classes) for the ensemble (iou) or one member alone (member index)."""
    out = {}
    for stem, im in data["images"].items():
        if member is not None:
            b, s, c = im["views"][member]
        else:
            b, s, c = pp.weighted_boxes_fusion(im["views"], n_views=len(im["views"]), iou_thr=iou)
        o = np.argsort(-s)[:300]
        out[stem] = (b[o], s[o], c[o])
    return out


def _tp(b, c, gt_cls, gt_box, iouv):
    if len(gt_cls) and len(c):
        iou = box_iou(torch.from_numpy(gt_box).float(), torch.from_numpy(b).float()).numpy()
        return ep.match(c, gt_cls, iou, iouv)
    return np.zeros((len(c), len(iouv)), bool)


def evaluate(data, preds, thresholds: dict | None = None, subset: str | None = None) -> dict:
    """mAP (Ultralytics maths) over all boxes; P/R/F1 and image-level results at the given per-class thresholds."""
    st = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
    cnt = {c: {"tp": 0, "fp": 0, "gt": 0} for c in range(len(NAMES))}
    img = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "type_ok": 0}
    for stem, (b, s, c) in preds.items():
        if subset and not stem.startswith(subset):
            continue
        gt_cls, gt_box = data["images"][stem]["gt"]
        st["tp"].append(_tp(b, c, gt_cls, gt_box, ep.IOUV)); st["conf"].append(s)
        st["pred_cls"].append(c); st["target_cls"].append(gt_cls)
        if thresholds is None:
            continue
        keep = np.array([s[i] >= thresholds[int(c[i])] for i in range(len(s))], bool)
        tk = _tp(b[keep], c[keep], gt_cls, gt_box, np.array([0.5]))[:, 0]
        for k in range(len(NAMES)):
            cnt[k]["tp"] += int(tk[c[keep] == k].sum()); cnt[k]["fp"] += int((~tk[c[keep] == k]).sum())
            cnt[k]["gt"] += int((gt_cls == k).sum())
        flagged = bool(keep.any())
        img[("tp" if flagged else "fn") if len(gt_cls) else ("fp" if flagged else "tn")] += 1
        if len(gt_cls) and flagged:
            img["type_ok"] += bool(set(c[keep].astype(int)) & set(gt_cls.astype(int)))
    a = {k: np.concatenate(v, 0) for k, v in st.items()}
    r = ap_per_class(a["tp"], a["conf"], a["pred_cls"], a["target_cls"])
    ap, cls_ids, f1_curve, conf_axis = r[5], r[6], r[9], r[10]
    res = {"images": len(st["conf"]), "mAP50": round(float(ap[:, 0].mean()), 4), "mAP50-95": round(float(ap.mean()), 4),
           "mean_best_f1": round(float(f1_curve.max(1).mean()), 4),
           "per_class_mAP50": {NAMES[int(k)]: round(float(ap[i, 0]), 4) for i, k in enumerate(cls_ids)},
           "_f1": (f1_curve, conf_axis, cls_ids)}
    if thresholds is not None:
        pc = {}
        for k, v in cnt.items():
            p = v["tp"] / max(1, v["tp"] + v["fp"]); rc = v["tp"] / max(1, v["gt"])
            pc[NAMES[k]] = {"precision": round(p, 4), "recall": round(rc, 4),
                            "f1": round(2 * p * rc / max(1e-9, p + rc), 4), "gt": v["gt"]}
        res["at_thresholds"] = {"per_class": pc,
                                "precision": round(float(np.mean([x["precision"] for x in pc.values()])), 4),
                                "recall": round(float(np.mean([x["recall"] for x in pc.values()])), 4),
                                "f1": round(float(np.mean([x["f1"] for x in pc.values()])), 4)}
        dmg, clean = img["tp"] + img["fn"], img["fp"] + img["tn"]
        res["image_level"] = {"damaged_photos": dmg, "clean_photos": clean,
                              "damaged_flagged": round(img["tp"] / max(1, dmg), 4),
                              "clean_false_alarm": round(img["fp"] / max(1, clean), 4),
                              "clean_correct": round(img["tn"] / max(1, clean), 4),
                              "photo_accuracy": round((img["tp"] + img["tn"]) / max(1, dmg + clean), 4),
                              "right_type_when_flagged": round(img["type_ok"] / max(1, img["tp"]), 4)}
    return res


def best_thresholds(res) -> dict:
    f1_curve, conf_axis, cls_ids = res["_f1"]
    return pp.best_f1_thresholds(f1_curve, conf_axis, cls_ids)


def isotonic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: non-decreasing least-squares fit of y (sorted by x)."""
    vals, wts, sizes = [], [], []
    for v in y.astype(float):
        vals.append(v); wts.append(1.0); sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            wts[-2] = w; sizes[-2] += sizes[-1]
            vals.pop(); wts.pop(); sizes.pop()
    return np.repeat(vals, sizes)


def calibration(data, preds) -> dict:
    """Per class: monotone table fused score -> share of boxes that matched a real damage (IoU >= 0.5)."""
    xs = {k: [] for k in range(len(NAMES))}; ys = {k: [] for k in range(len(NAMES))}
    for stem, (b, s, c) in preds.items():
        gt_cls, gt_box = data["images"][stem]["gt"]
        tp = _tp(b, c, gt_cls, gt_box, np.array([0.5]))[:, 0]
        for k in range(len(NAMES)):
            m = c == k
            xs[k].extend(s[m]); ys[k].extend(tp[m])
    out = {}
    for k in range(len(NAMES)):
        x, y = np.asarray(xs[k]), np.asarray(ys[k], float)
        o = np.argsort(x); x, y = x[o], y[o]
        fit = isotonic(x, y)
        # shrink each pooled block towards 1/2 by its size (Laplace: (hits + 1) / (n + 2)), so a small block of
        # high-score boxes that were all right shows e.g. 96 %, never a certain-sounding 100 %
        starts = np.r_[0, np.flatnonzero(np.diff(fit)) + 1]
        for a, b in zip(starts, np.r_[starts[1:], len(fit)]):
            n = b - a
            fit[a:b] = (fit[a] * n + 1) / (n + 2)
        # knots where the fitted step changes, plus the ends; interpolating between them gives a smooth curve
        idx = np.unique(np.r_[0, np.flatnonzero(np.diff(fit)) + 1, len(x) - 1])
        kx, ky = x[idx], fit[idx]
        if len(kx) > 40:
            sel = np.unique(np.linspace(0, len(kx) - 1, 40).round().astype(int))
            kx, ky = kx[sel], ky[sel]
        kx = np.r_[0.0, kx, 1.0]; ky = np.r_[ky[0], ky, ky[-1]]
        out[NAMES[k]] = {"score": [round(float(v), 4) for v in kx], "precision": [round(float(v), 4) for v in ky],
                         "boxes": int(len(x))}
    return out


def reliability(data, preds, cal: dict, thresholds: dict) -> dict:
    """Calibrated confidence vs. how often kept boxes were real damage (test check)."""
    conf, hit = [], []
    for stem, (b, s, c) in preds.items():
        keep = np.array([s[i] >= thresholds[int(c[i])] for i in range(len(s))], bool)
        if not keep.any():
            continue
        gt_cls, gt_box = data["images"][stem]["gt"]
        tp = _tp(b[keep], c[keep], gt_cls, gt_box, np.array([0.5]))[:, 0]
        conf.extend(float(np.interp(sc, cal[NAMES[int(k)]]["score"], cal[NAMES[int(k)]]["precision"]))
                    for sc, k in zip(s[keep], c[keep]))
        hit.extend(tp)
    conf, hit = np.asarray(conf), np.asarray(hit, float)
    bins, ece = [], 0.0
    for lo, hi in ((0, .3), (.3, .45), (.45, .6), (.6, .75), (.75, 1.01)):
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            bins.append({"range": [lo, min(hi, 1.0)], "boxes": int(m.sum()), "mean_confidence": round(float(conf[m].mean()), 3),
                         "actually_correct": round(float(hit[m].mean()), 3)})
            ece += m.sum() / len(conf) * abs(conf[m].mean() - hit[m].mean())
    return {"bins": bins, "expected_calibration_error": round(float(ece), 4), "boxes": int(len(conf))}


def strip(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.split == "val":
        det = EnsembleDetector(device=args.device, config=None)      # untuned defaults
        data = collect(det, "val")
        trials = {}
        for iou in IOU_CANDIDATES:
            r = evaluate(data, fused(data, iou))
            trials[iou] = r
            print(f"WBF IoU {iou:.2f}: mAP50 {r['mAP50']:.4f}  mAP50-95 {r['mAP50-95']:.4f}  mean best F1 {r['mean_best_f1']:.4f}")
        iou = max(IOU_CANDIDATES, key=lambda i: (trials[i]["mean_best_f1"], trials[i]["mAP50"]))
        preds = fused(data, iou)
        th = best_thresholds(trials[iou])
        ens = evaluate(data, preds, th)
        ref_preds = fused(data, None, member=0)
        ref_th = best_thresholds(evaluate(data, ref_preds))
        ref = evaluate(data, ref_preds, ref_th)
        cal = calibration(data, preds)
        bright = np.array([im["brightness"] for im in data["images"].values()])
        sharp = np.array([im["sharpness"] for im in data["images"].values()])
        quality = {"min_brightness": round(float(np.percentile(bright, 0.5)), 1),
                   "max_brightness": round(float(np.percentile(bright, 99.5)), 1),
                   "min_sharpness": round(float(np.percentile(sharp, 0.5)), 1), "min_side": 360,
                   "_about": "0.5th / 99.5th percentiles of the India + Japan val photos (all usable road photos)."}
        cfg = {"_about": "RoadSense ensemble settings, written by scripts/tune_ensemble.py --split val "
                         "(India + Japan val). Thresholds apply to the fused WBF score; calibration maps that score "
                         "to the share of such boxes that were real damage on val (IoU >= 0.5).",
               "members": det.settings["members"], "member_min_conf": det.settings["member_min_conf"],
               "wbf_iou": iou, "thresholds": {NAMES[k]: v for k, v in th.items()},
               "default_score": round(float(np.mean(list(th.values()))), 3),
               "reference": {"model": det.settings["members"][0], "thresholds": {NAMES[k]: v for k, v in ref_th.items()}},
               "calibration": cal, "quality": quality,
               "metrics": [["India+Japan val", ens["mAP50"], ens["mAP50-95"]]],
               "val": {"ensemble": strip(ens), "bsw_alone": strip(ref),
                       "wbf_trials": {str(i): {k: t[k] for k in ("mAP50", "mAP50-95", "mean_best_f1")}
                                      for i, t in trials.items()}},
               "tuned": str(date.today())}
        CONFIG.write_text(json.dumps(cfg, indent=1))
        (OUT / "val.json").write_text(json.dumps(cfg["val"], indent=1))
        print(f"\nChosen WBF IoU {iou}; thresholds {cfg['thresholds']}")
        for name, r in (("ensemble", ens), ("BSW alone", ref)):
            a, il = r["at_thresholds"], r["image_level"]
            print(f"{name:10s} mAP50 {r['mAP50']:.4f}  mAP50-95 {r['mAP50-95']:.4f}  P {a['precision']:.3f} R {a['recall']:.3f} "
                  f"F1 {a['f1']:.3f} | photos: damaged flagged {il['damaged_flagged']:.1%}, clean false alarm "
                  f"{il['clean_false_alarm']:.1%}, accuracy {il['photo_accuracy']:.1%}, right type {il['right_type_when_flagged']:.1%}")
        print(f"Quality limits: {quality}\nSaved: {CONFIG}, {OUT / 'val.json'}")
        return

    cfg = json.loads(CONFIG.read_text())
    det = EnsembleDetector(device=args.device)
    data = collect(det, "test")
    th = {NAMES.index(n): v for n, v in cfg["thresholds"].items()}
    ref_th = {NAMES.index(n): v for n, v in cfg["reference"]["thresholds"].items()}
    preds, ref_preds = fused(data, cfg["wbf_iou"]), fused(data, None, member=0)
    report = {"settings": {"wbf_iou": cfg["wbf_iou"], "thresholds": cfg["thresholds"], "tuned_on": "India+Japan val"}}
    for subset, label in ((None, "combined"), ("India", "india"), ("Japan", "japan")):
        report[label] = {"ensemble": strip(evaluate(data, preds, th, subset)),
                         "bsw_alone": strip(evaluate(data, ref_preds, ref_th, subset))}
    report["calibration_check"] = reliability(data, preds, cfg["calibration"], th)
    (OUT / "test.json").write_text(json.dumps(report, indent=1))
    c = report["combined"]["ensemble"]; i = report["india"]["ensemble"]; j = report["japan"]["ensemble"]
    cfg["metrics"] = [["India+Japan val", cfg["val"]["ensemble"]["mAP50"], cfg["val"]["ensemble"]["mAP50-95"]],
                      ["India+Japan test", c["mAP50"], c["mAP50-95"]], ["India test", i["mAP50"], i["mAP50-95"]],
                      ["Japan test", j["mAP50"], j["mAP50-95"]]]
    cfg["test"] = {"date": str(date.today()), "file": "runs/ensemble/test.json"}
    CONFIG.write_text(json.dumps(cfg, indent=1))
    for label in ("combined", "india", "japan"):
        for name in ("ensemble", "bsw_alone"):
            r = report[label][name]; a, il = r["at_thresholds"], r["image_level"]
            print(f"{label:9s} {name:10s} mAP50 {r['mAP50']:.4f}  mAP50-95 {r['mAP50-95']:.4f}  P {a['precision']:.3f} "
                  f"R {a['recall']:.3f} F1 {a['f1']:.3f} | photos: flagged {il['damaged_flagged']:.1%}, false alarm "
                  f"{il['clean_false_alarm']:.1%}, accuracy {il['photo_accuracy']:.1%}, right type {il['right_type_when_flagged']:.1%}")
    print("Calibration check (test):", json.dumps(report["calibration_check"]))
    print(f"Saved: {OUT / 'test.json'}; test metrics added to {CONFIG}")


if __name__ == "__main__":
    main()

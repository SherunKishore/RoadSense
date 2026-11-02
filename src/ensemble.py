"""RoadSense ensemble: three India + Japan YOLO11s models fused with weighted boxes fusion (WBF).

Members (all use guided-filter input, 720 px processing, 640 px model input):
  india_japan  BSW-YOLO11s            runs/final_bsw_s     (strongest single model, also used for HiResCAM)
  gia_cdsa     YOLO11s + GIA + CDSA   runs/gdgf_gia_cdsa_s
  plain_s      YOLO11s                runs/abl2_a0_base
The image is resized and guided-filtered once, each member predicts, and boxes of the same class that overlap
are averaged (score-weighted). A box only one model sees keeps a third of its score, so agreement counts.

Settings tuned on the India + Japan validation split by scripts/tune_ensemble.py live in configs/ensemble.json:
  wbf_iou      fusion IoU
  thresholds   per-class cut-offs on the fused score (best F1 on val)
  calibration  per class, fused score -> how often a box with that score was real damage on val (IoU >= 0.5).
               Detection.conf is this calibrated value, so 0.7 means "about 7 in 10 such boxes were right";
               the raw fused score is kept in Detection.score.
  quality      brightness / sharpness limits below which a photo is flagged for a retake

    from ensemble import EnsembleDetector
    det = EnsembleDetector()          # same interface as detector.Detector (predict, detect_raw, names, cfg, key)
    dets = det.predict(bgr)           # EnsembleDetection list
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

import postprocess as pp
from detector import MIN_CONF, Detection, Detector
from preprocess import enhance

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "ensemble.json"
MEMBERS = ("india_japan", "gia_cdsa", "plain_s")
DEFAULTS = {"members": list(MEMBERS), "member_min_conf": 0.02, "wbf_iou": 0.55,
            "thresholds": None, "calibration": None, "quality": None}


@dataclass
class EnsembleDetection(Detection):
    score: float = 0.0      # raw fused WBF score (what the thresholds apply to)
    agree: int = 1          # how many member models found this box


def calibrate(table: dict | None, score: float) -> float:
    """Fused score -> calibrated confidence, by linear interpolation in a monotone table."""
    if not table:
        return score
    return float(np.interp(score, table["score"], table["precision"]))


def photo_quality(bgr: np.ndarray, limits: dict | None) -> dict:
    """Brightness (mean grey, 0-255) and sharpness (variance of the Laplacian at 720 px) with retake warnings."""
    h, w = bgr.shape[:2]
    s = min(1.0, 720 / max(h, w))
    grey = cv2.cvtColor(cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr,
                        cv2.COLOR_BGR2GRAY)
    brightness, sharpness = float(grey.mean()), float(cv2.Laplacian(grey, cv2.CV_64F).var())
    warnings = []
    if limits:
        if brightness < limits["min_brightness"]:
            warnings.append("Photo is very dark; damage may be missed. Retake in daylight if possible.")
        elif brightness > limits["max_brightness"]:
            warnings.append("Photo is overexposed; damage may be missed.")
        if sharpness < limits["min_sharpness"]:
            warnings.append("Photo looks blurred; hold the camera steady or slow down and retake.")
        if min(h, w) < limits.get("min_side", 360):
            warnings.append(f"Photo is small ({w}x{h}); use at least 720 px for reliable results.")
    return {"brightness": round(brightness, 1), "sharpness": round(sharpness, 1), "warnings": warnings}


class EnsembleDetector:
    """WBF ensemble of registered Detector models that share the same preprocessing."""

    def __init__(self, device="auto", config: Path | str | None = CONFIG):
        path = Path(config) if config else None
        self.settings = {**DEFAULTS, **(json.loads(path.read_text()) if path and path.exists() else {})}
        self.members = [Detector(k, device=device) for k in self.settings["members"]]
        first = self.members[0]
        for m in self.members[1:]:
            assert (m.cfg["enhance"], m.cfg["proc_size"]) == (first.cfg["enhance"], first.cfg["proc_size"]), \
                "ensemble members must share preprocessing"
            assert m.names == first.names, "ensemble members must share class names"
        self.key, self.names, self.device = "ensemble", dict(first.names), first.device
        th = self.settings["thresholds"]
        self.class_thresholds = {c: th[n] for c, n in self.names.items()} if th else None
        self.per_class = self.class_thresholds is not None
        self.conf = self.settings.get("default_score", 0.1)
        self.cfg = {"title": "RoadSense ensemble: " + " + ".join(m.cfg["short"] for m in self.members) + " (WBF)",
                    "short": "RoadSense ensemble", "weights": path or CONFIG, "epoch": None,
                    "enhance": first.cfg["enhance"], "proc_size": first.cfg["proc_size"], "imgsz": first.cfg["imgsz"],
                    "thresholds": th, "default_conf": self.conf, "metrics": self.settings.get("metrics", []),
                    "note": "Three India + Japan models fused with weighted boxes fusion; settings tuned on val.",
                    "test_dir": first.cfg["test_dir"]}

    @property
    def explainer(self) -> Detector:
        """The member used for HiResCAM explanations (BSW-YOLO11s)."""
        return self.members[0]

    def threshold(self, cls: int) -> float:
        return self.class_thresholds[cls] if self.per_class else self.conf

    def member_views(self, bgr: np.ndarray):
        """Each member's raw boxes in original-image pixels, the shared model input, and timings in ms."""
        h0, w0 = bgr.shape[:2]
        s = min(1.0, self.cfg["proc_size"] / max(h0, w0))
        small = cv2.resize(bgr, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        t0 = time.perf_counter()
        inp = enhance(small) if self.cfg["enhance"] else small
        t1 = time.perf_counter()
        views = []
        for m in self.members:
            b, c, k = m._run(inp, self.settings["member_min_conf"])
            views.append((np.asarray(b, np.float64) / s, np.asarray(c, np.float64), np.asarray(k, np.float64)))
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        return views, inp, {"preprocess": round((t1 - t0) * 1000, 1), "inference": round((t2 - t1) * 1000, 1)}

    def fuse(self, views, hw, min_score: float = MIN_CONF / 5) -> list[EnsembleDetection]:
        h0, w0 = hw
        boxes, scores, classes, agree = pp.weighted_boxes_fusion(views, n_views=len(views),
                                                                 iou_thr=self.settings["wbf_iou"], return_sources=True)
        cal = self.settings["calibration"] or {}
        dets = []
        for xyxy, sc, c, n in zip(boxes, scores, classes, agree):
            if sc < min_score:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w0), x2), min(float(h0), y2)
            name = self.names[int(c)]
            dets.append(EnsembleDetection(int(c), name, round(calibrate(cal.get(name), float(sc)), 4),
                                          [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                                          [round(x1 / w0, 5), round(y1 / h0, 5), round(x2 / w0, 5), round(y2 / h0, 5)],
                                          round((x2 - x1) * (y2 - y1) / (w0 * h0), 5),
                                          score=round(float(sc), 4), agree=int(n)))
        dets.sort(key=lambda d: -d.score)
        return dets

    def detect_raw(self, bgr: np.ndarray, min_conf: float = MIN_CONF / 5):
        """All fused detections with score >= min_conf, the preprocessed model input, and timings in ms."""
        views, inp, ms = self.member_views(bgr)
        t = time.perf_counter()
        dets = self.fuse(views, bgr.shape[:2], min_conf)
        ms["fusion"] = round((time.perf_counter() - t) * 1000, 1)
        return dets, inp, ms

    def keep(self, d: EnsembleDetection) -> bool:
        return d.score >= self.threshold(d.cls)

    def predict(self, bgr: np.ndarray) -> list[EnsembleDetection]:
        """Fused detections that pass the tuned per-class thresholds."""
        dets, _, _ = self.detect_raw(bgr)
        return [d for d in dets if self.keep(d)]

    def quality(self, bgr: np.ndarray) -> dict:
        return photo_quality(bgr, self.settings["quality"])

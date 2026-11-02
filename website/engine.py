"""The website's analysis engine: ensemble detection (src/ensemble.py) -> severity (Module 4) -> repair priority
(Module 5), plus annotated images and HiResCAM explanations (Module 6).

All model calls go through one lock, so live uploads and background survey jobs never run the networks at once.
"""
import base64
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector import PRETTY, draw  # noqa: E402
from ensemble import EnsembleDetector  # noqa: E402
from gradcam import GradCAM, overlay  # noqa: E402
from priority import build_plan, group_records, load_config as load_priority_config, prioritise  # noqa: E402
from severity import load_config as load_severity_config, score_record  # noqa: E402

# Damage-class colours for the website (colour-blind-checked set; no red/green, which mean P1/P4 priority)
CLASS_HEX = {0: "#2a78d6", 1: "#1baf7a", 2: "#8a5a00", 3: "#db2777"}
CLASS_BGR = {c: tuple(int(h[i:i + 2], 16) for i in (5, 3, 1)) for c, h in CLASS_HEX.items()}


def jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def data_uri(img: np.ndarray, quality: int = 80) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg(img, quality)).decode()


def fit(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


class Engine:
    def __init__(self, device="cpu"):
        self.det = EnsembleDetector(device=device)
        self.names = self.det.names
        self.sev_cfg, self.pri_cfg = load_severity_config(), load_priority_config()
        self.lock = threading.Lock()
        self._cam = None
        self.device = device

    # ------------------------------------------------------------ single photo
    def analyse(self, bgr: np.ndarray, source: str = "photo", gps=None, road_class: str = "unknown") -> dict:
        """Detections (with severity) above the tuned thresholds, near-misses, the repair priority and photo quality."""
        h, w = bgr.shape[:2]
        with self.lock:
            quality = self.det.quality(bgr)
            dets, _, ms = self.det.detect_raw(bgr)
        kept = [d for d in dets if self.det.keep(d)]
        # near-misses: boxes within 60 % of their class threshold, shown on request as "uncertain"
        uncertain = [d.to_dict() for d in dets if not self.det.keep(d) and d.score >= 0.6 * self.det.threshold(d.cls)][:8]
        rec = score_record({"source": source, "frame": None, "time_s": None, "width": w, "height": h, "gps": gps,
                            "detections": [d.to_dict() for d in kept]}, self.sev_cfg, True)
        return {"width": w, "height": h, "detections": rec["detections"], "uncertain": uncertain,
                "image_severity": rec["image_severity"], "plan": self.plan(rec, road_class), "quality": quality,
                "ms": ms, "record": rec}

    def plan(self, rec: dict, road_class: str) -> dict | None:
        p = build_plan([rec], self.pri_cfg, road_class) if rec["detections"] else []
        return p[0] if p else None

    def plan_for_records(self, records: list[dict], road_class: str):
        """Survey grouping: [(location, plan item)] with locations from priority.group_records (25 m GPS clusters,
        5 s video windows, or single photos without GPS)."""
        out = []
        for loc in group_records([r for r in records if r["detections"]], self.pri_cfg):
            p = prioritise(loc, self.pri_cfg, road_class)
            if p:
                out.append((loc, p))
        return out

    def annotate(self, bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
        from detector import Detection
        objs = [Detection(d["cls"], d["name"], d["conf"], d["box_xyxy"], d.get("box_norm", [0, 0, 0, 0]),
                          d.get("area_ratio", 0.0)) for d in detections]
        return draw(bgr, objs, CLASS_BGR)

    # ------------------------------------------------------------ explanations
    def explain(self, bgr: np.ndarray, top_k: int = 3) -> dict:
        with self.lock:
            if self._cam is None:
                self._cam = GradCAM(self.det.explainer)
            res = self._cam.explain(bgr, top_k=top_k)
        small = fit(bgr, 1280)
        s = small.shape[1] / bgr.shape[1]
        rs = lambda cam: cv2.resize(cam, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        return {"model": self.det.explainer.cfg["short"], "combined": data_uri(overlay(small, rs(res["combined"]))),
                "detections": [{"name": d["name"], "conf": d["conf"], "peak_in_box": d["peak_in_box"],
                                "cam_in_box": d["cam_in_box"], "box": [v * s for v in d["box_xyxy"]],
                                "overlay": data_uri(overlay(small, rs(c)), 75)}
                               for d, c in zip(res["detections"], res["cams"])]}


def pretty(name: str) -> str:
    return PRETTY.get(name, name)

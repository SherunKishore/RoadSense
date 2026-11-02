"""Module 3 - shared road-damage inference core.

One place that knows how each trained model must be fed (preprocessing, image size, thresholds)
and turns an image into structured detections. Used by src/predict.py (CLI), the dashboard, and
later by the severity / priority / Grad-CAM / GPS modules.

    from detector import Detector
    det = Detector("india_japan")            # or "gia_cdsa", "plain_s", "china", "usc_rddc2020"
    dets = det.predict(bgr_image)            # list[Detection], thresholds applied
    annotated = draw(bgr_image, dets)
"""
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from PIL.ExifTags import IFD

import bsw_yolo  # noqa: F401  (BSW-YOLO11 classes, needed to load BSW checkpoints)
import gc_yolo  # noqa: F401  (YOLO11 + GIA + CDSA classes, needed to load those checkpoints)
from preprocess import enhance
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MIN_CONF = 0.05  # raw detections below this are never returned
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PRETTY = {"D00_Longitudinal_Crack": "D00 Longitudinal crack", "D10_Transverse_Crack": "D10 Transverse crack",
          "D20_Alligator_Crack": "D20 Alligator crack", "D40_Pothole": "D40 Pothole"}
COLORS_BGR = {0: (246, 130, 59), 1: (11, 158, 245), 2: (68, 68, 239), 3: (129, 185, 16)}  # same as the dashboard


def _thresholds(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


MODELS = {
    "india_japan": {
        "title": "BSW-YOLO11s (SnakeConv + BRA-DropKey + Wise-IoU) on India + Japan", "short": "BSW-YOLO11s",
        "weights": ROOT / "runs" / "final_bsw_s" / "weights" / "best.pt",
        "epoch": 87, "enhance": True, "proc_size": 720, "imgsz": 640, "default_conf": 0.29,
        "thresholds": _thresholds(ROOT / "configs" / "thresholds_final_bsw_s.json"),
        "test_dir": ROOT / "data" / "india_japan" / "test",
        "metrics": [["India+Japan test", 0.498, 0.207], ["India test", 0.365, 0.154], ["Japan test", 0.456, 0.195]],
        "note": "Final model for Indian roads, tested once on the India + Japan test split.",
    },
    "gia_cdsa": {
        "title": "YOLO11s + GIA + CDSA (gradient-guided + contextual dilated attention) on India + Japan",
        "short": "YOLO11s + GIA + CDSA",
        "weights": ROOT / "runs" / "gdgf_gia_cdsa_s" / "weights" / "best.pt",
        "epoch": 81, "enhance": True, "proc_size": 720, "imgsz": 640, "default_conf": 0.26,
        "thresholds": _thresholds(ROOT / "configs" / "thresholds_gdgf_gia_cdsa_s.json"),
        "test_dir": ROOT / "data" / "india_japan" / "test",
        "metrics": [["India+Japan test", 0.493, 0.209], ["India test", 0.359, 0.155], ["Japan test", 0.466, 0.206]],
        "note": "Alternative India + Japan model (no BSW parts), tested once on the India + Japan test split.",
    },
    "plain_s": {   # third member of the website's ensemble (src/ensemble.py)
        "title": "YOLO11s (no attention modules) on India + Japan, guided-filter input", "short": "YOLO11s (plain)",
        "weights": ROOT / "runs" / "abl2_a0_base" / "weights" / "best.pt",
        "epoch": 58, "enhance": True, "proc_size": 720, "imgsz": 640, "default_conf": 0.25,
        "thresholds": None,
        "test_dir": ROOT / "data" / "india_japan" / "test",
        "metrics": [["India+Japan test", 0.484, 0.201], ["India test", 0.354, 0.146], ["Japan test", 0.453, 0.196]],
        "note": "Ablation baseline A0 (60 epochs), the same recipe without BSW or GIA/CDSA.",
    },
    "china": {
        "title": "YOLO11s baseline on RDD-China (base-paper dataset)", "short": "YOLO11s (China)",
        "weights": ROOT / "runs" / "china_c0_base" / "weights" / "best.pt",
        "epoch": 263, "enhance": False, "proc_size": 512, "imgsz": 512, "default_conf": 0.25,
        "thresholds": None,
        "test_dir": ROOT / "data" / "rdd_china" / "test",
        "metrics": [["China test", 0.860, 0.564], ["Base paper BSW-YOLO (China test)", 0.905, None]],
        "note": "Tested once on the RDD-China test split. Does not transfer to Indian roads (India test mAP50 0.003).",
    },
    "usc_rddc2020": {   # external model, for comparison only (src/usc_yolov5.py, third_party/rddc2020_yolov5)
        "title": "YOLOv5x by USC IMSC (GRDDC 2020 challenge), trained on RDD2020 India + Japan + Czech",
        "short": "YOLOv5x (USC, RDD2020)", "backend": "usc_yolov5",
        "weights": ROOT / "models" / "usc_rddc2020" / "last_95.pt",
        "epoch": None, "enhance": False, "proc_size": 720, "imgsz": 640, "default_conf": 0.20,
        "thresholds": None,
        "test_dir": ROOT / "data" / "india_japan" / "test",
        "metrics": [["GRDDC'20 test1 (authors' ensemble)", 0.675, None, "F1"],
                    ["GRDDC'20 test2 (authors' ensemble)", 0.666, None, "F1"]],
        "note": "External model from github.com/USC-InfoLab/rddc2020 (GPL-3.0), best single checkpoint last_95.pt.",
        "warning": "Trained on RDD2020 India + Japan images, which overlap with this project's test split: its "
                   "found/missed counts on test samples are optimistic. Compare on new photos.",
    },
}
USC_NAMES = {0: "D00_Longitudinal_Crack", 1: "D10_Transverse_Crack", 2: "D20_Alligator_Crack", 3: "D40_Pothole"}


@dataclass
class Detection:
    cls: int
    name: str
    conf: float
    box_xyxy: list          # original-image pixels
    box_norm: list          # xyxy / (w, h, w, h)
    area_ratio: float       # box area / image area

    def to_dict(self) -> dict:
        return asdict(self)


class Detector:
    """Loads one registered model and runs it with that model's own preprocessing."""

    def __init__(self, model: str = "india_japan", device="auto", conf: float | None = None,
                 per_class: bool | None = None):
        if model not in MODELS:
            raise ValueError(f"unknown model '{model}', choose from {list(MODELS)}")
        self.key, self.cfg = model, MODELS[model]
        self.device = (0 if torch.cuda.is_available() else "cpu") if device == "auto" else \
            ("cpu" if str(device) == "cpu" else int(device))
        self.backend = self.cfg.get("backend", "ultralytics")
        if self.backend == "usc_yolov5":
            from usc_yolov5 import USCYOLOv5
            self.model = USCYOLOv5(self.cfg["weights"], device=self.device, imgsz=self.cfg["imgsz"])
            assert [n[:3] for n in self.model.names] == [USC_NAMES[i][:3] for i in range(4)], self.model.names
            self.names = dict(USC_NAMES)
        else:
            self.model = YOLO(str(self.cfg["weights"]))
            self.names = {int(k): v for k, v in self.model.names.items()}
        th = self.cfg["thresholds"]
        self.class_thresholds = {c: th[n] for c, n in self.names.items() if n in th} if th else None
        self.conf = conf if conf is not None else self.cfg["default_conf"]
        # per-class thresholds by default when the model has them and no global conf was forced
        self.per_class = (self.class_thresholds is not None and conf is None) if per_class is None else \
            (per_class and self.class_thresholds is not None)
        sz = self.cfg["imgsz"]
        self._run(np.zeros((sz, sz, 3), np.uint8), MIN_CONF)  # warm-up

    def _run(self, img: np.ndarray, min_conf: float):
        """Raw model output on the preprocessed image: (xyxy, conf, cls) arrays in that image's pixels."""
        if self.backend == "usc_yolov5":
            return self.model.predict(img, conf=min_conf)
        r = self.model.predict(img, imgsz=self.cfg["imgsz"], conf=min_conf, iou=0.7, device=self.device, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()

    def threshold(self, cls: int) -> float:
        return self.class_thresholds[cls] if self.per_class else self.conf

    def detect_raw(self, bgr: np.ndarray, min_conf: float = MIN_CONF):
        """All detections >= min_conf, the preprocessed model input, and timings in ms."""
        import time
        h0, w0 = bgr.shape[:2]
        s = min(1.0, self.cfg["proc_size"] / max(h0, w0))
        small = cv2.resize(bgr, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        t0 = time.perf_counter()
        inp = enhance(small) if self.cfg["enhance"] else small
        t1 = time.perf_counter()
        boxes, confs, classes = self._run(inp, min_conf)
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        dets = []
        for xyxy, p, c in zip(boxes, confs, classes):
            x1, y1, x2, y2 = (float(v) / s for v in xyxy)
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w0), x2), min(float(h0), y2)
            dets.append(Detection(int(c), self.names[int(c)], round(float(p), 4),
                                  [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                                  [round(x1 / w0, 5), round(y1 / h0, 5), round(x2 / w0, 5), round(y2 / h0, 5)],
                                  round((x2 - x1) * (y2 - y1) / (w0 * h0), 5)))
        dets.sort(key=lambda d: -d.conf)
        return dets, inp, {"preprocess": round((t1 - t0) * 1000, 1), "inference": round((t2 - t1) * 1000, 1)}

    def predict(self, bgr: np.ndarray) -> list[Detection]:
        """Detections that pass the model's thresholds (per-class or global)."""
        dets, _, _ = self.detect_raw(bgr)
        return [d for d in dets if d.conf >= self.threshold(d.cls)]


# ------------------------------------------------------------------ image I/O
def decode(data: bytes) -> np.ndarray:
    """Image bytes -> BGR array, honouring EXIF rotation from phone cameras (as browsers do)."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return np.ascontiguousarray(np.asarray(img)[..., ::-1])


def read_image(path) -> np.ndarray:
    return decode(Path(path).read_bytes())


def _dms_to_deg(dms, ref) -> float:
    d, m, s = (float(x) for x in dms)
    deg = d + m / 60 + s / 3600
    return -deg if str(ref).upper() in ("S", "W") else deg


def read_gps(path) -> dict | None:
    """Decimal latitude/longitude from JPEG EXIF, or None when the photo has no GPS."""
    try:
        gps = Image.open(path).getexif().get_ifd(IFD.GPSInfo)
        if not gps or 2 not in gps or 4 not in gps:
            return None
        return {"lat": round(_dms_to_deg(gps[2], gps.get(1, "N")), 7),
                "lon": round(_dms_to_deg(gps[4], gps.get(3, "E")), 7)}
    except Exception:
        return None


# ------------------------------------------------------------------ drawing
def draw(bgr: np.ndarray, dets: list[Detection], colors: dict | None = None) -> np.ndarray:
    """Boxes with 'class conf' labels; highest-confidence label drawn last so it stays readable."""
    colors = colors or COLORS_BGR
    out = bgr.copy()
    h, w = out.shape[:2]
    lw = max(2, round(w / 320))
    fs = max(0.45, w / 1400)
    th = max(1, round(fs * 2))
    for d in sorted(dets, key=lambda d: d.conf):
        x1, y1, x2, y2 = (int(round(v)) for v in d.box_xyxy)
        c = colors.get(d.cls, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, lw)
        text = f"{PRETTY.get(d.name, d.name).split(' ', 1)[-1]} {d.conf:.2f}"
        (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ty = y1 - tht - base - 4 if y1 - tht - base - 4 >= 0 else y1
        cv2.rectangle(out, (x1, ty), (x1 + tw + 8, ty + tht + base + 4), c, -1)
        cv2.putText(out, text, (x1 + 4, ty + tht + 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return out

"""Module 4 - damage severity estimation.

Scores each detection 0-100 and labels it Low / Medium / High from:
  - damage type (potholes and alligator cracking are structurally worse than linear cracks),
  - extent (box area for potholes / alligator cracking, box length for linear cracks),
  - perspective (dashcam view: damage higher in the frame is farther away, so larger than it looks),
  - detection confidence (uncertain detections are never scored as severe).
Each image / frame also gets its worst severity, counts per level and a 0-100 condition index.

RDD2022 has no severity labels, so the rules are transparent heuristics kept in configs/severity.json
(adjustable, and calibratable later against engineer ratings).

Input: a Module 3 output folder (runs/predict/<name>/detections.json).
Output: severity.json (records + severity fields) and severity.csv (one row per detection) in that folder.

Examples:
    python src/severity.py runs/predict/ij_test
    python src/severity.py runs/predict/ij_test --perspective off
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ["Low", "Medium", "High"]


def load_config(path=None) -> dict:
    return json.loads(Path(path or ROOT / "configs" / "severity.json").read_text())


def perspective_factor(box_xyxy, width, height, cfg, enabled: bool) -> float:
    """Linear size correction for a flat road seen from a dashcam (1 = no correction)."""
    if not enabled:
        return 1.0
    p = cfg["perspective"]
    yh = p["horizon"] * height
    yc = (box_xyxy[1] + box_xyxy[3]) / 2
    if yc <= yh + 1e-6:
        return p["max_factor"]
    return min(p["max_factor"], max(1.0, (height - yh) / (yc - yh)))


def extent_score(det: dict, width: int, height: int, f: float, cfg: dict) -> float:
    """Log-scale position of the (perspective-corrected) size within this damage type's training range."""
    e = cfg["extent"]
    x1, y1, x2, y2 = det["box_xyxy"]
    if det["name"] in e["area_classes"]:
        size = max(det["area_ratio"], 1e-9) * f * f
    else:
        size = max(max((x2 - x1) / width, (y2 - y1) / height) * f, 1e-9)
    lo, hi = e["ranges"].get(det["name"], [0.01, 1.0])
    return min(1.0, max(0.0, math.log(size / lo) / math.log(hi / lo)))


def level_of(score: float, cfg: dict) -> str:
    lv = cfg["levels"]
    return "High" if score >= lv["high"] else "Medium" if score >= lv["medium"] else "Low"


def score_detection(det: dict, width: int, height: int, cfg: dict, perspective: bool) -> dict:
    f = perspective_factor(det["box_xyxy"], width, height, cfg, perspective)
    ext = extent_score(det, width, height, f, cfg)
    s = cfg["score"]
    w = cfg["type_weight"].get(det["name"], 0.7)
    score = 100 * w * (s["base"] + (1 - s["base"]) * ext) * (s["conf_floor"] + (1 - s["conf_floor"]) * det["conf"])
    score = round(min(100.0, max(0.0, score)), 1)
    return {"score": score, "level": level_of(score, cfg), "extent": round(ext, 3), "perspective_factor": round(f, 3)}


def score_record(rec: dict, cfg: dict, perspective: bool) -> dict:
    out = dict(rec)
    dets = []
    for d in rec["detections"]:
        d = dict(d)
        d["severity"] = score_detection(d, rec["width"], rec["height"], cfg, perspective)
        dets.append(d)
    out["detections"] = dets
    scores = [d["severity"]["score"] for d in dets]
    cond = 100.0
    for s in scores:
        cond *= 1 - cfg["condition"]["decay"] * s / 100
    counts = Counter(d["severity"]["level"] for d in dets)
    worst = max(scores, default=0.0)
    out["image_severity"] = {"max_score": worst, "max_level": level_of(worst, cfg) if dets else "None",
                             "counts": {lv: counts.get(lv, 0) for lv in LEVELS},
                             "condition_index": round(cond, 1)}
    return out


def perspective_for(run_dir: Path, mode: str, cfg: dict) -> bool:
    if mode in ("on", "off"):
        return mode == "on"
    summ = run_dir / "summary.json"
    model = json.loads(summ.read_text()).get("model") if summ.exists() else None
    return bool(cfg["perspective"]["models"].get(model, False))


def run_severity(run: Path, perspective: str = "auto", config=None) -> tuple[list[dict], bool]:
    """Score run/detections.json and write severity.json + severity.csv. Returns (records, perspective_on)."""
    cfg = load_config(config)
    persp = perspective_for(run, perspective, cfg)
    records = [score_record(r, cfg, persp) for r in json.loads((run / "detections.json").read_text())]
    (run / "severity.json").write_text(json.dumps(records, indent=1))
    with open(run / "severity.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["source", "frame", "time_s", "lat", "lon", "name", "conf", "score", "level", "extent",
                     "perspective_factor", "x1", "y1", "x2", "y2", "image_max_level", "image_condition_index"])
        for r in records:
            g = r["gps"] or {}
            for d in r["detections"]:
                s = d["severity"]
                wr.writerow([Path(r["source"]).name, r["frame"], r["time_s"], g.get("lat"), g.get("lon"), d["name"],
                             d["conf"], s["score"], s["level"], s["extent"], s["perspective_factor"], *d["box_xyxy"],
                             r["image_severity"]["max_level"], r["image_severity"]["condition_index"]])
    return records, persp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="Module 3 output folder containing detections.json")
    ap.add_argument("--perspective", choices=["auto", "on", "off"], default="auto",
                    help="auto: on for the dashcam India+Japan model, off for China (drone/close-up)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    records, persp = run_severity(run, args.perspective, args.config)
    det_levels = Counter(d["severity"]["level"] for r in records for d in r["detections"])
    img_levels = Counter(r["image_severity"]["max_level"] for r in records)
    conds = [r["image_severity"]["condition_index"] for r in records]
    print(f"Perspective correction: {'on' if persp else 'off'}")
    print(f"Detections by severity: {{'Low': {det_levels['Low']}, 'Medium': {det_levels['Medium']}, 'High': {det_levels['High']}}}")
    print(f"Images by worst severity: {{'None': {img_levels['None']}, 'Low': {img_levels['Low']}, "
          f"'Medium': {img_levels['Medium']}, 'High': {img_levels['High']}}}")
    if conds:
        print(f"Condition index: mean {sum(conds) / len(conds):.1f}, min {min(conds):.1f}")
    print(f"Saved: {run / 'severity.json'}, {run / 'severity.csv'}")


if __name__ == "__main__":
    main()

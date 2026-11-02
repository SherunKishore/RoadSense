"""Module 3 - run road-damage detection on images, folders or videos.

Outputs in runs/predict/<name>/:
  annotated images (same file names) or <video>_annotated.mp4
  detections.json   one record per image / processed frame (input for severity, priority, GPS modules):
                    {source, frame, time_s, width, height, gps, detections: [{cls, name, conf, box_xyxy,
                     box_norm, area_ratio}]}
  detections.csv    one row per detection
  summary.json      counts per class, images/frames with damage, speed, model and thresholds

Examples:
    python src/predict.py --source photo.jpg
    python src/predict.py --source data/india_japan/test/images --name ij_test
    python src/predict.py --source dashcam.mp4 --vid-stride 5
    python src/predict.py --source photos/ --model china --conf 0.3
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from detector import IMG_EXT, MODELS, Detector, draw, read_gps, read_image

ROOT = Path(__file__).resolve().parents[1]
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def out_dir(name: str) -> Path:
    base = ROOT / "runs" / "predict"
    d, i = base / name, 2
    while d.exists():
        d, i = base / f"{name}{i}", i + 1
    d.mkdir(parents=True)
    return d


def run_images(det: Detector, files: list[Path], out: Path, save_media: bool) -> list[dict]:
    records = []
    for n, f in enumerate(files, 1):
        try:
            img = read_image(f)
        except Exception as e:
            print(f"  skipped {f.name}: not a readable image ({e.__class__.__name__})")
            continue
        dets = det.predict(img)
        records.append({"source": str(f), "frame": None, "time_s": None, "width": img.shape[1], "height": img.shape[0],
                        "gps": read_gps(f), "detections": [d.to_dict() for d in dets]})
        if save_media:
            cv2.imwrite(str(out / f"{f.stem}.jpg"), draw(img, dets), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if n % 25 == 0 or n == len(files):
            print(f"  {n}/{len(files)} images")
    return records


def run_video(det: Detector, video: Path, out: Path, stride: int, save_media: bool) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(out / f"{video.stem}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) \
        if save_media else None
    records, last, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            last = det.predict(frame)
            records.append({"source": str(video), "frame": i, "time_s": round(i / fps, 3), "width": w, "height": h,
                            "gps": None, "detections": [d.to_dict() for d in last]})
        if writer is not None:
            writer.write(draw(frame, last))   # between processed frames, show the latest detections
        i += 1
        if i % 300 == 0:
            print(f"  {i}/{total or '?'} frames")
    cap.release()
    if writer is not None:
        writer.release()
    print(f"  {i} frames read, {len(records)} processed (every {stride}), fps {fps:.1f}")
    return records


def write_outputs(out: Path, records: list[dict], det: Detector, args, seconds: float) -> dict:
    (out / "detections.json").write_text(json.dumps(records, indent=1))
    with open(out / "detections.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["source", "frame", "time_s", "lat", "lon", "cls", "name", "conf", "x1", "y1", "x2", "y2", "area_ratio"])
        for r in records:
            g = r["gps"] or {}
            for d in r["detections"]:
                wr.writerow([Path(r["source"]).name, r["frame"], r["time_s"], g.get("lat"), g.get("lon"), d["cls"],
                             d["name"], d["conf"], *d["box_xyxy"], d["area_ratio"]])
    counts = Counter(d["name"] for r in records for d in r["detections"])
    summary = {
        "model": det.key, "title": det.cfg["title"], "weights": str(Path(det.cfg["weights"]).relative_to(ROOT)),
        "device": "cpu" if det.device == "cpu" else "gpu",
        "thresholds": ({det.names[c]: t for c, t in det.class_thresholds.items()} if det.per_class
                       else {"all_classes": det.conf}),
        "items": len(records), "items_with_damage": sum(1 for r in records if r["detections"]),
        "detections": sum(counts.values()), "per_class": {n: counts.get(n, 0) for n in det.names.values()},
        "with_gps": sum(1 for r in records if r["gps"]), "seconds": round(seconds, 1),
        "ms_per_item": round(1000 * seconds / max(1, len(records)), 1), "source": str(args.source),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="image file, folder of images, or video file")
    ap.add_argument("--model", default="india_japan", choices=list(MODELS))
    ap.add_argument("--conf", type=float, default=None, help="one confidence for all classes (turns off per-class)")
    ap.add_argument("--per-class", action="store_true", help="force per-class thresholds (default when available)")
    ap.add_argument("--device", default="auto", help="auto, cpu, or GPU index")
    ap.add_argument("--vid-stride", type=int, default=5, help="run detection on every Nth video frame")
    ap.add_argument("--name", default=None, help="output folder name under runs/predict/")
    ap.add_argument("--no-save-media", action="store_true", help="only write JSON/CSV, no annotated images/video")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")
    det = Detector(args.model, device=args.device, conf=args.conf, per_class=True if args.per_class else None)
    out = out_dir(args.name or src.stem)
    print(f"Model: {det.cfg['title']} | thresholds: "
          f"{'per-class ' + str({det.names[c]: t for c, t in det.class_thresholds.items()}) if det.per_class else det.conf}")
    t0 = time.perf_counter()
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.is_file())
        imgs = [p for p in files if p.suffix.lower() in IMG_EXT]
        for p in files:
            if p.suffix.lower() not in IMG_EXT:
                print(f"  skipped {p.name}: not an image file")
        records = run_images(det, imgs, out, not args.no_save_media)
    elif src.suffix.lower() in VID_EXT:
        records = run_video(det, src, out, max(1, args.vid_stride), not args.no_save_media)
    else:
        records = run_images(det, [src], out, not args.no_save_media)
    s = write_outputs(out, records, det, args, time.perf_counter() - t0)
    print(f"\n{s['items']} items, {s['items_with_damage']} with damage, {s['detections']} detections {s['per_class']}")
    print(f"GPS tagged: {s['with_gps']} | {s['ms_per_item']} ms per item | output: {out}")


if __name__ == "__main__":
    main()

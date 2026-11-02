"""RoadSense end-to-end pipeline (Modules 3-7) in one command.

  detection (M3) -> [GPS for video from a GPX track] -> severity (M4) -> repair priority (M5)
  -> HiResCAM explanations for the most urgent locations (M6) -> GPS map (M7) -> report.md

Everything lands in runs/roadsense/<name>/: annotated media, detections/severity/priority JSON+CSV,
repair_plan.md, explanations/, damage_map.html + damage_locations.geojson (when GPS exists),
report.md and summary.json.

Examples:
    python src/roadsense.py --source survey_photos/ --name ward12
    python src/roadsense.py --source dashcam.mp4 --gpx dashcam.gpx --video-start 2026-10-01T08:15:00Z
    python src/roadsense.py --source photos/ --road-class arterial --explain 10
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2

import predict as m3
from detector import IMG_EXT, MODELS, Detector, read_image
from geo_map import run_map
from gradcam import GradCAM, explanation_sheet
from priority import build_plan, load_config as load_priority_config, write_outputs as write_priority
from severity import run_severity

ROOT = Path(__file__).resolve().parents[1]


def explain_top(det: Detector, run: Path, plan: list[dict], n: int, video: Path | None, paths: dict) -> int:
    """HiResCAM sheets for the images/frames of the n most urgent locations (paths: file name -> full path)."""
    if n <= 0 or not plan:
        return 0
    cam, out, done = GradCAM(det), run / "explanations", 0
    out.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video)) if video else None
    for p in plan[:n]:
        items = [(f, None) for f in p["frames"][:1]] if p.get("frames") else [(None, s) for s in p["sources"][:1]]
        for frame, src in items:
            if frame is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, img = cap.read()
                if not ok:
                    continue
                stem = f"{video.stem}_frame{frame:06d}"
            else:
                path = Path(paths.get(src, src))
                if not path.exists():
                    continue
                img, stem = read_image(path), path.stem
            res = cam.explain(img, top_k=3)
            cv2.imwrite(str(out / f"rank{p['rank']:03d}_{p['priority']}_{stem}.jpg"), explanation_sheet(img, res),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            done += 1
    if cap:
        cap.release()
    return done


def write_report(run: Path, summary: dict, plan: list[dict]) -> None:
    s = summary
    lines = [f"# RoadSense report: {s['name']}", "",
             f"- Source: `{s['source']}` ({s['items']} {'frames analysed' if s['video'] else 'images'})",
             f"- Model: {s['model_title']}",
             f"- Damage found in {s['items_with_damage']} of {s['items']} {'frames' if s['video'] else 'images'}; "
             f"{s['detections']} detections: " + ", ".join(f"{n.split('_', 1)[1].replace('_', ' ').lower()} {c}"
                                                         for n, c in s["per_class"].items()),
             f"- Severity of detections: Low {s['severity']['Low']}, Medium {s['severity']['Medium']}, High {s['severity']['High']}",
             f"- Repair locations: {len(plan)} — " + ", ".join(f"{k} {s['priority'].get(k, 0)}" for k in ("P1", "P2", "P3", "P4")),
             f"- GPS: {s['map']['mapped']} locations mapped" + (" → `damage_map.html`" if s['map']['mapped'] else " (no GPS in source)"),
             f"- Explanations (HiResCAM) for the {s['explained']} most urgent locations → `explanations/`",
             f"- Processing time: {s['seconds']:.0f} s", "", "## Top repair jobs", "",
             "| Rank | Priority | Location | Damage | First action |", "|---|---|---|---|---|"]
    for p in plan[:15]:
        a = p["actions"][0]
        lines.append(f"| {p['rank']} | {p['priority']} ({p['deadline']}) | {p['label']} | "
                     f"{', '.join(f'{c}× {n}' for n, c in p['damages'].items())} | {a['damage']} ({a['severity']}): {a['action']} |")
    lines += ["", "Full plan: `repair_plan.md` / `priority.csv`. Detections: `detections.csv`, severity: `severity.csv`.",
              "", "_Severity and priority are rule-based estimates from image evidence (configs/severity.json, "
              "configs/priority.json); confirm on site before committing repair budgets._"]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="image, folder of images, or video")
    ap.add_argument("--model", default="india_japan", choices=list(MODELS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--conf", type=float, default=None, help="one confidence for all classes (default: per-class)")
    ap.add_argument("--vid-stride", type=int, default=5)
    ap.add_argument("--gpx", default=None, help="GPX track for a video source")
    ap.add_argument("--video-start", default=None, help="ISO time of the first video frame (default: track start)")
    ap.add_argument("--gpx-offset", type=float, default=0.0)
    ap.add_argument("--road-class", default="unknown", choices=["highway", "arterial", "local", "unknown"])
    ap.add_argument("--explain", type=int, default=5, help="HiResCAM sheets for the N most urgent locations (0 = off)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-save-media", action="store_true")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")
    t0 = time.perf_counter()
    base = ROOT / "runs" / "roadsense"
    name = args.name or src.stem
    run, i = base / name, 2
    while run.exists():
        run, i = base / f"{name}{i}", i + 1
    run.mkdir(parents=True)
    det = Detector(args.model, device=args.device, conf=args.conf)
    is_video = src.is_file() and src.suffix.lower() in m3.VID_EXT

    print("[1/5] detection")
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)
        records = m3.run_images(det, files, run, not args.no_save_media)
    elif is_video:
        records = m3.run_video(det, src, run, max(1, args.vid_stride), not args.no_save_media)
    else:
        records = m3.run_images(det, [src], run, not args.no_save_media)
    m3.write_outputs(run, records, det, args, time.perf_counter() - t0)

    print("[2/5] severity")
    sev, persp = run_severity(run)
    print(f"      perspective correction {'on' if persp else 'off'}")

    print("[3/5] repair priority")
    pcfg = load_priority_config()
    plan = build_plan(sev, pcfg, args.road_class)
    write_priority(run, plan, pcfg, args.road_class, len(sev))

    print("[4/5] GPS map")
    mapinfo = run_map(run, args.gpx if is_video else None, args.video_start, args.gpx_offset, args.road_class)
    plan = json.loads((run / "priority.json").read_text())   # rebuilt with GPS clusters if a track was given

    print("[5/5] explanations")
    sev = json.loads((run / "severity.json").read_text())
    paths = {Path(r["source"]).name: r["source"] for r in sev}
    if det.cfg.get("backend", "ultralytics") == "ultralytics":
        explained = explain_top(det, run, plan, args.explain, src if is_video else None, paths)
    else:
        explained = 0
        print(f"      skipped: HiResCAM is not available for {det.cfg['short']}")

    summ = json.loads((run / "summary.json").read_text())
    summary = {"name": run.name, "source": str(src), "video": is_video, "model": det.key, "model_title": det.cfg["title"],
               "items": len(sev), "items_with_damage": sum(1 for r in sev if r["detections"]),
               "detections": summ["detections"], "per_class": summ["per_class"],
               "severity": {lv: sum(1 for r in sev for d in r["detections"] if d["severity"]["level"] == lv)
                            for lv in ("Low", "Medium", "High")},
               "priority": dict(Counter(p["priority"] for p in plan)), "map": mapinfo, "explained": explained,
               "seconds": round(time.perf_counter() - t0, 1)}
    (run / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(run, summary, plan)
    print(f"\nDone in {summary['seconds']:.0f} s: {summary['items_with_damage']}/{summary['items']} with damage, "
          f"{len(plan)} repair locations {summary['priority']}, {mapinfo['mapped']} mapped, {explained} explained")
    print(f"Report: {run / 'report.md'}")


if __name__ == "__main__":
    main()

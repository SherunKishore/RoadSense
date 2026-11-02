"""Background survey jobs: many geotagged photos, or a dash-cam video with a GPX track.

Photos: each photo is analysed and geotagged from its EXIF; damaged photos are grouped into repair locations
(priority.group_records: 25 m GPS clusters, photos without GPS stay single) and each location becomes one report.
Video: every Nth frame is analysed; frame times are matched to the GPX track (src/geo_map.py, linear
interpolation), so locations are again 25 m clusters (5 s windows when there is no track).
A location's report shows its worst photo/frame; up to 6 other sightings are kept as extra media.
"""
import json
import shutil
import sys
import time
import traceback
import uuid
from datetime import timedelta
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector import decode  # noqa: E402
from geo_map import interpolate, parse_time, read_gpx  # noqa: E402

import geo  # noqa: E402
from engine import Engine, fit, jpeg  # noqa: E402
from store import Store, now  # noqa: E402

MAX_MEDIA = 6


def _update(store: Store, sid: int, **kw) -> None:
    if "stats" in kw and not isinstance(kw["stats"], str):
        kw["stats"] = json.dumps(kw["stats"])
    with store.db() as c:
        c.execute(f"UPDATE surveys SET {', '.join(f'{k} = ?' for k in kw)} WHERE id = ?", [*kw.values(), sid])


def save_images(store: Store, engine: Engine, bgr, detections: list[dict], original: bytes | None = None) -> dict:
    """Original (as uploaded, or a JPEG of the frame), annotated (max 1600 px) and thumbnail (480 px) files."""
    key = uuid.uuid4().hex
    orig = f"uploads/{key}.jpg"
    store.file(orig).write_bytes(original if original is not None else jpeg(fit(bgr, 2560), 90))
    ann = fit(engine.annotate(bgr, detections), 1600)
    store.file(f"annotated/{key}.jpg").write_bytes(jpeg(ann, 85))
    store.file(f"thumbs/{key}.jpg").write_bytes(jpeg(fit(ann, 480), 78))
    return {"image": orig, "annotated": f"annotated/{key}.jpg", "thumb": f"thumbs/{key}.jpg"}


def report_fields(plan: dict | None, rec: dict) -> dict:
    """Columns of `reports` derived from a priority-plan item and the record whose photo is shown."""
    names = sorted({d["name"] for d in rec["detections"]})
    if not plan:
        return {"priority": None, "priority_name": None, "deadline": None, "priority_score": None, "worst_level": None,
                "worst_damage": None, "condition_index": rec["image_severity"]["condition_index"], "n_detections": 0,
                "damage_types": names, "plan": None}
    names = sorted(set(names) | {d for d in plan.get("_names", [])})
    return {"priority": plan["priority"], "priority_name": plan["priority_name"], "deadline": plan["deadline"],
            "priority_score": plan["priority_score"], "worst_level": plan["worst"]["level"],
            "worst_damage": plan["worst"]["damage"], "condition_index": rec["image_severity"]["condition_index"],
            "n_detections": len(rec["detections"]), "damage_types": names,
            "plan": {k: v for k, v in plan.items() if not k.startswith("_")}}


def run_survey(store: Store, engine: Engine, sid: int, user_id: int, files: list[Path], kind: str, road_class: str,
               gpx: Path | None = None, video_start: str | None = None, stride: int = 5) -> None:
    work = store.root / "surveys" / str(sid)
    try:
        t0 = time.time()
        records = _photos(store, engine, sid, files, t0) if kind == "photos" else \
            _video(store, engine, sid, files[0], gpx, video_start, stride, t0)
        _update(store, sid, message="Grouping damage into repair locations", progress=0.92)
        located = engine.plan_for_records(records, road_class)
        ids = []
        for loc, plan in located:
            members = loc["records"]
            best = max(members, key=lambda r: max((d["severity"]["score"] for d in r["detections"]), default=0))
            plan["_names"] = sorted({d["name"] for r in members for d in r["detections"]})
            img = cv2.imread(str(best["_image"])) if best.get("_frame_file") else decode(best["_image"].read_bytes())
            original = None if best.get("_frame_file") else best["_image"].read_bytes()
            files_ = save_images(store, engine, img, best["detections"], original)
            media = []
            for r in sorted((m for m in members if m is not best), key=lambda m: -len(m["detections"]))[:MAX_MEDIA]:
                im = cv2.imread(str(r["_image"])) if r.get("_frame_file") else decode(r["_image"].read_bytes())
                key = f"media/{uuid.uuid4().hex}.jpg"
                store.file(key).write_bytes(jpeg(fit(engine.annotate(im, r["detections"]), 1280), 82))
                media.append({"path": key, "label": r["_label"], "n_detections": len(r["detections"])})
            g = loc["gps"]
            acc = [r["_gps_meta"].get("accuracy_m") for r in members if r.get("_gps_meta", {}).get("accuracy_m")]
            rep = {**files_, **report_fields(plan, best), "survey_id": sid, "lat": g["lat"] if g else None,
                   "lon": g["lon"] if g else None, "gps_source": best.get("_gps_source") if g else None,
                   "accuracy_m": max(acc) if acc else None, "road_class": road_class,
                   "taken_at": best.get("_taken_at"), "width": best["width"], "height": best["height"],
                   "source_name": best["_label"], "quality": best.get("_quality")}
            ids.append(store.insert_report(rep, best["detections"], user_id, media))
        _update(store, sid, message="Looking up street addresses", progress=0.96)
        for rid in ids:
            r = store.one("SELECT lat, lon FROM reports WHERE id = ?", (rid,))
            if r["lat"] is not None:
                addr = geo.reverse_geocode(r["lat"], r["lon"], store.geocode_get, store.geocode_put)
                if addr:
                    with store.db() as c:
                        c.execute("UPDATE reports SET address = ? WHERE id = ?", (addr, rid))
        stats = {"items": len(records), "with_damage": sum(1 for r in records if r["detections"]),
                 "with_gps": sum(1 for r in records if r.get("gps")), "reports": len(ids),
                 "reports_without_location": sum(1 for loc, _ in located if not loc["gps"]),
                 "seconds": round(time.time() - t0, 1)}
        _update(store, sid, status="done", progress=1.0, finished_at=now(), stats=stats,
                message=f"{len(ids)} repair location{'s' if len(ids) != 1 else ''} from {len(records)} "
                        f"{'photos' if kind == 'photos' else 'frames'}")
    except Exception as e:
        traceback.print_exc()
        _update(store, sid, status="failed", finished_at=now(), message=f"{e.__class__.__name__}: {e}")
    finally:
        shutil.rmtree(work / "frames", ignore_errors=True)


def _eta(done: int, total: int, t0: float) -> str:
    if not done:
        return ""
    left = (time.time() - t0) / done * (total - done)
    return f", about {int(left // 60)} min {int(left % 60)} s left" if left >= 1 else ""


def _photos(store: Store, engine: Engine, sid: int, files: list[Path], t0: float) -> list[dict]:
    records = []
    for i, f in enumerate(files, 1):
        data = f.read_bytes()
        try:
            bgr = decode(data)
        except Exception:
            continue
        meta = geo.exif_gps(data) or {}
        gps = {"lat": meta["lat"], "lon": meta["lon"]} if "lat" in meta else None
        res = engine.analyse(bgr, source=f.name, gps=gps)
        rec = res["record"]
        label = f.name[6:] if len(f.name) > 6 and f.name[:5].isdigit() and f.name[5] == "_" else f.name  # drop upload index
        rec.update(source=label, _image=f, _label=label, _gps_meta=meta, _gps_source="exif", _taken_at=meta.get("taken_at"),
                   _quality=res["quality"])
        records.append(rec)
        _update(store, sid, done=i, progress=round(0.9 * i / len(files), 4),
                message=f"Analysed {i} of {len(files)} photos{_eta(i, len(files), t0)}")
    return records


def _video(store: Store, engine: Engine, sid: int, video: Path, gpx: Path | None, video_start: str | None,
           stride: int, t0: float) -> list[dict]:
    track = read_gpx(gpx) if gpx else None
    start = parse_time(video_start) if video_start else (track[0][0] if track else None)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("cannot open the video (use MP4/H.264, AVI or MOV)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    todo = max(1, (total + stride - 1) // stride) if total else 0
    _update(store, sid, total=todo)
    frames_dir = store.root / "surveys" / str(sid) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records, i, n = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            t = round(i / fps, 3)
            gps = interpolate(track, start + timedelta(seconds=t)) if track else None
            res = engine.analyse(frame, source=video.name, gps=gps)
            rec = res["record"]
            rec.update(frame=i, time_s=t, _label=f"{video.name} {int(t // 60):02d}:{t % 60:05.2f}",
                       _gps_source="gpx" if gps else None, _gps_meta={}, _quality=res["quality"],
                       _taken_at=(start + timedelta(seconds=t)).isoformat() if start else None)
            if rec["detections"]:
                p = frames_dir / f"{i:07d}.jpg"
                cv2.imwrite(str(p), fit(frame, 1920), [cv2.IMWRITE_JPEG_QUALITY, 92])
                # boxes were found on the full frame; keep them consistent with the stored (possibly smaller) image
                s = min(1.0, 1920 / max(frame.shape[:2]))
                if s < 1:
                    for d in rec["detections"]:
                        d["box_xyxy"] = [round(v * s, 1) for v in d["box_xyxy"]]
                    rec["width"], rec["height"] = round(frame.shape[1] * s), round(frame.shape[0] * s)
                rec.update(_image=p, _frame_file=True)
            records.append(rec)
            n += 1
            if n % 5 == 0 or n == todo:
                _update(store, sid, done=n, progress=round(0.9 * n / max(1, todo), 4),
                        message=f"Analysed {n} of {todo or '?'} frames{_eta(n, todo, t0) if todo else ''}")
        i += 1
    cap.release()
    if track:
        with_gps = sum(1 for r in records if r.get("gps"))
        if not with_gps:
            raise ValueError("no video frame falls inside the GPX track's time range; check the video start time")
    return records

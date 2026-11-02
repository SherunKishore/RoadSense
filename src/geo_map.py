"""Module 7 - GPS-based road-damage mapping.

GPS sources:
  - photos: EXIF GPS, already read by src/predict.py into each record's "gps";
  - video:  a GPX track (--gpx). Frame times are matched to the track by timestamp
            (--video-start, or the track's first point + --gpx-offset seconds) and interpolated
            linearly between track points; frames outside the track get no GPS.

Outputs in the run folder:
  damage_map.html            interactive map (Leaflet + OpenStreetMap), repair locations coloured by
                             priority, with damage, actions, deadline and photo thumbnails in popups
  damage_locations.geojson   the same locations as GeoJSON points (QGIS, Google Earth, geojson.io)
When a GPX track is given, severity.json records are geotagged and the repair plan (Module 5) is
rebuilt with GPS clustering first.

Examples:
    python src/geo_map.py runs/predict/city_photos
    python src/geo_map.py runs/predict/dashcam --gpx drive.gpx --video-start 2026-10-01T08:15:00Z
"""
import argparse
import html
import json
import xml.etree.ElementTree as ET
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from priority import build_plan, load_config as load_priority_config, write_outputs as write_priority

ROOT = Path(__file__).resolve().parents[1]
P_COLORS = {"P1": "#d62728", "P2": "#ff7f0e", "P3": "#e0b000", "P4": "#2ca02c"}


# ------------------------------------------------------------------ GPX
def parse_time(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def read_gpx(path) -> list[tuple[datetime, float, float]]:
    """Track points (time, lat, lon) sorted by time; namespace-agnostic (GPX 1.0 / 1.1)."""
    pts = []
    for el in ET.parse(path).getroot().iter():
        if el.tag.split("}")[-1] in ("trkpt", "rtept") and "lat" in el.attrib:
            t = next((c.text for c in el if c.tag.split("}")[-1] == "time"), None)
            if t:
                pts.append((parse_time(t), float(el.attrib["lat"]), float(el.attrib["lon"])))
    pts.sort(key=lambda p: p[0])
    if len(pts) < 2:
        raise SystemExit(f"{path}: need at least 2 timestamped track points, found {len(pts)}")
    return pts


def interpolate(track, t: datetime):
    """Linear interpolation of (lat, lon) at time t; None outside the track."""
    times = [p[0] for p in track]
    if t < times[0] or t > times[-1]:
        return None
    i = bisect_left(times, t)
    if times[i] == t:
        return {"lat": round(track[i][1], 7), "lon": round(track[i][2], 7)}
    (t0, la0, lo0), (t1, la1, lo1) = track[i - 1], track[i]
    f = (t - t0).total_seconds() / (t1 - t0).total_seconds()
    return {"lat": round(la0 + f * (la1 - la0), 7), "lon": round(lo0 + f * (lo1 - lo0), 7)}


def geotag_video(records: list[dict], track, start: datetime) -> int:
    n = 0
    for r in records:
        if r.get("frame") is not None and r.get("time_s") is not None:
            r["gps"] = interpolate(track, start + timedelta(seconds=r["time_s"]))
            n += r["gps"] is not None
    return n


# ------------------------------------------------------------------ outputs
def to_geojson(plan: list[dict]) -> dict:
    feats = []
    for p in plan:
        if not p.get("gps"):
            continue
        props = {k: p[k] for k in ("rank", "priority", "priority_name", "deadline", "priority_score", "label")}
        props.update({"damages": p["damages"], "worst": p["worst"], "sources": p["sources"],
                      "actions": [f"{a['damage']} ({a['severity']}): {a['action']}" for a in p["actions"]],
                      "marker-color": P_COLORS[p["priority"]]})
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["gps"]["lon"], p["gps"]["lat"]]},
                      "properties": props})
    return {"type": "FeatureCollection", "features": feats}


def _thumb(path: Path, width: int = 280) -> str | None:
    """Small JPEG embedded as a data URI, so the map stays a single self-contained file."""
    import base64
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def map_html(plan: list[dict], run: Path, track=None, title: str = "RoadSense damage map", max_thumbs: int = 400) -> str:
    pts = []
    for p in plan:
        if not p.get("gps"):
            continue
        thumb = None
        if len(pts) < max_thumbs:   # plan is in priority order, so the most urgent locations get photos
            src = next((run / f"{Path(s).stem}.jpg" for s in p["sources"] if (run / f"{Path(s).stem}.jpg").exists()), None)
            thumb = _thumb(src) if src else None
        pts.append({"lat": p["gps"]["lat"], "lon": p["gps"]["lon"], "p": p["priority"], "rank": p["rank"],
                    "score": p["priority_score"], "deadline": p["deadline"], "name": p["priority_name"],
                    "damages": ", ".join(f"{c}× {n}" for n, c in p["damages"].items()),
                    "actions": [f"{a['damage']} ({a['severity']}): {a['action']}" for a in p["actions"]],
                    "thumbs": [thumb] if thumb else [], "label": p["label"]})
    line = [[la, lo] for _, la, lo in track] if track else []
    counts = {k: sum(1 for q in pts if q["p"] == k) for k in P_COLORS}
    data = json.dumps({"points": pts, "track": line, "colors": P_COLORS, "counts": counts})
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body, #map {{ height: 100%; margin: 0; font: 14px/1.4 system-ui, sans-serif; }}
  .panel {{ background: #fff; padding: 8px 12px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.3); }}
  .panel h3 {{ margin: 0 0 4px; font-size: 15px; }} .sw {{ display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:6px; vertical-align:-1px; }}
  .pop b {{ font-size: 14px; }} .pop ul {{ margin: 4px 0 4px 16px; padding: 0; }} .pop img {{ width: 220px; display:block; margin-top:4px; border-radius:4px; }}
</style></head><body><div id="map"></div>
<script>
const D = {data};
const map = L.map("map");
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}}).addTo(map);
const layer = L.featureGroup().addTo(map);
if (D.track.length) L.polyline(D.track, {{color: "#3b82f6", weight: 3, opacity: .6}}).addTo(layer);
const esc = s => String(s).replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c]));
const order = {{P4: 0, P3: 1, P2: 2, P1: 3}};
D.points.sort((a, b) => order[a.p] - order[b.p]).forEach(q => {{
  L.circleMarker([q.lat, q.lon], {{radius: q.p === "P1" ? 10 : q.p === "P2" ? 8 : 6, color: "#222", weight: 1,
    fillColor: D.colors[q.p], fillOpacity: .9}}).addTo(layer).bindPopup(
    `<div class="pop"><b>#${{q.rank}} ${{q.p}} ${{esc(q.name)}}</b> · repair ${{esc(q.deadline)}}<br>` +
    `score ${{q.score}} · ${{esc(q.damages)}}<ul>${{q.actions.map(a => `<li>${{esc(a)}}</li>`).join("")}}</ul>` +
    `<small>${{esc(q.label)}}</small>${{q.thumbs.map(t => `<img src="${{t}}" alt="annotated photo">`).join("")}}</div>`,
    {{maxWidth: 260}});
}});
if (layer.getLayers().length) map.fitBounds(layer.getBounds().pad(0.15)); else map.setView([20.6, 78.9], 5);
const legend = L.control({{position: "bottomleft"}});
legend.onAdd = () => {{ const d = L.DomUtil.create("div", "panel");
  d.innerHTML = "<h3>Repair priority</h3>" + [["P1", "Urgent · 48 h"], ["P2", "High · 7 days"], ["P3", "Medium · 30 days"], ["P4", "Routine"]]
    .map(([k, t]) => `<div><span class="sw" style="background:${{D.colors[k]}}"></span>${{k}} ${{t}} (${{D.counts[k]}})</div>`).join("");
  return d; }};
legend.addTo(map);
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="folder with severity.json / priority.json (Modules 4-5)")
    ap.add_argument("--gpx", default=None, help="GPX track for video runs")
    ap.add_argument("--video-start", default=None, help="ISO time of the video's first frame, e.g. 2026-10-01T08:15:00Z")
    ap.add_argument("--gpx-offset", type=float, default=0.0, help="seconds after the track's first point (if no --video-start)")
    ap.add_argument("--road-class", default="unknown", choices=["highway", "arterial", "local", "unknown"])
    args = ap.parse_args()
    run_map(Path(args.run_dir), args.gpx, args.video_start, args.gpx_offset, args.road_class)


def run_map(run: Path, gpx=None, video_start=None, gpx_offset: float = 0.0, road_class: str = "unknown") -> dict:
    """Geotag video frames from a GPX track (rebuilding the repair plan), then write the map + GeoJSON."""
    track = None
    if gpx:
        track = read_gpx(gpx)
        start = parse_time(video_start) if video_start else track[0][0] + timedelta(seconds=gpx_offset)
        records = json.loads((run / "severity.json").read_text())
        n = geotag_video(records, track, start)
        (run / "severity.json").write_text(json.dumps(records, indent=1))
        cfg = load_priority_config()
        plan = build_plan(records, cfg, road_class)
        write_priority(run, plan, cfg, road_class, len(records))
        print(f"GPX: {len(track)} track points; geotagged {n} of {sum(1 for r in records if r.get('frame') is not None)} "
              f"video frames; repair plan rebuilt with GPS clustering")
    plan = json.loads((run / "priority.json").read_text())
    located = [p for p in plan if p.get("gps")]
    if not located:
        print("No GPS positions in this run (photos without EXIF GPS and no --gpx track), so no map was made.")
        return {"mapped": 0, "without_gps": len(plan)}
    (run / "damage_locations.geojson").write_text(json.dumps(to_geojson(plan), indent=1))
    (run / "damage_map.html").write_text(map_html(plan, run, track), encoding="utf-8")
    c = Counter(p["priority"] for p in located)
    print(f"Mapped {len(located)} repair locations ({dict(sorted(c.items()))}); {len(plan) - len(located)} without GPS")
    print(f"Saved: {run / 'damage_map.html'}, {run / 'damage_locations.geojson'}")
    return {"mapped": len(located), "without_gps": len(plan) - len(located), "by_priority": dict(c)}


if __name__ == "__main__":
    main()

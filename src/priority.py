"""Module 5 - repair-priority recommendations.

Turns Module 4 severity output into a ranked repair plan:
  1. group detections into locations: GPS clusters (within radius_m), else video time windows,
     else one location per photo;
  2. score each location: worst severity + a bonus per extra Medium/High damage + a pothole safety
     bonus, times an optional road-class factor;
  3. map to P1 Urgent (48 h) / P2 High (7 days) / P3 Medium (30 days) / P4 Routine, with overrides
     (e.g. any High-severity pothole is always P1);
  4. attach a recommended treatment per damage type at its worst severity.
Rules live in configs/priority.json.

Input: a folder with severity.json (from src/severity.py).
Output in that folder: priority.json, priority.csv, repair_plan.md.

Examples:
    python src/priority.py runs/predict/ij_test
    python src/priority.py runs/predict/city_survey --road-class arterial
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEV_RANK = {"Low": 0, "Medium": 1, "High": 2}
P_RANK = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
SHORT = {"D00_Longitudinal_Crack": "longitudinal crack", "D10_Transverse_Crack": "transverse crack",
         "D20_Alligator_Crack": "alligator crack", "D40_Pothole": "pothole"}


def load_config(path=None) -> dict:
    return json.loads(Path(path or ROOT / "configs" / "priority.json").read_text())


def haversine_m(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def group_records(records: list[dict], cfg: dict) -> list[dict]:
    """Return locations: {kind, records, gps, label}.

    GPS records use leader clustering: in survey order, a record joins the nearest existing location
    whose anchor (first point) is within radius_m, else starts a new one. Unlike single-linkage this
    cannot chain a whole street of photos taken every ~10 m into one giant location.
    """
    g = cfg["grouping"]
    locs = []
    gps_recs = sorted((r for r in records if r.get("gps")),
                      key=lambda r: (r["source"], r["frame"] if r.get("frame") is not None else -1))
    if gps_recs:
        anchors, clusters = [], []
        for r in gps_recs:
            near = [(haversine_m(a, r["gps"]), k) for k, a in enumerate(anchors)]
            d, k = min(near) if near else (float("inf"), -1)
            if d <= g["radius_m"]:
                clusters[k].append(r)
            else:
                anchors.append(r["gps"])
                clusters.append([r])
        for members in clusters:
            lat = sum(r["gps"]["lat"] for r in members) / len(members)
            lon = sum(r["gps"]["lon"] for r in members) / len(members)
            locs.append({"kind": "gps", "records": members, "gps": {"lat": round(lat, 7), "lon": round(lon, 7)},
                         "label": f"{lat:.5f}, {lon:.5f}"})
    video = defaultdict(list)
    for r in records:
        if r.get("gps"):
            continue
        if r.get("frame") is not None:
            video[(r["source"], int(r["time_s"] // g["video_window_s"]))].append(r)
        else:
            locs.append({"kind": "photo", "records": [r], "gps": None, "label": Path(r["source"]).name})
    for (src, w), members in sorted(video.items()):
        t0, t1 = w * g["video_window_s"], (w + 1) * g["video_window_s"]
        locs.append({"kind": "video", "records": members, "gps": None,
                     "label": f"{Path(src).name} {t0 // 60:02.0f}:{t0 % 60:02.0f}-{t1 // 60:02.0f}:{t1 % 60:02.0f}"})
    return locs


def level_for(score: float, cfg: dict) -> dict:
    for lv in cfg["levels"]:
        if score >= lv["min_score"]:
            return lv
    return cfg["levels"][-1]


def prioritise(loc: dict, cfg: dict, road_class: str) -> dict | None:
    dets = [d for r in loc["records"] for d in r["detections"]]
    if not dets:
        return None
    s = cfg["score"]
    worst = max(dets, key=lambda d: d["severity"]["score"])
    # The same damage is usually seen in several consecutive frames / nearby photos, so counts per location
    # use the most seen in any single image or frame (a lower bound on distinct damages), not the sum.
    serious = max(sum(1 for d in r["detections"] if d["severity"]["level"] in ("Medium", "High"))
                  for r in loc["records"])
    per_type = Counter()
    for r in loc["records"]:
        for name, c in Counter(SHORT.get(d["name"], d["name"]) for d in r["detections"]).items():
            per_type[name] = max(per_type[name], c)
    bonus = min(s["density_bonus_max"], s["density_bonus_per_extra"] * max(0, serious - 1))
    safety = s["pothole_safety_bonus"] if any(d["name"] == "D40_Pothole" and d["severity"]["level"] != "Low"
                                              for d in dets) else 0
    factor = cfg["road_factor"].get(road_class, 1.0)
    score = round(min(100.0, (worst["severity"]["score"] + bonus + safety) * factor), 1)
    lv = level_for(score, cfg)
    # per damage type: worst severity and its action
    worst_by_type = {}
    for d in dets:
        cur = worst_by_type.get(d["name"])
        if cur is None or d["severity"]["score"] > cur["severity"]["score"]:
            worst_by_type[d["name"]] = d
    reasons = []
    for name, d in worst_by_type.items():   # overrides: e.g. a High pothole is always at least P1
        forced = cfg["overrides"].get(name, {}).get(d["severity"]["level"])
        if forced and P_RANK[forced] < P_RANK[lv["level"]]:
            lv = next(x for x in cfg["levels"] if x["level"] == forced)
            reasons.append(f"{d['severity']['level']} {SHORT.get(name, name)}")
    actions = [{"damage": SHORT.get(n, n), "severity": d["severity"]["level"], "score": d["severity"]["score"],
                "action": cfg["actions"].get(n, {}).get(d["severity"]["level"], "Inspect")}
               for n, d in sorted(worst_by_type.items(), key=lambda kv: -kv[1]["severity"]["score"])]
    return {"label": loc["label"], "kind": loc["kind"], "gps": loc["gps"], "priority": lv["level"],
            "priority_name": lv["name"], "deadline": lv["deadline"], "priority_score": score,
            "override": ", ".join(reasons) or None,
            "damages": dict(per_type),
            "sightings": len(dets), "seen_in": sum(1 for r in loc["records"] if r["detections"]),
            "severity_counts": dict(Counter(d["severity"]["level"] for d in dets)),
            "worst": {"damage": SHORT.get(worst["name"], worst["name"]), "score": worst["severity"]["score"],
                      "level": worst["severity"]["level"]},
            "actions": actions, "sources": sorted({Path(r["source"]).name for r in loc["records"]}),
            "frames": sorted({r["frame"] for r in loc["records"] if r.get("frame") is not None}) or None}


def build_plan(records: list[dict], cfg: dict, road_class: str = "unknown") -> list[dict]:
    plan = [p for p in (prioritise(l, cfg, road_class) for l in group_records(records, cfg)) if p]
    plan.sort(key=lambda p: (P_RANK[p["priority"]], -p["priority_score"]))
    for i, p in enumerate(plan, 1):
        p["rank"] = i
    return plan


def write_outputs(run: Path, plan: list[dict], cfg: dict, road_class: str, n_records: int) -> None:
    (run / "priority.json").write_text(json.dumps(plan, indent=1))
    with open(run / "priority.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["rank", "priority", "deadline", "priority_score", "location", "lat", "lon", "damages",
                     "worst_damage", "worst_severity", "actions"])
        for p in plan:
            g = p["gps"] or {}
            wr.writerow([p["rank"], p["priority"], p["deadline"], p["priority_score"], p["label"], g.get("lat"),
                         g.get("lon"), "; ".join(f"{n} x{c}" for n, c in p["damages"].items()),
                         p["worst"]["damage"], p["worst"]["level"],
                         " | ".join(f"{a['damage']} ({a['severity']}): {a['action']}" for a in p["actions"])])
    counts = Counter(p["priority"] for p in plan)
    lines = ["# Repair plan", "",
             f"{n_records} images/frames analysed, {len(plan)} location{'s' if len(plan) != 1 else ''} "
             f"need{'s' if len(plan) == 1 else ''} attention (road class: {road_class}).", "",
             "| Priority | Meaning | Locations |", "|---|---|---|"]
    for lv in cfg["levels"]:
        lines.append(f"| {lv['level']} {lv['name']} | {lv['deadline']} | {counts.get(lv['level'], 0)} |")
    lines += ["", "| Rank | Priority | Location | Damage found | Recommended action |", "|---|---|---|---|---|"]
    for p in plan[:100]:
        dmg = ", ".join(f"{c}× {n}" for n, c in p["damages"].items())
        if p["seen_in"] > 1:
            dmg += f" (seen in {p['seen_in']} {'frames' if p['kind'] == 'video' or p.get('frames') else 'images'})"
        act = "<br>".join(f"**{a['damage']} ({a['severity']})**: {a['action']}" for a in p["actions"])
        lines.append(f"| {p['rank']} | {p['priority']} ({p['deadline']}) | {p['label']} | {dmg} | {act} |")
    if len(plan) > 100:
        lines.append(f"\n…{len(plan) - 100} more locations in priority.csv")
    (run / "repair_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="folder containing severity.json")
    ap.add_argument("--road-class", default="unknown", choices=["highway", "arterial", "local", "unknown"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run, cfg = Path(args.run_dir), load_config(args.config)
    records = json.loads((run / "severity.json").read_text())
    plan = build_plan(records, cfg, args.road_class)
    write_outputs(run, plan, cfg, args.road_class, len(records))
    counts = Counter(p["priority"] for p in plan)
    print(f"{len(records)} images/frames -> {len(plan)} location{'s' if len(plan) != 1 else ''} with damage: "
          + ", ".join(f"{lv['level']} {counts.get(lv['level'], 0)}" for lv in cfg["levels"]))
    for p in plan[:5]:
        print(f"  #{p['rank']} {p['priority']} score {p['priority_score']:5.1f} {p['label']}: "
              f"{p['actions'][0]['damage']} ({p['actions'][0]['severity']}) -> {p['actions'][0]['action']}")
    print(f"Saved: {run / 'priority.json'}, priority.csv, repair_plan.md")


if __name__ == "__main__":
    main()

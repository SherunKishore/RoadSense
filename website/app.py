"""RoadSense website: road-damage reporting with GPS mapping, repair priorities and explainable AI.

Run (from the FY1 folder):
  python website/app.py                      http://localhost:8000 (this computer only)
  python website/app.py --https              https://<LAN IP>:8443 for phones on the same Wi-Fi
                                             (needs website/certs/, made by: bash scripts/make_cert.sh)
  python website/app.py --device auto        use the GPU when it is free (default: CPU)
First start: open the site from this computer and create the admin account, or use
`python website/manage.py create-user`. Data lives in website/data/ (SQLite + images).
"""
import argparse
import csv
import io
import ipaddress
import json
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
sys.path.insert(0, str(WEB))
sys.path.insert(0, str(ROOT / "src"))
import auth  # noqa: E402
import geo  # noqa: E402
from detector import PRETTY, decode  # noqa: E402
from engine import CLASS_HEX, Engine, fit, jpeg  # noqa: E402
from store import STATUSES, Store, now  # noqa: E402
from surveys import report_fields, run_survey, save_images  # noqa: E402

MAX_PHOTO = 25 * 2 ** 20
MAX_VIDEO = 500 * 2 ** 20
MAX_SURVEY_PHOTOS = 500
ROAD_CLASSES = ("unknown", "highway", "arterial", "local")
GPS_SOURCES = ("exif", "device", "map")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--device", default="cpu", help="cpu (default), auto (GPU if available) or a GPU index")
ap.add_argument("--https", action="store_true", help="serve HTTPS on all interfaces for phones on the same Wi-Fi")
ap.add_argument("--host", default=None, help="default 127.0.0.1, or 0.0.0.0 with --https")
ap.add_argument("--port", type=int, default=None, help="default 8000, or 8443 with --https")
ap.add_argument("--data", default=str(WEB / "data"), help="folder for the database and images")
ARGS, _ = ap.parse_known_args()

STORE = Store(Path(ARGS.data))
ENGINE = Engine(device=ARGS.device)
JOBS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="survey")
SECURE = ARGS.https
MODEL_CARD = json.loads((ROOT / "configs" / "ensemble.json").read_text())
TEST_REPORT = ROOT / "runs" / "ensemble" / "test.json"

with STORE.db() as _c:   # surveys interrupted by a restart cannot resume
    _c.execute("UPDATE surveys SET status = 'failed', message = 'Stopped because the server restarted.', "
               "finished_at = ? WHERE status IN ('queued', 'running')", (now(),))

app = FastAPI(title="RoadSense", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")

CSP = ("default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; "
       "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
       "font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob: https://*.tile.openstreetmap.org; "
       "connect-src 'self' data: blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")


@app.middleware("http")
async def security(request: Request, call_next):
    # API writes must carry a custom header, which other sites cannot send without CORS (CSRF protection)
    if request.url.path.startswith("/api/") and request.method not in ("GET", "HEAD") \
            and request.headers.get("x-roadsense") != "1":
        return JSONResponse({"error": "Missing X-RoadSense header."}, status_code=403)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Content-Security-Policy"] = CSP
    resp.headers["Permissions-Policy"] = "geolocation=(self), camera=(self)"
    if request.url.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-store")
    elif request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"     # revalidate (cheap 304s), so updated scripts load at once
    return resp


def err(msg: str, code: int = 400):
    raise HTTPException(status_code=code, detail=msg)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


# ------------------------------------------------------------------ auth
def current_user(request: Request) -> dict:
    u = auth.user_for(STORE, request.cookies.get(auth.COOKIE))
    if not u:
        err("Please sign in.", 401)
    return u


def admin_user(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        err("Only admins can do that.", 403)
    return user


def is_local(request: Request) -> bool:
    try:
        return ipaddress.ip_address(request.client.host).is_loopback
    except (ValueError, AttributeError):
        return False


def set_session(resp: Response, token: str) -> None:
    resp.set_cookie(auth.COOKIE, token, max_age=auth.SESSION_DAYS * 86400, httponly=True, samesite="lax",
                    secure=SECURE, path="/")


@app.get("/api/session")
def session(request: Request):
    u = auth.user_for(STORE, request.cookies.get(auth.COOKIE))
    no_users = STORE.one("SELECT COUNT(*) AS n FROM users")["n"] == 0
    return {"user": u, "setup_needed": no_users, "setup_allowed": no_users and is_local(request),
            "secure": SECURE}


@app.post("/api/setup")
async def setup(request: Request):
    """Create the first admin. Only while there are no accounts, and only from this computer."""
    if STORE.one("SELECT COUNT(*) AS n FROM users")["n"]:
        err("Setup is already done; sign in instead.", 409)
    if not is_local(request):
        err("Create the first admin from the computer running RoadSense.", 403)
    b = await request.json()
    try:
        auth.create_user(STORE, b.get("username", ""), b.get("name", ""), "admin", b.get("password", ""))
    except ValueError as e:
        err(str(e))
    u, token = auth.login(STORE, b["username"], b["password"], "setup")
    resp = JSONResponse({"user": u})
    set_session(resp, token)
    return resp


@app.post("/api/login")
async def login(request: Request):
    b = await request.json()
    u, token = auth.login(STORE, str(b.get("username", "")), str(b.get("password", "")),
                          f"{str(b.get('username', '')).lower()}@{request.client.host if request.client else '?'}")
    if not u:
        err(token, 401)
    resp = JSONResponse({"user": u})
    set_session(resp, token)
    return resp


@app.post("/api/logout")
def logout(request: Request):
    auth.logout(STORE, request.cookies.get(auth.COOKIE))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(auth.COOKIE, path="/")
    return resp


@app.post("/api/me/password")
async def change_password(request: Request, user: dict = Depends(current_user)):
    b = await request.json()
    full = STORE.one("SELECT pw_hash FROM users WHERE id = ?", (user["id"],))
    if not auth.check_password(str(b.get("current", "")), full["pw_hash"]):
        err("Current password is wrong.")
    try:
        auth.set_password(STORE, user["id"], str(b.get("new", "")), end_sessions=False)
    except ValueError as e:
        err(str(e))
    return {"ok": True}


@app.get("/api/users")
def users(user: dict = Depends(admin_user)):
    rows = STORE.all("SELECT u.id, u.username, u.name, u.role, u.active, u.created_at, u.last_login, "
                     "(SELECT COUNT(*) FROM reports r WHERE r.user_id = u.id AND r.deleted = 0) AS reports "
                     "FROM users u ORDER BY u.role, u.username")
    return {"users": rows}


@app.post("/api/users")
async def add_user(request: Request, user: dict = Depends(admin_user)):
    b = await request.json()
    try:
        uid = auth.create_user(STORE, str(b.get("username", "")), str(b.get("name", "")), str(b.get("role", "")),
                               str(b.get("password", "")))
    except ValueError as e:
        err(str(e))
    return {"id": uid}


@app.patch("/api/users/{uid}")
async def edit_user(uid: int, request: Request, user: dict = Depends(admin_user)):
    b = await request.json()
    target = STORE.one("SELECT * FROM users WHERE id = ?", (uid,))
    if not target:
        err("No such user.", 404)
    if uid == user["id"] and (b.get("active") is False or b.get("role") == "inspector"):
        err("You can't deactivate or demote your own account.")
    if b.get("password"):
        try:
            auth.set_password(STORE, uid, str(b["password"]))
        except ValueError as e:
            err(str(e))
    with STORE.db() as c:
        if "name" in b and str(b["name"]).strip():
            c.execute("UPDATE users SET name = ? WHERE id = ?", (str(b["name"]).strip(), uid))
        if b.get("role") in auth.ROLES:
            c.execute("UPDATE users SET role = ? WHERE id = ?", (b["role"], uid))
        if "active" in b:
            c.execute("UPDATE users SET active = ? WHERE id = ?", (1 if b["active"] else 0, uid))
            if not b["active"]:
                c.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
    return {"ok": True}


# ------------------------------------------------------------------ info
@app.get("/api/info")
def info(user: dict = Depends(current_user)):
    names = ENGINE.names
    test = json.loads(TEST_REPORT.read_text()) if TEST_REPORT.exists() else None
    card = {"title": ENGINE.det.cfg["title"], "members": [m.cfg["title"] for m in ENGINE.det.members],
            "wbf_iou": MODEL_CARD["wbf_iou"], "thresholds": MODEL_CARD["thresholds"], "metrics": MODEL_CARD["metrics"],
            "val": MODEL_CARD["val"], "test": test, "quality": MODEL_CARD["quality"], "tuned": MODEL_CARD.get("tuned")}
    return {"classes": [{"id": c, "name": n, "label": PRETTY.get(n, n).split(" ", 1)[-1], "code": n[:3],
                         "color": CLASS_HEX[c]} for c, n in names.items()],
            "priority_levels": ENGINE.pri_cfg["levels"], "statuses": list(STATUSES), "road_classes": list(ROAD_CLASSES),
            "severity_levels": ENGINE.sev_cfg["levels"], "model": card,
            "device": "CPU" if ENGINE.det.device == "cpu" else "GPU", "secure": SECURE}


@app.get("/api/geocode")
def geocode(lat: float, lon: float, user: dict = Depends(current_user)):
    if not geo.valid(lat, lon):
        err("Invalid coordinates.")
    return {"address": geo.reverse_geocode(lat, lon, STORE.geocode_get, STORE.geocode_put)}


# ------------------------------------------------------------------ analysis
async def read_upload(f: UploadFile, limit: int) -> bytes:
    data = await f.read(limit + 1)
    if len(data) > limit:
        err(f"File too large (max {limit // 2 ** 20} MB).", 413)
    return data


def _tmp(token: str) -> tuple[Path, Path]:
    if not (len(token) == 32 and all(ch in "0123456789abcdef" for ch in token)):
        err("Unknown analysis.", 404)
    return STORE.file(f"tmp/{token}.img"), STORE.file(f"tmp/{token}.json")


def _clean_tmp(max_age_h: float = 6) -> None:
    cutoff = time.time() - max_age_h * 3600
    for p in (STORE.root / "tmp").glob("*"):
        if p.stat().st_mtime < cutoff:
            p.unlink(missing_ok=True)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), road_class: str = Form("unknown"), user: dict = Depends(current_user)):
    data = await read_upload(file, MAX_PHOTO)
    try:
        bgr = decode(data)
    except Exception:
        err("That file isn't a photo RoadSense can read (use JPEG, PNG or WebP; iPhone HEIC: export as JPEG).")
    meta = geo.exif_gps(data) or {}
    gps = {"lat": meta["lat"], "lon": meta["lon"]} if "lat" in meta else None
    res = ENGINE.analyse(bgr, source=file.filename or "photo", gps=gps,
                         road_class=road_class if road_class in ROAD_CLASSES else "unknown")
    token = uuid.uuid4().hex
    img_p, js_p = _tmp(token)
    img_p.write_bytes(data)
    keep = {k: res[k] for k in ("width", "height", "detections", "uncertain", "image_severity", "plan", "quality", "ms")}
    js_p.write_text(json.dumps({**keep, "record": res["record"], "exif": meta, "filename": file.filename}))
    _clean_tmp()
    return {"token": token, "exif": meta or None, "filename": file.filename, **keep}


@app.get("/api/analyze/{token}/enhanced")
def analyze_enhanced(token: str, user: dict = Depends(current_user)):
    img_p, _ = _tmp(token)
    if not img_p.exists():
        err("This analysis has expired; analyse the photo again.", 404)
    from preprocess import enhance
    bgr = fit(decode(img_p.read_bytes()), 1280)
    return Response(content=jpeg(enhance(bgr), 85), media_type="image/jpeg")


@app.post("/api/analyze/{token}/explain")
def analyze_explain(token: str, user: dict = Depends(current_user)):
    img_p, _ = _tmp(token)
    if not img_p.exists():
        err("This analysis has expired; analyse the photo again.", 404)
    return ENGINE.explain(decode(img_p.read_bytes()))


@app.post("/api/analyze/{token}/plan")
async def analyze_plan(token: str, request: Request, user: dict = Depends(current_user)):
    """Repair priority for another road type (no re-detection)."""
    _, js_p = _tmp(token)
    if not js_p.exists():
        err("This analysis has expired; analyse the photo again.", 404)
    b = await request.json()
    road = b.get("road_class") if b.get("road_class") in ROAD_CLASSES else "unknown"
    return {"plan": ENGINE.plan(json.loads(js_p.read_text())["record"], road)}


@app.post("/api/reports")
async def create_report(request: Request, user: dict = Depends(current_user)):
    b = await request.json()
    img_p, js_p = _tmp(str(b.get("token", "")))
    if not js_p.exists():
        err("This analysis has expired; analyse the photo again.", 404)
    a = json.loads(js_p.read_text())
    lat, lon, src = b.get("lat"), b.get("lon"), b.get("gps_source")
    if lat is not None or lon is not None:
        if not geo.valid(lat, lon) or src not in GPS_SOURCES:
            err("The location is not valid; pick it on the map.")
        lat, lon = round(float(lat), 7), round(float(lon), 7)
    else:
        src = None
    acc = b.get("accuracy_m")
    acc = round(float(acc), 1) if isinstance(acc, (int, float)) and acc >= 0 else None
    road = b.get("road_class") if b.get("road_class") in ROAD_CLASSES else "unknown"
    rec = a["record"]
    rec["gps"] = {"lat": lat, "lon": lon} if lat is not None else None
    plan = ENGINE.plan(rec, road)
    bgr = decode(img_p.read_bytes())
    files = save_images(STORE, ENGINE, bgr, rec["detections"], img_p.read_bytes())
    rep = {**files, **report_fields(plan, rec), "lat": lat, "lon": lon, "gps_source": src, "accuracy_m": acc,
           "road_class": road, "taken_at": (a.get("exif") or {}).get("taken_at"), "width": a["width"],
           "height": a["height"], "source_name": a.get("filename"), "quality": a["quality"],
           "note": str(b.get("note") or "").strip()[:2000] or None}
    if lat is not None:
        rep["address"] = geo.reverse_geocode(lat, lon, STORE.geocode_get, STORE.geocode_put)
    rid = STORE.insert_report(rep, rec["detections"], user["id"])
    img_p.unlink(missing_ok=True); js_p.unlink(missing_ok=True)
    return {"id": rid}


# ------------------------------------------------------------------ reports
def filters(request: Request) -> dict:
    q = request.query_params
    f = {k: q.get(k) for k in ("status", "priority", "damage", "since", "until", "survey_id", "q", "has_location")
         if q.get(k)}
    if q.get("mine") in ("1", "true"):
        f["user_id"] = current_user(request)["id"]
    return f


@app.get("/api/reports")
def list_reports(request: Request, page: int = 1, per_page: int = Query(25, le=200), sort: str = "priority",
                 user: dict = Depends(current_user)):
    rows, total = STORE.query_reports(filters(request), per_page, (max(1, page) - 1) * per_page, sort)
    return {"rows": rows, "total": total, "page": page, "per_page": per_page}


def _report_or_404(rid: int) -> dict:
    r = STORE.report(rid)
    if not r:
        err("Report not found.", 404)
    return r


@app.get("/api/reports/{rid}")
def get_report(rid: int, user: dict = Depends(current_user)):
    r = _report_or_404(rid)
    nearby = []
    if r["lat"] is not None:
        d = 0.0005   # ~55 m box, then exact distance
        for o in STORE.all("SELECT id, lat, lon, priority, status, worst_damage, created_at FROM reports "
                           "WHERE deleted = 0 AND id != ? AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
                           (rid, r["lat"] - d, r["lat"] + d, r["lon"] - d, r["lon"] + d)):
            m = geo.haversine_m(r["lat"], r["lon"], o["lat"], o["lon"])
            if m <= ENGINE.pri_cfg["grouping"]["radius_m"]:
                nearby.append({**o, "distance_m": round(m, 1)})
    r["nearby"] = sorted(nearby, key=lambda o: o["distance_m"])
    r["can_edit"] = user["role"] == "admin" or r["user_id"] == user["id"]
    for k in ("image", "annotated", "thumb"):
        r.pop(k, None)
    return r


@app.patch("/api/reports/{rid}")
async def edit_report(rid: int, request: Request, user: dict = Depends(current_user)):
    r = _report_or_404(rid)
    b = await request.json()
    is_admin = user["role"] == "admin"
    if not (is_admin or r["user_id"] == user["id"]):
        err("You can only edit your own reports.", 403)
    changes, events = {}, []
    if "status" in b:
        if not is_admin:
            err("Only admins can change the status.", 403)
        if b["status"] not in STATUSES:
            err("Unknown status.")
        if b["status"] != r["status"]:
            changes["status"] = b["status"]
            events.append(("status", {"from": r["status"], "to": b["status"], "comment": (b.get("comment") or "")[:500] or None}))
    if "note" in b:
        note = str(b["note"] or "").strip()[:2000] or None
        if note != r["note"]:
            changes["note"] = note
            events.append(("note", None))
    if "lat" in b or "lon" in b:
        if not geo.valid(b.get("lat"), b.get("lon")):
            err("The location is not valid.")
        lat, lon = round(float(b["lat"]), 7), round(float(b["lon"]), 7)
        changes.update(lat=lat, lon=lon, gps_source=b.get("gps_source") if b.get("gps_source") in GPS_SOURCES else "map",
                       accuracy_m=b.get("accuracy_m") if isinstance(b.get("accuracy_m"), (int, float)) else None,
                       address=geo.reverse_geocode(lat, lon, STORE.geocode_get, STORE.geocode_put))
        events.append(("location", {"from": [r["lat"], r["lon"]], "to": [lat, lon], "source": changes["gps_source"]}))
    if b.get("road_class") in ROAD_CLASSES and b["road_class"] != r["road_class"] and r["n_detections"]:
        rec = {"source": r["source_name"] or "photo", "frame": None, "time_s": None, "width": r["width"],
               "height": r["height"], "gps": None,
               "detections": [{"cls": d["cls"], "name": d["name"], "conf": d["conf"], "box_xyxy": d["box"],
                               "severity": {"score": d["severity_score"], "level": d["severity_level"]}}
                              for d in r["detections"]]}
        if r["plan"] and len(r["plan"].get("sources", [])) <= 1:   # single photo: recompute; survey locations keep theirs
            plan = ENGINE.plan(rec, b["road_class"])
            if plan:
                changes.update(priority=plan["priority"], priority_name=plan["priority_name"], deadline=plan["deadline"],
                               priority_score=plan["priority_score"], plan=json.dumps(plan))
        changes["road_class"] = b["road_class"]
        events.append(("road_class", {"from": r["road_class"], "to": b["road_class"]}))
    elif b.get("road_class") in ROAD_CLASSES and b["road_class"] != r["road_class"]:
        changes["road_class"] = b["road_class"]
        events.append(("road_class", {"from": r["road_class"], "to": b["road_class"]}))
    if changes:
        changes["updated_at"] = now()
        with STORE.db() as c:
            c.execute(f"UPDATE reports SET {', '.join(f'{k} = ?' for k in changes)} WHERE id = ?", [*changes.values(), rid])
            for action, detail in events:
                STORE.event(c, rid, user["id"], action, detail)
    return {"ok": True, "changed": [k for k in changes if k != "updated_at"]}


@app.delete("/api/reports/{rid}")
def delete_report(rid: int, user: dict = Depends(admin_user)):
    _report_or_404(rid)
    with STORE.db() as c:
        c.execute("UPDATE reports SET deleted = 1, updated_at = ? WHERE id = ?", (now(), rid))
        STORE.event(c, rid, user["id"], "deleted")
    return {"ok": True}


@app.get("/api/reports/{rid}/image/{kind}")
def report_image(rid: int, kind: str, user: dict = Depends(current_user)):
    if kind not in ("original", "annotated", "thumb"):
        err("Unknown image.", 404)
    r = STORE.one("SELECT image, annotated, thumb FROM reports WHERE id = ? AND deleted = 0", (rid,))
    if not r:
        err("Report not found.", 404)
    p = STORE.file(r[{"original": "image", "annotated": "annotated", "thumb": "thumb"}[kind]])
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/media/{mid}")
def media(mid: int, user: dict = Depends(current_user)):
    m = STORE.one("SELECT m.path FROM report_media m JOIN reports r ON r.id = m.report_id "
                  "WHERE m.id = ? AND r.deleted = 0", (mid,))
    if not m:
        err("Not found.", 404)
    return FileResponse(STORE.file(m["path"]), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.post("/api/reports/{rid}/explain")
def report_explain(rid: int, user: dict = Depends(current_user)):
    r = STORE.one("SELECT image FROM reports WHERE id = ? AND deleted = 0", (rid,))
    if not r:
        err("Report not found.", 404)
    return ENGINE.explain(decode(STORE.file(r["image"]).read_bytes()))


@app.get("/api/map")
def map_points(request: Request, user: dict = Depends(current_user)):
    f = {**filters(request), "has_location": "1"}
    rows, total = STORE.query_reports(f, None, 0, "priority")
    return {"points": [{k: r[k] for k in ("id", "lat", "lon", "priority", "priority_name", "status", "worst_damage",
                                          "worst_level", "address", "created_at", "damage_types", "gps_source",
                                          "accuracy_m", "deadline")} for r in rows],
            "without_location": STORE.query_reports({**filters(request), "has_location": "0"}, 1)[1]}


def _export_rows(request: Request):
    rows, _ = STORE.query_reports(filters(request), None, 0, "priority")
    return rows


@app.get("/api/export.csv")
def export_csv(request: Request, user: dict = Depends(current_user)):
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["id", "created_at", "status", "priority", "priority_name", "deadline", "priority_score", "latitude",
                 "longitude", "gps_source", "gps_accuracy_m", "address", "road_class", "worst_damage", "worst_severity",
                 "damage_types", "detections", "condition_index", "reporter", "note"])
    for r in _export_rows(request):
        wr.writerow([r["id"], r["created_at"], r["status"], r["priority"], r["priority_name"], r["deadline"],
                     r["priority_score"], r["lat"], r["lon"], r["gps_source"], r["accuracy_m"], r["address"],
                     r["road_class"], r["worst_damage"], r["worst_level"], "; ".join(r["damage_types"]),
                     r["n_detections"], r["condition_index"], r["reporter"], r["note"]])
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Response(buf.getvalue().encode("utf-8-sig"), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="roadsense_reports_{stamp}.csv"'})


@app.get("/api/export.geojson")
def export_geojson(request: Request, user: dict = Depends(current_user)):
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
              "properties": {k: r[k] for k in ("id", "created_at", "status", "priority", "priority_name", "deadline",
                                               "priority_score", "address", "road_class", "worst_damage", "worst_level",
                                               "damage_types", "gps_source", "accuracy_m", "note")}}
             for r in _export_rows(request) if r["lat"] is not None]
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Response(json.dumps({"type": "FeatureCollection", "features": feats}, indent=1),
                    media_type="application/geo+json",
                    headers={"Content-Disposition": f'attachment; filename="roadsense_reports_{stamp}.geojson"'})


@app.get("/api/stats")
def stats(user: dict = Depends(current_user)):
    q = STORE.all
    live = "deleted = 0"
    open_ = f"{live} AND status IN ('open', 'in_progress')"
    month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    since = (datetime.now(timezone.utc) - timedelta(days=29)).date().isoformat()
    kpi = q(f"SELECT COUNT(*) AS total, SUM(status IN ('open', 'in_progress')) AS open, "
            f"SUM(status IN ('open', 'in_progress') AND priority = 'P1') AS urgent, "
            f"SUM(status = 'repaired' AND updated_at >= ?) AS repaired_month, "
            f"SUM(lat IS NULL) AS no_location, AVG(condition_index) AS condition FROM reports WHERE {live}", (month,))[0]
    pot = q(f"SELECT COUNT(*) AS n FROM reports WHERE {open_} AND damage_types LIKE '%D40_Pothole%'")[0]["n"]
    by_pri = {r["priority"] or "none": r["n"] for r in q(f"SELECT priority, COUNT(*) AS n FROM reports WHERE {open_} GROUP BY priority")}
    by_status = {r["status"]: r["n"] for r in q(f"SELECT status, COUNT(*) AS n FROM reports WHERE {live} GROUP BY status")}
    by_damage = {}
    for r in q(f"SELECT damage_types FROM reports WHERE {open_}"):
        for n in json.loads(r["damage_types"] or "[]"):
            by_damage[n] = by_damage.get(n, 0) + 1
    per_day = {r["d"]: r["n"] for r in q(f"SELECT substr(created_at, 1, 10) AS d, COUNT(*) AS n FROM reports "
                                         f"WHERE {live} AND created_at >= ? GROUP BY d", (since,))}
    days = [(datetime.now(timezone.utc) - timedelta(days=i)).date().isoformat() for i in range(29, -1, -1)]
    urgent, _ = STORE.query_reports({"status": "open,in_progress", "priority": "P1,P2"}, 6, 0, "priority")
    recent_surveys = q("SELECT id, name, kind, status, progress, created_at, stats FROM surveys ORDER BY id DESC LIMIT 3")
    return {"kpi": {**kpi, "potholes_open": pot}, "by_priority": by_pri, "by_status": by_status, "by_damage": by_damage,
            "per_day": [{"date": d, "n": per_day.get(d, 0)} for d in days], "urgent": urgent,
            "surveys": [{**s, "stats": json.loads(s["stats"]) if s["stats"] else None} for s in recent_surveys]}


# ------------------------------------------------------------------ surveys
@app.post("/api/surveys")
async def create_survey(name: str = Form(""), kind: str = Form(...), road_class: str = Form("unknown"),
                        stride: int = Form(5), video_start: str = Form(""), photos: list[UploadFile] | None = File(None),
                        video: UploadFile | None = File(None), gpx: UploadFile | None = File(None),
                        user: dict = Depends(current_user)):
    if kind not in ("photos", "video"):
        err("Choose photos or video.")
    road_class = road_class if road_class in ROAD_CLASSES else "unknown"
    with STORE.db() as c:
        sid = c.execute("INSERT INTO surveys (user_id, name, kind, status, created_at, message) VALUES (?, ?, ?, ?, ?, ?)",
                        (user["id"], name.strip()[:120] or f"Survey {datetime.now():%d %b %Y %H:%M}", kind, "queued",
                         now(), "Uploading")).lastrowid
    work = STORE.root / "surveys" / str(sid)
    work.mkdir(parents=True, exist_ok=True)
    try:
        files, gpx_path = [], None
        if kind == "photos":
            if not photos:
                err("Add some photos.")
            if len(photos) > MAX_SURVEY_PHOTOS:
                err(f"At most {MAX_SURVEY_PHOTOS} photos per survey.")
            for i, f in enumerate(photos):
                p = work / f"{i:05d}_{Path(f.filename or 'photo').stem[:60]}{Path(f.filename or '.jpg').suffix[:6].lower()}"
                p.write_bytes(await read_upload(f, MAX_PHOTO))
                files.append(p)
        else:
            if not video:
                err("Add a video.")
            stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in Path(video.filename or "video").stem)[:60]
            p = work / f"{stem or 'video'}{Path(video.filename or '.mp4').suffix[:6].lower()}"
            p.write_bytes(await read_upload(video, MAX_VIDEO))
            files.append(p)
            if gpx and gpx.filename:
                gpx_path = work / "track.gpx"
                gpx_path.write_bytes(await read_upload(gpx, 20 * 2 ** 20))
                try:
                    from geo_map import read_gpx
                    read_gpx(gpx_path)
                except BaseException as e:   # read_gpx exits with SystemExit on a bad track
                    err(f"The GPX file has no usable timestamped track ({e}).")
            if video_start:
                try:
                    from geo_map import parse_time
                    parse_time(video_start)
                except Exception:
                    err("Video start time must look like 2026-10-21T08:15:00Z.")
    except HTTPException:
        shutil.rmtree(work, ignore_errors=True)
        with STORE.db() as c:
            c.execute("DELETE FROM surveys WHERE id = ?", (sid,))
        raise
    with STORE.db() as c:
        c.execute("UPDATE surveys SET status = 'running', total = ?, message = 'Waiting for the model' WHERE id = ?",
                  (len(files) if kind == "photos" else None, sid))
    JOBS.submit(run_survey, STORE, ENGINE, sid, user["id"], files, kind, road_class, gpx_path, video_start or None,
                max(1, min(60, stride)))
    return {"id": sid}


@app.get("/api/surveys")
def list_surveys(user: dict = Depends(current_user)):
    rows = STORE.all("SELECT s.*, u.name AS user FROM surveys s LEFT JOIN users u ON u.id = s.user_id "
                     "ORDER BY s.id DESC LIMIT 50")
    return {"surveys": [{**s, "stats": json.loads(s["stats"]) if s["stats"] else None} for s in rows]}


@app.get("/api/surveys/{sid}")
def get_survey(sid: int, user: dict = Depends(current_user)):
    s = STORE.one("SELECT * FROM surveys WHERE id = ?", (sid,))
    if not s:
        err("Survey not found.", 404)
    s["stats"] = json.loads(s["stats"]) if s["stats"] else None
    return s


# ------------------------------------------------------------------ pages
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((WEB / "static" / "index.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-cache"})


@app.get("/favicon.ico")
def favicon():
    return FileResponse(WEB / "static" / "img" / "favicon.svg", media_type="image/svg+xml")


if __name__ == "__main__":
    host = ARGS.host or ("0.0.0.0" if ARGS.https else "127.0.0.1")
    port = ARGS.port or (8443 if ARGS.https else 8000)
    kw = {}
    if ARGS.https:
        cert, key = WEB / "certs" / "cert.pem", WEB / "certs" / "key.pem"
        if not (cert.exists() and key.exists()):
            sys.exit("No certificate yet. Run: bash scripts/make_cert.sh")
        kw = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    print(f"RoadSense on {'https' if ARGS.https else 'http'}://{'localhost' if host == '127.0.0.1' else host}:{port} "
          f"(models on {'CPU' if ENGINE.det.device == 'cpu' else 'GPU'})")
    uvicorn.run(app, host=host, port=port, log_level="warning", **kw)

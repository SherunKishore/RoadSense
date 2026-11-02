"""RoadSense website tests (no pytest needed):  python website/tests/test_website.py

Covers EXIF GPS parsing, accounts and roles, the full report flow through the HTTP API (analyse -> save -> list ->
status -> export -> delete), a background photo survey, the comparison API, and that the ensemble is deterministic
and its members match the stand-alone detectors. Uses a temporary data folder; the real website/data is untouched.
"""
import io
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

from PIL import Image
from PIL.ExifTags import IFD

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
TMP = Path(tempfile.mkdtemp(prefix="roadsense_test_"))
sys.argv = [sys.argv[0], "--data", str(TMP / "data")]
sys.path.insert(0, str(WEB))
sys.path.insert(0, str(ROOT / "src"))

import auth  # noqa: E402
import geo  # noqa: E402
from store import Store  # noqa: E402

DEMO = sorted((ROOT / "data" / "demo" / "survey_synthetic_gps").glob("*.jpg"))
TEST_IMG = ROOT / "data" / "india_japan" / "test" / "images"


def jpeg_with_gps(gps: dict | None, size=(96, 72)) -> bytes:
    img = Image.new("RGB", size, (90, 90, 90))
    ex = Image.Exif()
    if gps is not None:
        ex.get_ifd(IFD.GPSInfo).update(gps)
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=ex)
    return buf.getvalue()


# ------------------------------------------------------------------ GPS
def test_exif_north_east():
    g = geo.exif_gps(jpeg_with_gps({1: "N", 2: (12.0, 55.0, 40.44), 3: "E", 4: (77.0, 37.0, 37.56), 5: b"\x00", 6: 920.5, 31: 4.0}))
    assert abs(g["lat"] - (12 + 55 / 60 + 40.44 / 3600)) < 1e-6 and abs(g["lon"] - (77 + 37 / 60 + 37.56 / 3600)) < 1e-6, g
    assert g["altitude_m"] == 920.5 and g["accuracy_m"] == 4.0, g


def test_exif_south_west():
    g = geo.exif_gps(jpeg_with_gps({1: "S", 2: (33.0, 52.0, 4.8), 3: "W", 4: (151.0, 12.0, 26.4)}))
    assert g["lat"] < 0 and g["lon"] < 0 and abs(g["lat"] + 33.868) < 1e-3 and abs(g["lon"] + 151.2073) < 1e-3, g


def test_exif_missing_zero_and_bad():
    assert geo.exif_gps(jpeg_with_gps(None)) is None
    z = geo.exif_gps(jpeg_with_gps({1: "N", 2: (0.0, 0.0, 0.0), 3: "E", 4: (0.0, 0.0, 0.0)}))
    assert z and "lat" not in z and z.get("invalid_gps"), z
    bad = geo.exif_gps(jpeg_with_gps({1: "N", 2: (95.0, 0.0, 0.0), 3: "E", 4: (10.0, 0.0, 0.0)}))
    assert bad is not None and "lat" not in bad, bad
    assert geo.exif_gps(b"not an image") is None


def test_exif_matches_pipeline_reader():
    from detector import read_gps
    for p in DEMO[:5]:
        a, b = geo.exif_gps(p.read_bytes()), read_gps(p)
        assert abs(a["lat"] - b["lat"]) < 1e-7 and abs(a["lon"] - b["lon"]) < 1e-7, (p.name, a, b)


def test_valid_and_distance():
    assert geo.valid(12.9, 77.6) and not geo.valid(0, 0) and not geo.valid(91, 0) and not geo.valid("x", 1)
    assert abs(geo.haversine_m(12.9279, 77.6271, 12.9279 + 1 / 111195 * 25, 77.6271) - 25) < 0.1


# ------------------------------------------------------------------ accounts
def test_passwords_and_users():
    st = Store(TMP / "authdb")
    h = auth.hash_password("roadsense42")
    assert auth.check_password("roadsense42", h) and not auth.check_password("roadsense43", h)
    assert auth.password_problem("short1") and auth.password_problem("onlyletters") and not auth.password_problem("letters123")
    auth.create_user(st, "alice", "Alice", "inspector", "letters123")
    for bad in (("alice", "inspector", "letters123"), ("x", "inspector", "letters123"), ("bob", "boss", "letters123")):
        try:
            auth.create_user(st, bad[0], "B", bad[1], bad[2])
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    u, tok = auth.login(st, "alice", "letters123", "k1")
    assert u and auth.user_for(st, tok)["username"] == "alice"
    auth.logout(st, tok)
    assert auth.user_for(st, tok) is None
    for _ in range(5):
        assert auth.login(st, "alice", "wrong-pass1", "k2")[0] is None
    u, msg = auth.login(st, "alice", "letters123", "k2")
    assert u is None and "Too many" in msg


# ------------------------------------------------------------------ HTTP API
def client():
    from fastapi.testclient import TestClient
    import app as webapp
    return TestClient(webapp.app), webapp


H = {"X-RoadSense": "1"}


def test_api_flow():
    c, webapp = client()
    s = c.get("/api/session").json()
    assert s["user"] is None and s["setup_needed"] is True
    r = c.post("/api/setup", json={"username": "boss", "name": "Boss", "password": "letters123"}, headers=H)
    assert r.status_code == 403, r.text                                     # not from this computer
    auth.create_user(webapp.STORE, "boss", "Boss", "admin", "letters123")
    auth.create_user(webapp.STORE, "insp", "Inspector", "inspector", "letters123")
    assert c.get("/api/reports").status_code == 401
    assert c.post("/api/login", json={"username": "boss", "password": "letters123"}).status_code == 403   # no CSRF header
    assert c.post("/api/login", json={"username": "boss", "password": "nope12345"}, headers=H).status_code == 401
    assert c.post("/api/login", json={"username": "insp", "password": "letters123"}, headers=H).status_code == 200
    info = c.get("/api/info").json()
    assert len(info["classes"]) == 4 and info["model"]["test"] is not None

    # analyse a test photo, then save it with a device location
    img = sorted(TEST_IMG.glob("India_*.jpg"))[3]
    a = c.post("/api/analyze", files={"file": (img.name, img.read_bytes(), "image/jpeg")}, data={"road_class": "local"}, headers=H)
    assert a.status_code == 200, a.text
    a = a.json()
    assert a["exif"] is None and a["quality"]["warnings"] == [] and all(0 <= d["conf"] <= 1 for d in a["detections"])
    assert c.post(f"/api/analyze/{a['token']}/plan", json={"road_class": "highway"}, headers=H).status_code == 200
    assert c.get(f"/api/analyze/{a['token']}/enhanced").headers["content-type"] == "image/jpeg"
    bad = c.post("/api/reports", json={"token": a["token"], "lat": 0, "lon": 0, "gps_source": "device"}, headers=H)
    assert bad.status_code == 400
    webapp.geo.reverse_geocode = lambda *a_, **k: "Test Road, Bengaluru"   # no network calls in tests
    rid = c.post("/api/reports", json={"token": a["token"], "lat": 12.9716, "lon": 77.5946, "gps_source": "device",
                                       "accuracy_m": 8.4, "road_class": "local", "note": "<b>near school</b>"}, headers=H).json()["id"]
    rep = c.get(f"/api/reports/{rid}").json()
    assert rep["gps_source"] == "device" and rep["accuracy_m"] == 8.4 and rep["address"] == "Test Road, Bengaluru"
    assert rep["note"] == "<b>near school</b>" and rep["n_detections"] == len(a["detections"]) and rep["can_edit"]
    assert c.get(f"/api/reports/{rid}/image/thumb").status_code == 200
    for probe in (f"/api/reports/{rid}/image/../../roadsense.db", f"/api/reports/{rid}/image/..%2F..%2Froadsense.db"):
        r = c.get(probe)
        assert r.status_code in (404, 422) and b"SQLite" not in r.content, (probe, r.status_code)

    # a photo with EXIF GPS keeps it; a second report 10 m away shows up as nearby
    gps_img = DEMO[0]
    a2 = c.post("/api/analyze", files={"file": (gps_img.name, gps_img.read_bytes(), "image/jpeg")}, headers=H).json()
    assert a2["exif"]["lat"] and a2["exif"]["lon"]
    rid2 = c.post("/api/reports", json={"token": a2["token"], "lat": a2["exif"]["lat"], "lon": a2["exif"]["lon"],
                                        "gps_source": "exif"}, headers=H).json()["id"]
    a3 = c.post("/api/analyze", files={"file": (gps_img.name, gps_img.read_bytes(), "image/jpeg")}, headers=H).json()
    rid3 = c.post("/api/reports", json={"token": a3["token"], "lat": a2["exif"]["lat"] + 0.00009, "lon": a2["exif"]["lon"],
                                        "gps_source": "map"}, headers=H).json()["id"]
    near = c.get(f"/api/reports/{rid3}").json()["nearby"]
    assert [n["id"] for n in near] == [rid2] and 9 < near[0]["distance_m"] < 11, near

    # roles: inspector can't change status or delete
    assert c.patch(f"/api/reports/{rid}", json={"status": "repaired"}, headers=H).status_code == 403
    assert c.delete(f"/api/reports/{rid}", headers=H).status_code == 403
    assert c.patch(f"/api/reports/{rid}", json={"note": "updated"}, headers=H).status_code == 200
    c.post("/api/logout", headers=H)
    assert c.post("/api/login", json={"username": "boss", "password": "letters123"}, headers=H).status_code == 200
    assert c.patch(f"/api/reports/{rid}", json={"status": "in_progress", "comment": "crew assigned"}, headers=H).status_code == 200
    assert c.patch(f"/api/reports/{rid}", json={"lat": 12.97, "lon": 77.59}, headers=H).json()["changed"]
    ev = [e["action"] for e in c.get(f"/api/reports/{rid}").json()["events"]]
    assert ev == ["created", "note", "status", "location"], ev

    # lists, filters, map, stats, exports
    lst = c.get("/api/reports?status=in_progress").json()
    assert lst["total"] == 1 and lst["rows"][0]["id"] == rid
    assert c.get("/api/reports?status=x").json()["total"] == 0
    assert c.get("/api/reports?q=updated").json()["total"] == 1
    assert len(c.get("/api/map").json()["points"]) == 3
    stats = c.get("/api/stats").json()
    assert stats["kpi"]["total"] == 3 and len(stats["per_day"]) == 30
    csv_text = c.get("/api/export.csv").content.decode("utf-8-sig")
    assert csv_text.count("\n") == 4 and "latitude" in csv_text
    gj = json.loads(c.get("/api/export.geojson").content)
    assert len(gj["features"]) == 3 and gj["features"][0]["geometry"]["type"] == "Point"
    assert c.delete(f"/api/reports/{rid3}", headers=H).status_code == 200
    assert c.get(f"/api/reports/{rid3}").status_code == 404 and len(c.get("/api/map").json()["points"]) == 2

    # users
    assert c.post("/api/users", json={"username": "new_insp", "name": "N", "role": "inspector", "password": "letters123"}, headers=H).status_code == 200
    me = c.get("/api/session").json()["user"]
    assert c.patch(f"/api/users/{me['id']}", json={"active": False}, headers=H).status_code == 400


def test_photo_survey():
    c, webapp = client()
    webapp.geo.reverse_geocode = lambda *a_, **k: None
    if c.get("/api/session").json()["user"] is None:
        assert c.post("/api/login", json={"username": "boss", "password": "letters123"}, headers=H).status_code == 200
    files = [("photos", (p.name, p.read_bytes(), "image/jpeg")) for p in DEMO[:8]]
    sid = c.post("/api/surveys", data={"kind": "photos", "name": "test survey"}, files=files, headers=H).json()["id"]
    for _ in range(240):
        s = c.get(f"/api/surveys/{sid}").json()
        if s["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert s["status"] == "done", s
    assert s["stats"]["items"] == 8 and s["stats"]["with_gps"] == 8 and s["stats"]["reports"] >= 1, s
    reps = c.get(f"/api/reports?survey_id={sid}&per_page=50").json()["rows"]
    assert len(reps) == s["stats"]["reports"] and all(r["lat"] is not None and r["gps_source"] == "exif" for r in reps)
    lats = [r["lat"] for r in reps]
    assert all(12.9 < x < 13.0 for x in lats), lats


def test_ensemble_consistency():
    import numpy as np
    from detector import Detector, read_image
    import app as webapp
    det = webapp.ENGINE.det
    img = read_image(sorted(TEST_IMG.glob("Japan_*.jpg"))[2])
    a, b = det.predict(img), det.predict(img)
    assert [d.to_dict() for d in a] == [d.to_dict() for d in b]
    solo = Detector("india_japan", device="cpu")
    views, _, _ = det.member_views(img)
    sb, sc, sk = solo._run(__import__("preprocess").enhance(img), det.settings["member_min_conf"])
    assert np.allclose(views[0][0], sb, atol=1e-3) and np.allclose(views[0][1], sc, atol=1e-5)


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t:.1f} s)")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

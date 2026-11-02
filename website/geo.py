"""GPS for the RoadSense website: EXIF GPS from uploaded photo bytes, coordinate checks, and street addresses.

Location sources, in the order the website prefers them (each report stores which one it used):
  exif    GPS written into the photo by the camera / phone (most reliable for photos taken on site)
  device  the browser's live location (phone GPS or laptop Wi-Fi positioning), with its accuracy in metres
  map     a pin the user dropped or dragged on the map
  gpx     a dash-cam video frame matched to a GPX track (survey uploads, src/geo_map.py)

Street addresses come from OpenStreetMap's Nominatim service (one request per second at most, cached by
coordinates rounded to ~1 m, a descriptive User-Agent as the usage policy asks). Failures fall back to coordinates.
"""
import io
import json
import math
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime

from PIL import Image
from PIL.ExifTags import IFD

NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "RoadSense-FY1/1.0 (road-damage survey website, final-year project)"
_geocode_lock = threading.Lock()
_last_request = [0.0]


def _num(v) -> float:
    """EXIF rational / tuple / number -> float (NaN for 0/0 rationals)."""
    if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, int) for x in v):
        return v[0] / v[1] if v[1] else math.nan
    try:
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return math.nan


def _deg(dms, ref) -> float:
    if isinstance(dms, (tuple, list)) and len(dms) == 3:
        d, m, s = (_num(x) for x in dms)
        deg = d + m / 60 + s / 3600
    else:                      # a few writers store decimal degrees directly
        deg = _num(dms)
    ref = (ref.decode() if isinstance(ref, bytes) else str(ref or "")).strip().upper()
    return -abs(deg) if ref in ("S", "W") else deg


def valid(lat, lon) -> bool:
    """A usable position: finite, in range, and not the (0, 0) placeholder some cameras write."""
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
            and not (abs(lat) < 1e-6 and abs(lon) < 1e-6))


def exif_gps(data: bytes) -> dict | None:
    """{'lat', 'lon', 'altitude_m', 'accuracy_m', 'gps_time', 'taken_at'} from JPEG/PNG/WebP EXIF, or None."""
    try:
        exif = Image.open(io.BytesIO(data)).getexif()
    except Exception:
        return None
    out = {}
    try:
        dto = exif.get_ifd(IFD.Exif).get(0x9003) or exif.get(0x0132)   # DateTimeOriginal, else DateTime
        if dto:
            out["taken_at"] = datetime.strptime(str(dto).strip("\x00 "), "%Y:%m:%d %H:%M:%S").isoformat()
    except Exception:
        pass
    try:
        gps = exif.get_ifd(IFD.GPSInfo)
    except Exception:
        gps = None
    if not gps or 2 not in gps or 4 not in gps:
        return {"taken_at": out["taken_at"]} if out.get("taken_at") else None
    try:
        lat, lon = _deg(gps[2], gps.get(1, "N")), _deg(gps[4], gps.get(3, "E"))
    except Exception:
        lat = lon = math.nan
    if not valid(lat, lon):
        return {"taken_at": out["taken_at"], "invalid_gps": True} if out.get("taken_at") else {"invalid_gps": True}
    out.update(lat=round(lat, 7), lon=round(lon, 7))
    if 6 in gps:
        alt = _num(gps[6])
        if math.isfinite(alt):
            ref = gps.get(5, 0)
            ref = ref[0] if isinstance(ref, bytes) and ref else ref
            out["altitude_m"] = round(-alt if ref == 1 else alt, 1)
    if 31 in gps:                                # GPSHPositioningError, metres
        acc = _num(gps[31])
        if math.isfinite(acc) and acc >= 0:
            out["accuracy_m"] = round(acc, 1)
    if 29 in gps and 7 in gps:                   # GPSDateStamp + GPSTimeStamp (UTC)
        try:
            h, m, s = (_num(x) for x in gps[7])
            d = datetime.strptime(str(gps[29]).strip("\x00 "), "%Y:%m:%d")
            out["gps_time"] = d.replace(hour=int(h), minute=int(m), second=int(s)).isoformat() + "Z"
        except Exception:
            pass
    return out


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def _short_address(js: dict) -> str | None:
    a = js.get("address") or {}
    road = a.get("road") or a.get("pedestrian") or a.get("footway") or a.get("path")
    area = a.get("suburb") or a.get("neighbourhood") or a.get("quarter") or a.get("village") or a.get("hamlet")
    city = a.get("city") or a.get("town") or a.get("municipality") or a.get("county") or a.get("state_district")
    parts = [p for p in (road, area, city, a.get("state"), a.get("postcode")) if p]
    seen, uniq = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return ", ".join(uniq) or js.get("display_name")


def reverse_geocode(lat: float, lon: float, cache_get=None, cache_put=None, timeout: float = 6.0) -> str | None:
    """Street address for a position (OpenStreetMap Nominatim), or None if unavailable."""
    if not valid(lat, lon):
        return None
    key = f"{lat:.5f},{lon:.5f}"
    if cache_get:
        hit = cache_get(key)
        if hit is not None:
            return hit or None
    q = urllib.parse.urlencode({"format": "jsonv2", "lat": f"{lat:.6f}", "lon": f"{lon:.6f}", "zoom": 18,
                                "addressdetails": 1, "accept-language": "en"})
    req = urllib.request.Request(f"{NOMINATIM}?{q}", headers={"User-Agent": USER_AGENT})
    with _geocode_lock:                           # Nominatim policy: at most 1 request per second
        wait = 1.05 - (time.monotonic() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                js = json.loads(r.read().decode("utf-8"))
            addr = _short_address(js) if "error" not in js else ""
        except Exception:
            return None                           # offline or rate-limited: try again next time
        finally:
            _last_request[0] = time.monotonic()
    if cache_put:
        cache_put(key, addr or "")
    return addr or None

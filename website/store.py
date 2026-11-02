"""SQLite storage for the RoadSense website (website/data/roadsense.db) plus image files next to it.

Tables
  users          accounts (role 'admin' or 'inspector', scrypt password hash)
  sessions       login sessions (only a SHA-256 of the cookie token is stored)
  reports        one per damage location: position, GPS source and accuracy, address, priority, status...
  detections     boxes for a report's main photo (class, confidence, severity, models agreeing)
  report_media   extra photos / video frames of the same location (surveys)
  report_events  status changes and edits (timeline / audit trail)
  surveys        batch jobs (many photos, or video + GPX) and their progress
  geocode_cache  street addresses by rounded position
Images live in data/{uploads,annotated,thumbs,media}/ under random names; the API serves them by report id only.
"""
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL COLLATE NOCASE, name TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('admin', 'inspector')), pw_hash TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, last_login TEXT);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS surveys (
  id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id), name TEXT NOT NULL, kind TEXT NOT NULL,
  status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, message TEXT, total INTEGER, done INTEGER DEFAULT 0,
  stats TEXT, created_at TEXT NOT NULL, finished_at TEXT);
CREATE TABLE IF NOT EXISTS reports (
  id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, user_id INTEGER REFERENCES users(id),
  survey_id INTEGER REFERENCES surveys(id), status TEXT NOT NULL DEFAULT 'open',
  lat REAL, lon REAL, gps_source TEXT, accuracy_m REAL, address TEXT, road_class TEXT NOT NULL DEFAULT 'unknown',
  taken_at TEXT, image TEXT NOT NULL, annotated TEXT NOT NULL, thumb TEXT NOT NULL, width INTEGER, height INTEGER,
  source_name TEXT, priority TEXT, priority_name TEXT, deadline TEXT, priority_score REAL, worst_level TEXT,
  worst_damage TEXT, condition_index REAL, n_detections INTEGER NOT NULL DEFAULT 0, damage_types TEXT,
  plan TEXT, quality TEXT, note TEXT, deleted INTEGER NOT NULL DEFAULT 0, updated_at TEXT);
CREATE INDEX IF NOT EXISTS reports_status ON reports(status, deleted);
CREATE INDEX IF NOT EXISTS reports_created ON reports(created_at);
CREATE TABLE IF NOT EXISTS detections (
  id INTEGER PRIMARY KEY, report_id INTEGER NOT NULL REFERENCES reports(id), cls INTEGER NOT NULL,
  name TEXT NOT NULL, conf REAL NOT NULL, score REAL, agree INTEGER, box TEXT NOT NULL,
  severity_score REAL, severity_level TEXT, extent REAL);
CREATE INDEX IF NOT EXISTS detections_report ON detections(report_id);
CREATE TABLE IF NOT EXISTS report_media (
  id INTEGER PRIMARY KEY, report_id INTEGER NOT NULL REFERENCES reports(id), path TEXT NOT NULL,
  label TEXT, n_detections INTEGER);
CREATE TABLE IF NOT EXISTS report_events (
  id INTEGER PRIMARY KEY, report_id INTEGER NOT NULL REFERENCES reports(id), user_id INTEGER REFERENCES users(id),
  at TEXT NOT NULL, action TEXT NOT NULL, detail TEXT);
CREATE TABLE IF NOT EXISTS geocode_cache (key TEXT PRIMARY KEY, address TEXT NOT NULL, at TEXT NOT NULL);
"""
STATUSES = ("open", "in_progress", "repaired", "rejected")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        for sub in ("uploads", "annotated", "thumbs", "media", "tmp", "surveys"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.path = self.root / "roadsense.db"
        self._lock = threading.RLock()
        with self.db() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def db(self):
        """A connection per use; writes are serialised so background jobs and requests don't collide."""
        with self._lock:
            c = sqlite3.connect(self.path, timeout=30)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys = ON")
            c.execute("PRAGMA journal_mode = WAL")
            try:
                yield c
                c.commit()
            finally:
                c.close()

    # ------------------------------------------------------------ helpers
    def file(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError("path outside the data folder")
        return p

    def one(self, sql: str, args=()) -> dict | None:
        with self.db() as c:
            r = c.execute(sql, args).fetchone()
        return dict(r) if r else None

    def all(self, sql: str, args=()) -> list[dict]:
        with self.db() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def event(self, c, report_id: int, user_id, action: str, detail=None) -> None:
        c.execute("INSERT INTO report_events (report_id, user_id, at, action, detail) VALUES (?, ?, ?, ?, ?)",
                  (report_id, user_id, now(), action, json.dumps(detail) if detail is not None else None))

    # ------------------------------------------------------------ geocode cache
    def geocode_get(self, key: str):
        r = self.one("SELECT address FROM geocode_cache WHERE key = ?", (key,))
        return r["address"] if r else None

    def geocode_put(self, key: str, address: str) -> None:
        with self.db() as c:
            c.execute("INSERT OR REPLACE INTO geocode_cache (key, address, at) VALUES (?, ?, ?)", (key, address, now()))

    # ------------------------------------------------------------ reports
    def insert_report(self, rep: dict, detections: list[dict], user_id, media: list[dict] | None = None) -> int:
        cols = ["created_at", "user_id", "survey_id", "status", "lat", "lon", "gps_source", "accuracy_m", "address",
                "road_class", "taken_at", "image", "annotated", "thumb", "width", "height", "source_name", "priority",
                "priority_name", "deadline", "priority_score", "worst_level", "worst_damage", "condition_index",
                "n_detections", "damage_types", "plan", "quality", "note", "updated_at"]
        row = {**rep, "created_at": rep.get("created_at") or now(), "user_id": user_id, "updated_at": now(),
               "status": rep.get("status", "open")}
        for k in ("plan", "quality", "damage_types"):
            if row.get(k) is not None and not isinstance(row[k], str):
                row[k] = json.dumps(row[k])
        with self.db() as c:
            cur = c.execute(f"INSERT INTO reports ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                            [row.get(k) for k in cols])
            rid = cur.lastrowid
            for d in detections:
                sev = d.get("severity") or {}
                c.execute("INSERT INTO detections (report_id, cls, name, conf, score, agree, box, severity_score, "
                          "severity_level, extent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (rid, d["cls"], d["name"], d["conf"], d.get("score"), d.get("agree"), json.dumps(d["box_xyxy"]),
                           sev.get("score"), sev.get("level"), sev.get("extent")))
            for m in media or []:
                c.execute("INSERT INTO report_media (report_id, path, label, n_detections) VALUES (?, ?, ?, ?)",
                          (rid, m["path"], m.get("label"), m.get("n_detections")))
            self.event(c, rid, user_id, "created", {"gps_source": rep.get("gps_source"), "survey_id": rep.get("survey_id")})
        return rid

    def report(self, rid: int, include_deleted: bool = False) -> dict | None:
        r = self.one("SELECT r.*, u.name AS reporter, u.username AS reporter_username FROM reports r "
                     "LEFT JOIN users u ON u.id = r.user_id WHERE r.id = ?" + ("" if include_deleted else " AND r.deleted = 0"),
                     (rid,))
        if not r:
            return None
        for k in ("plan", "quality", "damage_types"):
            r[k] = json.loads(r[k]) if r.get(k) else None
        r["detections"] = [{**d, "box": json.loads(d["box"])} for d in
                           self.all("SELECT * FROM detections WHERE report_id = ? ORDER BY conf DESC", (rid,))]
        r["media"] = self.all("SELECT id, label, n_detections FROM report_media WHERE report_id = ? ORDER BY id", (rid,))
        r["events"] = [{**e, "detail": json.loads(e["detail"]) if e["detail"] else None} for e in self.all(
            "SELECT e.at, e.action, e.detail, u.name AS user FROM report_events e LEFT JOIN users u ON u.id = e.user_id "
            "WHERE e.report_id = ? ORDER BY e.id", (rid,))]
        return r

    def query_reports(self, f: dict, limit: int | None = None, offset: int = 0, sort: str = "priority"):
        """Filtered reports (no detections), plus the total count. f keys: status, priority, damage, since, until,
        user_id, survey_id, q, has_location."""
        where, args = ["r.deleted = 0"], []
        if f.get("status"):
            st = [s for s in str(f["status"]).split(",") if s in STATUSES]
            if st:
                where.append(f"r.status IN ({', '.join('?' * len(st))})"); args += st
            else:
                where.append("0")
        if f.get("priority"):
            pr = [p for p in str(f["priority"]).split(",") if p in ("P1", "P2", "P3", "P4", "none")]
            cond = [f"r.priority IN ({', '.join('?' * len([p for p in pr if p != 'none']))})"] if any(p != "none" for p in pr) else []
            if "none" in pr:
                cond.append("r.priority IS NULL")
            if cond:
                where.append("(" + " OR ".join(cond) + ")"); args += [p for p in pr if p != "none"]
            else:
                where.append("0")
        if f.get("damage"):
            dm = [d for d in str(f["damage"]).split(",") if d]
            where.append("(" + " OR ".join("r.damage_types LIKE ?" for _ in dm) + ")"); args += [f'%"{d}"%' for d in dm]
        if f.get("since"):
            where.append("r.created_at >= ?"); args.append(f["since"])
        if f.get("until"):
            where.append("r.created_at < ?"); args.append(f["until"])
        if f.get("user_id"):
            where.append("r.user_id = ?"); args.append(int(f["user_id"]))
        if f.get("survey_id"):
            where.append("r.survey_id = ?"); args.append(int(f["survey_id"]))
        if f.get("has_location") in ("1", "true", True):
            where.append("r.lat IS NOT NULL")
        if f.get("has_location") in ("0", "false"):
            where.append("r.lat IS NULL")
        if f.get("q"):
            q = f"%{f['q']}%"
            where.append("(r.address LIKE ? OR r.note LIKE ? OR r.source_name LIKE ? OR CAST(r.id AS TEXT) = ?)")
            args += [q, q, q, str(f["q"]).lstrip("#")]
        order = {"priority": "CASE WHEN r.priority IS NULL THEN 9 ELSE CAST(SUBSTR(r.priority, 2) AS INTEGER) END, "
                             "r.priority_score DESC, r.created_at DESC",
                 "newest": "r.created_at DESC, r.id DESC", "oldest": "r.created_at ASC, r.id ASC",
                 "score": "r.priority_score DESC"}.get(sort, "r.created_at DESC")
        sql = (" FROM reports r LEFT JOIN users u ON u.id = r.user_id WHERE " + " AND ".join(where))
        with self.db() as c:
            total = c.execute("SELECT COUNT(*)" + sql, args).fetchone()[0]
            q = ("SELECT r.id, r.created_at, r.status, r.lat, r.lon, r.gps_source, r.accuracy_m, r.address, r.road_class,"
                 " r.priority, r.priority_name, r.deadline, r.priority_score, r.worst_level, r.worst_damage,"
                 " r.condition_index, r.n_detections, r.damage_types, r.survey_id, r.source_name, r.note,"
                 " u.name AS reporter" + sql + f" ORDER BY {order}")
            if limit:
                q += f" LIMIT {int(limit)} OFFSET {int(offset)}"
            rows = [dict(r) for r in c.execute(q, args).fetchall()]
        for r in rows:
            r["damage_types"] = json.loads(r["damage_types"]) if r["damage_types"] else []
        return rows, total

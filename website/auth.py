"""Accounts and sessions for the RoadSense website.

Passwords: scrypt (stdlib hashlib) with a random 16-byte salt. Sessions: a random token in an HttpOnly,
SameSite=Lax cookie (Secure over HTTPS); only its SHA-256 is stored. Roles:
  inspector  analyse photos, create reports and surveys, edit notes / location of their own reports
  admin      everything an inspector can do, plus change report status, delete reports and manage users
Failed logins are rate-limited per username + address (5 tries, then a 60 s wait).
"""
import hashlib
import hmac
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone

from store import Store, now

COOKIE = "roadsense_session"
SESSION_DAYS = 7
ROLES = ("admin", "inspector")
_N, _R, _P = 2 ** 14, 8, 1
_fails: dict[str, list[float]] = {}
_fails_lock = threading.Lock()


def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${h.hex()}"


def check_password(pw: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, h = stored.split("$")
        got = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=len(h) // 2)
        return hmac.compare_digest(got.hex(), h)
    except Exception:
        return False


def password_problem(pw: str) -> str | None:
    if len(pw) < 8:
        return "Password must be at least 8 characters."
    if pw.isdigit() or pw.isalpha():
        return "Use letters and numbers (or symbols) in the password."
    return None


def _tok_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_user(store: Store, username: str, name: str, role: str, password: str) -> int:
    if role not in ROLES:
        raise ValueError("role must be admin or inspector")
    username = username.strip()
    if not (3 <= len(username) <= 32) or not all(ch.isalnum() or ch in "._-" for ch in username):
        raise ValueError("Username: 3-32 letters, digits, dots, dashes or underscores.")
    problem = password_problem(password)
    if problem:
        raise ValueError(problem)
    with store.db() as c:
        if c.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            raise ValueError("That username is taken.")
        cur = c.execute("INSERT INTO users (username, name, role, pw_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                        (username, name.strip() or username, role, hash_password(password), now()))
        return cur.lastrowid


def rate_limited(key: str) -> float:
    """Seconds to wait before another login attempt for this key (0 = allowed)."""
    with _fails_lock:
        recent = [t for t in _fails.get(key, []) if time.time() - t < 60]
        _fails[key] = recent
        return round(60 - (time.time() - recent[0]), 0) if len(recent) >= 5 else 0


def login(store: Store, username: str, password: str, key: str):
    """(user, token) on success, else (None, message)."""
    wait = rate_limited(key)
    if wait:
        return None, f"Too many failed attempts. Try again in {int(wait)} s."
    u = store.one("SELECT * FROM users WHERE username = ?", (username.strip(),))
    if not u or not u["active"] or not check_password(password, u["pw_hash"]):
        with _fails_lock:
            _fails.setdefault(key, []).append(time.time())
        return None, "Wrong username or password." if not u or u["active"] else "This account is deactivated."
    with _fails_lock:
        _fails.pop(key, None)
    token = secrets.token_urlsafe(32)
    exp = (datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)).replace(microsecond=0).isoformat()
    with store.db() as c:
        c.execute("DELETE FROM sessions WHERE expires_at < ?", (now(),))
        c.execute("INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                  (_tok_hash(token), u["id"], now(), exp))
        c.execute("UPDATE users SET last_login = ? WHERE id = ?", (now(), u["id"]))
    return public(u), token


def logout(store: Store, token: str | None) -> None:
    if token:
        with store.db() as c:
            c.execute("DELETE FROM sessions WHERE token_hash = ?", (_tok_hash(token),))


def user_for(store: Store, token: str | None) -> dict | None:
    if not token:
        return None
    u = store.one("SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ? "
                  "AND s.expires_at > ? AND u.active = 1", (_tok_hash(token), now()))
    return public(u) if u else None


def public(u: dict) -> dict:
    return {k: u[k] for k in ("id", "username", "name", "role", "active", "created_at", "last_login") if k in u}


def set_password(store: Store, user_id: int, password: str, end_sessions: bool = True) -> None:
    problem = password_problem(password)
    if problem:
        raise ValueError(problem)
    with store.db() as c:
        c.execute("UPDATE users SET pw_hash = ? WHERE id = ?", (hash_password(password), user_id))
        if end_sessions:
            c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

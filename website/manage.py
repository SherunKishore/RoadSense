"""RoadSense website administration from the terminal.

    python website/manage.py create-user --username admin --name "Road Admin" --role admin   (asks for the password)
    python website/manage.py list-users
    python website/manage.py set-password --username admin                                    (asks for the password)
    python website/manage.py seed-dev       test accounts for development; passwords are written to
                                            website/dev_accounts.json (not shown on screen)
    python website/manage.py backup-reset   moves website/data to website/data_backup_<time> and starts empty
"""
import argparse
import getpass
import json
import secrets
import shutil
import string
import sys
from datetime import datetime
from pathlib import Path

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB))
import auth  # noqa: E402
from store import Store  # noqa: E402

DEV_FILE = WEB / "dev_accounts.json"
DEV_USERS = [("test_admin", "Test Admin", "admin"), ("test_inspector", "Test Inspector", "inspector")]


def ask_password() -> str:
    pw = getpass.getpass("Password (8+ characters, letters and numbers): ")
    if pw != getpass.getpass("Repeat password: "):
        sys.exit("Passwords don't match.")
    return pw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["create-user", "list-users", "set-password", "seed-dev", "backup-reset"])
    ap.add_argument("--username")
    ap.add_argument("--name", default="")
    ap.add_argument("--role", choices=auth.ROLES, default="inspector")
    ap.add_argument("--data", default=str(WEB / "data"))
    a = ap.parse_args()

    if a.command == "backup-reset":
        src = Path(a.data)
        if src.exists():
            dst = src.with_name(f"data_backup_{datetime.now():%Y%m%d_%H%M%S}")
            shutil.move(str(src), dst)
            print(f"Moved {src} -> {dst}. The website starts empty next time.")
        return
    store = Store(Path(a.data))
    if a.command == "list-users":
        for u in store.all("SELECT username, name, role, active, last_login FROM users ORDER BY role, username"):
            print(f"{u['username']:20s} {u['role']:10s} {'active' if u['active'] else 'inactive':9s} "
                  f"last login {u['last_login'] or '-':25s} {u['name']}")
    elif a.command == "create-user":
        if not a.username:
            sys.exit("--username is required")
        uid = auth.create_user(store, a.username, a.name or a.username, a.role, ask_password())
        print(f"Created {a.role} '{a.username}' (id {uid}).")
    elif a.command == "set-password":
        u = store.one("SELECT id FROM users WHERE username = ?", (a.username or "",))
        if not u:
            sys.exit("No such user.")
        auth.set_password(store, u["id"], ask_password())
        print("Password changed; that user's sessions were signed out.")
    elif a.command == "seed-dev":
        creds = json.loads(DEV_FILE.read_text()) if DEV_FILE.exists() else {}
        for username, name, role in DEV_USERS:
            if store.one("SELECT id FROM users WHERE username = ?", (username,)):
                continue
            pw = creds.get(username) or "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(14)) + "7a"
            auth.create_user(store, username, name, role, pw)
            creds[username] = pw
        DEV_FILE.write_text(json.dumps(creds, indent=2))
        print(f"Test accounts: {', '.join(u for u, _, _ in DEV_USERS)}; passwords in {DEV_FILE}")


if __name__ == "__main__":
    main()

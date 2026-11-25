"""Persistence layer for PneumoScan AI (SQLite, standard library only).

Tables
  users      - accounts with role-based access (physician / admin); passwords stored as salted PBKDF2 hashes
  patients   - pseudonymised patient records
  studies    - one row per analysed X-ray: image, AI result, uncertainty, report, sign-off status
  audit_log  - append-only trail of who did what and when (login, view, analyse, sign, export)

The database file is created on first run at data/pneumoscan.db with two demo accounts.
"""
import hashlib
import hmac
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Istanbul")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "pneumoscan.db"

ROLES = {"physician": "Physician", "admin": "Administrator"}
DEMO_USERS = [  # username, password, role, full name
    ("dr.demo", "demo123", "physician", "Dr. Demo Radiologist"),
    ("admin", "admin123", "admin", "System Administrator"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY, full_name TEXT NOT NULL, role TEXT NOT NULL,
    salt BLOB NOT NULL, pw_hash BLOB NOT NULL, active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS patients (
    pid TEXT PRIMARY KEY, display_name TEXT, birth_year INTEGER, sex TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS studies (
    id INTEGER PRIMARY KEY AUTOINCREMENT, pid TEXT NOT NULL REFERENCES patients(pid),
    created_at TEXT NOT NULL, created_by TEXT NOT NULL, source TEXT, file_name TEXT, indication TEXT,
    image_png BLOB, prob REAL, threshold REAL, level TEXT, unc_std REAL, confidence TEXT,
    report TEXT, status TEXT NOT NULL DEFAULT 'pending', signed_by TEXT, signed_at TEXT);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, username TEXT, action TEXT NOT NULL,
    pid TEXT, detail TEXT);
"""


def _now():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")


def connect(path=DB_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    if con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        for u, pw, role, name in DEMO_USERS:
            create_user(con, u, pw, role, name)
    return con


# ------------------------------------------------------------------ users
def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)


def create_user(con, username, password, role, full_name):
    salt = os.urandom(16)
    con.execute("INSERT INTO users (username, full_name, role, salt, pw_hash, created_at) VALUES (?,?,?,?,?,?)",
                (username, full_name, role, salt, _hash(password, salt), _now()))
    con.commit()


def authenticate(con, username, password):
    row = con.execute("SELECT * FROM users WHERE username=? AND active=1", (username,)).fetchone()
    if row and hmac.compare_digest(_hash(password, row["salt"]), row["pw_hash"]):
        return {"username": row["username"], "full_name": row["full_name"], "role": row["role"]}
    return None


def list_users(con):
    return con.execute("SELECT username, full_name, role, active, created_at FROM users ORDER BY username").fetchall()


def set_active(con, username, active):
    con.execute("UPDATE users SET active=? WHERE username=?", (int(active), username)); con.commit()


# ------------------------------------------------------------------ audit
def log(con, username, action, pid=None, detail=""):
    con.execute("INSERT INTO audit_log (ts, username, action, pid, detail) VALUES (?,?,?,?,?)",
                (_now(), username, action, pid, detail))
    con.commit()


def audit_entries(con, limit=500, username=None, action=None):
    q, a = "SELECT * FROM audit_log WHERE 1=1", []
    if username:
        q += " AND username=?"; a.append(username)
    if action:
        q += " AND action=?"; a.append(action)
    return con.execute(q + " ORDER BY id DESC LIMIT ?", (*a, limit)).fetchall()


# ------------------------------------------------------------------ patients & studies
def upsert_patient(con, pid, display_name=None, birth_year=None, sex=None):
    if con.execute("SELECT 1 FROM patients WHERE pid=?", (pid,)).fetchone():
        con.execute("UPDATE patients SET display_name=COALESCE(?, display_name), birth_year=COALESCE(?, birth_year), "
                    "sex=COALESCE(?, sex) WHERE pid=?", (display_name, birth_year, sex, pid))
    else:
        con.execute("INSERT INTO patients VALUES (?,?,?,?,?)", (pid, display_name, birth_year, sex, _now()))
    con.commit()


def list_patients(con, search=""):
    s = f"%{search}%"
    return con.execute("""
        SELECT p.*, COUNT(s.id) AS n_studies, MAX(s.created_at) AS last_study,
               (SELECT level FROM studies WHERE pid=p.pid ORDER BY id DESC LIMIT 1) AS last_level
        FROM patients p LEFT JOIN studies s ON s.pid=p.pid
        WHERE p.pid LIKE ? OR IFNULL(p.display_name,'') LIKE ?
        GROUP BY p.pid ORDER BY last_study DESC""", (s, s)).fetchall()


def get_patient(con, pid):
    return con.execute("SELECT * FROM patients WHERE pid=?", (pid,)).fetchone()


def add_study(con, **kw):
    created = kw.pop("created_at", None) or _now()
    cols = ", ".join(kw); qs = ", ".join("?" * len(kw))
    cur = con.execute(f"INSERT INTO studies (created_at, {cols}) VALUES (?, {qs})", (created, *kw.values()))
    con.commit()
    return cur.lastrowid


def update_study(con, study_id, **kw):
    sets = ", ".join(f"{k}=?" for k in kw)
    con.execute(f"UPDATE studies SET {sets} WHERE id=?", (*kw.values(), study_id)); con.commit()


def sign_study(con, study_id, username, report):
    update_study(con, study_id, report=report, status="signed", signed_by=username, signed_at=_now())


def get_study(con, study_id):
    return con.execute("SELECT * FROM studies WHERE id=?", (study_id,)).fetchone()


def patient_studies(con, pid):
    return con.execute("SELECT * FROM studies WHERE pid=? ORDER BY id DESC", (pid,)).fetchall()


LEVEL_ORDER = "CASE level WHEN 'urgent' THEN 0 WHEN 'review' THEN 1 ELSE 2 END"


def worklist(con, status=None):
    q = "SELECT * FROM studies" + (" WHERE status=?" if status else "")
    return con.execute(q + f" ORDER BY {LEVEL_ORDER}, CASE confidence WHEN 'low' THEN 0 ELSE 1 END, id DESC",
                       (status,) if status else ()).fetchall()


def stats(con):
    r = con.execute("""SELECT COUNT(*) n, SUM(status='pending') pending, SUM(status='signed') signed,
                       SUM(level='urgent') urgent, SUM(confidence='low') low_conf FROM studies""").fetchone()
    return {k: (r[k] or 0) for k in r.keys()} | {
        "patients": con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]}

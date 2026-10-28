"""Persistent alert log: every live alert from every session in one SQLite file
(results/alerts.db), so a dashboard can read it while live.py is still writing.

SQLite rather than CSV: Excel locks an open CSV on Windows, which would make live
logging fail mid-session. Use the dashboard's "Download CSV" for spreadsheets.

Tables
  sessions(session, mode, started_at, ended_at, camera, resolution, config_json, deleted)
  alerts(id, session, mode, timestamp, t_session_s, track_id, event_type, alert_name,
         message, confidence, confidence_level, positive_ratio, episode_start_s,
         duration_s, evidence, review, reviewed_at)
  review = 'confirmed' / 'dismissed' / NULL (not reviewed yet); accuracy of an alert
  type = confirmed / reviewed.
"""

import csv
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

EVENT_NAMES = {"phone_in_hand": "Phone in hand", "phone_on_desk": "Phone on desk",
               "body_movement": "Body movement", "unusual_seat_movement": "Unusual seat movement",
               "head_turned": "Head turned away"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session TEXT PRIMARY KEY, mode TEXT, started_at TEXT, ended_at TEXT,
    camera INTEGER, resolution TEXT, config_json TEXT, deleted INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, mode TEXT, timestamp TEXT,
    t_session_s REAL, track_id INTEGER, event_type TEXT, alert_name TEXT, message TEXT,
    confidence REAL, confidence_level TEXT, positive_ratio REAL, episode_start_s REAL,
    duration_s REAL, evidence TEXT, review TEXT, reviewed_at TEXT);
CREATE INDEX IF NOT EXISTS alerts_session ON alerts(session);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, timestamp TEXT, t_session_s REAL,
    point TEXT, kind TEXT, note TEXT);
"""

# Columns added in week 5 (robot route, consolidation, clips). Old databases are migrated on open.
NEW_ALERT_COLUMNS = {"observation_point": "TEXT", "sightings": "INTEGER DEFAULT 1", "last_seen_s": "REAL",
                     "clips": "TEXT", "region": "TEXT", "context_note": "TEXT", "seat": "TEXT"}


def confidence_level(conf):
    if conf is None or conf != conf:
        return "n/a"
    return "High" if conf >= 0.75 else "Medium" if conf >= 0.5 else "Low"


def connect(db_path):
    con = sqlite3.connect(str(db_path), timeout=5)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")      # dashboard can read while live.py writes
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(alerts)")}
    for col, typ in NEW_ALERT_COLUMNS.items():
        if col not in have:
            con.execute(f"ALTER TABLE alerts ADD COLUMN {col} {typ}")
    if "deleted" not in {r[1] for r in con.execute("PRAGMA table_info(sessions)")}:   # older db files
        con.execute("ALTER TABLE sessions ADD COLUMN deleted INTEGER DEFAULT 0")
    con.commit()
    return con


def delete_alert(con, alert_id):
    with con:
        return con.execute("DELETE FROM alerts WHERE id=?", (int(alert_id),)).rowcount


def delete_session(con, session):
    """Remove a session's alerts and robot observations from the log. The session row is kept,
    marked deleted, so the dashboard does not re-import it from results/live_*/alerts.csv."""
    with con:
        n = con.execute("DELETE FROM alerts WHERE session=?", (session,)).rowcount
        con.execute("DELETE FROM observations WHERE session=?", (session,))
        con.execute("UPDATE sessions SET deleted=1 WHERE session=?", (session,))
    return n


class AlertLog:
    def __init__(self, db_path, session, mode, started_at: datetime, camera=None, resolution="", config=None):
        self.con, self.session, self.mode, self.started_at = connect(db_path), session, mode, started_at
        with self.con:
            self.con.execute("INSERT OR REPLACE INTO sessions (session, mode, started_at, ended_at, camera,"
                             " resolution, config_json) VALUES (?,?,?,?,?,?,?)",
                             (session, mode, started_at.isoformat(timespec="seconds"), None, camera,
                              resolution, json.dumps(config or {})))

    def add(self, alert, evidence="", point=None, region=None, clip=None, context_note=None):
        """alert: temporal.Alert. evidence / clip: paths relative to results/. Returns the row id."""
        conf = None if alert.confidence != alert.confidence else float(alert.confidence)
        ratio = None if alert.positive_ratio != alert.positive_ratio else float(alert.positive_ratio)
        ts = self.started_at + timedelta(seconds=alert.t_alert)
        with self.con:
            cur = self.con.execute(
                "INSERT INTO alerts (session, mode, timestamp, t_session_s, track_id, event_type, alert_name,"
                " message, confidence, confidence_level, positive_ratio, episode_start_s, duration_s, evidence,"
                " observation_point, sightings, last_seen_s, clips, region, context_note, seat)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.session, self.mode, ts.isoformat(timespec="seconds"), round(alert.t_alert, 2),
                 alert.track_id if isinstance(alert.track_id, str) else int(alert.track_id), alert.event_type, EVENT_NAMES.get(alert.event_type, alert.event_type),
                 alert.message(ts.strftime("%H:%M:%S")), conf, confidence_level(conf), ratio,
                 round(alert.t_start, 2), round(alert.t_alert - alert.t_start, 2), evidence,
                 point, 1, round(alert.t_alert, 2), json.dumps([clip] if clip else []),
                 json.dumps(region) if region else None, context_note,
                 alert.track_id if isinstance(alert.track_id, str) else None))
        return cur.lastrowid

    def update_event(self, row_id, sightings=None, last_seen_s=None, add_clip=None):
        """A repeated sighting was consolidated into this record."""
        with self.con:
            row = self.con.execute("SELECT sightings, last_seen_s, clips FROM alerts WHERE id=?", (row_id,)).fetchone()
            if row is None:
                return
            clips = json.loads(row["clips"] or "[]")
            if add_clip and add_clip not in clips:
                clips.append(add_clip)
            self.con.execute("UPDATE alerts SET sightings=?, last_seen_s=?, clips=? WHERE id=?",
                             (sightings if sightings is not None else row["sightings"],
                              round(last_seen_s, 2) if last_seen_s is not None else row["last_seen_s"],
                              json.dumps(clips), row_id))

    def observe(self, t_session, point, kind, note=""):
        """Robot observation log: arrival at a point, insufficient view, leaving a point."""
        ts = self.started_at + timedelta(seconds=t_session)
        with self.con:
            self.con.execute("INSERT INTO observations (session, timestamp, t_session_s, point, kind, note)"
                             " VALUES (?,?,?,?,?,?)",
                             (self.session, ts.isoformat(timespec="seconds"), round(t_session, 2), point, kind, note))

    def close(self):
        with self.con:
            self.con.execute("UPDATE sessions SET ended_at=? WHERE session=?",
                             (datetime.now().isoformat(timespec="seconds"), self.session))
        self.con.close()


def import_live_dirs(db_path, results_dir):
    """Add older results/live_* sessions (from before this log existed) from their alerts.csv."""
    from .temporal import Alert
    con = connect(db_path)
    known = {r[0] for r in con.execute("SELECT session FROM sessions")}
    con.close()
    added = []
    for d in sorted(Path(results_dir).glob("live_*")):
        f = d / "alerts.csv"
        if d.name in known or not f.exists():
            continue
        try:                                       # live_YYYYmmdd_HHMMSS_mode
            _, day, hms, mode = d.name.split("_", 3)
            started = datetime.strptime(day + hms, "%Y%m%d%H%M%S")
        except ValueError:
            continue
        log = AlertLog(db_path, d.name, mode, started)
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                a = Alert(int(r["track_id"]), r["event_type"], float(r["t_alert"]), float(r["t_start"]),
                          float(r["confidence"] or "nan"), float(r["positive_ratio"] or "nan"))
                log.add(a, f"{d.name}/evidence/{r['evidence']}" if r.get("evidence") else "")
        log.con.execute("UPDATE sessions SET ended_at='' WHERE session=?", (d.name,))   # end unknown
        log.con.commit()
        log.con.close()
        added.append(d.name)
    return added

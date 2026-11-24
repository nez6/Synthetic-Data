"""SQLite store for every HL7 message the listener receives (and what it did with it)."""
import json
import sqlite3
from datetime import datetime, timezone

from src.config import load_config, resolve

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at TEXT, control_id TEXT, message_type TEXT, patient_id TEXT,
    ack_code TEXT, ack_text TEXT, dqi INTEGER, issues TEXT,
    prediction TEXT, fhir_bundle TEXT, raw TEXT
)"""


def _connect():
    path = resolve(load_config()["paths"]["hl7_inbox_db"])
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def save(**row):
    row["received_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for k in ("issues", "prediction", "fhir_bundle"):
        if k in row and row[k] is not None:
            row[k] = json.dumps(row[k])
    cols = ", ".join(row)
    with _connect() as conn:
        conn.execute(f"INSERT INTO messages ({cols}) VALUES ({', '.join('?' * len(row))})", list(row.values()))


def recent(limit: int = 200) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("issues", "prediction", "fhir_bundle"):
            d[k] = json.loads(d[k]) if d[k] else None
        out.append(d)
    return out


def clear():
    with _connect() as conn:
        conn.execute("DELETE FROM messages")

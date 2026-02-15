# app/utils/parameter_store.py
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

DB_PATH = Path(__file__).resolve().parent.parent / "hems.db"

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hems_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        conn.commit()

def save_parameters(params: Dict[str, Any]) -> int:
    """Insert a new config snapshot."""
    init_db()
    created_at = datetime.utcnow().isoformat()
    payload = json.dumps(params, default=str)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO hems_runs (created_at, payload) VALUES (?, ?)",
            (created_at, payload),
        )
        conn.commit()
        return int(cur.lastrowid)

def load_latest_parameters() -> Dict[str, Any]:
    """Load the most recent snapshot (or {} if none)."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT payload FROM hems_runs ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
    if not row:
        return {}
    try:
        return json.loads(row[0])
    except Exception:
        return {}

def overwrite_latest_parameters(new_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overwrite by inserting a fresh snapshot.
    Keeps history (recommended for debugging / evaluation).
    """
    save_parameters(new_params)
    return new_params

def list_snapshots(limit: int = 20) -> list[Tuple[int, str]]:
    """Return [(id, created_at), ...] newest first."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT id, created_at FROM hems_runs ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )
        return [(int(r[0]), str(r[1])) for r in cur.fetchall()]

def load_snapshot(snapshot_id: int) -> Dict[str, Any]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT payload FROM hems_runs WHERE id = ?", (int(snapshot_id),))
        row = cur.fetchone()
    if not row:
        return {}
    try:
        return json.loads(row[0])
    except Exception:
        return {}

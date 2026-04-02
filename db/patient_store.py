# db/patient_store.py
# SPECTER v5 — Patient SQLite Store
# Handles longitudinal patient history, pre-call brief persistence,
# and next-call strategy storage.

import sqlite3
import json
import os
from datetime import date

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patients.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Call once on server startup."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id          TEXT PRIMARY KEY,
            name                TEXT NOT NULL,
            dob                 TEXT,
            call_history        TEXT DEFAULT '[]',
            risk_trend          TEXT DEFAULT '[]',
            known_sensitivities TEXT DEFAULT '[]',
            last_brief          TEXT DEFAULT '',
            next_call_strategy  TEXT DEFAULT ''
        )
    """)
    conn.commit()

    # ── Seed demo patient if not already present ──
    existing = conn.execute(
        "SELECT patient_id FROM patients WHERE patient_id = 'elena_vance'"
    ).fetchone()

    if not existing:
        call_history = json.dumps([
            {
                "date":       "2026-02-28",
                "churn_score": 52,
                "risk_level": "MEDIUM",
                "key_flags":  ["pricing", "nausea"],
                "outcome":    "completed"
            },
            {
                "date":       "2026-03-14",
                "churn_score": 61,
                "risk_level": "MEDIUM",
                "key_flags":  ["pricing", "missed_doses"],
                "outcome":    "completed"
            }
        ])
        risk_trend = json.dumps([
            {"call": 1, "churn": 52},
            {"call": 2, "churn": 61}
        ])
        sensitivities = json.dumps(["pricing", "nausea", "weight_discussion"])
        conn.execute(
            """INSERT INTO patients
               (patient_id, name, dob, call_history, risk_trend, known_sensitivities)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("elena_vance", "Elena Vance", "1985-03-14",
             call_history, risk_trend, sensitivities)
        )
        conn.commit()
        print("[DB] Demo patient 'elena_vance' seeded.")

    conn.close()


def get_patient(patient_id: str) -> dict | None:
    """Return full patient record as dict, or None if not found."""
    conn = _get_conn()
    row  = conn.execute(
        "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "patient_id":          row["patient_id"],
        "name":                row["name"],
        "dob":                 row["dob"],
        "call_history":        json.loads(row["call_history"]   or "[]"),
        "risk_trend":          json.loads(row["risk_trend"]     or "[]"),
        "known_sensitivities": json.loads(row["known_sensitivities"] or "[]"),
        "last_brief":          row["last_brief"]         or "",
        "next_call_strategy":  row["next_call_strategy"] or "",
    }


def save_call_summary(patient_id: str, summary: dict):
    """
    Append a call summary to the patient's history and update risk trend.
    summary = {
        "date": "2026-03-30",
        "churn_score": 72,
        "risk_level": "HIGH",
        "key_flags": ["pricing"],
        "outcome": "completed"
    }
    """
    conn = _get_conn()
    row  = conn.execute(
        "SELECT call_history, risk_trend FROM patients WHERE patient_id = ?",
        (patient_id,)
    ).fetchone()
    if not row:
        conn.close()
        return

    history    = json.loads(row["call_history"] or "[]")
    risk_trend = json.loads(row["risk_trend"]   or "[]")

    summary.setdefault("date", str(date.today()))
    history.append(summary)

    call_num = len(history)
    risk_trend.append({"call": call_num, "churn": summary.get("churn_score", 0)})

    conn.execute(
        "UPDATE patients SET call_history = ?, risk_trend = ? WHERE patient_id = ?",
        (json.dumps(history), json.dumps(risk_trend), patient_id)
    )
    conn.commit()
    conn.close()
    print(f"[DB] Call summary saved for {patient_id} (call #{call_num})")


def update_next_strategy(patient_id: str, strategy: str, last_brief: str = ""):
    """Save the post-call next-call strategy and the brief that was used."""
    conn = _get_conn()
    conn.execute(
        "UPDATE patients SET next_call_strategy = ?, last_brief = ? WHERE patient_id = ?",
        (strategy, last_brief, patient_id)
    )
    conn.commit()
    conn.close()
    print(f"[DB] Next-call strategy updated for {patient_id}")


def upsert_patient(patient_id: str, name: str, dob: str = ""):
    """Create a patient record if it doesn't exist."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR IGNORE INTO patients (patient_id, name, dob)
           VALUES (?, ?, ?)""",
        (patient_id, name, dob)
    )
    conn.commit()
    conn.close()
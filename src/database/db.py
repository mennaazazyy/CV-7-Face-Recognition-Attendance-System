from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import DATABASE_PATH, EXPORTS_DIR, SCHEMA_PATH, ensure_project_dirs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    ensure_project_dirs()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _apply_lightweight_migrations(conn)


def add_student(student_id: str, full_name: str, email: str | None = None) -> None:
    with connect() as conn:
        columns = _column_names(conn, "students")
        if "created_at" in columns:
            conn.execute(
                """
                INSERT INTO students (student_id, full_name, email, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(student_id) DO UPDATE SET
                    full_name = excluded.full_name,
                    email = excluded.email,
                    is_active = 1
                """,
                (student_id, full_name, email, utc_now()),
            )
        else:
            conn.execute(
                """
                INSERT OR IGNORE INTO students (student_id, full_name, email, enrolled_at)
                VALUES (?, ?, ?, ?)
                """,
                (student_id, full_name, email, utc_now()),
            )
            conn.execute(
                "UPDATE students SET full_name = ?, email = ? WHERE student_id = ?",
                (full_name, email, student_id),
            )


def save_template(student_id: str, model_name: str, template: np.ndarray) -> None:
    template = np.asarray(template, dtype=np.float32).reshape(-1)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO face_templates (student_id, model_name, template, template_dim, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(student_id, model_name) DO UPDATE SET
                template = excluded.template,
                template_dim = excluded.template_dim,
                created_at = excluded.created_at
            """,
            (student_id, model_name, template.tobytes(), int(template.size), utc_now()),
        )


def load_gallery(model_name: str) -> list[tuple[str, np.ndarray]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT student_id, template FROM face_templates WHERE model_name = ?",
            (model_name,),
        ).fetchall()
    return [(row["student_id"], np.frombuffer(row["template"], dtype=np.float32)) for row in rows]


def create_session(session_id: str, course_code: str, session_date: str, model_name: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO attendance_sessions
                (session_id, course_code, session_date, model_name, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, course_code, session_date, model_name, utc_now()),
        )


def mark_attendance(session_id: str, student_id: str, confidence: float) -> bool:
    with connect() as conn:
        try:
            conn.execute(
                """
                INSERT INTO attendance_records (session_id, student_id, marked_at, confidence)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, student_id, utc_now(), confidence),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def log_recognition_event(
    predicted_label: str,
    confidence: float,
    event_type: str,
    session_id: str | None = None,
    student_id: str | None = None,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO recognition_events
                (session_id, student_id, predicted_label, confidence, event_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, student_id, predicted_label, confidence, event_type, utc_now()),
        )


def get_attendance_rows(session_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT ar.session_id, ar.student_id, s.full_name, ar.marked_at, ar.confidence
            FROM attendance_records ar
            JOIN students s ON s.student_id = ar.student_id
            WHERE ar.session_id = ?
            ORDER BY ar.marked_at
            """,
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def export_attendance_csv(session_id: str, output_path: Path | None = None) -> Path:
    rows = get_attendance_rows(session_id)
    output_path = output_path or EXPORTS_DIR / f"attendance_{session_id}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["session_id", "student_id", "full_name", "marked_at", "confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def _apply_lightweight_migrations(conn: sqlite3.Connection) -> None:
    """Keeps early foundation databases usable while the schema is still settling."""
    student_columns = _column_names(conn, "students")
    if student_columns and "email" not in student_columns:
        conn.execute("ALTER TABLE students ADD COLUMN email TEXT")
    if student_columns and "is_active" not in student_columns:
        conn.execute("ALTER TABLE students ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
    if student_columns and "created_at" not in student_columns and "enrolled_at" not in student_columns:
        conn.execute("ALTER TABLE students ADD COLUMN created_at TEXT")


def _column_names(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}

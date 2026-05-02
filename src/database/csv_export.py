from pathlib import Path

from src.database.db import export_attendance_csv


def export_session_csv(session_id: str, output_path: Path | None = None) -> Path:
    """Compatibility wrapper for CSV export."""
    return export_attendance_csv(session_id=session_id, output_path=output_path)

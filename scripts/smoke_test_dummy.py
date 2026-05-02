from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db import (  # noqa: E402
    add_student,
    create_session,
    export_attendance_csv,
    init_db,
    load_gallery,
    mark_attendance,
    save_template,
)
from src.models import create_model  # noqa: E402


def main() -> None:
    model_name = "dummy"
    student_id = "student_dummy"
    session_id = f"smoke_dummy_{uuid4().hex[:8]}"

    init_db()
    model = create_model(model_name)

    face_crop = np.full((120, 120, 3), 127, dtype=np.uint8)
    encoding = model.encode(face_crop)
    assert set(encoding) == {"model_name", "embedding"}
    assert encoding["model_name"] == model_name
    assert isinstance(encoding["embedding"], np.ndarray)

    add_student(student_id=student_id, full_name="Dummy Student")
    save_template(student_id=student_id, model_name=model_name, template=encoding["embedding"])

    gallery = load_gallery(model_name)
    assert any(row_student_id == student_id for row_student_id, _ in gallery)

    prediction = model.predict(face_crop, gallery)
    assert set(prediction) == {"student_id", "confidence", "status"}
    assert prediction["student_id"] == student_id
    assert prediction["status"] == "known"

    create_session(
        session_id=session_id,
        course_code="CV7",
        session_date=date.today().isoformat(),
        model_name=model_name,
    )
    first_mark = mark_attendance(session_id, student_id, prediction["confidence"])
    duplicate_mark = mark_attendance(session_id, student_id, prediction["confidence"])
    assert first_mark is True
    assert duplicate_mark is False

    csv_path = export_attendance_csv(session_id)
    assert csv_path.exists()

    print("Dummy smoke test passed")
    print(f"Prediction: {prediction}")
    print(f"CSV export: {csv_path}")


if __name__ == "__main__":
    main()

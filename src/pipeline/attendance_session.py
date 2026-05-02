from __future__ import annotations

from datetime import date

import cv2

from src.config import ACTIVE_MODEL, FRAME_SKIP, UNKNOWN_LABEL, WEBCAM_INDEX
from src.database.db import create_session, export_attendance_csv, log_recognition_event, mark_attendance
from src.pipeline.recognize import Recognizer, is_unknown


def run_attendance_session(
    session_id: str,
    course_code: str,
    model_name: str = ACTIVE_MODEL,
    camera_index: int = WEBCAM_INDEX,
    show_preview: bool = True,
) -> dict:
    create_session(
        session_id=session_id,
        course_code=course_code,
        session_date=date.today().isoformat(),
        model_name=model_name,
    )
    recognizer = Recognizer(model_name=model_name)

    cap = cv2.VideoCapture(camera_index)
    present: set[str] = set()
    duplicate_count = 0
    unknown_count = 0
    frame_number = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_number += 1
        if frame_number % FRAME_SKIP != 0:
            if _should_stop_preview(frame, show_preview):
                break
            continue

        for result in recognizer.recognize_frame(frame):
            x, y, w, h = result.bbox

            if is_unknown(result):
                unknown_count += 1
                log_recognition_event(
                    session_id=session_id,
                    student_id=None,
                    predicted_label=UNKNOWN_LABEL,
                    confidence=result.confidence,
                    event_type="unknown",
                )
                color = (0, 0, 255)
            else:
                assert result.student_id is not None
                is_new = mark_attendance(session_id, result.student_id, result.confidence)
                if is_new:
                    present.add(result.student_id)
                else:
                    duplicate_count += 1

                log_recognition_event(
                    session_id=session_id,
                    student_id=result.student_id,
                    predicted_label=result.student_id,
                    confidence=result.confidence,
                    event_type="recognized",
                )
                color = (0, 180, 0)

            if show_preview:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, result.label, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if _should_stop_preview(frame, show_preview):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_path = export_attendance_csv(session_id)

    return {
        "session_id": session_id,
        "present_count": len(present),
        "present_student_ids": sorted(present),
        "duplicate_count": duplicate_count,
        "unknown_count": unknown_count,
        "csv_path": str(csv_path),
    }


def _should_stop_preview(frame, show_preview: bool) -> bool:
    if not show_preview:
        return False
    cv2.imshow("Attendance", frame)
    return cv2.waitKey(1) & 0xFF == ord("q")

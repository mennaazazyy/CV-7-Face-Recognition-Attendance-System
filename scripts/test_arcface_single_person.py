from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db import (  # noqa: E402
    add_student,
    connect,
    create_session,
    init_db,
    load_gallery,
    log_recognition_event,
    mark_attendance,
    save_template,
)
from src.detection import crop_faces  # noqa: E402
from src.models.arcface_encoder import ArcFaceEncoder  # noqa: E402


MODEL_NAME = "arcface"
SAMPLE_COUNT = 10


def main() -> None:
    init_db()
    student_id = input("Student ID: ").strip()
    full_name = input("Student name: ").strip()
    if not student_id or not full_name:
        raise ValueError("Student ID and name are required.")

    model = ArcFaceEncoder()
    template = capture_and_enroll(student_id, full_name, model)
    print(f"Saved ArcFace template for {student_id} with shape {template.shape}")

    run_live_recognition(model)


def capture_and_enroll(student_id: str, full_name: str, model: ArcFaceEncoder) -> np.ndarray:
    add_student(student_id=student_id, full_name=full_name)
    cap = open_camera()
    embeddings: list[np.ndarray] = []

    print("Enrollment started. Look at the camera. Press q to cancel.")
    while len(embeddings) < SAMPLE_COUNT:
        ok, frame = cap.read()
        if not ok:
            break

        face_items = crop_faces(frame)
        status_text = f"Captured {len(embeddings)}/{SAMPLE_COUNT}"

        try:
            bbox, face_bgr = get_best_face_for_arcface(frame, face_items)
            encoding = model.encode(face_bgr)
            embeddings.append(encoding["embedding"])
            status_text = f"Captured {len(embeddings)}/{SAMPLE_COUNT}"
            color = (0, 180, 0)
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        except ValueError as exc:
            status_text = str(exc)
            color = (0, 0, 255)

        cv2.putText(frame, status_text[:80], (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("ArcFace Enrollment", frame)
        if cv2.waitKey(250) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not embeddings:
        raise RuntimeError("No ArcFace embeddings captured.")

    template = np.mean(np.vstack(embeddings), axis=0).astype(np.float32)
    template = model.l2_normalize(template)
    save_template(student_id=student_id, model_name=MODEL_NAME, template=template)
    return template


def run_live_recognition(model: ArcFaceEncoder) -> None:
    gallery = load_gallery(MODEL_NAME)
    session_id = f"arcface_single_{date.today().isoformat()}"
    create_session(
        session_id=session_id,
        course_code="CV7",
        session_date=date.today().isoformat(),
        model_name=MODEL_NAME,
    )

    cap = open_camera()
    print("Recognition started. Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        face_items = crop_faces(frame)
        try:
            bbox, face_bgr = get_best_face_for_arcface(frame, face_items)
            x, y, w, h = bbox
            prediction = model.predict(face_bgr, gallery)
        except ValueError:
            prediction = {"student_id": None, "confidence": 0.0, "status": "unknown"}
            bbox = (0, 0, frame.shape[1], frame.shape[0])
            x, y, w, h = bbox

        student_id = prediction["student_id"]
        confidence = float(prediction["confidence"])
        status = prediction["status"]
        name = get_student_name(student_id) if student_id else None

        if status == "known" and student_id is not None:
            mark_attendance(session_id, student_id, confidence)
            log_recognition_event(
                session_id=session_id,
                student_id=student_id,
                predicted_label=student_id,
                confidence=confidence,
                event_type="recognized",
            )
            color = (0, 180, 0)
            label = f"{student_id} {name or ''} {confidence:.2f} known"
        else:
            log_recognition_event(
                session_id=session_id,
                student_id=None,
                predicted_label="Unknown",
                confidence=confidence,
                event_type="unknown",
            )
            color = (0, 0, 255)
            label = f"Unknown {confidence:.2f}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label.strip(), (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("ArcFace Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def largest_crop(face_items: list[tuple[tuple[int, int, int, int], np.ndarray]]):
    return max(face_items, key=lambda item: item[0][2] * item[0][3])


def get_best_face_for_arcface(
    frame: np.ndarray,
    face_items: list[tuple[tuple[int, int, int, int], np.ndarray]],
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    if face_items:
        return largest_crop(face_items)
    return (0, 0, frame.shape[1], frame.shape[0]), frame


def open_camera() -> cv2.VideoCapture:
    for camera_index in (0, 1, 2):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"Using camera index {camera_index}")
                return cap
        cap.release()

    raise RuntimeError(
        "Could not open a webcam. Enable Camera access for Visual Studio Code or Terminal "
        "in System Settings > Privacy & Security > Camera, then restart VS Code."
    )


def get_student_name(student_id: str | None) -> str | None:
    if student_id is None:
        return None
    with connect() as conn:
        row = conn.execute(
            "SELECT full_name FROM students WHERE student_id = ?",
            (student_id,),
        ).fetchone()
    return row["full_name"] if row else None


if __name__ == "__main__":
    main()

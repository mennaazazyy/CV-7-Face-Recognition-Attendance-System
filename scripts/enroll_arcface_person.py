from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import cv2
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db import add_student, init_db, save_template  # noqa: E402
from src.detection import crop_faces  # noqa: E402
from src.models.arcface_encoder import ArcFaceEncoder  # noqa: E402


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DEFAULT_SAMPLE_COUNT = 10
MODEL_NAME = "arcface"


def main() -> None:
    args = parse_args()
    init_db()
    model = ArcFaceEncoder()
    template = capture_template(args.student_id, args.full_name, model, args.samples)
    add_student(student_id=args.student_id, full_name=args.full_name)
    save_template(student_id=args.student_id, model_name=MODEL_NAME, template=template)
    print(f"Saved ArcFace enrollment for {args.student_id} {args.full_name} ({args.samples} samples requested).")


def parse_args():
    parser = ArgumentParser(description="Enroll one person into the ArcFace gallery using the webcam.")
    parser.add_argument("--id", required=True, dest="student_id", help="Student ID, e.g. 22-101004")
    parser.add_argument("--name", required=True, dest="full_name", help="Full name, e.g. \"Sara Ahmed\"")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT, help="Number of face samples to capture.")
    return parser.parse_args()


def capture_template(
    student_id: str,
    full_name: str,
    model: ArcFaceEncoder,
    sample_count: int,
) -> np.ndarray:
    cap = open_camera()
    embeddings: list[np.ndarray] = []

    print(f"Enrollment started for {student_id} {full_name}. Press q to cancel.")
    while len(embeddings) < sample_count:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame could not be read.")
            break

        face_items = crop_faces(frame)
        try:
            bbox, face_bgr = largest_crop(face_items)
            encoding = model.encode(face_bgr)
            embeddings.append(encoding["embedding"])
            color = (0, 180, 0)
            status_text = f"Captured {len(embeddings)}/{sample_count}"
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        except ValueError as exc:
            color = (0, 0, 255)
            status_text = str(exc)

        cv2.putText(frame, status_text[:80], (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("ArcFace Enrollment", frame)
        if cv2.waitKey(250) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not embeddings:
        raise RuntimeError("No ArcFace embeddings captured. Enrollment was not saved.")

    template = np.mean(np.vstack(embeddings), axis=0).astype(np.float32)
    return model.l2_normalize(template)


def largest_crop(face_items: list[tuple[tuple[int, int, int, int], np.ndarray]]):
    if not face_items:
        raise ValueError("No face detected.")
    return max(face_items, key=lambda item: item[0][2] * item[0][3])


def open_camera() -> cv2.VideoCapture:
    for camera_index in (0, 1, 2):
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 15)
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


if __name__ == "__main__":
    main()

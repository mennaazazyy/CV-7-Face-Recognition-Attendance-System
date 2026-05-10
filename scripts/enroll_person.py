#!/usr/bin/env python3
"""
Generic webcam enrollment that works with any model.

Usage:
    python scripts/enroll_person.py --id "22-101004" --name "Sara Ahmed" --model arcface
    python scripts/enroll_person.py --id "22-101005" --name "Ali Hassan" --model facenet
    python scripts/enroll_person.py --id "22-101006" --name "Nour Wael"  --model dlib
    python scripts/enroll_person.py --id "22-101007" --name "Yara Tarek" --model lbph
    python scripts/enroll_person.py --id "22-101004" --name "Sara Ahmed" --model arcface --verify
"""
from __future__ import annotations

import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import cv2
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SUPPORTED_MODELS, ACTIVE_MODEL  # noqa: E402
from src.database.db import add_student, init_db, load_gallery, save_template  # noqa: E402
from src.detection import crop_faces  # noqa: E402
from src.models import create_model  # noqa: E402
from src.models.base_model import FaceRecognitionModel  # noqa: E402


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DEFAULT_SAMPLE_COUNT = 20
MIN_CAPTURE_INTERVAL = 0.4
DIVERSITY_THRESHOLD = 0.995
GUIDANCE = [
    "Look straight",
    "Turn head slightly left",
    "Turn head slightly right",
    "Tilt head up a little",
    "Tilt head down a little",
]
VERIFY_ATTEMPTS = 15


def main() -> None:
    args = parse_args()
    init_db()
    model = create_model(args.model)
    template = capture_template(args.student_id, args.full_name, model, args.samples)
    add_student(student_id=args.student_id, full_name=args.full_name)
    save_template(student_id=args.student_id, model_name=args.model, template=template)
    print(f"Saved {args.model} enrollment for {args.student_id} {args.full_name} ({args.samples} samples).")

    if args.verify:
        gallery = load_gallery(args.model)
        verify_enrollment(args.student_id, args.full_name, model, gallery)


def parse_args():
    parser = ArgumentParser(description="Enroll one person into the gallery using the webcam.")
    parser.add_argument("--id", required=True, dest="student_id", help="Student ID, e.g. 22-101004")
    parser.add_argument("--name", required=True, dest="full_name", help='Full name, e.g. "Sara Ahmed"')
    parser.add_argument(
        "--model",
        default=ACTIVE_MODEL,
        choices=SUPPORTED_MODELS,
        help=f"Recognition model to use (default: {ACTIVE_MODEL}).",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT, help="Number of face samples to capture.")
    parser.add_argument("--verify", action="store_true", help="After enrollment, verify the model recognises the person.")
    return parser.parse_args()


def capture_template(
    student_id: str,
    full_name: str,
    model: FaceRecognitionModel,
    sample_count: int,
) -> np.ndarray:
    cap = open_camera()
    embeddings: list[np.ndarray] = []
    last_capture_time = 0.0
    last_embedding: np.ndarray | None = None

    print(f"[{model.name}] Enrollment started for {student_id} {full_name}.")
    print("Follow the on-screen guidance to capture diverse poses. Press q to cancel early.")

    while len(embeddings) < sample_count:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame could not be read.")
            break

        guidance_text = GUIDANCE[len(embeddings) % len(GUIDANCE)]
        face_items = crop_faces(frame)
        now = time.monotonic()
        try:
            bbox, face_bgr = largest_crop(face_items)
            encoding = model.encode(face_bgr)
            new_embedding = encoding["embedding"]

            too_soon = (now - last_capture_time) < MIN_CAPTURE_INTERVAL
            too_similar = (
                last_embedding is not None
                and FaceRecognitionModel.cosine_similarity(new_embedding, last_embedding) > DIVERSITY_THRESHOLD
            )

            if too_soon or too_similar:
                color = (0, 200, 255)
                status_text = f"{len(embeddings)}/{sample_count} - {guidance_text}"
            else:
                embeddings.append(new_embedding)
                last_embedding = new_embedding
                last_capture_time = now
                color = (0, 180, 0)
                status_text = f"Captured {len(embeddings)}/{sample_count} - {guidance_text}"

            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        except ValueError as exc:
            color = (0, 0, 255)
            status_text = str(exc)

        cv2.putText(frame, status_text[:90], (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.imshow(f"{model.name} Enrollment", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not embeddings:
        raise RuntimeError("No embeddings captured. Enrollment was not saved.")

    print(f"Captured {len(embeddings)} diverse samples.")
    template = np.mean(np.vstack(embeddings), axis=0).astype(np.float32)
    if not model.lower_score_is_better:
        template = model.l2_normalize(template)
    return template


def verify_enrollment(
    student_id: str,
    full_name: str,
    model: FaceRecognitionModel,
    gallery: list[tuple[str, np.ndarray]],
) -> None:
    print(f"\nVerifying enrollment for {student_id} {full_name} — show your face to the camera.")
    print(f"Collecting {VERIFY_ATTEMPTS} test frames… press q to skip.")

    cap = open_camera()
    correct = 0
    attempts = 0
    frame_count = 0

    while attempts < VERIFY_ATTEMPTS:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        if frame_count % 5 != 0:
            cv2.imshow("Verification", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        face_items = crop_faces(frame)
        try:
            _bbox, face_bgr = largest_crop(face_items)
            prediction = model.predict(face_bgr, gallery)
            attempts += 1
            matched = prediction["student_id"] == student_id
            if matched:
                correct += 1
            conf = float(prediction["confidence"])
            color = (0, 180, 0) if matched else (0, 0, 255)
            label = f"{'OK' if matched else 'MISS'} {correct}/{attempts}  conf={conf:.2f}"
        except ValueError:
            label = "No face detected"
            color = (0, 200, 255)

        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Verification", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if attempts == 0:
        print("Verification skipped — no faces detected.")
        return

    pct = correct / attempts * 100
    status = "PASS" if pct >= 70 else "WARN" if pct >= 40 else "FAIL"
    print(f"\nVerification result [{status}]: {correct}/{attempts} recognised correctly ({pct:.0f}%)")
    if pct < 70:
        print("  Tip: re-enroll with --samples 30 and vary your pose during capture.")


def largest_crop(face_items):
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

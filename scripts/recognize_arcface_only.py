from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("DEEPFACE_HOME", str(PROJECT_ROOT / ".deepface-cache"))

import cv2

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_THRESHOLDS  # noqa: E402
from src.antispoof.minifasnet import AntiSpoofChecker, BlinkChallengeAntiSpoofChecker  # noqa: E402
from src.database.db import connect, load_gallery  # noqa: E402
from src.detection import crop_faces  # noqa: E402
from src.models.arcface_encoder import ArcFaceEncoder  # noqa: E402


DEFAULT_PROCESS_EVERY_N_FRAMES = 10
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def main() -> None:
    args = parse_args()
    model = ArcFaceEncoder()
    model.threshold = args.threshold
    liveness_checker = BlinkChallengeAntiSpoofChecker() if args.antispoof else None
    liveness_state = None
    deep_antispoof_checker = AntiSpoofChecker() if args.deep_antispoof else None
    deep_antispoof_state: bool | None = None
    process_every = min(args.process_every, 3) if args.antispoof else args.process_every
    gallery = [] if args.empty_gallery else load_gallery("arcface")

    if not gallery:
        print("No ArcFace templates loaded. Every detected face will be labeled Unknown.")
    else:
        print(f"Loaded {len(gallery)} ArcFace template(s). Threshold: {model.threshold:.2f}")
        for student_id, _ in gallery:
            name = get_student_name(student_id)
            print(f"  - {student_id} {name or ''}".strip())

    cap = open_camera()
    print("ArcFace recognition test. Show your face for known, then another person for Unknown. Press q to quit.")
    if liveness_checker is not None:
        print("Anti-spoofing is ON. Open your eyes, blink once, then open your eyes again.")
    if deep_antispoof_checker is not None:
        print("Deep anti-spoofing is ON. This may run slower on CPU.")
    frame_count = 0
    last_prediction = {"student_id": None, "confidence": 0.0, "status": "unknown"}
    last_bbox: tuple[int, int, int, int] | None = None
    no_face_detected = True

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame could not be read.")
            break

        frame_count += 1
        if frame_count % process_every == 0:
            face_items = crop_faces(frame)
            no_face_detected = not face_items
            try:
                last_bbox, face_bgr = largest_crop(face_items)
                if liveness_checker is not None:
                    liveness_state = liveness_checker.update(frame, last_bbox)
                liveness_passed = liveness_checker is None or liveness_state.is_live
                if liveness_passed and deep_antispoof_checker is not None:
                    deep_antispoof_state = deep_antispoof_checker.is_real(face_bgr)
                deep_passed = deep_antispoof_checker is None or deep_antispoof_state is True

                if liveness_passed and deep_passed:
                    last_prediction = model.predict(face_bgr, gallery)
                else:
                    last_prediction = {"student_id": None, "confidence": 0.0, "status": "unknown"}
            except ValueError:
                last_bbox = None
                if liveness_checker is not None:
                    liveness_state = liveness_checker.update(frame, None)
                deep_antispoof_state = None
                last_prediction = {"student_id": None, "confidence": 0.0, "status": "unknown"}

        prediction = last_prediction
        student_id = prediction["student_id"]
        confidence = float(prediction["confidence"])
        status = prediction["status"]
        name = get_student_name(student_id) if student_id else None

        if liveness_checker is not None and (liveness_state is None or not liveness_state.is_live):
            color = (0, 200, 255)
            progress = 0 if liveness_state is None else int(liveness_state.progress * 100)
            message = "Open your eyes, blink once" if liveness_state is None else liveness_state.message
            label = f"Liveness {progress}% - {message}"
        elif deep_antispoof_checker is not None and deep_antispoof_state is not True:
            color = (0, 0, 255)
            label = "Spoof detected" if deep_antispoof_state is False else "Checking spoof..."
        elif status == "known" and student_id is not None:
            color = (0, 180, 0)
            label = f"{student_id} {name or ''} {confidence:.2f} known"
        elif no_face_detected:
            color = (0, 200, 255)
            label = "No face detected"
        else:
            color = (0, 0, 255)
            label = f"Unknown {confidence:.2f}"

        if last_bbox is not None:
            x, y, w, h = last_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_origin = (x, max(y - 10, 25))
        else:
            label_origin = (30, 45)

        cv2.putText(frame, label.strip(), label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("ArcFace Recognition Only", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = ArgumentParser(description="Run a lightweight ArcFace webcam recognition test.")
    parser.add_argument(
        "--empty-gallery",
        action="store_true",
        help="Ignore saved templates and verify that detected faces are labeled Unknown.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=MODEL_THRESHOLDS["arcface"],
        help="Cosine similarity threshold for known matches. Higher is stricter.",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=DEFAULT_PROCESS_EVERY_N_FRAMES,
        help="Run recognition every N frames to reduce CPU usage.",
    )
    parser.add_argument(
        "--antispoof",
        action="store_true",
        help="Require a lightweight blink liveness check before recognizing a face.",
    )
    parser.add_argument(
        "--deep-antispoof",
        action="store_true",
        help="Require DeepFace anti-spoofing before recognizing a face.",
    )
    return parser.parse_args()


def largest_crop(face_items: list[tuple[tuple[int, int, int, int], object]]):
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

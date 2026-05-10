#!/usr/bin/env python3
"""
Generic live webcam recognition that works with any model.

Usage:
    python scripts/recognize_live.py --model arcface
    python scripts/recognize_live.py --model facenet --threshold 0.60
    python scripts/recognize_live.py --model dlib
    python scripts/recognize_live.py --model lbph
    python scripts/recognize_live.py --model arcface --show-scores
"""
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

from src.config import MODEL_THRESHOLDS, SUPPORTED_MODELS, ACTIVE_MODEL  # noqa: E402
from src.database.db import connect, load_gallery  # noqa: E402
from src.detection import crop_faces  # noqa: E402
from src.models import create_model  # noqa: E402
from src.models.base_model import FaceRecognitionModel  # noqa: E402


DEFAULT_PROCESS_EVERY = 10
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def main() -> None:
    args = parse_args()
    model = create_model(args.model)
    if args.threshold is not None:
        model.threshold = args.threshold
    gallery = [] if args.empty_gallery else load_gallery(args.model)
    show_scores = args.show_scores
    process_every = args.process_every

    if not gallery:
        print(f"[{model.name}] No templates loaded. Every detected face will be labeled Unknown.")
    else:
        print(f"[{model.name}] Loaded {len(gallery)} template(s). Threshold: {model.threshold:.2f}")
        for student_id, _ in gallery:
            name = get_student_name(student_id)
            print(f"  - {student_id} {name or ''}".strip())

    cap = open_camera()
    print(f"[{model.name}] Recognition running. Press q to quit.")
    frame_count = 0
    last_prediction: dict = {"student_id": None, "confidence": 0.0, "status": "unknown"}
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
                last_prediction = model.predict(face_bgr, gallery)

                if show_scores and gallery:
                    try:
                        enc = model.encode(face_bgr)
                        query = enc["embedding"]
                        scores = []
                        for sid, emb in gallery:
                            if model.lower_score_is_better:
                                sc = float(np.linalg.norm(
                                    np.asarray(query, dtype=np.float32).reshape(-1)
                                    - np.asarray(emb, dtype=np.float32).reshape(-1)
                                ))
                            else:
                                sc = FaceRecognitionModel.cosine_similarity(query, emb)
                            scores.append((sid, sc))
                        scores.sort(key=lambda x: x[1], reverse=not model.lower_score_is_better)
                        last_prediction["_scores"] = scores[:3]
                    except Exception:
                        pass
            except ValueError:
                last_bbox = None
                last_prediction = {"student_id": None, "confidence": 0.0, "status": "unknown"}

        prediction = last_prediction
        student_id = prediction["student_id"]
        confidence = float(prediction["confidence"])
        status = prediction["status"]
        name = get_student_name(student_id) if student_id else None

        if status == "known" and student_id is not None:
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

        if show_scores:
            scores_list = prediction.get("_scores", [])
            for i, (sid, sc) in enumerate(scores_list):
                name_i = get_student_name(sid) or sid
                if model.lower_score_is_better:
                    above = sc <= model.threshold
                else:
                    above = sc >= model.threshold
                score_label = f"{name_i}: {sc:.3f}"
                score_color = (0, 200, 255) if above else (150, 150, 150)
                cv2.putText(frame, score_label, (10, CAMERA_HEIGHT - 20 - i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, score_color, 1)

        model_label = f"[{model.name}] thr={model.threshold:.2f}"
        cv2.putText(frame, model_label, (CAMERA_WIDTH - 250, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(f"{model.name} Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = ArgumentParser(description="Run live webcam recognition with any model.")
    parser.add_argument(
        "--model",
        default=ACTIVE_MODEL,
        choices=SUPPORTED_MODELS,
        help=f"Recognition model to use (default: {ACTIVE_MODEL}).",
    )
    parser.add_argument("--empty-gallery", action="store_true", help="Ignore saved templates (test unknown rejection).")
    parser.add_argument("--threshold", type=float, default=None, help="Override the default similarity threshold.")
    parser.add_argument("--process-every", type=int, default=DEFAULT_PROCESS_EVERY, help="Run recognition every N frames.")
    parser.add_argument("--show-scores", action="store_true", help="Overlay top-3 gallery match scores for debugging.")
    return parser.parse_args()


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

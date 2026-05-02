from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import ACTIVE_MODEL, ENROLLMENT_IMAGES_PER_STUDENT, ENROLLMENT_DIR, WEBCAM_INDEX
from src.database.db import add_student, init_db, save_template
from src.detection import crop_faces
from src.models import create_model


def enroll_from_images(
    student_id: str,
    full_name: str,
    image_paths: list[Path],
    model_name: str = ACTIVE_MODEL,
    email: str | None = None,
) -> np.ndarray:
    """Enroll one student from images containing their face."""
    init_db()
    add_student(student_id=student_id, full_name=full_name, email=email)
    model = create_model(model_name)

    embeddings = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        face_crops = crop_faces(image)
        if not face_crops:
            raise ValueError(f"No face detected in enrollment image: {image_path}")

        _, face_bgr = face_crops[0]
        encoding = model.encode(face_bgr)
        embeddings.append(encoding["embedding"])

    if not embeddings:
        raise ValueError("Enrollment needs at least one valid face image.")

    template = np.mean(np.vstack(embeddings), axis=0).astype(np.float32)
    if not model.lower_score_is_better:
        template = model.l2_normalize(template)

    save_template(student_id=student_id, model_name=model_name, template=template)
    return template


def capture_enrollment_images(
    student_id: str,
    count: int = ENROLLMENT_IMAGES_PER_STUDENT,
    camera_index: int = WEBCAM_INDEX,
) -> list[Path]:
    """Capture full webcam frames. Face cropping happens during enrollment."""
    output_dir = ENROLLMENT_DIR / student_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    saved_paths: list[Path] = []

    while len(saved_paths) < count:
        ok, frame = cap.read()
        if not ok:
            break

        image_path = output_dir / f"{student_id}_{len(saved_paths) + 1:03d}.jpg"
        cv2.imwrite(str(image_path), frame)
        saved_paths.append(image_path)

        cv2.imshow("Enrollment capture - press q to stop", frame)
        if cv2.waitKey(250) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved_paths

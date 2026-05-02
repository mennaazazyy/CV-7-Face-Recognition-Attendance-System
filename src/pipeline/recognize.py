from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import ACTIVE_MODEL, UNKNOWN_LABEL, UNKNOWN_STATUS
from src.database.db import load_gallery
from src.detection import crop_faces
from src.models import create_model


@dataclass(frozen=True)
class RecognitionResult:
    student_id: str | None
    confidence: float
    status: str
    bbox: tuple[int, int, int, int]

    @property
    def label(self) -> str:
        return self.student_id if self.student_id is not None else UNKNOWN_LABEL


class Recognizer:
    """Loads one model and its SQLite gallery, then recognizes faces in frames."""

    def __init__(self, model_name: str = ACTIVE_MODEL) -> None:
        self.model_name = model_name
        self.model = create_model(model_name)
        self.gallery = load_gallery(model_name)

    def reload_gallery(self) -> None:
        self.gallery = load_gallery(self.model_name)

    def recognize_frame(self, frame_bgr: np.ndarray) -> list[RecognitionResult]:
        results: list[RecognitionResult] = []
        for bbox, face_bgr in crop_faces(frame_bgr):
            prediction = self.model.predict(face_bgr, self.gallery)
            results.append(
                RecognitionResult(
                    student_id=prediction["student_id"],
                    confidence=prediction["confidence"],
                    status=prediction["status"],
                    bbox=bbox,
                )
            )
        return results

    def recognize_face_crop(self, face_bgr: np.ndarray) -> RecognitionResult:
        prediction = self.model.predict(face_bgr, self.gallery)
        return RecognitionResult(
            student_id=prediction["student_id"],
            confidence=prediction["confidence"],
            status=prediction["status"],
            bbox=(0, 0, face_bgr.shape[1], face_bgr.shape[0]),
        )


def is_unknown(result: RecognitionResult) -> bool:
    return result.status == UNKNOWN_STATUS or result.student_id is None

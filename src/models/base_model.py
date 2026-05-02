from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, TypedDict

import cv2
import numpy as np

from src.config import KNOWN_STATUS, UNKNOWN_STATUS


class EncodingOutput(TypedDict):
    model_name: str
    embedding: np.ndarray


class PredictionOutput(TypedDict):
    student_id: str | None
    confidence: float
    status: Literal["known", "unknown"]


class FaceRecognitionModel(ABC):
    """Frozen interface shared by all recognition models.

    Every encoder receives a cropped BGR face image and returns:
        {"model_name": "...", "embedding": np.ndarray}

    Every gallery prediction returns:
        {"student_id": "student_001", "confidence": 0.87, "status": "known"}
    or:
        {"student_id": None, "confidence": 0.31, "status": "unknown"}
    """

    name: str
    threshold: float
    lower_score_is_better: bool = False

    @abstractmethod
    def encode(self, face_bgr: np.ndarray) -> EncodingOutput:
        """Return the standard encoding output for one cropped face."""

    def train(self, student_faces: dict[str, list[np.ndarray]]) -> None:
        """Optional hook for trainable models such as LBPH."""

    def save(self, path: Path) -> None:
        """Optional hook for model-specific files."""

    def load(self, path: Path) -> None:
        """Optional hook for model-specific files."""

    def predict(self, face_bgr: np.ndarray, gallery: list[tuple[str, np.ndarray]]) -> PredictionOutput:
        encoding = self.encode(face_bgr)
        return self.match_embedding(encoding["embedding"], gallery)

    def match_embedding(self, query_embedding: np.ndarray, gallery: list[tuple[str, np.ndarray]]) -> PredictionOutput:
        if not gallery:
            return {"student_id": None, "confidence": 0.0, "status": UNKNOWN_STATUS}

        best_student_id: str | None = None
        best_score: float | None = None

        for student_id, stored_embedding in gallery:
            score = self.compare(query_embedding, stored_embedding)
            if best_score is None or self._is_better(score, best_score):
                best_student_id = student_id
                best_score = score

        assert best_score is not None
        is_known = best_score <= self.threshold if self.lower_score_is_better else best_score >= self.threshold
        if not is_known:
            return {"student_id": None, "confidence": float(best_score), "status": UNKNOWN_STATUS}
        return {"student_id": best_student_id, "confidence": float(best_score), "status": KNOWN_STATUS}

    def compare(self, query_embedding: np.ndarray, stored_embedding: np.ndarray) -> float:
        if self.lower_score_is_better:
            return float(np.linalg.norm(self._as_vector(query_embedding) - self._as_vector(stored_embedding)))
        return self.cosine_similarity(query_embedding, stored_embedding)

    def make_output(self, embedding: np.ndarray) -> EncodingOutput:
        return {"model_name": self.name, "embedding": self._as_vector(embedding)}

    def preprocess(self, face_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty face image received.")
        return cv2.resize(face_bgr, size)

    @staticmethod
    def l2_normalize(vector: np.ndarray) -> np.ndarray:
        vector = FaceRecognitionModel._as_vector(vector)
        return vector / (np.linalg.norm(vector) + 1e-8)

    @staticmethod
    def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left = FaceRecognitionModel._as_vector(left)
        right = FaceRecognitionModel._as_vector(right)
        return float(np.dot(left, right) / ((np.linalg.norm(left) * np.linalg.norm(right)) + 1e-8))

    @staticmethod
    def _as_vector(vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, dtype=np.float32).reshape(-1)

    def _is_better(self, score: float, best_score: float) -> bool:
        return score < best_score if self.lower_score_is_better else score > best_score

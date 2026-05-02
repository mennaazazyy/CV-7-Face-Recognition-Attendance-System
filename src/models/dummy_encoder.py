from __future__ import annotations

import hashlib

import cv2
import numpy as np

from src.config import IMAGE_SIZE, MODEL_THRESHOLDS
from src.models.base_model import EncodingOutput, FaceRecognitionModel


class DummyEncoder(FaceRecognitionModel):
    """Deterministic fake encoder used only to test the foundation pipeline."""

    name = "dummy"
    threshold = MODEL_THRESHOLDS[name]

    def encode(self, face_bgr: np.ndarray) -> EncodingOutput:
        face_bgr = self.preprocess(face_bgr, IMAGE_SIZE)
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (8, 8))
        digest = hashlib.sha256(small.tobytes()).digest()
        values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        embedding = np.tile(values, 4)[:128]
        return self.make_output(self.l2_normalize(embedding))

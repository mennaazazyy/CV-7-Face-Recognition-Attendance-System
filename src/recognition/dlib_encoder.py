import cv2
import numpy as np
import face_recognition
from .encoder import Encoder


class DlibEncoder(Encoder):
    """128-d dlib ResNet embeddings via the face_recognition library."""

    def encode(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if not encs:
            raise ValueError("No face encoding found — ensure the crop contains a face.")
        return self._l2_normalize(encs[0])

    def compare(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        return self.cosine_similarity(emb_a, emb_b)

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from .encoder import Encoder
from src.config import INSIGHTFACE_MODEL


class ArcFaceEncoder(Encoder):
    """InsightFace buffalo_l ArcFace — primary model (512-d embeddings)."""

    def __init__(self, ctx_id: int = 0):
        self._app = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def encode(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        faces = self._app.get(face_crop_bgr)
        if not faces:
            raise ValueError("No face detected in crop — pass a tighter crop.")
        emb = faces[0].embedding
        return self._l2_normalize(emb)

    def compare(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        return self.cosine_similarity(emb_a, emb_b)

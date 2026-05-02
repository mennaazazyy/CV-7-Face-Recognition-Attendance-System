from __future__ import annotations

import os

import cv2
import numpy as np

from src.config import ROOT_DIR
from src.config import MODEL_THRESHOLDS
from src.models.base_model import EncodingOutput, FaceRecognitionModel

os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".matplotlib-cache"))


class ArcFaceEncoder(FaceRecognitionModel):
    name = "arcface"
    threshold = MODEL_THRESHOLDS[name]
    lower_score_is_better = False
    embedding_dim = 512

    def __init__(self) -> None:
        self._app = None

    def encode(self, face_bgr: np.ndarray) -> EncodingOutput:
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("ArcFace received an empty face image.")

        prepared_face = self._prepare_crop(face_bgr)
        faces = self._get_app().get(prepared_face)
        if not faces:
            raise ValueError("ArcFace could not detect a face in the provided crop.")

        face = self._largest_face(faces)
        if not hasattr(face, "embedding") or face.embedding is None:
            raise ValueError("ArcFace failed to extract an embedding.")

        embedding = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError("ArcFace embedding must be 512-dimensional")

        embedding = self.l2_normalize(embedding)
        return self.make_output(embedding)

    def compare(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        return self.cosine_similarity(embedding_a, embedding_b)

    def _get_app(self):
        if self._app is None:
            try:
                from insightface.app import FaceAnalysis
            except ImportError as exc:
                raise ImportError(
                    "ArcFaceEncoder requires InsightFace. Install it with: "
                    "pip install insightface onnxruntime"
                ) from exc

            try:
                app = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"],
                    allowed_modules=["detection", "recognition"],
                )
            except TypeError:
                app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._app = app
        return self._app

    @staticmethod
    def _largest_face(faces):
        return max(
            faces,
            key=lambda face: max(float(face.bbox[2] - face.bbox[0]), 0.0)
            * max(float(face.bbox[3] - face.bbox[1]), 0.0),
        )

    @staticmethod
    def _prepare_crop(face_bgr: np.ndarray) -> np.ndarray:
        min_side = min(face_bgr.shape[:2])
        if min_side < 160:
            scale = 160 / max(min_side, 1)
            width = int(face_bgr.shape[1] * scale)
            height = int(face_bgr.shape[0] * scale)
            face_bgr = cv2.resize(face_bgr, (width, height))

        pad_y = max(int(face_bgr.shape[0] * 0.25), 20)
        pad_x = max(int(face_bgr.shape[1] * 0.25), 20)
        return cv2.copyMakeBorder(
            face_bgr,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

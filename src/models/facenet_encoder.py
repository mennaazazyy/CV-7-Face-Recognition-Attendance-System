from __future__ import annotations

import cv2
import numpy as np

from src.config import IMAGE_SIZE, MODEL_THRESHOLDS
from src.models.base_model import EncodingOutput, FaceRecognitionModel


class FaceNetEncoder(FaceRecognitionModel):
    name = "facenet"
    threshold = MODEL_THRESHOLDS[name]
    lower_score_is_better = False
    embedding_dim = 512

    def __init__(self) -> None:
        self._model = None
        self._device = None

    def encode(self, face_bgr: np.ndarray) -> EncodingOutput:
        face_bgr = self.preprocess(face_bgr, IMAGE_SIZE)

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._to_tensor(face_rgb)

        import torch

        model = self._get_model()
        with torch.no_grad():
            embedding = model(tensor).cpu().numpy().reshape(-1)

        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"FaceNet embedding must be {self.embedding_dim}-dimensional, got {embedding.shape[0]}"
            )

        embedding = self.l2_normalize(embedding)
        return self.make_output(embedding)

    def _to_tensor(self, face_rgb: np.ndarray):
        import torch

        face = face_rgb.astype(np.float32)
        face = (face - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))
        tensor = torch.from_numpy(face).unsqueeze(0)
        return tensor.to(self._device)

    def _get_model(self):
        if self._model is None:
            try:
                import torch
                from facenet_pytorch import InceptionResnetV1
            except ImportError as exc:
                raise ImportError(
                    "FaceNetEncoder requires facenet-pytorch. Install it with: "
                    "pip install facenet-pytorch"
                ) from exc

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self._device)
        return self._model

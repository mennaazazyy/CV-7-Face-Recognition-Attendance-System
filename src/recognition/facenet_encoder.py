import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from .encoder import Encoder
from src.config import FACENET_MODEL


class FaceNetEncoder(Encoder):
    """FaceNet via facenet-pytorch (512-d embeddings, VGGFace2 weights)."""

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mtcnn = MTCNN(image_size=160, margin=0, device=self._device)
        self._model = (
            InceptionResnetV1(pretrained=FACENET_MODEL).eval().to(self._device)
        )

    def encode(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image

        pil = Image.fromarray(rgb)
        tensor = self._mtcnn(pil)
        if tensor is None:
            # fall back: resize and normalise manually
            resized = cv2.resize(rgb, (160, 160))
            tensor = torch.tensor(resized).permute(2, 0, 1).float() / 255.0
            tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model(tensor).cpu().numpy().flatten()
        return self._l2_normalize(emb)

    def compare(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        return self.cosine_similarity(emb_a, emb_b)

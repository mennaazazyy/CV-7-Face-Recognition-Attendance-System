from abc import ABC, abstractmethod
import numpy as np


class Encoder(ABC):
    """Abstract interface every encoder must implement."""

    @abstractmethod
    def encode(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        """Return an L2-normalised 1-D embedding vector."""

    @abstractmethod
    def compare(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Return cosine similarity in [0, 1].  Higher = more similar."""

    # ── shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

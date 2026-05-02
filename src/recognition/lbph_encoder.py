import cv2
import numpy as np
from .encoder import Encoder


class LBPHEncoder(Encoder):
    """Local Binary Pattern Histogram — traditional baseline (OpenCV-contrib)."""

    def encode(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))

        # Compute LBP histogram manually for a fixed-size feature vector.
        # opencv-contrib's LBPH recogniser is a full classifier, not a standalone
        # feature extractor, so we roll our own here.
        radius, n_points = 1, 8
        lbp = np.zeros_like(gray, dtype=np.uint8)
        for i in range(radius, gray.shape[0] - radius):
            for j in range(radius, gray.shape[1] - radius):
                center = gray[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    ni = int(round(i - radius * np.sin(angle)))
                    nj = int(round(j + radius * np.cos(angle)))
                    code |= (int(gray[ni, nj]) >= int(center)) << k
                lbp[i, j] = code

        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        return self._l2_normalize(hist.astype(np.float32))

    def compare(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        return self.cosine_similarity(emb_a, emb_b)

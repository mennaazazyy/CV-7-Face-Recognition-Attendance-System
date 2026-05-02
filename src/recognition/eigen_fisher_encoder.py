import cv2
import numpy as np
from .encoder import Encoder


class _PCAEncoder(Encoder):
    """Shared PCA fitting logic for Eigenfaces and Fisherfaces."""

    FACE_SIZE = (100, 100)

    def __init__(self, n_components: int = 150):
        self.n_components = n_components
        self._recognizer = None  # set by subclass after fit()

    def fit(self, images: list[np.ndarray], labels: list[int]) -> None:
        raise NotImplementedError

    def encode(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def compare(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        return self.cosine_similarity(emb_a, emb_b)

    @staticmethod
    def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, _PCAEncoder.FACE_SIZE)


class EigenfacesEncoder(_PCAEncoder):
    """OpenCV EigenFaceRecognizer — keeps projection as embedding."""

    def fit(self, images: list[np.ndarray], labels: list[int]) -> None:
        self._recognizer = cv2.face.EigenFaceRecognizer_create(
            num_components=self.n_components
        )
        grays = [self._preprocess(img) for img in images]
        self._recognizer.train(grays, np.array(labels))

    def encode(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        if self._recognizer is None:
            raise RuntimeError("Call fit() before encode().")
        gray = self._preprocess(face_crop_bgr)
        _, proj = self._recognizer.predict(gray)
        # proj is the reconstruction confidence; extract weights via project()
        mean = self._recognizer.getMean()
        evs = self._recognizer.getEigenVectors()
        flat = gray.flatten().astype(np.float64)
        weights = evs.T @ (flat - mean)
        return self._l2_normalize(weights.astype(np.float32))


class FisherfacesEncoder(_PCAEncoder):
    """OpenCV FisherFaceRecognizer — keeps LDA projection as embedding."""

    def fit(self, images: list[np.ndarray], labels: list[int]) -> None:
        self._recognizer = cv2.face.FisherFaceRecognizer_create(
            num_components=self.n_components
        )
        grays = [self._preprocess(img) for img in images]
        self._recognizer.train(grays, np.array(labels))

    def encode(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        if self._recognizer is None:
            raise RuntimeError("Call fit() before encode().")
        gray = self._preprocess(face_crop_bgr)
        mean = self._recognizer.getMean()
        evs = self._recognizer.getEigenVectors()
        flat = gray.flatten().astype(np.float64)
        weights = evs.T @ (flat - mean)
        return self._l2_normalize(weights.astype(np.float32))

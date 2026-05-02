import cv2
import numpy as np
from src.config import HAAR_CASCADE_PATH


class HaarDetector:
    """OpenCV Haar cascade – fast but least accurate; kept as ablation baseline."""

    def __init__(self):
        self._cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        faces = []
        for x, y, w, h in boxes:
            faces.append(
                {"bbox": (x, y, x + w, y + h), "landmarks": None, "score": 1.0}
            )
        return faces

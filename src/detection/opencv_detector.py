from __future__ import annotations

import cv2
import numpy as np

from src.config import MIN_FACE_SIZE


_DETECTOR: cv2.CascadeClassifier | None = None


def _get_detector() -> cv2.CascadeClassifier:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _DETECTOR


def detect_faces(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Input: full BGR image/frame.
    Output: list of bounding boxes as (x, y, w, h).
    """
    if frame is None or frame.size == 0:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = _get_detector().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes]


def crop_faces(frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    crops = []
    for x, y, w, h in detect_faces(frame):
        crops.append(((x, y, w, h), frame[y : y + h, x : x + w]))
    return crops

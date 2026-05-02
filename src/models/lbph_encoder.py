from pathlib import Path

import cv2
import numpy as np

from src.config import IMAGE_SIZE, MODEL_THRESHOLDS
from src.models.base_model import EncodingOutput, FaceRecognitionModel


class LBPHEncoder(FaceRecognitionModel):
    name = "lbph"
    threshold = MODEL_THRESHOLDS[name]
    lower_score_is_better = True

    def __init__(self) -> None:
        if not hasattr(cv2, "face"):
            raise ImportError("LBPH requires opencv-contrib-python.")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_to_student: dict[int, str] = {}
        self.student_to_label: dict[str, int] = {}
        self._trained = False

    def encode(self, face_bgr: np.ndarray) -> EncodingOutput:
        face_bgr = self.preprocess(face_bgr, IMAGE_SIZE)
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        return self.make_output(gray.astype(np.float32))

    def train(self, student_faces: dict[str, list[np.ndarray]]) -> None:
        images: list[np.ndarray] = []
        labels: list[int] = []

        for label, student_id in enumerate(sorted(student_faces)):
            self.label_to_student[label] = student_id
            self.student_to_label[student_id] = label
            for face in student_faces[student_id]:
                face = self.preprocess(face, IMAGE_SIZE)
                images.append(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
                labels.append(label)

        if not images:
            raise ValueError("No images provided for LBPH training.")

        self.recognizer.train(images, np.asarray(labels, dtype=np.int32))
        self._trained = True

    def save(self, path: Path) -> None:
        if not self._trained:
            raise RuntimeError("Cannot save LBPH model before training.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.recognizer.write(str(path))

    def load(self, path: Path) -> None:
        self.recognizer.read(str(path))
        self._trained = True

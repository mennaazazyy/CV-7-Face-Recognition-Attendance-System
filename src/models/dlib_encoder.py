import numpy as np

from src.config import IMAGE_SIZE, MODEL_THRESHOLDS
from src.models.base_model import EncodingOutput, FaceRecognitionModel


class DlibEncoder(FaceRecognitionModel):
    name = "dlib"
    threshold = MODEL_THRESHOLDS[name]

    def encode(self, face_bgr: np.ndarray) -> EncodingOutput:
        face_bgr = self.preprocess(face_bgr, IMAGE_SIZE)
        raise NotImplementedError(
            "Person 3: convert BGR to RGB, use face_recognition to extract a 128D embedding, "
            "L2-normalize it, and return self.make_output(embedding)."
        )

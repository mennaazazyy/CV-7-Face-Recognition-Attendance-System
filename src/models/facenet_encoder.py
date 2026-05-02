import numpy as np

from src.config import IMAGE_SIZE, MODEL_THRESHOLDS
from src.models.base_model import EncodingOutput, FaceRecognitionModel


class FaceNetEncoder(FaceRecognitionModel):
    name = "facenet"
    threshold = MODEL_THRESHOLDS[name]

    def encode(self, face_bgr: np.ndarray) -> EncodingOutput:
        face_bgr = self.preprocess(face_bgr, IMAGE_SIZE)
        raise NotImplementedError(
            "Person 2: load FaceNet through facenet-pytorch or DeepFace, extract an embedding, "
            "L2-normalize it, and return self.make_output(embedding)."
        )

import numpy as np

from src.models.base_model import FaceRecognitionModel


class CountingModel(FaceRecognitionModel):
    name = "counting"
    threshold = 0.5

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, face_bgr):
        self.encode_calls += 1
        return self.make_output(np.ones(4, dtype=np.float32))


def test_predict_empty_gallery_returns_unknown_without_encoding():
    model = CountingModel()
    face = np.zeros((10, 10, 3), dtype=np.uint8)

    prediction = model.predict(face, [])

    assert prediction == {"student_id": None, "confidence": 0.0, "status": "unknown"}
    assert model.encode_calls == 0

from src.models.arcface_encoder import ArcFaceEncoder
from src.models.dlib_encoder import DlibEncoder
from src.models.dummy_encoder import DummyEncoder
from src.models.facenet_encoder import FaceNetEncoder
from src.models.lbph_encoder import LBPHEncoder


MODEL_REGISTRY = {
    "dummy": DummyEncoder,
    "arcface": ArcFaceEncoder,
    "facenet": FaceNetEncoder,
    "dlib": DlibEncoder,
    "lbph": LBPHEncoder,
}


def create_model(model_name: str):
    try:
        return MODEL_REGISTRY[model_name]()
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}") from exc

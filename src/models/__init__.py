MODEL_REGISTRY = {
    "dummy": "src.models.dummy_encoder.DummyEncoder",
    "arcface": "src.models.arcface_encoder.ArcFaceEncoder",
    "facenet": "src.models.facenet_encoder.FaceNetEncoder",
    "dlib": "src.models.dlib_encoder.DlibEncoder",
    "lbph": "src.models.lbph_encoder.LBPHEncoder",
}


def create_model(model_name: str):
    try:
        import_path = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}") from exc

    module_name, class_name = import_path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)()

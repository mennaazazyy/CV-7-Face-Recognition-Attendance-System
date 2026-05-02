from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ENROLLMENT_DIR = DATA_DIR / "enrollment"
EVALUATION_DIR = DATA_DIR / "evaluation"
MODELS_DIR = ROOT_DIR / "models"
EXPORTS_DIR = ROOT_DIR / "exports"

DATABASE_PATH = ROOT_DIR / "attendance.db"
SCHEMA_PATH = ROOT_DIR / "src" / "database" / "schema.sql"

ACTIVE_MODEL = "arcface"
UNKNOWN_LABEL = "Unknown"
KNOWN_STATUS = "known"
UNKNOWN_STATUS = "unknown"

IMAGE_SIZE = (160, 160)
MIN_FACE_SIZE = 40
ENROLLMENT_IMAGES_PER_STUDENT = 10
WEBCAM_INDEX = 0
FRAME_SKIP = 3

MODEL_THRESHOLDS = {
    "dummy": 0.50,
    "arcface": 0.45,
    "facenet": 0.65,
    "dlib": 0.60,
    "lbph": 70.0,
}

SUPPORTED_MODELS = ("dummy", "arcface", "facenet", "dlib", "lbph")


def ensure_project_dirs() -> None:
    for path in (DATA_DIR, RAW_DATA_DIR, ENROLLMENT_DIR, EVALUATION_DIR, MODELS_DIR, EXPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)

# CV-7 Face Recognition Attendance System

Foundation for a 5-week university Computer Vision project. The system is one attendance application with interchangeable recognition models, not four separate applications.

## A. Architecture

```text
Student images / webcam frames
        |
        v
src/detection/opencv_detector.py
OpenCV Haar Cascade detection -> [(x, y, w, h), ...]
        |
        v
Cropped BGR face images
        |
        v
Frozen FaceRecognitionModel interface
        |
        +--> DummyEncoder        foundation testing only
        +--> ArcFaceEncoder      Person 1
        +--> FaceNetEncoder      Person 2
        +--> DlibEncoder         Person 3
        +--> LBPHEncoder         Person 4
        |
        v
SQLite gallery: students + face_templates
        |
        v
Standard prediction:
{"student_id": "...", "confidence": 0.87, "status": "known"}
or
{"student_id": null, "confidence": 0.31, "status": "unknown"}
        |
        v
Attendance session
        |
        +--> UNIQUE(session_id, student_id) prevents duplicates
        +--> recognition_events log
        +--> attendance CSV export
```

## B. Folder Structure

```text
CV-7-Face-Attendance/
├── data/
│   ├── raw/
│   ├── enrollment/
│   └── evaluation/
├── exports/
├── models/
├── notebooks/
├── scripts/
│   └── smoke_test_dummy.py
├── src/
│   ├── config.py
│   ├── database/
│   │   ├── csv_export.py
│   │   ├── db.py
│   │   └── schema.sql
│   ├── detection/
│   │   ├── __init__.py
│   │   └── opencv_detector.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── dummy_encoder.py
│   │   ├── arcface_encoder.py
│   │   ├── facenet_encoder.py
│   │   ├── dlib_encoder.py
│   │   └── lbph_encoder.py
│   └── pipeline/
│       ├── enroll.py
│       ├── recognize.py
│       └── attendance_session.py
├── tests/
├── requirements.txt
└── README.md
```

## C. Requirements

Install the minimal foundation dependencies first:

```bash
pip install -r requirements.txt
```

`requirements.txt` intentionally uses only `opencv-contrib-python`, not both OpenCV packages. Heavy model libraries stay optional until each member implements their model.

## D. Frozen Model Interface

All model classes must subclass `FaceRecognitionModel`.

Encoding output:

```python
{"model_name": "arcface", "embedding": embedding}
```

Known prediction:

```python
{"student_id": "student_001", "confidence": 0.87, "status": "known"}
```

Unknown prediction:

```python
{"student_id": None, "confidence": 0.31, "status": "unknown"}
```

The UI, database, enrollment pipeline, recognition pipeline, and attendance pipeline must not contain model-specific logic.

## E. Dummy Model

`src/models/dummy_encoder.py` returns deterministic image-content-based fake embeddings. The same input image produces the same embedding, so smoke tests are repeatable. Use it to test:

- enrollment
- template saving
- gallery loading
- recognition flow
- attendance marking
- duplicate prevention
- CSV export

Do not use the dummy model in final evaluation.

## Config

`src/config.py` controls the active backend:

```python
ACTIVE_MODEL = "arcface"
```

For foundation-only testing this can be changed back to:

```python
ACTIVE_MODEL = "dummy"
```

Later this can also be changed to:

```python
ACTIVE_MODEL = "facenet"
ACTIVE_MODEL = "dlib"
ACTIVE_MODEL = "lbph"
```

Model thresholds live in one place:

```python
MODEL_THRESHOLDS = {
    "dummy": 0.50,
    "arcface": 0.45,
    "facenet": 0.65,
    "dlib": 0.60,
    "lbph": 70.0,
}
```

ArcFace, FaceNet, and Dlib use similarity scores where higher is better. LBPH uses distance scores where lower is better.

## F. Detector Foundation

`src/detection/opencv_detector.py` exposes:

```python
detect_faces(frame) -> list[tuple[int, int, int, int]]
crop_faces(frame) -> list[tuple[bbox, face_crop]]
```

The first detector is a simple Haar Cascade so the project stays easy to run in Week 1.

## G. Database Schema

SQLite tables:

- `students`
- `face_templates`
- `attendance_sessions`
- `attendance_records`
- `recognition_events`

`face_templates` stores one template per student per model:

```text
student_id
model_name
template
template_dim
created_at
```

This allows the same student to have separate ArcFace, FaceNet, Dlib, and LBPH templates.

Duplicate attendance prevention is enforced by:

```sql
UNIQUE(session_id, student_id)
```

## H. Pipeline Flow

Enrollment:

```text
read image -> detect face -> crop face -> model.encode(face)
-> average embeddings/templates -> save in face_templates
```

Recognition:

```text
read frame -> detect faces -> crop each face -> model.predict(face, gallery)
-> return student_id/confidence/status/bbox
```

Attendance:

```text
create session -> recognize frame -> if known, mark attendance
-> if duplicate, database rejects it -> export CSV
```

## I. Team Coding Rules

Assigned files:

- Person 1: `src/models/arcface_encoder.py`
- Person 2: `src/models/facenet_encoder.py`
- Person 3: `src/models/dlib_encoder.py`
- Person 4: `src/models/lbph_encoder.py`

Do not change these shared files without team discussion:

- `src/models/base_model.py`
- `src/database/db.py`
- `src/database/schema.sql`
- `src/pipeline/enroll.py`
- `src/pipeline/recognize.py`
- `src/pipeline/attendance_session.py`
- `src/config.py`

Model-specific next steps:

- ArcFace: load InsightFace `FaceAnalysis`, extract 512D embeddings, use cosine similarity, return the standard format.
- FaceNet: load FaceNet with `facenet-pytorch` or DeepFace, extract embeddings, use cosine similarity, return the standard format.
- Dlib ResNet: use `face_recognition`, convert BGR to RGB, return 128D embeddings, return the standard format.
- LBPH: use OpenCV `LBPHFaceRecognizer`, train on enrollment images, handle lower-distance-is-better matching, return the standard format.

## J. Checklist Before Model Division

- Git repository initialized.
- Folder structure finalized.
- `src/detection/opencv_detector.py` added.
- `requirements.txt` cleaned.
- Database initialized successfully.
- Dummy model tested.
- Enrollment pipeline tested.
- Recognition pipeline tested.
- Attendance marking tested.
- Duplicate attendance prevention tested.
- CSV export tested.
- Model interface frozen.
- Dataset folder structure agreed.
- README updated with relative paths only.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m compileall src/config.py src/detection src/models src/database src/pipeline src/evaluation
python3 -c "from src.database.db import init_db; init_db(); print('database ready')"
python3 scripts/smoke_test_dummy.py
```

The dummy smoke test verifies database initialization, deterministic dummy encoding, template saving, gallery loading, standard prediction output, attendance marking, duplicate attendance prevention, and CSV export.

## ArcFace Test

Install the foundation and ArcFace dependencies:

```bash
pip install -r requirements.txt
pip install insightface onnxruntime
```

Compile-check the project:

```bash
python3 -m compileall src/config.py src/detection src/models src/database src/pipeline src/evaluation
```

Run the one-person ArcFace webcam test:

```bash
python3 scripts/test_arcface_single_person.py
```

The script registers one student, captures 10 webcam face samples, saves an ArcFace template under `model_name = "arcface"`, reopens recognition mode, displays `student_id`, student name, confidence, and status, and marks attendance when recognized. Press `q` to quit.

Enroll an ArcFace student without immediately starting attendance:

```bash
python3 scripts/enroll_arcface_person.py --id "22-101004" --name "Student Name"
```

Run ArcFace recognition only:

```bash
python3 scripts/recognize_arcface_only.py
```

Tune the known/unknown threshold:

```bash
python3 scripts/recognize_arcface_only.py --threshold 0.55
```

Run the blink liveness challenge before recognition:

```bash
python3 scripts/recognize_arcface_only.py --antispoof
```

Run DeepFace anti-spoofing before recognition:

```bash
python3 scripts/recognize_arcface_only.py --deep-antispoof
```

Use both anti-spoofing checks:

```bash
python3 scripts/recognize_arcface_only.py --antispoof --deep-antispoof
```

DeepFace anti-spoofing requires the optional packages listed in `requirements.txt`.

## Git Setup

Initialize the repository before splitting model work:

```bash
git init
git add .
git commit -m "Initial project foundation"
```

Then create a GitHub repository and push this foundation before each member starts their assigned encoder.

## Final Team Message

The foundation is ready.

Everyone should pull the same GitHub repo before starting.

Each person will only work inside their assigned model file:

- Person 1: `src/models/arcface_encoder.py`
- Person 2: `src/models/facenet_encoder.py`
- Person 3: `src/models/dlib_encoder.py`
- Person 4: `src/models/lbph_encoder.py`

Do not change shared files like `base_model.py`, `config.py`, `db.py`, `schema.sql`, `enroll.py`, `recognize.py`, or `attendance_session.py` without team agreement.

Our goal is one attendance system with four interchangeable models, not four separate systems.

First, we will test everything using `DummyEncoder`. After the dummy flow works, each person can implement their real model.

## What Not To Add Yet

Do not add anti-spoofing, bias analysis, YOLO detection, detector ablation, or full Streamlit UI implementation during the foundation phase.

# Team Handoff — Face Attendance Recognition

## Project Overview

We're building a face recognition attendance system that supports **multiple models** so we can compare them in our presentation. The architecture is fully built — each teammate only needs to implement one `encode()` method in their assigned file.

---

## What's Already Working

- Full pipeline: enrollment → recognition → attendance marking → CSV export
- Database (SQLite): students, face templates, attendance sessions, records
- Face detection: OpenCV Haar Cascade
- ArcFace model: fully implemented and tested
- LBPH model: fully implemented (just needs testing)
- Anti-spoofing: blink detection + DeepFace liveness (optional)
- Generic scripts that work with any model via `--model` flag
- Benchmark script for side-by-side model comparison

---

## Setup

```bash
git pull origin main
pip install -r requirements.txt
```

---

## Who Does What

| Person | Model | File | Status |
|--------|-------|------|--------|
| **Menna** | ArcFace | `src/models/arcface_encoder.py` | Done |
| **Rahma** | FaceNet | `src/models/facenet_encoder.py` | Done |
| **Seif** | dlib | `src/models/dlib_encoder.py` | Needs implementation |
| **Ahmed** | LBPH | `src/models/lbph_encoder.py` | Already implemented — needs testing + evaluation dataset |

**Do NOT edit** `base_model.py`, `config.py`, `db.py`, or any pipeline files without telling the team first.

---

## Detailed Instructions Per Person

### Rahma — FaceNet ✅ Implemented

**Setup:**
```bash
git pull origin main
pip install facenet-pytorch
```

**Implementation notes (`src/models/facenet_encoder.py`):**

- Lazily loads `InceptionResnetV1(pretrained="vggface2")` on first call (cached on the instance) — first encode pays the download cost, subsequent calls are fast.
- Auto-selects CUDA if available, otherwise CPU.
- Pipeline: BGR → resize to `IMAGE_SIZE` (160×160 via `self.preprocess`) → BGR→RGB → normalize `(x - 127.5) / 128.0` → CHW tensor → forward pass → 512-D embedding → `l2_normalize` → `make_output`.
- Uses cosine similarity (default `lower_score_is_better = False`).

**Phase 2 testing — done (non-interactive smoke test):**

Ran `tests/smoke_facenet.py` which validates the encoder pipeline end-to-end without a webcam:

| Check | Result |
|---|---|
| Output shape / dtype | `(512,) float32` ✅ |
| `model_name` field | `"facenet"` ✅ |
| L2 norm of embedding | `1.000000` ✅ |
| Determinism (same image → same embedding) | cosine `1.000000` ✅ |
| Different inputs → lower similarity than self | ✅ |
| `predict()` returns `known` for matching gallery entry | ✅ |
| `predict()` returns `unknown` on empty gallery | ✅ |
| `predict()` returns `unknown` when threshold is unreachable | ✅ |
| First encode (incl. `vggface2` weight download + model build) | ~65 s |
| Warm encode latency (CPU) | **~30 ms/face** |

Run it yourself any time:
```bash
PYTHONPATH=. python tests/smoke_facenet.py
```

**Still to do (requires a webcam / real faces):**

1. Webcam smoke test:
   ```bash
   python scripts/enroll_person.py --id "22-XXXXXX" --name "Rahma" --model facenet --verify
   python scripts/recognize_live.py --model facenet --show-scores
   ```
2. Tune `MODEL_THRESHOLDS["facenet"]` in `src/config.py` if 0.65 doesn't separate known/unknown cleanly on real faces. Note: random-noise inputs give cosine ~0.97-0.98 between distinct images (no facial structure to separate them); with real faces the spread is much wider, so the 0.65 default should be roughly right but worth confirming.

**Dependency note:** `facenet-pytorch` pins `numpy<2.0`. If `pip install facenet-pytorch` upgrades/downgrades numpy and breaks something else, run `pip install "numpy<2.0" --force-reinstall` afterwards.

---

### Seif — dlib

**Setup:**
```bash
git pull origin main
pip install face_recognition dlib
```

**What to do:**

1. Open `src/models/dlib_encoder.py`
2. Implement `encode()`:
   - Convert BGR face to RGB (`cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)`)
   - Use `face_recognition.face_encodings(rgb_image)` to get a 128D embedding
   - Handle the case where no encoding is returned (raise `ValueError`)
   - L2-normalize: `embedding = self.l2_normalize(embedding)`
   - Return: `self.make_output(embedding)`
3. Test:
   ```bash
   python scripts/enroll_person.py --id "22-XXXXXX" --name "Seif" --model dlib --verify
   python scripts/recognize_live.py --model dlib --show-scores
   ```
4. If the default threshold (0.60) doesn't work well, update `MODEL_THRESHOLDS["dlib"]` in `src/config.py`

---

### Ahmed — LBPH

LBPH is already fully implemented. Your job is to **test it** and **build the shared evaluation dataset**.

**Setup:**
```bash
git pull origin main
pip install opencv-contrib-python
```

**What to do:**

1. Read `src/models/lbph_encoder.py` to understand how it works
   - Note: LBPH uses `lower_score_is_better = True` (distance-based, not cosine similarity)
   - It has a `train()` method and needs to be trained before prediction works through the gallery
2. Test:
   ```bash
   python scripts/enroll_person.py --id "22-XXXXXX" --name "Ahmed" --model lbph --verify
   python scripts/recognize_live.py --model lbph --show-scores
   ```
3. **Extra task — Build the evaluation dataset:**
   - Take 5-10 photos of each team member (different angles, lighting)
   - Save them into `data/evaluation/<student_id>/img1.jpg`, `img2.jpg`, etc.
   - Get 5+ photos of people NOT on the team and save them in `data/evaluation/_unknown/`
   - This dataset is what we all use for the final benchmark comparison

---

### Menna — Architecture + ArcFace + Final Comparison

ArcFace is done. Your remaining tasks:

1. Help teammates if they get stuck with the architecture
2. Once everyone's model works, make sure everyone enrolls the **same people** with their model
3. Run the final benchmark:
   ```bash
   python scripts/benchmark_models.py
   ```
4. Prepare the architecture overview section of the presentation

---

## How to Implement Your Model

Open your assigned file. You'll see a stub like this:

```python
class FaceNetEncoder(FaceRecognitionModel):
    name = "facenet"
    threshold = MODEL_THRESHOLDS[name]

    def encode(self, face_bgr: np.ndarray) -> EncodingOutput:
        face_bgr = self.preprocess(face_bgr, IMAGE_SIZE)
        raise NotImplementedError("...")
```

Replace the `raise NotImplementedError(...)` with your actual model logic. Your `encode()` must:

1. Take a BGR face crop (numpy array)
2. Run it through your model to get an embedding vector
3. L2-normalize it: `embedding = self.l2_normalize(embedding)`
4. Return: `self.make_output(embedding)`

That's it. The base class handles prediction, gallery matching, and thresholding automatically.

### Output format (enforced by base class)

Encoding:
```python
{"model_name": "facenet", "embedding": np.ndarray}  # shape (N,)
```

Prediction (handled automatically, don't override):
```python
{"student_id": "22-101004", "confidence": 0.87, "status": "known"}
{"student_id": None, "confidence": 0.31, "status": "unknown"}
```

### Thresholds

Default thresholds are in `src/config.py`:

| Model | Threshold | Direction |
|-------|-----------|-----------|
| arcface | 0.45 | higher = more similar (cosine) |
| facenet | 0.65 | higher = more similar (cosine) |
| dlib | 0.60 | higher = more similar (cosine) |
| lbph | 70.0 | lower = more similar (distance) |

If your model needs a different threshold, update your entry in `MODEL_THRESHOLDS` in `src/config.py`.

If your model uses distance (lower = better match) instead of similarity, set `lower_score_is_better = True` in your class (like LBPH does).

---

## How to Test Your Model

### Step 1: Enroll people

```bash
# Enroll yourself
python scripts/enroll_person.py --id "22-101001" --name "Your Name" --model <your_model>

# Enroll with verification (recommended)
python scripts/enroll_person.py --id "22-101001" --name "Your Name" --model <your_model> --verify

# Enroll more people
python scripts/enroll_person.py --id "22-101002" --name "Friend Name" --model <your_model>
```

During enrollment, follow the on-screen guidance (look straight, turn left/right, tilt up/down). The script captures 20 diverse samples and averages them into one template.

### Step 2: Live recognition test

```bash
# Basic test
python scripts/recognize_live.py --model <your_model>

# See match scores for debugging
python scripts/recognize_live.py --model <your_model> --show-scores

# Test with custom threshold
python scripts/recognize_live.py --model <your_model> --threshold 0.55

# Test unknown rejection (empty gallery)
python scripts/recognize_live.py --model <your_model> --empty-gallery
```

### Step 3: Check these things

- [ ] Can it enroll a person? (green box, "Captured X/20")
- [ ] Can it recognize an enrolled person? (green box, name + confidence)
- [ ] Does it reject someone not enrolled as "Unknown"? (red box)
- [ ] What threshold gives the best balance?
- [ ] How fast/slow does it feel?

---

## Shared Evaluation Dataset (for the presentation)

Before we can run the benchmark comparison, we need a shared dataset. Everyone should contribute images.

### How to set it up

Create this folder structure:

```
data/evaluation/
    22-101001/
        img1.jpg
        img2.jpg
        img3.jpg
    22-101002/
        img1.jpg
        img2.jpg
    22-101003/
        img1.jpg
    _unknown/
        stranger1.jpg
        stranger2.jpg
```

- Each subfolder = one enrolled student's ID
- `_unknown/` = faces of people who are NOT enrolled (for false accept rate)
- Use 5-10 images per person, different angles/lighting if possible
- All models get tested against the exact same images

### Running the benchmark

```bash
# Test all models
python scripts/benchmark_models.py

# Test specific models
python scripts/benchmark_models.py --models arcface facenet
```

This prints a comparison table:

```
Model       Threshold   Accuracy    Known Acc   FAR      FRR      MisID   Avg ms    Avg Conf
arcface     0.45        95.0%       93.3%       0.0%     6.7%     0       42.3      0.782
facenet     0.65        88.3%       86.7%       5.0%     8.3%     1       38.1      0.701
dlib        0.60        83.3%       80.0%       10.0%    10.0%    2       15.2      0.634
lbph        70.00       75.0%       73.3%       15.0%    11.7%    3       5.1       58.200
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/models/base_model.py` | Abstract base class — do not edit |
| `src/models/<your_model>.py` | Your encoder — edit only this |
| `src/config.py` | Thresholds, paths, constants |
| `src/detection/opencv_detector.py` | Face detection (Haar Cascade) |
| `src/database/db.py` | All database operations |
| `src/pipeline/attendance_session.py` | Full attendance session runner |
| `scripts/enroll_person.py` | Generic enrollment (any model) |
| `scripts/recognize_live.py` | Generic live recognition (any model) |
| `scripts/benchmark_models.py` | Side-by-side model comparison |
| `scripts/enroll_arcface_person.py` | ArcFace-specific enrollment (legacy) |
| `scripts/recognize_arcface_only.py` | ArcFace-specific recognition (legacy) |

---

## Common Issues

**"No face detected" during enrollment**
- Make sure you're in good lighting
- Face the camera directly, don't be too far away
- Min face size is 40x40 pixels

**ImportError when creating your model**
- Install your model's dependencies (e.g. `pip install facenet-pytorch` or `pip install face_recognition`)
- Add any new pip packages to `requirements.txt`

**Low recognition accuracy**
- Re-enroll with more samples: `--samples 30`
- Move your head slowly during enrollment for more diverse captures
- Try adjusting the threshold with `--threshold`
- Use `--show-scores` to see why matches are failing

**"No templates loaded"**
- You need to enroll people with your specific model first
- Each model has its own templates in the database

---

## Timeline

### Phase 1 — Implement (Days 1-2, everyone in parallel)
- **Everyone**: `git pull origin main`, install dependencies
- **Rahma**: Implement FaceNet `encode()`
- **Seif**: Implement dlib `encode()`
- **Ahmed**: Test LBPH, start collecting evaluation photos
- **Menna**: Available to help with architecture questions

### Phase 2 — Test Individually (Days 2-3)
- **Rahma, Seif, Ahmed**: Enroll yourself + 2 other people with your model, test recognition, test unknown rejection, find the best threshold
- Write down: accuracy, speed, best threshold, any issues

### Phase 3 — Shared Enrollment + Evaluation Dataset (Day 4, must be done together)
- Pick 4-6 people as shared test subjects (the 4 of us + a couple of friends)
- **Each person enrolls the SAME people** with their model
- Ahmed finalises the evaluation dataset in `data/evaluation/`
- Make sure `data/evaluation/_unknown/` has photos of non-enrolled people

### Phase 4 — Benchmark (Day 5)
- **Menna** runs `python scripts/benchmark_models.py`
- Everyone reviews the comparison table
- Tune thresholds if needed, re-run

### Phase 5 — Presentation (Day 6)
- **Menna**: Architecture overview + ArcFace results
- **Rahma**: FaceNet — how it works, setup, results
- **Seif**: dlib — how it works, setup, results
- **Ahmed**: LBPH — how it works, results + evaluation dataset methodology

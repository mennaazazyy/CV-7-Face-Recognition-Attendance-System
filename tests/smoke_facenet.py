"""Non-interactive smoke test for FaceNetEncoder.

Validates:
- model loads, encode() runs end-to-end
- output shape is 512 and L2-normalized
- identical inputs -> identical embeddings (cosine ~1.0)
- distinct inputs -> lower cosine similarity (separable)
- predict() routes through gallery matching correctly (known + unknown)
"""
from __future__ import annotations

import sys
import time

import numpy as np

from src.models.facenet_encoder import FaceNetEncoder


def fake_face(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # 200x200 BGR uint8 — bigger than IMAGE_SIZE so preprocess() resizes
    return rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)


def main() -> int:
    enc = FaceNetEncoder()

    img_a1 = fake_face(1)
    img_a2 = img_a1.copy()
    img_b = fake_face(99)
    img_c = fake_face(7)

    t0 = time.perf_counter()
    out_a1 = enc.encode(img_a1)
    first_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    out_a2 = enc.encode(img_a2)
    out_b = enc.encode(img_b)
    out_c = enc.encode(img_c)
    avg_ms = ((time.perf_counter() - t0) * 1000) / 3

    emb_a1, emb_a2, emb_b, emb_c = (
        out_a1["embedding"],
        out_a2["embedding"],
        out_b["embedding"],
        out_c["embedding"],
    )

    # Shape + model_name
    assert out_a1["model_name"] == "facenet", out_a1
    assert emb_a1.shape == (512,), emb_a1.shape
    assert emb_a1.dtype == np.float32, emb_a1.dtype
    print(f"[OK] shape={emb_a1.shape}, dtype={emb_a1.dtype}, model_name={out_a1['model_name']}")

    # L2-normalized
    norm = float(np.linalg.norm(emb_a1))
    assert abs(norm - 1.0) < 1e-4, norm
    print(f"[OK] L2 norm = {norm:.6f}")

    # Determinism (same image -> same embedding)
    sim_same = float(np.dot(emb_a1, emb_a2))
    assert sim_same > 0.9999, sim_same
    print(f"[OK] determinism cosine = {sim_same:.6f}")

    # Different images: lower similarity than self
    sim_diff_b = float(np.dot(emb_a1, emb_b))
    sim_diff_c = float(np.dot(emb_a1, emb_c))
    print(f"[OK] cross-image cosine: A-B={sim_diff_b:.4f}  A-C={sim_diff_c:.4f}  (self={sim_same:.4f})")
    assert sim_same > sim_diff_b and sim_same > sim_diff_c

    # Gallery / predict() routing through base class
    gallery = [("22-101001", emb_a1), ("22-101002", emb_b)]

    pred_known = enc.predict(img_a2, gallery)
    print(f"[OK] predict(self) -> {pred_known}")
    assert pred_known["status"] == "known"
    assert pred_known["student_id"] == "22-101001"

    # Force unknown via empty gallery
    pred_empty = enc.predict(img_c, [])
    print(f"[OK] predict(empty gallery) -> {pred_empty}")
    assert pred_empty["status"] == "unknown" and pred_empty["student_id"] is None

    # Force unknown via impossibly high threshold
    enc_strict = FaceNetEncoder()
    enc_strict._model = enc._model
    enc_strict._device = enc._device
    enc_strict.threshold = 0.999999
    pred_strict = enc_strict.predict(img_c, gallery)
    print(f"[OK] predict(threshold=0.999999) -> {pred_strict}")
    assert pred_strict["status"] == "unknown"

    print(f"\nTiming: first encode {first_ms:.1f} ms (incl. model load), warm avg {avg_ms:.1f} ms/face")
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

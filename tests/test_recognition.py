import numpy as np
import pytest
from src.recognition.encoder import Encoder


class DummyEncoder(Encoder):
    """Minimal concrete encoder for testing the base class helpers."""

    def encode(self, face_crop_bgr):
        flat = face_crop_bgr.flatten().astype(np.float32)
        return self._l2_normalize(flat[:128])

    def compare(self, emb_a, emb_b):
        return self.cosine_similarity(emb_a, emb_b)


def test_l2_normalize_unit_length():
    enc = DummyEncoder()
    v = np.array([3.0, 4.0])
    normed = enc._l2_normalize(v)
    assert abs(np.linalg.norm(normed) - 1.0) < 1e-6


def test_cosine_same_vector():
    enc = DummyEncoder()
    v = np.random.rand(128).astype(np.float32)
    sim = enc.cosine_similarity(v, v)
    assert abs(sim - 1.0) < 1e-5


def test_cosine_orthogonal():
    enc = DummyEncoder()
    a = np.zeros(4, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(4, dtype=np.float32)
    b[1] = 1.0
    sim = enc.cosine_similarity(a, b)
    assert abs(sim) < 1e-6


def test_encode_output_shape():
    enc = DummyEncoder()
    crop = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    emb = enc.encode(crop)
    assert emb.ndim == 1
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

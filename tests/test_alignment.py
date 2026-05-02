import numpy as np
import cv2
import pytest
from src.alignment.similarity_transform import align_face


@pytest.fixture
def dummy_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_landmarks():
    # Approximate 5-point landmarks for a face centred at (320, 240)
    return np.array(
        [[280, 210], [360, 210], [320, 250], [290, 290], [350, 290]],
        dtype=np.float32,
    )


def test_align_output_shape(dummy_image, dummy_landmarks):
    aligned = align_face(dummy_image, dummy_landmarks, output_size=(112, 112))
    assert aligned.shape == (112, 112, 3)


def test_align_output_shape_custom_size(dummy_image, dummy_landmarks):
    aligned = align_face(dummy_image, dummy_landmarks, output_size=(160, 160))
    assert aligned.shape == (160, 160, 3)


def test_align_dtype_preserved(dummy_image, dummy_landmarks):
    aligned = align_face(dummy_image, dummy_landmarks)
    assert aligned.dtype == np.uint8

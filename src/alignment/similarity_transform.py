import cv2
import numpy as np

# Standard 5-point landmark positions for a 112×112 aligned face
# (ArcFace / InsightFace convention)
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.6963],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.3655],
    ],
    dtype=np.float32,
)


def align_face(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    output_size: tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Apply a 5-point similarity transform to warp the face into a canonical
    112×112 crop.

    Args:
        image_bgr:   Full BGR frame.
        landmarks:   Shape (5, 2) float array — (left_eye, right_eye, nose,
                     left_mouth, right_mouth).
        output_size: Target crop dimensions (width, height).

    Returns:
        Aligned BGR face crop of shape (output_size[1], output_size[0], 3).
    """
    src = landmarks.astype(np.float32)
    dst = _ARCFACE_DST * np.array(output_size) / 112.0

    transform = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
    aligned = cv2.warpAffine(image_bgr, transform, output_size, flags=cv2.INTER_LINEAR)
    return aligned

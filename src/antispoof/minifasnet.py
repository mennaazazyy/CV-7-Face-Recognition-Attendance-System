import cv2
import numpy as np


class AntiSpoofChecker:
    """
    Thin wrapper around DeepFace's anti-spoofing flag.

    DeepFace calls MiniFASNet internally when anti_spoofing=True.
    We expose a simple is_real(face_crop) → bool interface so the rest
    of the pipeline never imports DeepFace directly.
    """

    def is_real(self, face_crop_bgr: np.ndarray) -> bool:
        """Return True if the face appears to be live (not a photo / screen)."""
        from deepface import DeepFace

        try:
            result = DeepFace.analyze(
                face_crop_bgr,
                actions=["emotion"],   # any action; we only care about anti_spoofing
                anti_spoofing=True,
                silent=True,
            )
            # DeepFace raises ValueError("Spoof detected") if it fails
            return True
        except ValueError as exc:
            if "spoof" in str(exc).lower():
                return False
            raise

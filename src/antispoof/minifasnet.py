import cv2
import numpy as np
from dataclasses import dataclass


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
            face_objs = DeepFace.extract_faces(
                img_path=face_crop_bgr,
                anti_spoofing=True,
                detector_backend="skip",
                enforce_detection=False,
                silent=True,
            )
            return bool(face_objs) and all(face_obj.get("is_real") is True for face_obj in face_objs)
        except ValueError as exc:
            if "spoof" in str(exc).lower():
                return False
            raise


@dataclass(frozen=True)
class LivenessState:
    is_live: bool
    progress: float
    message: str


class MotionChallengeAntiSpoofChecker:
    """Lightweight webcam liveness challenge.

    This is not a full production anti-spoofing model. It is a practical demo
    gate that asks for natural face movement before recognition is allowed.
    Static photos and still screen images should stay blocked.
    """

    def __init__(self, required_motion: float = 0.12, window_size: int = 12) -> None:
        self.required_motion = required_motion
        self.window_size = window_size
        self._centers: list[tuple[float, float]] = []
        self._areas: list[float] = []

    def reset(self) -> None:
        self._centers.clear()
        self._areas.clear()

    def update(self, bbox: tuple[int, int, int, int] | None) -> LivenessState:
        if bbox is None:
            self.reset()
            return LivenessState(False, 0.0, "No face detected")

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            self.reset()
            return LivenessState(False, 0.0, "No face detected")

        self._centers.append((x + w / 2.0, y + h / 2.0))
        self._areas.append(float(w * h))
        self._centers = self._centers[-self.window_size :]
        self._areas = self._areas[-self.window_size :]

        if len(self._centers) < max(4, self.window_size // 2):
            return LivenessState(False, len(self._centers) / self.window_size, "Move your head slightly")

        center_motion = self._center_motion_ratio(w, h)
        scale_motion = self._scale_motion_ratio()
        motion_score = max(center_motion, scale_motion)
        progress = min(motion_score / self.required_motion, 1.0)

        if motion_score >= self.required_motion:
            return LivenessState(True, 1.0, "Live face verified")
        return LivenessState(False, progress, "Move your head left/right or closer/farther")

    def _center_motion_ratio(self, width: int, height: int) -> float:
        xs = [center[0] for center in self._centers]
        ys = [center[1] for center in self._centers]
        movement = max(max(xs) - min(xs), max(ys) - min(ys))
        return float(movement / max(width, height, 1))

    def _scale_motion_ratio(self) -> float:
        min_area = min(self._areas)
        max_area = max(self._areas)
        return float((max_area - min_area) / max(max_area, 1.0))


class BlinkChallengeAntiSpoofChecker:
    """Simple blink challenge: open eyes -> closed eyes -> open eyes.

    This blocks basic printed-photo and phone-photo attacks better than the
    motion-only gate because moving a static image does not create a blink.
    It still remains a classroom/demo liveness check, not a production PAD model.
    """

    def __init__(self, min_closed_frames: int = 1) -> None:
        self.min_closed_frames = min_closed_frames
        self._eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
        self._stage = "need_open"
        self._closed_frames = 0

    def reset(self) -> None:
        self._stage = "need_open"
        self._closed_frames = 0

    def update(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int] | None) -> LivenessState:
        if bbox is None:
            self.reset()
            return LivenessState(False, 0.0, "No face detected")

        eyes_open = self._eyes_are_open(frame_bgr, bbox)

        if self._stage == "need_open":
            if eyes_open:
                self._stage = "need_closed"
                return LivenessState(False, 0.33, "Blink now")
            return LivenessState(False, 0.0, "Open your eyes")

        if self._stage == "need_closed":
            if eyes_open:
                self._closed_frames = 0
                return LivenessState(False, 0.33, "Blink now")

            self._closed_frames += 1
            if self._closed_frames >= self.min_closed_frames:
                self._stage = "need_reopen"
                return LivenessState(False, 0.66, "Open your eyes again")
            return LivenessState(False, 0.5, "Keep eyes closed briefly")

        if self._stage == "need_reopen":
            if eyes_open:
                self._stage = "verified"
                return LivenessState(True, 1.0, "Blink verified")
            return LivenessState(False, 0.66, "Open your eyes again")

        return LivenessState(True, 1.0, "Blink verified")

    def _eyes_are_open(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return False

        frame_h, frame_w = frame_bgr.shape[:2]
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, frame_w)
        y2 = min(y + int(h * 0.60), frame_h)
        upper_face = frame_bgr[y1:y2, x1:x2]
        if upper_face.size == 0:
            return False

        gray = cv2.cvtColor(upper_face, cv2.COLOR_BGR2GRAY)
        eyes = self._eye_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(max(int(w * 0.10), 12), max(int(h * 0.05), 8)),
        )
        return len(eyes) >= 1

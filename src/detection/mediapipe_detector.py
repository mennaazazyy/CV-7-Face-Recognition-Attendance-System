import numpy as np
import mediapipe as mp


class MediaPipeDetector:
    """Wraps MediaPipe Face Detection for fast CPU inference."""

    def __init__(self, min_confidence: float = 0.5):
        self._mp = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_confidence
        )

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        result = self._mp.process(rgb)
        faces = []
        if result.detections:
            for det in result.detections:
                bb = det.location_data.relative_bounding_box
                x1 = int(bb.xmin * w)
                y1 = int(bb.ymin * h)
                x2 = int((bb.xmin + bb.width) * w)
                y2 = int((bb.ymin + bb.height) * h)
                faces.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "landmarks": None,
                        "score": det.score[0],
                    }
                )
        return faces

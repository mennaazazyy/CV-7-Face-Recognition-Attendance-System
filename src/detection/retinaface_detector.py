import numpy as np
from retinaface import RetinaFace


class RetinaFaceDetector:
    """Wraps RetinaFace to return (bbox, landmarks) lists."""

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Returns a list of dicts with keys:
            bbox       – (x1, y1, x2, y2) int
            landmarks  – dict of 5 facial keypoints (or None)
            score      – float confidence
        """
        results = RetinaFace.detect_faces(frame_bgr)
        faces = []
        if isinstance(results, dict):
            for face_data in results.values():
                area = face_data["facial_area"]
                faces.append(
                    {
                        "bbox": (area[0], area[1], area[2], area[3]),
                        "landmarks": face_data.get("landmarks"),
                        "score": face_data.get("score", 1.0),
                    }
                )
        return faces

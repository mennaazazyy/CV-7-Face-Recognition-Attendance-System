#!/usr/bin/env python3
"""
Sweep cosine similarity thresholds from 0.30 to 0.70 on the validation set
and report Top-1 accuracy, FAR, FRR at each step.

Usage:
    python scripts/threshold_tuning.py --encoder arcface
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import VALIDATION_DIR, UNKNOWN_LABEL
from src.utils.metrics import top1_accuracy, compute_far_frr


def load_validation_data(encoder_name: str):
    """
    Returns (y_true, embeddings) from the validation split.
    Assumes structure: data/splits/validation/<student_id>/<image>.jpg
    """
    from src.recognition import (
        ArcFaceEncoder, FaceNetEncoder, DlibEncoder,
        LBPHEncoder, EigenfacesEncoder, FisherfacesEncoder,
    )
    import cv2

    enc_map = {
        "arcface": ArcFaceEncoder, "facenet": FaceNetEncoder,
        "dlib": DlibEncoder, "lbph": LBPHEncoder,
        "eigenfaces": EigenfacesEncoder, "fisherfaces": FisherfacesEncoder,
    }
    encoder = enc_map[encoder_name]()

    y_true, embeddings = [], []
    for student_dir in sorted(VALIDATION_DIR.iterdir()):
        if not student_dir.is_dir():
            continue
        for img_path in student_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            try:
                emb = encoder.encode(img)
                y_true.append(student_dir.name)
                embeddings.append(emb)
            except ValueError:
                pass

    return y_true, embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="arcface")
    parser.add_argument("--low", type=float, default=0.30)
    parser.add_argument("--high", type=float, default=0.70)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Loading validation data for {args.encoder}…")
    y_true, embeddings = load_validation_data(args.encoder)
    from src.database.db import get_all_embeddings
    gallery = get_all_embeddings(args.encoder)

    print(f"\n{'Threshold':>10} {'Top-1':>8} {'FAR':>8} {'FRR':>8}")
    thresholds = np.arange(args.low, args.high + args.step / 2, args.step)

    best_threshold, best_acc = args.low, 0.0

    for thresh in thresholds:
        y_pred = []
        for emb in embeddings:
            best_id, best_score = UNKNOWN_LABEL, -1.0
            for sid, g_emb in gallery:
                from src.recognition.encoder import Encoder
                score = float(np.dot(emb, g_emb))
                if score > best_score:
                    best_score, best_id = score, sid
            y_pred.append(best_id if best_score >= thresh else UNKNOWN_LABEL)

        acc = top1_accuracy(y_true, y_pred)
        far, frr = compute_far_frr(y_true, y_pred)
        print(f"{thresh:>10.2f} {acc:>8.4f} {far:>8.4f} {frr:>8.4f}")
        if acc > best_acc:
            best_acc, best_threshold = acc, thresh

    print(f"\n→ Recommended threshold for {args.encoder}: {best_threshold:.2f}  (acc={best_acc:.4f})")


if __name__ == "__main__":
    main()

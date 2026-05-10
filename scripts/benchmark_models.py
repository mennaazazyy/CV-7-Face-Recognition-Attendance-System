#!/usr/bin/env python3
"""
Run all implemented models against the same evaluation dataset and print
a side-by-side comparison table.

Dataset layout (put images here before running):

    data/evaluation/
        22-101001/
            img1.jpg
            img2.jpg
        22-101002/
            img1.jpg
        _unknown/          <-- faces of people NOT enrolled (optional)
            img1.jpg

Each subfolder name must match a student_id that was enrolled for the models
being tested.  The special folder "_unknown" (optional) contains faces of
people who should NOT be recognised — used to measure false-accept rate.

Usage:
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --models arcface facenet
    python scripts/benchmark_models.py --eval-dir path/to/custom/eval
"""
from __future__ import annotations

import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import cv2
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EVALUATION_DIR, SUPPORTED_MODELS, UNKNOWN_LABEL  # noqa: E402
from src.database.db import load_gallery  # noqa: E402
from src.detection import crop_faces  # noqa: E402
from src.models import create_model  # noqa: E402

UNKNOWN_FOLDER = "_unknown"


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)

    if not eval_dir.exists() or not any(eval_dir.iterdir()):
        print(f"Evaluation directory is empty: {eval_dir}")
        print()
        print("To use this script, create subfolders for each enrolled student:")
        print(f"  {eval_dir}/<student_id>/img1.jpg")
        print(f"  {eval_dir}/<student_id>/img2.jpg")
        print(f"  {eval_dir}/{UNKNOWN_FOLDER}/img1.jpg   (optional, for FAR)")
        sys.exit(1)

    images = load_eval_images(eval_dir)
    if not images:
        print("No images found in evaluation directory.")
        sys.exit(1)

    known_count = sum(1 for label, _ in images if label != UNKNOWN_LABEL)
    unknown_count = sum(1 for label, _ in images if label == UNKNOWN_LABEL)
    print(f"Loaded {len(images)} evaluation images ({known_count} known, {unknown_count} unknown)")
    print()

    models_to_test = args.models
    results: list[dict] = []

    for model_name in models_to_test:
        print(f"--- Testing {model_name} ---")
        result = evaluate_model(model_name, images)
        if result is not None:
            results.append(result)
        print()

    if not results:
        print("No models could be evaluated.")
        sys.exit(1)

    print_comparison_table(results)


def parse_args():
    parser = ArgumentParser(description="Benchmark all models on the same evaluation dataset.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        help=f"Models to benchmark (default: all). Choices: {', '.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--eval-dir",
        default=str(EVALUATION_DIR),
        help=f"Path to evaluation images directory (default: {EVALUATION_DIR}).",
    )
    return parser.parse_args()


def load_eval_images(eval_dir: Path) -> list[tuple[str, np.ndarray]]:
    """Return list of (true_label, face_bgr) pairs from the evaluation directory."""
    images: list[tuple[str, np.ndarray]] = []
    for student_dir in sorted(eval_dir.iterdir()):
        if not student_dir.is_dir():
            continue
        label = UNKNOWN_LABEL if student_dir.name == UNKNOWN_FOLDER else student_dir.name
        for img_path in sorted(student_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Warning: could not read {img_path}")
                continue
            face_items = crop_faces(img)
            if not face_items:
                print(f"  Warning: no face detected in {img_path}")
                continue
            _, face_bgr = max(face_items, key=lambda item: item[0][2] * item[0][3])
            images.append((label, face_bgr))
    return images


def evaluate_model(model_name: str, images: list[tuple[str, np.ndarray]]) -> dict | None:
    """Run a single model against all evaluation images and return metrics."""
    try:
        model = create_model(model_name)
    except Exception as exc:
        print(f"  Skipping {model_name}: could not create model — {exc}")
        return None

    gallery = load_gallery(model_name)
    if not gallery:
        print(f"  Skipping {model_name}: no enrolled templates found. Enroll people first with --model {model_name}.")
        return None

    print(f"  Gallery: {len(gallery)} template(s), threshold: {model.threshold:.2f}")

    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    encode_times: list[float] = []
    errors = 0

    for true_label, face_bgr in images:
        try:
            t0 = time.perf_counter()
            prediction = model.predict(face_bgr, gallery)
            t1 = time.perf_counter()
            encode_times.append(t1 - t0)

            pred_label = prediction["student_id"] if prediction["student_id"] is not None else UNKNOWN_LABEL
            y_true.append(true_label)
            y_pred.append(pred_label)
            confidences.append(float(prediction["confidence"]))
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"  Error on image (true={true_label}): {exc}")

    if not y_true:
        print(f"  No successful predictions for {model_name}.")
        return None

    # Compute metrics
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)

    known_mask = [t != UNKNOWN_LABEL for t in y_true]
    unknown_mask = [t == UNKNOWN_LABEL for t in y_true]

    # True positive: known person correctly identified (exact ID match)
    tp = sum(t == p for t, p, m in zip(y_true, y_pred, known_mask) if m)
    # False reject: known person labelled Unknown
    fr = sum(p == UNKNOWN_LABEL for p, m in zip(y_pred, known_mask) if m)
    # False accept: unknown person labelled as someone
    fa = sum(p != UNKNOWN_LABEL for p, m in zip(y_pred, unknown_mask) if m)
    # Misidentify: known person identified as wrong known person
    misid = sum(
        t != p and p != UNKNOWN_LABEL
        for t, p, m in zip(y_true, y_pred, known_mask) if m
    )

    known_total = sum(known_mask)
    unknown_total = sum(unknown_mask)

    frr = fr / max(known_total, 1)
    far = fa / max(unknown_total, 1)
    known_acc = tp / max(known_total, 1)

    avg_time_ms = np.mean(encode_times) * 1000 if encode_times else 0
    avg_conf = np.mean(confidences) if confidences else 0

    result = {
        "model": model_name,
        "gallery_size": len(gallery),
        "eval_images": len(y_true),
        "accuracy": accuracy,
        "known_accuracy": known_acc,
        "far": far,
        "frr": frr,
        "misidentifications": misid,
        "avg_confidence": avg_conf,
        "avg_time_ms": avg_time_ms,
        "threshold": model.threshold,
        "errors": errors,
    }

    print(f"  Accuracy: {accuracy:.1%}  |  Known acc: {known_acc:.1%}  |  FAR: {far:.1%}  |  FRR: {frr:.1%}")
    print(f"  Avg time: {avg_time_ms:.1f}ms/face  |  Errors: {errors}")
    return result


def print_comparison_table(results: list[dict]) -> None:
    print("=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)

    headers = ["Model", "Threshold", "Accuracy", "Known Acc", "FAR", "FRR", "MisID", "Avg ms", "Avg Conf"]
    widths = [10, 10, 10, 10, 8, 8, 6, 8, 9]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        row = [
            r["model"].ljust(10),
            f"{r['threshold']:.2f}".ljust(10),
            f"{r['accuracy']:.1%}".ljust(10),
            f"{r['known_accuracy']:.1%}".ljust(10),
            f"{r['far']:.1%}".ljust(8),
            f"{r['frr']:.1%}".ljust(8),
            str(r["misidentifications"]).ljust(6),
            f"{r['avg_time_ms']:.1f}".ljust(8),
            f"{r['avg_confidence']:.3f}".ljust(9),
        ]
        print("  ".join(row))

    print()
    best = max(results, key=lambda x: x["accuracy"])
    fastest = min(results, key=lambda x: x["avg_time_ms"])
    print(f"Best accuracy : {best['model']} ({best['accuracy']:.1%})")
    print(f"Fastest       : {fastest['model']} ({fastest['avg_time_ms']:.1f}ms/face)")

    far_candidates = [r for r in results if r["far"] == 0]
    if far_candidates:
        print(f"Zero FAR      : {', '.join(r['model'] for r in far_candidates)}")

    print()
    print("Legend:")
    print("  Accuracy  = overall correct predictions (known + unknown)")
    print("  Known Acc = known people correctly identified by exact ID")
    print("  FAR       = false accept rate (unknown person accepted as someone)")
    print("  FRR       = false reject rate (known person rejected as unknown)")
    print("  MisID     = known person identified as a different known person")


if __name__ == "__main__":
    main()

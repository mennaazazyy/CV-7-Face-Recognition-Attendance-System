#!/usr/bin/env python3
"""CLI: python scripts/enroll_student.py --id 001 --name "Menna Azazy" """
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ACTIVE_MODEL, ENROLLMENT_IMAGES_PER_STUDENT
from src.pipeline.enroll import capture_enrollment_images, enroll_from_images

parser = argparse.ArgumentParser(description="Enrol a student via webcam.")
parser.add_argument("--id", required=True, dest="student_id", help="Student ID")
parser.add_argument("--name", required=True, dest="full_name", help="Full name")
parser.add_argument("--frames", type=int, default=ENROLLMENT_IMAGES_PER_STUDENT, help="Frames to capture")
parser.add_argument("--model", default=ACTIVE_MODEL, help="Model name (default: arcface)")
args = parser.parse_args()

print(f"Capturing {args.frames} frames for {args.student_id} {args.full_name}…")
image_paths = capture_enrollment_images(args.student_id, count=args.frames)
print(f"Captured {len(image_paths)} images. Enrolling with {args.model}…")
enroll_from_images(
    student_id=args.student_id,
    full_name=args.full_name,
    image_paths=image_paths,
    model_name=args.model,
)
print(f"Enrolled {args.student_id} {args.full_name} with {args.model}.")

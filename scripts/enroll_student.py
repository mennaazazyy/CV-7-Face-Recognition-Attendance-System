#!/usr/bin/env python3
"""CLI: python scripts/enroll_student.py --id 001 --name "Menna Azazy" """
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.enroll import enroll_student

parser = argparse.ArgumentParser(description="Enrol a student via webcam.")
parser.add_argument("--id", required=True, dest="student_id", help="Student ID")
parser.add_argument("--name", required=True, dest="full_name", help="Full name")
parser.add_argument("--frames", type=int, default=25, help="Frames to capture")
args = parser.parse_args()

enroll_student(args.student_id, args.full_name, n_frames=args.frames)

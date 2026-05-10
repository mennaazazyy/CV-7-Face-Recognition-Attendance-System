#!/usr/bin/env python3
"""CLI: python scripts/run_attendance.py --class CS401"""
import argparse
import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.attendance_session import run_attendance_session

parser = argparse.ArgumentParser(description="Run a live attendance session.")
parser.add_argument("--class", required=True, dest="class_id", help="Class identifier e.g. CS401")
parser.add_argument("--date", default=str(date.today()), help="Session date (YYYY-MM-DD)")
args = parser.parse_args()

session_id = f"{args.class_id}-{args.date}"
summary = run_attendance_session(session_id=session_id, course_code=args.class_id)
print(f"\nSession {session_id} complete.")
print(f"  Present : {summary['present_count']}")
print(f"  Unknowns: {summary['unknown_count']}")
print(f"  Dupes   : {summary['duplicate_count']}")
print(f"  CSV     : {summary['csv_path']}")

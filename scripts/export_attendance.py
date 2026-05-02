#!/usr/bin/env python3
"""CLI: python scripts/export_attendance.py --session CS401-2025-05-12"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.csv_export import export_session_csv

parser = argparse.ArgumentParser(description="Export attendance to CSV.")
parser.add_argument("--session", required=True, help="Session ID")
args = parser.parse_args()

out = export_session_csv(args.session)
print(f"Exported to {out}")

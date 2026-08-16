#!/usr/bin/env python3
"""Run the independent Mast Cells extraction task."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from lineage_workflow import run_extraction


if __name__ == "__main__":
    run_extraction("Mast Cells", "mast", Path(__file__))

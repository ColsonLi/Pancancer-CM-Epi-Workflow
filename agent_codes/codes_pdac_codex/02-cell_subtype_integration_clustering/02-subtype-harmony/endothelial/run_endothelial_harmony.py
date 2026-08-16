#!/usr/bin/env python3
"""Run the independent Endothelial Cells Harmony task."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from lineage_workflow import run_harmony


if __name__ == "__main__":
    run_harmony("Endothelial Cells", "endothelial", Path(__file__))

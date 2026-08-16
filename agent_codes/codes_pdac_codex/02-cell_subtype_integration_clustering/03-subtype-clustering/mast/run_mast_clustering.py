#!/usr/bin/env python3
"""Run direct subtype clustering for Mast Cells."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from lineage_workflow import run_direct_clustering


if __name__ == "__main__":
    run_direct_clustering(
        "Mast Cells", "mast", "mast", Path(__file__), fixed_resolution=0.2
    )

#!/usr/bin/env python3
"""Export complete raw-cluster DEGs for T/NK Cells."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from lineage_workflow import run_raw_cluster_deg


if __name__ == "__main__":
    run_raw_cluster_deg("T/NK Cells", "t_nk", "t_nk", Path(__file__))

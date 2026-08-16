#!/usr/bin/env python3
"""Prepare stromal subtype candidates without writing final labels."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from candidate_annotation import prepare_candidate_annotation


if __name__ == "__main__":
    prepare_candidate_annotation("stromal", Path(__file__))

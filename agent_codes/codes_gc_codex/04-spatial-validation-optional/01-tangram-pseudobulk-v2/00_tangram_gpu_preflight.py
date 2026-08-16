#!/usr/bin/env python3
"""Fail-fast Tangram/CUDA preflight for the GC spatial validation."""

from __future__ import annotations

import json
import platform
import random
from pathlib import Path

import numpy as np
import scanpy as sc
import tangram as tg
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

WF = Path(__file__).resolve().parents[3]
OUT = WF / "tables/04-spatial-validation-optional/01-tangram-pseudobulk-v2/00-gpu-preflight"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Tangram canonical spatial validation requires cuda:0")
    probe = torch.arange(16, dtype=torch.float32, device="cuda:0").reshape(4, 4)
    value = float((probe @ probe.T).sum().item())
    report = {
        "status": "passed",
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "tangram": getattr(tg, "__version__", "unknown"),
        "scanpy": sc.__version__,
        "python": platform.python_version(),
        "seed": SEED,
        "cuda_probe_value": value,
    }
    (OUT / "tangram_gpu_preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

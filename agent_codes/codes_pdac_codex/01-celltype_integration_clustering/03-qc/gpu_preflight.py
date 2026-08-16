#!/usr/bin/env python3
"""Record the repaired project-local RAPIDS environment and GPU roundtrip."""

from __future__ import annotations

import importlib.metadata
import json
import random
import sys
from pathlib import Path

import anndata as ad
import cugraph
import cudf
import cuml
import cupy as cp
import cuvs
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
from scipy import sparse


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
cp.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
TABLE_DIR = PROJECT_ROOT / "tables/01-celltype_integration_clustering/03-qc/00-gpu-preflight"


def main() -> None:
    outputs = [
        TABLE_DIR / "gpu_preflight.json",
        TABLE_DIR / "environment_repairs.csv",
        TABLE_DIR / "package_versions.txt",
    ]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    source = sparse.csr_matrix(
        np.asarray([[1, 0, 2], [0, 3, 0], [4, 0, 1]], dtype=np.float32)
    )
    adata = ad.AnnData(source.copy())
    rsc.get.anndata_to_GPU(adata)
    gpu_matrix_type = f"{type(adata.X).__module__}.{type(adata.X).__name__}"
    rsc.get.anndata_to_CPU(adata)
    roundtrip_exact = bool((adata.X != source).nnz == 0)
    if not roundtrip_exact:
        raise RuntimeError("RAPIDS CPU->GPU->CPU sparse roundtrip changed values")
    device = cp.cuda.runtime.getDeviceProperties(0)
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    report = {
        "status": "passed",
        "device_name": device["name"].decode(),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "free_vram_bytes": int(free_bytes),
        "total_vram_bytes": int(total_bytes),
        "gpu_matrix_type": gpu_matrix_type,
        "sparse_roundtrip_exact": roundtrip_exact,
        "rapids_singlecell_version": rsc.__version__,
        "cupy_version": cp.__version__,
        "cudf_version": cudf.__version__,
        "cuml_version": cuml.__version__,
        "cugraph_version": cugraph.__version__,
        "cuvs_version": cuvs.__version__,
        "random_seed": SEED,
        "code_file": str(Path(__file__).resolve()),
    }
    outputs[0].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "attempt": 1,
                "status": "failed_import",
                "error_summary": "ModuleNotFoundError: No module named 'dask'",
                "repair": "installed project-local dask, dask-cuda, dask-cudf, distributed and UCX stack",
            },
            {
                "attempt": 2,
                "status": "failed_import",
                "error_summary": "ModuleNotFoundError: No module named 'cuvs'",
                "repair": "installed project-local cuvs, cugraph, pylibcugraph and raft-dask stack",
            },
            {
                "attempt": 3,
                "status": "passed",
                "error_summary": "",
                "repair": "none",
            },
        ]
    ).to_csv(outputs[1], index=False)
    packages = [
        "rapids-singlecell-cu13", "cudf-cu13", "cuml-cu13", "cugraph-cu13",
        "cuvs-cu13", "cupy-cuda13x", "rmm-cu13", "dask", "dask-cuda",
        "dask-cudf-cu13", "distributed", "scanpy", "anndata", "numpy", "scipy",
    ]
    lines = [
        "environment_category=rapids",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        "install_manager=uv",
        "index_url=https://pypi.tuna.tsinghua.edu.cn/simple",
        f"random_seed={SEED}",
        f"code_file={Path(__file__).resolve()}",
    ]
    for package in packages:
        lines.append(f"{package}={importlib.metadata.version(package)}")
    outputs[2].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

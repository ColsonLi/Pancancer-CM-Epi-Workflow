#!/usr/bin/env python3
from __future__ import annotations

import csv
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import rapids_singlecell as rsc
from anndata import AnnData
from scipy import sparse


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "epi-cm-core-workflow/tables/01-celltype_integration_clustering/00-gpu-preflight"
OUT.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command_output(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


props = cp.cuda.runtime.getDeviceProperties(0)
device_name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
driver_cuda = command_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
).splitlines()[0]
nvcc_last = command_output(["/usr/bin/nvcc", "--version"]).splitlines()[-1]

rng = np.random.default_rng(20260802)
matrix = sparse.random(
    400,
    800,
    density=0.06,
    random_state=rng,
    data_rvs=lambda n: rng.integers(1, 6, size=n),
).tocsr().astype(np.float32)
adata = AnnData(matrix)
count_sum_before = int(adata.X.sum())
rsc.get.anndata_to_GPU(adata)
gpu_sparse = isinstance(adata.X, cpsp.csr_matrix)
count_sum_gpu = int(adata.X.sum().get())
rsc.pp.scrublet(
    adata,
    expected_doublet_rate=0.05,
    n_prin_comps=15,
    random_state=0,
    verbose=False,
)
finite_scores = int(np.isfinite(adata.obs["doublet_score"]).sum())
predicted_doublets = int(adata.obs["predicted_doublet"].sum())
rsc.get.anndata_to_CPU(adata)
cpu_sparse = sparse.issparse(adata.X)
count_sum_after = int(adata.X.sum())

checks = [
    {
        "check": "rapids_singlecell_import",
        "status": "pass",
        "observed": version("rapids-singlecell"),
        "expected": "import succeeds",
    },
    {
        "check": "gpu_visible",
        "status": "pass" if cp.cuda.runtime.getDeviceCount() >= 1 else "fail",
        "observed": f"{cp.cuda.runtime.getDeviceCount()} GPU; {device_name}; compute capability {cp.cuda.Device(0).compute_capability}",
        "expected": ">=1 CUDA GPU",
    },
    {
        "check": "anndata_cpu_gpu_cpu_roundtrip",
        "status": "pass" if gpu_sparse and cpu_sparse and count_sum_before == count_sum_gpu == count_sum_after else "fail",
        "observed": f"gpu_sparse={gpu_sparse}; cpu_sparse={cpu_sparse}; count_sums={count_sum_before},{count_sum_gpu},{count_sum_after}",
        "expected": "sparse type appropriate to backend and invariant count sum",
    },
    {
        "check": "gpu_scrublet_smoke",
        "status": "pass" if finite_scores == adata.n_obs and {"doublet_score", "predicted_doublet"}.issubset(adata.obs) else "fail",
        "observed": f"cells={adata.n_obs}; finite_scores={finite_scores}; predicted_doublets={predicted_doublets}",
        "expected": "all cells scored and both Scrublet output columns present",
    },
]
write_csv(OUT / "backend_capability.csv", checks)

versions = []
for package in [
    "rapids-singlecell",
    "cudf-cu12",
    "cugraph-cu12",
    "cuml-cu12",
    "cuvs-cu12",
    "cupy-cuda12x",
    "scanpy",
    "anndata",
    "numpy",
    "scipy",
]:
    versions.append({"component": package, "version": version(package)})
versions.extend(
    [
        {"component": "python", "version": platform.python_version()},
        {"component": "gpu", "version": device_name},
        {"component": "nvidia_driver", "version": driver_cuda},
        {"component": "cupy_runtime_version", "version": str(cp.cuda.runtime.runtimeGetVersion())},
        {"component": "cupy_driver_version", "version": str(cp.cuda.runtime.driverGetVersion())},
        {"component": "system_nvcc", "version": nvcc_last},
    ]
)
write_csv(OUT / "environment_versions.csv", versions)

attempts = [
    {
        "attempt": 1,
        "requested_backend": "rapids-singlecell 0.16.1 via configured TUNA mirror",
        "result": "fail_non_oom",
        "evidence": "source build selected; /usr/bin/nvcc CUDA 11.5; nvlink fatal elfLink linker library load error; GCC std_function parameter-pack errors",
        "decision": "do not switch silently to CPU; test official binary routes",
    },
    {
        "attempt": 2,
        "requested_backend": "rapids-singlecell 0.16.1 from official PyPI plus NVIDIA index with CUDA 12 RAPIDS wheels",
        "result": "fail_non_oom",
        "evidence": "0.16.1 itself has only an sdist and compiles 214 CUDA/C++ units; build again selected system nvcc 11.5",
        "decision": "check official conda package and last upstream wheel release",
    },
    {
        "attempt": 3,
        "requested_backend": "rapids-singlecell 0.16.1 from rapidsai, conda-forge, and nvidia channels",
        "result": "unavailable",
        "evidence": "micromamba search returned no matching entries",
        "decision": "use the last upstream pure-Python wheel and validate required APIs",
    },
    {
        "attempt": 4,
        "requested_backend": "rapids-singlecell 0.14.1 wheel plus CUDA 12 RAPIDS 25.12 binary stack",
        "result": "pass",
        "evidence": "import, RTX 2080 Ti compute, AnnData sparse roundtrip, and GPU Scrublet smoke test all passed",
        "decision": "approved GPU backend for the workflow; version downgrade remains explicit",
    },
]
write_csv(OUT / "backend_install_attempts.csv", attempts)

if not all(row["status"] == "pass" for row in checks):
    raise SystemExit("One or more GPU preflight checks failed")
print(f"GPU preflight passed; audit tables written to {OUT}")

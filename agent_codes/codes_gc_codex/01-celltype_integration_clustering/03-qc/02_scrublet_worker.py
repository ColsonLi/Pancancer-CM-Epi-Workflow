#!/usr/bin/env python3
"""One-sample Scrublet worker used by 01_run_qc.py.

The worker is intentionally process-isolated so a CUDA error can release all
GPU memory before the caller starts an explicitly audited CPU fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


SEED = 42


def package_versions(packages: list[str]) -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def obs_hash(adata: ad.AnnData) -> str:
    return hashlib.sha256("\n".join(adata.obs_names).encode()).hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("gpu", "cpu"))
    parser.add_argument("--expected-doublet-rate", type=float, default=0.05)
    parser.add_argument("--sim-doublet-ratio", type=float, default=2.0)
    parser.add_argument("--n-prin-comps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    adata = ad.read_h5ad(args.input)
    if adata.n_obs < 4 or adata.n_vars < 4:
        raise ValueError(f"Scrublet input is too small: {adata.shape}")
    if adata.obs_names.has_duplicates:
        raise ValueError("Scrublet input has duplicate obs_names")
    actual_pcs = min(args.n_prin_comps, adata.n_obs - 2, adata.n_vars - 1)
    if actual_pcs < 2:
        raise ValueError(f"Scrublet has fewer than two usable PCs: {adata.shape}")
    input_obs_hash = obs_hash(adata)

    if args.backend == "gpu":
        import cupy as cp
        import rapids_singlecell as rsc

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device visible")
        device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        # CuPy sparse does not accept integer dtypes. This is an isolated
        # working copy; values remain unnormalised counts and the caller's
        # merge/final QC h5ads keep their int32 X unchanged.
        adata.X = adata.X.astype(np.float32, copy=False)
        rsc.get.anndata_to_GPU(adata)
        rsc.pp.scrublet(
            adata,
            sim_doublet_ratio=args.sim_doublet_ratio,
            expected_doublet_rate=args.expected_doublet_rate,
            n_prin_comps=actual_pcs,
            random_state=args.seed,
            use_approx_neighbors=True,
            verbose=True,
        )
        # Only obs/uns are needed, but return all arrays to CPU before cleanup.
        rsc.get.anndata_to_CPU(adata, convert_all=True)
        versions = package_versions(
            [
                "anndata",
                "rapids-singlecell",
                "cupy-cuda12x",
                "cuml-cu12",
                "numpy",
                "pandas",
            ]
        )
    else:
        import scanpy as sc

        device = "CPU"
        sc.pp.scrublet(
            adata,
            sim_doublet_ratio=args.sim_doublet_ratio,
            expected_doublet_rate=args.expected_doublet_rate,
            n_prin_comps=actual_pcs,
            random_state=args.seed,
            use_approx_neighbors=True,
            verbose=True,
        )
        versions = package_versions(
            ["anndata", "scanpy", "scrublet", "numpy", "pandas"]
        )

    if "doublet_score" not in adata.obs or "predicted_doublet" not in adata.obs:
        raise RuntimeError("Scrublet returned without required obs columns")
    result = pd.DataFrame(
        {
            "obs_name": adata.obs_names.astype(str),
            "doublet_score": pd.to_numeric(adata.obs["doublet_score"], errors="raise"),
            "predicted_doublet": adata.obs["predicted_doublet"].astype(bool).to_numpy(),
        }
    )
    if result["doublet_score"].isna().any() or len(result) != adata.n_obs:
        raise RuntimeError("Scrublet returned incomplete scores")
    uns = adata.uns.get("scrublet", {})
    metadata = {
        "backend": args.backend,
        "device": device,
        "input": str(args.input),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_prin_comps_requested": int(args.n_prin_comps),
        "n_prin_comps_used": int(actual_pcs),
        "expected_doublet_rate": args.expected_doublet_rate,
        "sim_doublet_ratio": args.sim_doublet_ratio,
        "threshold": float(uns.get("threshold", np.nan)),
        "n_predicted_doublets": int(result["predicted_doublet"].sum()),
        "input_obs_names_sha256": input_obs_hash,
        "random_seed": args.seed,
        "downsampled": False,
        "worker_count_dtype": "float32" if args.backend == "gpu" else str(adata.X.dtype),
        "caller_raw_integer_counts_mutated": False,
        "package_versions": versions,
    }
    atomic_csv(result, args.output_csv)
    atomic_json(metadata, args.output_json)


if __name__ == "__main__":
    main()
    # This process owns no caller state. Explicit exit avoids a known hang in
    # interpreter teardown with the installed CPU Numba/TBB combination.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

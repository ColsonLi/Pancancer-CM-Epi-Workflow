#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import random
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc

SEED = 42
TARGET_SUM = 1e4
N_HVG = 3000
N_PCS = 50
ROOT = Path(__file__).resolve().parents[4]
WF = ROOT / "epi-cm-core-workflow"
INPUT = WF / "h5ad/01-celltype_integration_clustering/03-qc/adata_qc.h5ad"
OUTPUT = WF / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_harmony.h5ad"
TABLE = WF / "tables/01-celltype_integration_clustering/04-integration-harmony"
MANIFEST = WF / "tables/01-celltype_integration_clustering/01-sample-manifest/all_datasets_curated_xlsx_inclusion_v5.csv"
PREFLIGHT = TABLE / "gpu_preflight_attempt02.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def release() -> None:
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not PREFLIGHT.is_file() or not pd.read_csv(PREFLIGHT)["passed"].all():
        raise RuntimeError("GPU preflight is missing or failed")
    if not INPUT.is_file() or not MANIFEST.is_file():
        raise FileNotFoundError("QC h5ad or curated XLSX v5 manifest missing")
    adata = ad.read_h5ad(INPUT)
    input_shape = adata.shape
    input_ids = adata.obs_names.copy()
    if not input_ids.is_unique or adata.raw is not None:
        raise RuntimeError("Invalid clean QC input")
    for col in ["sample", "series", "status", "total_counts", "pct_counts_MT"]:
        if col not in adata.obs or adata.obs[col].isna().any():
            raise KeyError(f"QC input lacks complete {col}")
    if set(adata.obs["status"].astype(str)) != {"normal", "tumor"}:
        raise RuntimeError("status must be exactly normal/tumor")
    manifest_sha = sha256(MANIFEST)
    resources: list[dict[str, object]] = []

    # RAPIDS-SingleCell/CuPy sparse matrices do not accept integer dtypes.
    # Keep the QC h5ad on disk as raw int32 counts, but convert this freshly
    # loaded working copy before the first GPU transfer.  float32 exactly
    # represents the observed UMI counts and is the canonical GPU dtype here.
    input_x_dtype = str(adata.X.dtype)
    adata.X = adata.X.astype(np.float32, copy=False)
    gpu_x_dtype = str(adata.X.dtype)
    if gpu_x_dtype != "float32":
        raise RuntimeError(f"GPU working matrix conversion failed: {gpu_x_dtype}")

    def record(step: str, started: float) -> None:
        free, total = cp.cuda.runtime.memGetInfo()
        resources.append({"step": step, "elapsed_seconds": time.monotonic() - started,
                          "gpu_free_bytes": int(free), "gpu_total_bytes": int(total),
                          "cupy_pool_used_bytes": int(cp.get_default_memory_pool().used_bytes())})

    t = time.monotonic(); rsc.get.anndata_to_GPU(adata); record("transfer", t)
    t = time.monotonic(); rsc.pp.normalize_total(adata, target_sum=TARGET_SUM); record("normalize_total", t)
    t = time.monotonic(); rsc.pp.log1p(adata); record("log1p", t)
    t = time.monotonic(); rsc.pp.highly_variable_genes(
        adata, n_top_genes=N_HVG, flavor="seurat", batch_key="sample"
    ); record("highly_variable_genes", t)
    rsc.get.anndata_to_CPU(adata, convert_all=True); release()
    if int(adata.var["highly_variable"].sum()) != min(N_HVG, adata.n_vars):
        raise RuntimeError("Unexpected HVG count")
    hvg_table = adata.var.reset_index(names="gene")
    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    t = time.monotonic(); rsc.get.anndata_to_GPU(adata); record("transfer_hvg", t)
    t = time.monotonic(); rsc.pp.regress_out(adata, ["total_counts", "pct_counts_MT"], batchsize=100, verbose=True); record("regress_out", t)
    t = time.monotonic(); rsc.pp.scale(adata, max_value=10); record("scale", t)
    t = time.monotonic(); rsc.tl.pca(adata, n_comps=N_PCS, random_state=SEED); record("pca", t)
    clean_pca = adata.obsm["X_pca"].copy()
    attempts: list[dict[str, object]] = []
    accepted = False
    for max_iter in [10, 30]:
        adata.obsm["X_pca"] = clean_pca.copy()
        if "X_pca_inte" in adata.obsm:
            del adata.obsm["X_pca_inte"]
        t = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rsc.pp.harmony_integrate(
                adata, key="sample", basis="X_pca", adjusted_basis="X_pca_inte",
                correction_method="original", use_gemm=False,
                max_iter_harmony=max_iter, random_state=SEED, verbose=True,
            )
        messages = [str(x.message) for x in caught]
        failed = any("did not converge" in x.lower() for x in messages)
        attempts.append({"max_iter_harmony": max_iter, "warnings": " | ".join(messages),
                         "nonconvergence_warning": failed, "accepted": not failed,
                         "elapsed_seconds": time.monotonic() - t})
        record(f"harmony_maxiter_{max_iter}", t)
        if not failed:
            accepted = True
            break
    if not accepted:
        raise RuntimeError("Harmony did not converge from clean PCA at max_iter=30")
    rsc.get.anndata_to_CPU(adata, convert_all=True); release()
    if not np.isfinite(adata.obsm["X_pca_inte"]).all() or adata.obsm["X_pca_inte"].shape != (adata.n_obs, N_PCS):
        raise RuntimeError("Invalid Harmony basis")
    if not adata.obs_names.equals(input_ids) or adata.raw is None or adata.raw.n_vars != input_shape[1]:
        raise RuntimeError("Cell order/raw preservation failed")
    adata.uns["integration_parameters"] = {
        "manifest": str(MANIFEST.relative_to(WF)), "manifest_sha256": manifest_sha,
        "target_sum": TARGET_SUM, "log1p": True, "n_top_hvg": N_HVG,
        "hvg_flavor": "seurat", "hvg_batch_key": "sample",
        "raw_snapshot": "normalized_log_full_genes_before_hvg_subset",
        "regress_keys": ["total_counts", "pct_counts_MT"], "scale_max_value": 10,
        "n_pcs": N_PCS, "harmony_key": "sample", "harmony_basis": "X_pca",
        "harmony_adjusted_basis": "X_pca_inte", "random_seed": SEED,
        "backend": "rapids-singlecell GPU", "downsampled": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".tmp.h5ad")
    adata.write_h5ad(tmp, compression="gzip")
    os.replace(tmp, OUTPUT)
    TABLE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resources).to_csv(TABLE / "gpu_resource_log.csv", index=False)
    pd.DataFrame(
        [
            {
                "attempt": 1,
                "stage": "initial_gpu_transfer",
                "status": "failed_compatibility",
                "input_x_dtype": input_x_dtype,
                "gpu_working_x_dtype": "",
                "error_summary": (
                    "ValueError: Only bool, float32, float64, complex64 and "
                    "complex128 are supported"
                ),
                "repair": "reload clean QC input and cast working X to float32",
                "clean_input_reloaded": True,
            },
            {
                "attempt": 2,
                "stage": "full_gpu_integration",
                "status": "completed",
                "input_x_dtype": input_x_dtype,
                "gpu_working_x_dtype": gpu_x_dtype,
                "error_summary": "",
                "repair": "working-copy dtype conversion only; QC h5ad unchanged",
                "clean_input_reloaded": True,
            },
        ]
    ).to_csv(TABLE / "gpu_compatibility_attempts.csv", index=False)
    pd.DataFrame(attempts).to_csv(TABLE / "harmony_attempts.csv", index=False)
    hvg_table.to_csv(TABLE / "hvg_full_gene_table.csv", index=False)
    hvg_table[hvg_table["highly_variable"].astype(bool)].to_csv(TABLE / "hvg_selected_genes.csv", index=False)
    params = pd.DataFrame([{"input_h5ad": str(INPUT.relative_to(WF)), "output_h5ad": str(OUTPUT.relative_to(WF)),
                            "n_cells": adata.n_obs, "n_genes_input": input_shape[1], "n_hvg": adata.n_vars,
                            "target_sum": TARGET_SUM, "hvg_batch_key": "sample", "n_pcs": N_PCS,
                            "harmony_key": "sample", "manifest_sha256": manifest_sha,
                            "input_x_dtype": input_x_dtype, "gpu_working_x_dtype": gpu_x_dtype,
                            "random_seed": SEED, "backend": "rapids-singlecell GPU", "cpu_fallback": False,
                            "downsampled": False}])
    params.to_csv(TABLE / "integration_parameters.csv", index=False)
    code_file = str(Path(__file__).relative_to(ROOT))
    input_rel = str(INPUT.relative_to(WF))
    output_rel = str(OUTPUT.relative_to(WF))
    common = {
        "input_h5ad": input_rel,
        "output_h5ad_or_object": output_rel,
        "n_obs_before": input_shape[0],
        "n_obs_after": adata.n_obs,
        "backend_package": "rapids-singlecell GPU",
        "code_file": code_file,
        "random_seed": SEED,
        "notes_fallbacks": "full data; no downsampling; no fallback",
    }
    step_tables = [
        ("01_normalize_total_parameters.csv", "normalize_total", input_shape[1], input_shape[1],
         "target_sum=10000"),
        ("02_log1p_parameters.csv", "log1p", input_shape[1], input_shape[1],
         "natural log1p after normalize_total"),
        ("03_highly_variable_genes_parameters.csv", "highly_variable_genes", input_shape[1], input_shape[1],
         f"flavor=seurat;n_top_genes={N_HVG};batch_key=sample;actual_hvg={adata.n_vars}"),
        ("04_raw_assignment_and_hvg_subset_record.csv", "raw_assignment_and_hvg_subset", input_shape[1], adata.n_vars,
         "raw=normalized_log_full_genes;subset=highly_variable"),
        ("05_regress_out_parameters.csv", "regress_out", adata.n_vars, adata.n_vars,
         "keys=total_counts,pct_counts_MT;batchsize=100"),
        ("06_scale_parameters.csv", "scale", adata.n_vars, adata.n_vars,
         "max_value=10"),
        ("07_pca_parameters.csv", "pca", adata.n_vars, adata.n_vars,
         f"n_comps={N_PCS};zero_center=package_default"),
        ("08_harmony_integrate_parameters.csv", "harmony_integrate", adata.n_vars, adata.n_vars,
         "key=sample;basis=X_pca;adjusted_basis=X_pca_inte;max_iter=10_then_30_if_needed"),
    ]
    summary_rows = []
    for filename, step, n_vars_before, n_vars_after, exact_parameters in step_tables:
        row = {
            **common,
            "step": step,
            "n_vars_before": n_vars_before,
            "n_vars_after": n_vars_after,
            "exact_parameters": exact_parameters,
        }
        pd.DataFrame([row]).to_csv(TABLE / filename, index=False)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(TABLE / "integration_harmony_summary.csv", index=False)
    pd.DataFrame([
        {
            "step": row[1],
            "planned_backend": "rapids-singlecell GPU",
            "attempted_backend": "rapids-singlecell GPU",
            "status": "completed",
            "error_summary": "",
            "fallback_backend": "",
            "clean_input_reloaded": True,
            "final_backend_for_rerun": "rapids-singlecell GPU",
        }
        for row in step_tables
    ]).to_csv(TABLE / "gpu_backend_capability_summary.csv", index=False)
    validation = {"all_passed": True, "n_obs": adata.n_obs, "n_hvg": adata.n_vars,
                  "raw_n_vars": adata.raw.n_vars, "cell_order_sha256": hashlib.sha256("\n".join(input_ids).encode()).hexdigest(),
                  "harmony_shape": list(adata.obsm["X_pca_inte"].shape), "output_sha256": sha256(OUTPUT)}
    (TABLE / "integration_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    versions = {p: version(p) for p in ["anndata", "scanpy", "rapids-singlecell", "cupy-cuda12x", "numpy", "pandas"]}
    versions.update({"python": platform.python_version(), "seed": SEED})
    (TABLE / "package_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    (TABLE / "package_versions.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in sorted(versions.items()))
        + f"\nenvironment={ROOT / 'uv_envs/rapids/.venv'}\ncode_file={code_file}\n"
    )
    (TABLE / "readme.txt").write_text(
        "GPU full-cell normalization/log1p/3000 sample-aware HVG/raw/regression/scale/PCA50/Harmony-by-sample.\n"
        "No graph, UMAP, clustering, downsampling, or CPU fallback was used.\n"
    )


if __name__ == "__main__":
    main()

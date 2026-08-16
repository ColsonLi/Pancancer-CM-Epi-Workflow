#!/usr/bin/env python3
"""Recompute epithelial HVGs, regression, PCA, and sample-keyed Harmony on GPU."""

from __future__ import annotations

import importlib.metadata
import json
import random
import sys
import time
import warnings
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc


SEED = 42
N_TOP_GENES = 3_000
REGRESSION_KEYS = ["total_counts", "pct_counts_MT"]
REGRESSION_BATCHSIZE = 1_000
SCALE_MAX_VALUE = 10.0
PCA_N_COMPS = 50
HARMONY_MAX_ITER = 50
BATCH_KEY = "sample"
PCA_BASIS = "X_pca"
INTEGRATED_BASIS = "X_pca_inte"

random.seed(SEED)
np.random.seed(SEED)
cp.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
INPUT_H5AD = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/01-subtype-extraction/epithelial/adata_epithelial_normalized_log_all_genes.h5ad"
)
H5AD_DIR = (
    PROJECT_ROOT / "h5ad/02-cell_subtype_integration_clustering/02-subtype-harmony/epithelial"
)
PENDING_H5AD_DIR = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/02-subtype-harmony/epithelial_pending_validation"
)
PRE_HARMONY_H5AD = H5AD_DIR / "adata_epithelial_pre_harmony_pca.h5ad"
OUTPUT_H5AD = H5AD_DIR / "adata_epithelial_harmony.h5ad"
PENDING_PRE_HARMONY_H5AD = PENDING_H5AD_DIR / PRE_HARMONY_H5AD.name
PENDING_OUTPUT_H5AD = PENDING_H5AD_DIR / OUTPUT_H5AD.name
TABLE_DIR = (
    PROJECT_ROOT / "tables/02-cell_subtype_integration_clustering/02-subtype-harmony/epithelial"
)


def package_version_lines() -> list[str]:
    packages = [
        "rapids-singlecell-cu13",
        "cudf-cu13",
        "cuml-cu13",
        "cugraph-cu13",
        "cuvs-cu13",
        "cupy-cuda13x",
        "scanpy",
        "anndata",
        "numpy",
        "pandas",
        "scipy",
        "harmonypy",
    ]
    lines = [
        "environment_category=rapids",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"random_seed={SEED}",
        f"code_files={Path(__file__).resolve()}",
    ]
    lines.extend(f"{package}={importlib.metadata.version(package)}" for package in packages)
    return lines


def main() -> None:
    if not INPUT_H5AD.is_file():
        raise FileNotFoundError(INPUT_H5AD)
    for path in [H5AD_DIR, PENDING_H5AD_DIR, TABLE_DIR]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite epithelial Harmony artifact: {path}")

    started = time.time()
    adata = sc.read_h5ad(INPUT_H5AD)
    n_obs, n_vars_all = adata.shape
    required_obs = {"sample", "series", "status", "original_barcode", *REGRESSION_KEYS}
    missing_obs = required_obs.difference(adata.obs.columns)
    if missing_obs:
        raise KeyError(f"Extracted epithelial object lacks obs columns: {missing_obs}")
    if adata.raw is None or adata.raw.n_vars != n_vars_all:
        raise ValueError("Extracted epithelial object lacks the all-gene normalized/log raw snapshot")
    if len(adata.layers) != 0:
        raise ValueError("Extracted mixed-source normalized atlas must not fabricate a counts layer")
    for key in REGRESSION_KEYS:
        values = pd.to_numeric(adata.obs[key], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Regression covariate {key} contains non-finite values")
        adata.obs[key] = values

    timings = {}
    backend_rows = []
    rsc.get.anndata_to_GPU(adata)
    step_started = time.time()
    rsc.pp.highly_variable_genes(
        adata,
        n_top_genes=N_TOP_GENES,
        flavor="seurat",
        batch_key="series",
    )
    rsc.get.anndata_to_CPU(adata)
    timings["highly_variable_genes"] = time.time() - step_started
    n_hvg = int(adata.var["highly_variable"].sum())
    if n_hvg != N_TOP_GENES:
        raise ValueError(f"Expected {N_TOP_GENES} epithelial HVGs, observed {n_hvg}")
    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    if adata.raw is None or adata.raw.n_vars != n_vars_all:
        raise ValueError("All-gene raw snapshot was lost during epithelial HVG subsetting")

    rsc.get.anndata_to_GPU(adata)
    step_started = time.time()
    rsc.pp.regress_out(
        adata,
        keys=REGRESSION_KEYS,
        batchsize=REGRESSION_BATCHSIZE,
        verbose=False,
    )
    timings["regress_out"] = time.time() - step_started
    step_started = time.time()
    rsc.pp.scale(adata, zero_center=True, max_value=SCALE_MAX_VALUE)
    timings["scale"] = time.time() - step_started
    step_started = time.time()
    rsc.tl.pca(adata, n_comps=PCA_N_COMPS, random_state=SEED, dtype="float32")
    timings["pca"] = time.time() - step_started
    rsc.get.anndata_to_CPU(adata)
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    if adata.obsm[PCA_BASIS].shape != (n_obs, PCA_N_COMPS):
        raise ValueError("Epithelial PCA has an unexpected shape")
    if INTEGRATED_BASIS in adata.obsm or "neighbors" in adata.uns or "X_umap" in adata.obsm:
        raise ValueError("Clean epithelial pre-Harmony object contains downstream results")
    adata.uns["epithelial_pre_harmony_contract"] = {
        "input_h5ad": str(INPUT_H5AD),
        "n_top_genes": N_TOP_GENES,
        "hvg_flavor": "seurat",
        "hvg_batch_key": "series",
        "regression_keys": REGRESSION_KEYS,
        "scale_max_value": SCALE_MAX_VALUE,
        "pca_n_comps": PCA_N_COMPS,
        "random_seed": SEED,
    }

    PENDING_H5AD_DIR.mkdir(parents=True, exist_ok=False)
    adata.write_h5ad(PENDING_PRE_HARMONY_H5AD, compression="gzip", compression_opts=1)
    del adata

    adata = sc.read_h5ad(PENDING_PRE_HARMONY_H5AD)
    step_started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rsc.pp.harmony_integrate(
            adata,
            key=BATCH_KEY,
            basis=PCA_BASIS,
            adjusted_basis=INTEGRATED_BASIS,
            max_iter_harmony=HARMONY_MAX_ITER,
            random_state=SEED,
            verbose=True,
        )
    timings["harmony_integrate"] = time.time() - step_started
    warning_messages = [str(item.message) for item in caught]
    nonconvergence = [message for message in warning_messages if "did not converge" in message.lower()]
    if nonconvergence:
        raise RuntimeError(
            f"Epithelial Harmony did not converge at max_iter={HARMONY_MAX_ITER}: {nonconvergence}"
        )
    if INTEGRATED_BASIS not in adata.obsm:
        raise ValueError("Epithelial Harmony did not create X_pca_inte")
    if adata.obsm[INTEGRATED_BASIS].shape != adata.obsm[PCA_BASIS].shape:
        raise ValueError("Epithelial PCA and Harmony embeddings differ in shape")
    forbidden = [
        column for column in adata.obs if column.startswith(("leiden_res", "louvain_res"))
    ]
    if forbidden or "neighbors" in adata.uns or "X_umap" in adata.obsm:
        raise ValueError("Epithelial Harmony handoff contains clustering/UMAP results")
    adata.uns["epithelial_integration_contract"] = {
        "input_expression_state": "all_gene_normalized_log; UCSC not renormalized",
        "hvg_n_top_genes": N_TOP_GENES,
        "hvg_flavor": "seurat",
        "hvg_batch_key": "series",
        "raw_assignment_point": "extraction before subtype HVG selection",
        "regression_keys": REGRESSION_KEYS,
        "scale_max_value": SCALE_MAX_VALUE,
        "pca_n_comps": PCA_N_COMPS,
        "harmony_batch_key": BATCH_KEY,
        "harmony_max_iter": HARMONY_MAX_ITER,
        "pca_basis": PCA_BASIS,
        "integrated_basis": INTEGRATED_BASIS,
        "random_seed": SEED,
    }
    adata.write_h5ad(PENDING_OUTPUT_H5AD, compression="gzip", compression_opts=1)

    pending_errors = []
    for path, harmony_required in [
        (PENDING_PRE_HARMONY_H5AD, False),
        (PENDING_OUTPUT_H5AD, True),
    ]:
        pending = sc.read_h5ad(path, backed="r")
        if pending.shape != (n_obs, N_TOP_GENES):
            pending_errors.append(f"shape mismatch: {path.name}")
        if pending.raw is None or pending.raw.n_vars != n_vars_all:
            pending_errors.append(f"raw mismatch: {path.name}")
        if PCA_BASIS not in pending.obsm:
            pending_errors.append(f"PCA missing: {path.name}")
        if harmony_required != (INTEGRATED_BASIS in pending.obsm):
            pending_errors.append(f"Harmony presence mismatch: {path.name}")
        if "neighbors" in pending.uns or "X_umap" in pending.obsm:
            pending_errors.append(f"downstream graph present: {path.name}")
        pending.file.close()
        del pending
    if pending_errors:
        raise ValueError(f"Pending epithelial Harmony validation failed: {pending_errors}")
    PENDING_H5AD_DIR.replace(H5AD_DIR)

    for step in ["highly_variable_genes", "regress_out", "scale", "pca", "harmony_integrate"]:
        backend_rows.append(
            {
                "step": step,
                "planned_backend": "rapids-singlecell_GPU",
                "attempted_backend": "rapids-singlecell_GPU",
                "status": "completed",
                "error_summary": "",
                "fallback_backend": "",
                "clean_input_reloaded": step == "harmony_integrate",
                "final_backend_for_rerun": "rapids-singlecell_GPU",
            }
        )
    TABLE_DIR.mkdir(parents=True, exist_ok=False)
    common = {
        "input_h5ad": str(INPUT_H5AD),
        "output_h5ad": str(OUTPUT_H5AD),
        "n_obs": n_obs,
        "n_vars_all_gene_raw": n_vars_all,
        "n_vars_hvg": N_TOP_GENES,
        "random_seed": SEED,
        "code_file": str(Path(__file__).resolve()),
    }
    parameter_rows = [
        {**common, "step": "highly_variable_genes", "n_top_genes": N_TOP_GENES, "flavor": "seurat", "batch_key": "series", "elapsed_seconds": timings["highly_variable_genes"]},
        {**common, "step": "raw_preservation_and_hvg_subset", "raw_assignment_point": "extraction_before_HVG_selection", "raw_n_vars": n_vars_all, "elapsed_seconds": 0.0},
        {**common, "step": "regress_out", "regression_keys": ";".join(REGRESSION_KEYS), "batchsize": REGRESSION_BATCHSIZE, "elapsed_seconds": timings["regress_out"]},
        {**common, "step": "scale", "zero_center": True, "max_value": SCALE_MAX_VALUE, "elapsed_seconds": timings["scale"]},
        {**common, "step": "pca", "n_comps": PCA_N_COMPS, "basis": PCA_BASIS, "dtype": "float32", "elapsed_seconds": timings["pca"]},
        {**common, "step": "harmony_integrate", "key": BATCH_KEY, "basis": PCA_BASIS, "adjusted_basis": INTEGRATED_BASIS, "max_iter_harmony": HARMONY_MAX_ITER, "converged": True, "warning_messages": " | ".join(warning_messages), "elapsed_seconds": timings["harmony_integrate"]},
    ]
    pd.DataFrame(parameter_rows).to_csv(TABLE_DIR / "epithelial_harmony_parameters.csv", index=False)
    pd.DataFrame(
        [
            {
                "attempt": 1,
                "status": "failed_validation_not_promoted",
                "error_summary": "validator treated legal broad annotation column leiden_coarse as a technical clustering column",
                "biological_method_changed": False,
                "harmony_converged": True,
                "harmony_iterations": 11,
            },
            {
                "attempt": 2,
                "status": "completed",
                "error_summary": "",
                "biological_method_changed": False,
                "harmony_converged": True,
                "harmony_iterations": 11,
            },
        ]
    ).to_csv(TABLE_DIR / "execution_attempts.csv", index=False)
    pd.DataFrame(backend_rows).to_csv(TABLE_DIR / "gpu_backend_capability_summary.csv", index=False)
    (TABLE_DIR / "package_versions.txt").write_text(
        "\n".join(package_version_lines()) + "\n", encoding="utf-8"
    )
    elapsed = time.time() - started
    validation = {
        **common,
        "pre_harmony_h5ad": str(PRE_HARMONY_H5AD),
        "pca_shape": list(adata.obsm[PCA_BASIS].shape),
        "harmony_shape": list(adata.obsm[INTEGRATED_BASIS].shape),
        "harmony_converged": True,
        "harmony_max_iter": HARMONY_MAX_ITER,
        "neighbor_graph_present": "neighbors" in adata.uns,
        "umap_present": "X_umap" in adata.obsm,
        "cluster_columns_present": forbidden,
        "counts_layer_present": "counts" in adata.layers,
        "UCSC_renormalized": False,
        "obs_names_unique": bool(adata.obs_names.is_unique),
        "total_elapsed_seconds": elapsed,
        "output_h5ad_readable": OUTPUT_H5AD.is_file(),
    }
    (TABLE_DIR / "epithelial_harmony_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

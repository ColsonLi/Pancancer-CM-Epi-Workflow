#!/usr/bin/env python3
"""Run HVG -> raw assignment -> regress/scale/PCA -> Harmony on the full atlas."""

from __future__ import annotations

import importlib.metadata
import json
import random
import re
import sys
import time
from pathlib import Path

import anndata as ad
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
HARMONY_MAX_ITER = 10
BATCH_KEY = "sample"
PCA_BASIS = "X_pca"
INTEGRATED_BASIS = "X_pca_inte"

random.seed(SEED)
np.random.seed(SEED)
cp.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
INPUT_H5AD = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_normalized_log_all_genes.h5ad"
)
OUTPUT_H5AD = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_harmony.h5ad"
)
TABLE_DIR = PROJECT_ROOT / "tables/01-celltype_integration_clustering/04-integration-harmony"
NORMALIZE_BACKEND_TABLE = TABLE_DIR / "gpu_backend_normalize_ucsc_merge.csv"
ENSEMBL_GENE_RE = re.compile(r"^ENS[A-Z]*G[0-9]+(?:\.[0-9]+)?$")


def require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def parameter_row(step: str, n_obs: int, n_vars_before: int, n_vars_after: int) -> dict[str, object]:
    return {
        "step": step,
        "input_h5ad": str(INPUT_H5AD),
        "output_h5ad_or_object": str(OUTPUT_H5AD),
        "n_obs_before": n_obs,
        "n_obs_after": n_obs,
        "n_vars_before": n_vars_before,
        "n_vars_after": n_vars_after,
        "backend_package": "rapids-singlecell_GPU",
        "code_file": str(Path(__file__).resolve()),
        "random_seed": SEED,
        "notes_fallbacks": "none",
    }


def gene_audit(index: pd.Index, object_layer: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    values = pd.Index(index).astype(str)
    is_ensembl = values.to_series(index=range(len(values))).str.match(ENSEMBL_GENE_RE)
    fraction = float(is_ensembl.mean()) if len(values) else 0.0
    kind = "gene_id" if fraction >= 0.90 else (
        "gene_name" if fraction <= 0.10 else "mixed_or_unknown"
    )
    row = {
        "input_h5ad": str(OUTPUT_H5AD),
        "object_layer": object_layer,
        "var_names_type": kind,
        "n_features": len(values),
        "n_unique": int(values.nunique()),
        "n_duplicates": int(values.duplicated().sum()),
        "ensembl_like_n": int(is_ensembl.sum()),
        "ensembl_like_fraction": fraction,
        "candidate_gene_id_columns": "",
        "candidate_gene_name_columns": "var_names;gene_name",
        "selected_gene_id_source": "",
        "selected_gene_name_source": "var_names",
        "action_taken": "retained_unique_gene_names",
    }
    examples = [
        {"object_layer": object_layer, "example_order": i + 1, "var_name": value}
        for i, value in enumerate(values[:20])
    ]
    return row, examples


def main() -> None:
    if not INPUT_H5AD.is_file():
        raise FileNotFoundError(INPUT_H5AD)
    if not NORMALIZE_BACKEND_TABLE.is_file():
        raise FileNotFoundError(NORMALIZE_BACKEND_TABLE)
    outputs = [
        OUTPUT_H5AD,
        TABLE_DIR / "03_highly_variable_genes_parameters.csv",
        TABLE_DIR / "04_raw_assignment_and_hvg_subset_record.csv",
        TABLE_DIR / "05_regress_out_parameters.csv",
        TABLE_DIR / "06_scale_parameters.csv",
        TABLE_DIR / "07_pca_parameters.csv",
        TABLE_DIR / "08_harmony_integrate_parameters.csv",
        TABLE_DIR / "integration_harmony_summary.csv",
        TABLE_DIR / "gpu_backend_capability_summary.csv",
        TABLE_DIR / "gene_identifier_audit_harmony.csv",
        TABLE_DIR / "gene_identifier_examples_harmony.csv",
        TABLE_DIR / "package_versions.txt",
        TABLE_DIR / "harmony_validation.json",
    ]
    for path in outputs:
        require_new(path)
    OUTPUT_H5AD.parent.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    started = time.time()
    adata = sc.read_h5ad(INPUT_H5AD)
    n_obs, n_vars_all = adata.shape
    required_obs = {"sample", "series", "status", "original_barcode", *REGRESSION_KEYS}
    if not required_obs.issubset(adata.obs.columns):
        raise ValueError(f"Input lacks required obs columns: {required_obs - set(adata.obs.columns)}")
    for key in REGRESSION_KEYS:
        values = pd.to_numeric(adata.obs[key], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Regression covariate {key} contains non-finite values")
        adata.obs[key] = values
    if "counts" in adata.layers:
        raise ValueError("Combined normalized atlas must not contain a fabricated counts layer")

    backend_rows: list[dict[str, object]] = []
    rsc.get.anndata_to_GPU(adata)
    hvg_started = time.time()
    rsc.pp.highly_variable_genes(
        adata,
        n_top_genes=N_TOP_GENES,
        flavor="seurat",
        batch_key="series",
    )
    rsc.get.anndata_to_CPU(adata)
    hvg_elapsed = time.time() - hvg_started
    n_hvg = int(adata.var["highly_variable"].sum())
    if n_hvg < N_TOP_GENES:
        raise ValueError(f"Only {n_hvg} HVGs were selected; expected at least {N_TOP_GENES}")
    row = parameter_row("highly_variable_genes", n_obs, n_vars_all, n_vars_all)
    row.update(
        {
            "n_top_genes": N_TOP_GENES,
            "actual_hvg_count": n_hvg,
            "flavor": "seurat",
            "batch_key": "series",
            "elapsed_seconds": hvg_elapsed,
        }
    )
    pd.DataFrame([row]).to_csv(TABLE_DIR / "03_highly_variable_genes_parameters.csv", index=False)
    backend_rows.append(
        {
            "step": "highly_variable_genes",
            "planned_backend": "rapids-singlecell_GPU",
            "attempted_backend": "rapids-singlecell_GPU",
            "status": "completed",
            "error_summary": "",
            "fallback_backend": "",
            "clean_input_reloaded": False,
            "final_backend_for_rerun": "rapids-singlecell_GPU",
        }
    )

    adata.raw = adata.copy()
    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    raw_row = parameter_row("raw_assignment_and_hvg_subset", n_obs, n_vars_all, adata.n_vars)
    raw_row.update(
        {
            "raw_assignment_point": "after_normalize_log_and_HVG_marking_before_HVG_subset",
            "raw_matrix_role": "all_gene_normalized_log_expression_for_DEG",
            "raw_n_vars": adata.raw.n_vars,
            "HVG_subset_n_vars": adata.n_vars,
            "counts_layer_present": False,
            "non_UCSC_counts_h5ad": adata.uns.get("non_UCSC_raw_counts_h5ad", ""),
        }
    )
    pd.DataFrame([raw_row]).to_csv(
        TABLE_DIR / "04_raw_assignment_and_hvg_subset_record.csv", index=False
    )

    rsc.get.anndata_to_GPU(adata)
    regress_started = time.time()
    rsc.pp.regress_out(
        adata,
        keys=REGRESSION_KEYS,
        batchsize=REGRESSION_BATCHSIZE,
        verbose=False,
    )
    regress_elapsed = time.time() - regress_started
    row = parameter_row("regress_out", n_obs, n_hvg, n_hvg)
    row.update(
        {
            "regression_keys": ";".join(REGRESSION_KEYS),
            "batchsize": REGRESSION_BATCHSIZE,
            "elapsed_seconds": regress_elapsed,
        }
    )
    pd.DataFrame([row]).to_csv(TABLE_DIR / "05_regress_out_parameters.csv", index=False)
    backend_rows.append(
        {
            "step": "regress_out",
            "planned_backend": "rapids-singlecell_GPU",
            "attempted_backend": "rapids-singlecell_GPU",
            "status": "completed",
            "error_summary": "",
            "fallback_backend": "",
            "clean_input_reloaded": False,
            "final_backend_for_rerun": "rapids-singlecell_GPU",
        }
    )

    scale_started = time.time()
    rsc.pp.scale(adata, zero_center=True, max_value=SCALE_MAX_VALUE)
    scale_elapsed = time.time() - scale_started
    row = parameter_row("scale", n_obs, n_hvg, n_hvg)
    row.update(
        {
            "zero_center": True,
            "max_value": SCALE_MAX_VALUE,
            "elapsed_seconds": scale_elapsed,
        }
    )
    pd.DataFrame([row]).to_csv(TABLE_DIR / "06_scale_parameters.csv", index=False)
    backend_rows.append(
        {
            "step": "scale",
            "planned_backend": "rapids-singlecell_GPU",
            "attempted_backend": "rapids-singlecell_GPU",
            "status": "completed",
            "error_summary": "",
            "fallback_backend": "",
            "clean_input_reloaded": False,
            "final_backend_for_rerun": "rapids-singlecell_GPU",
        }
    )

    pca_started = time.time()
    rsc.tl.pca(adata, n_comps=PCA_N_COMPS, random_state=SEED, dtype="float32")
    pca_elapsed = time.time() - pca_started
    row = parameter_row("pca", n_obs, n_hvg, n_hvg)
    row.update(
        {
            "n_comps": PCA_N_COMPS,
            "basis": PCA_BASIS,
            "svd_solver": "rapids_default",
            "dtype": "float32",
            "elapsed_seconds": pca_elapsed,
        }
    )
    pd.DataFrame([row]).to_csv(TABLE_DIR / "07_pca_parameters.csv", index=False)
    backend_rows.append(
        {
            "step": "pca",
            "planned_backend": "rapids-singlecell_GPU",
            "attempted_backend": "rapids-singlecell_GPU",
            "status": "completed",
            "error_summary": "",
            "fallback_backend": "",
            "clean_input_reloaded": False,
            "final_backend_for_rerun": "rapids-singlecell_GPU",
        }
    )

    harmony_started = time.time()
    rsc.pp.harmony_integrate(
        adata,
        key=BATCH_KEY,
        basis=PCA_BASIS,
        adjusted_basis=INTEGRATED_BASIS,
        max_iter_harmony=HARMONY_MAX_ITER,
        random_state=SEED,
        verbose=True,
    )
    harmony_elapsed = time.time() - harmony_started
    if INTEGRATED_BASIS not in adata.obsm:
        raise ValueError(f"Harmony did not create obsm[{INTEGRATED_BASIS!r}]")
    row = parameter_row("harmony_integrate", n_obs, n_hvg, n_hvg)
    row.update(
        {
            "key": BATCH_KEY,
            "basis": PCA_BASIS,
            "adjusted_basis": INTEGRATED_BASIS,
            "flavor": "harmony2",
            "max_iter_harmony": HARMONY_MAX_ITER,
            "max_iter_clustering": 200,
            "tol_harmony": 0.0001,
            "tol_clustering": 0.00001,
            "elapsed_seconds": harmony_elapsed,
        }
    )
    pd.DataFrame([row]).to_csv(TABLE_DIR / "08_harmony_integrate_parameters.csv", index=False)
    backend_rows.append(
        {
            "step": "harmony_integrate",
            "planned_backend": "rapids-singlecell_GPU",
            "attempted_backend": "rapids-singlecell_GPU",
            "status": "completed",
            "error_summary": "",
            "fallback_backend": "",
            "clean_input_reloaded": False,
            "final_backend_for_rerun": "rapids-singlecell_GPU",
        }
    )
    rsc.get.anndata_to_CPU(adata)
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    if adata.obsm[PCA_BASIS].shape != (n_obs, PCA_N_COMPS):
        raise ValueError(f"Unexpected PCA shape: {adata.obsm[PCA_BASIS].shape}")
    if adata.obsm[INTEGRATED_BASIS].shape != (n_obs, PCA_N_COMPS):
        raise ValueError(f"Unexpected Harmony shape: {adata.obsm[INTEGRATED_BASIS].shape}")
    forbidden = [column for column in adata.obs.columns if column.startswith("leiden") or column.startswith("louvain")]
    if forbidden or "X_umap" in adata.obsm or "neighbors" in adata.uns:
        raise ValueError("Harmony handoff unexpectedly contains clustering/UMAP/neighbor outputs")
    adata.uns["integration_contract"] = {
        "normalization_target_sum": 10_000.0,
        "log1p": True,
        "hvg_n_top_genes": N_TOP_GENES,
        "hvg_batch_key": "series",
        "raw_assignment_point": "after_HVG_marking_before_HVG_subset",
        "regression_keys": REGRESSION_KEYS,
        "scale_max_value": SCALE_MAX_VALUE,
        "pca_n_comps": PCA_N_COMPS,
        "harmony_batch_key": BATCH_KEY,
        "pca_basis": PCA_BASIS,
        "integrated_basis": INTEGRATED_BASIS,
        "random_seed": SEED,
    }
    adata.write_h5ad(OUTPUT_H5AD, compression="gzip", compression_opts=1)

    prior_backend = pd.read_csv(NORMALIZE_BACKEND_TABLE)
    pd.concat([prior_backend, pd.DataFrame(backend_rows)], ignore_index=True).to_csv(
        TABLE_DIR / "gpu_backend_capability_summary.csv", index=False
    )
    audit_rows = []
    example_rows = []
    row, examples = gene_audit(adata.var_names, "var_names_HVG_scaled_matrix")
    audit_rows.append(row)
    example_rows.extend(examples)
    row, examples = gene_audit(adata.raw.var_names, "raw.var_names_all_gene_normalized_log")
    audit_rows.append(row)
    example_rows.extend(examples)
    pd.DataFrame(audit_rows).to_csv(TABLE_DIR / "gene_identifier_audit_harmony.csv", index=False)
    pd.DataFrame(example_rows).to_csv(TABLE_DIR / "gene_identifier_examples_harmony.csv", index=False)

    total_elapsed = time.time() - started
    pd.DataFrame(
        [
            {
                "step": "integration_harmony_summary",
                "input_h5ad": str(INPUT_H5AD),
                "output_h5ad_or_object": str(OUTPUT_H5AD),
                "n_obs_before": n_obs,
                "n_obs_after": adata.n_obs,
                "n_vars_before": n_vars_all,
                "n_vars_after": adata.n_vars,
                "raw_n_vars": adata.raw.n_vars,
                "HVG_count": n_hvg,
                "pca_shape": str(tuple(adata.obsm[PCA_BASIS].shape)),
                "harmony_shape": str(tuple(adata.obsm[INTEGRATED_BASIS].shape)),
                "batch_key": BATCH_KEY,
                "backend_package": "rapids-singlecell_GPU",
                "code_file": str(Path(__file__).resolve()),
                "random_seed": SEED,
                "total_elapsed_seconds": total_elapsed,
                "notes_fallbacks": "none_first_full_execution",
            }
        ]
    ).to_csv(TABLE_DIR / "integration_harmony_summary.csv", index=False)

    packages = [
        "rapids-singlecell-cu13", "cudf-cu13", "cuml-cu13", "cugraph-cu13",
        "cuvs-cu13", "cupy-cuda13x", "rmm-cu13", "scanpy", "anndata",
        "numpy", "pandas", "scipy", "h5py", "harmonypy",
    ]
    lines = [
        "environment_category=rapids",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        "install_manager=uv",
        "index_url=https://pypi.tuna.tsinghua.edu.cn/simple",
        f"random_seed={SEED}",
        f"code_files={PROJECT_ROOT / 'codes/01-celltype_integration_clustering/04-integration-harmony/build_normalized_combined_atlas.py'};{Path(__file__).resolve()}",
    ]
    for package in packages:
        lines.append(f"{package}={importlib.metadata.version(package)}")
    (TABLE_DIR / "package_versions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    validation = {
        "input_h5ad": str(INPUT_H5AD),
        "output_h5ad": str(OUTPUT_H5AD),
        "n_obs": adata.n_obs,
        "n_vars_HVG": adata.n_vars,
        "raw_n_vars_all_gene_normalized_log": adata.raw.n_vars,
        "raw_role": "all_gene_normalized_log_expression_for_DEG",
        "counts_layer_present": "counts" in adata.layers,
        "non_UCSC_raw_counts_h5ad": adata.uns.get("non_UCSC_raw_counts_h5ad", ""),
        "pca_shape": list(adata.obsm[PCA_BASIS].shape),
        "harmony_shape": list(adata.obsm[INTEGRATED_BASIS].shape),
        "neighbor_graph_present": "neighbors" in adata.uns,
        "umap_present": "X_umap" in adata.obsm,
        "cluster_columns_present": forbidden,
        "obs_names_unique": bool(adata.obs_names.is_unique),
        "required_obs_present": sorted(required_obs),
        "random_seed": SEED,
    }
    (TABLE_DIR / "harmony_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

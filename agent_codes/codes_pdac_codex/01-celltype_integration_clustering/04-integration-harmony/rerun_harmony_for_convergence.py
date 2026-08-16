#!/usr/bin/env python3
"""Preserve max-iter-10 output and rerun only Harmony at max_iter=50."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import random
import shutil
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
INITIAL_MAX_ITER = 10
INCREASED_MAX_ITER = 50
BATCH_KEY = "sample"
PCA_BASIS = "X_pca"
INTEGRATED_BASIS = "X_pca_inte"

random.seed(SEED)
np.random.seed(SEED)
cp.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
H5AD_DIR = PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/04-integration-harmony"
TABLE_DIR = PROJECT_ROOT / "tables/01-celltype_integration_clustering/04-integration-harmony"
CANONICAL_H5AD = H5AD_DIR / "adata_harmony.h5ad"
INITIAL_COPY = H5AD_DIR / "adata_harmony_nonconverged_maxiter10.h5ad"
PRE_HARMONY_H5AD = H5AD_DIR / "adata_pre_harmony_pca.h5ad"
PENDING_CANONICAL_H5AD = H5AD_DIR / "adata_harmony_maxiter50_pending_validation.h5ad"
INITIAL_TABLE = TABLE_DIR / "08_harmony_integrate_parameters.csv"
INITIAL_TABLE_COPY = TABLE_DIR / "08_harmony_integrate_parameters_initial_nonconverged.csv"
RERUN_TABLE = TABLE_DIR / "harmony_convergence_rerun.csv"
INITIAL_CANONICAL_TABLES = [
    TABLE_DIR / "integration_harmony_summary.csv",
    TABLE_DIR / "gpu_backend_capability_summary.csv",
    TABLE_DIR / "gene_identifier_audit_harmony.csv",
    TABLE_DIR / "gene_identifier_examples_harmony.csv",
    TABLE_DIR / "package_versions.txt",
    TABLE_DIR / "harmony_validation.json",
]


def require_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite preservation/rerun artifact: {path}")


def initial_backup_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_initial_nonconverged{path.suffix}")


def main() -> None:
    if not CANONICAL_H5AD.is_file() or not INITIAL_TABLE.is_file():
        raise FileNotFoundError("Initial nonconverged Harmony H5AD/parameter table is missing")
    missing_initial_tables = [path for path in INITIAL_CANONICAL_TABLES if not path.is_file()]
    if missing_initial_tables:
        raise FileNotFoundError(
            f"Initial Harmony summary/validation artifacts are missing: {missing_initial_tables}"
        )
    for path in [
        INITIAL_COPY,
        PRE_HARMONY_H5AD,
        PENDING_CANONICAL_H5AD,
        INITIAL_TABLE_COPY,
        RERUN_TABLE,
        *[initial_backup_path(path) for path in INITIAL_CANONICAL_TABLES],
    ]:
        require_absent(path)

    # Preserve every success-like table emitted by the initial max-iter-10 run
    # before replacing the canonical copies with convergence-aware records.
    for path in INITIAL_CANONICAL_TABLES:
        shutil.copy2(path, initial_backup_path(path))

    initial_table = pd.read_csv(INITIAL_TABLE)
    initial_table["converged"] = False
    initial_table["convergence_warning"] = (
        "Harmony did not converge. Consider increasing the number of iterations"
    )
    initial_table["accepted_as_canonical"] = False
    initial_table.to_csv(INITIAL_TABLE_COPY, index=False)

    # Preserve a byte-for-byte sibling before the specifically authorized
    # same-stage canonical Harmony convergence overwrite.
    shutil.copy2(CANONICAL_H5AD, INITIAL_COPY)
    initial = sc.read_h5ad(INITIAL_COPY)
    if PCA_BASIS not in initial.obsm or INTEGRATED_BASIS not in initial.obsm:
        raise ValueError("Initial output lacks PCA or the nonconverged Harmony embedding")
    if initial.obsm[PCA_BASIS].shape != initial.obsm[INTEGRATED_BASIS].shape:
        raise ValueError("Initial PCA/Harmony shapes differ")
    del initial.obsm[INTEGRATED_BASIS]
    for key in list(initial.uns):
        if "harmony" in str(key).lower():
            del initial.uns[key]
    initial.uns["pre_harmony_pca_contract"] = {
        "derived_from": str(INITIAL_COPY),
        "removed_nonconverged_basis": INTEGRATED_BASIS,
        "PCA_basis_preserved": PCA_BASIS,
        "initial_max_iter_harmony": INITIAL_MAX_ITER,
        "random_seed": SEED,
    }
    initial.write_h5ad(PRE_HARMONY_H5AD, compression="gzip", compression_opts=1)
    del initial
    gc.collect()

    adata = sc.read_h5ad(PRE_HARMONY_H5AD)
    if INTEGRATED_BASIS in adata.obsm:
        raise ValueError("Clean pre-Harmony object still contains the adjusted basis")
    started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rsc.pp.harmony_integrate(
            adata,
            key=BATCH_KEY,
            basis=PCA_BASIS,
            adjusted_basis=INTEGRATED_BASIS,
            max_iter_harmony=INCREASED_MAX_ITER,
            random_state=SEED,
            verbose=True,
        )
    elapsed = time.time() - started
    warning_messages = [str(item.message) for item in caught]
    nonconvergence = [message for message in warning_messages if "did not converge" in message.lower()]
    rerun_row = {
        "initial_h5ad_preserved": str(INITIAL_COPY),
        "clean_pre_harmony_pca_h5ad": str(PRE_HARMONY_H5AD),
        "canonical_h5ad": str(CANONICAL_H5AD),
        "original_max_iter_harmony": INITIAL_MAX_ITER,
        "increased_max_iter_harmony": INCREASED_MAX_ITER,
        "initial_converged": False,
        "rerun_converged": not bool(nonconvergence),
        "accepted_as_canonical": not bool(nonconvergence),
        "warning_messages_rerun": " | ".join(warning_messages),
        "elapsed_seconds_rerun": elapsed,
        "random_seed": SEED,
    }
    if nonconvergence:
        rerun_row["status"] = "nonconverged_stopped_before_clustering"
        pd.DataFrame([rerun_row]).to_csv(RERUN_TABLE, index=False)
        raise RuntimeError(
            f"Harmony still did not converge at max_iter={INCREASED_MAX_ITER}: {nonconvergence}"
        )
    if INTEGRATED_BASIS not in adata.obsm:
        raise ValueError("Convergence rerun did not create the adjusted basis")
    if adata.obsm[PCA_BASIS].shape != adata.obsm[INTEGRATED_BASIS].shape:
        raise ValueError("Converged PCA/Harmony shapes differ")
    adata.uns["harmony_convergence_rerun"] = {
        "initial_max_iter_harmony": INITIAL_MAX_ITER,
        "increased_max_iter_harmony": INCREASED_MAX_ITER,
        "initial_warning": "Harmony did not converge",
        "rerun_converged_without_warning": True,
        "clean_pre_harmony_input": str(PRE_HARMONY_H5AD),
        "preserved_nonconverged_output": str(INITIAL_COPY),
        "random_seed": SEED,
    }

    # Write and structurally re-open a pending file first. Only a warning-free,
    # readable result may atomically replace the Harmony-stage canonical H5AD.
    adata.write_h5ad(PENDING_CANONICAL_H5AD, compression="gzip", compression_opts=1)
    pending = sc.read_h5ad(PENDING_CANONICAL_H5AD, backed="r")
    pending_errors = []
    if (pending.n_obs, pending.n_vars) != (adata.n_obs, adata.n_vars):
        pending_errors.append("n_obs/n_vars mismatch")
    if PCA_BASIS not in pending.obsm or INTEGRATED_BASIS not in pending.obsm:
        pending_errors.append("PCA or Harmony basis missing")
    elif pending.obsm[PCA_BASIS].shape != pending.obsm[INTEGRATED_BASIS].shape:
        pending_errors.append("PCA/Harmony shape mismatch")
    if pending.raw is None or pending.raw.n_vars != adata.raw.n_vars:
        pending_errors.append("all-gene raw snapshot missing or mismatched")
    pending.file.close()
    del pending
    if pending_errors:
        raise ValueError(f"Pending converged Harmony H5AD failed validation: {pending_errors}")
    PENDING_CANONICAL_H5AD.replace(CANONICAL_H5AD)
    row = {
        "step": "harmony_integrate_convergence_rerun",
        "input_h5ad": str(PRE_HARMONY_H5AD),
        "output_h5ad_or_object": str(CANONICAL_H5AD),
        "n_obs_before": adata.n_obs,
        "n_obs_after": adata.n_obs,
        "n_vars_before": adata.n_vars,
        "n_vars_after": adata.n_vars,
        "key": BATCH_KEY,
        "basis": PCA_BASIS,
        "adjusted_basis": INTEGRATED_BASIS,
        "flavor": "harmony2",
        "original_max_iter_harmony": INITIAL_MAX_ITER,
        "max_iter_harmony": INCREASED_MAX_ITER,
        "max_iter_clustering": 200,
        "tol_harmony": 0.0001,
        "tol_clustering": 0.00001,
        "backend_package": "rapids-singlecell_GPU",
        "code_file": str(Path(__file__).resolve()),
        "random_seed": SEED,
        "elapsed_seconds": elapsed,
        "converged": True,
        "convergence_warning": "",
        "accepted_as_canonical": True,
        "notes_fallbacks": "same-method increased-iteration convergence rerun; no backend fallback",
    }
    pd.DataFrame([row]).to_csv(INITIAL_TABLE, index=False)
    rerun_row["status"] = "converged_and_accepted_as_canonical"
    pd.DataFrame([rerun_row]).to_csv(RERUN_TABLE, index=False)

    normalize_backend = pd.read_csv(TABLE_DIR / "gpu_backend_normalize_ucsc_merge.csv")
    backend_rows = [
        {
            "step": step,
            "planned_backend": "rapids-singlecell_GPU",
            "attempted_backend": "rapids-singlecell_GPU",
            "status": status,
            "error_summary": error,
            "fallback_backend": "",
            "clean_input_reloaded": clean,
            "final_backend_for_rerun": "rapids-singlecell_GPU",
        }
        for step, status, error, clean in [
            ("highly_variable_genes", "completed", "", False),
            ("regress_out", "completed", "", False),
            ("scale", "completed", "", False),
            ("pca", "completed", "", False),
            ("harmony_integrate_maxiter10", "nonconverged_not_canonical", "Harmony did not converge", False),
            ("harmony_integrate_maxiter50", "completed_converged", "", True),
        ]
    ]
    pd.concat([normalize_backend, pd.DataFrame(backend_rows)], ignore_index=True).to_csv(
        TABLE_DIR / "gpu_backend_capability_summary.csv", index=False
    )

    def audit(index, layer):
        values = pd.Index(index).astype(str)
        ensembl_n = int(values.str.match(r"^ENS[A-Z]*G[0-9]+(?:\.[0-9]+)?$").sum())
        return {
            "input_h5ad": str(CANONICAL_H5AD),
            "object_layer": layer,
            "var_names_type": "gene_name",
            "n_features": len(values),
            "n_unique": int(values.nunique()),
            "n_duplicates": int(values.duplicated().sum()),
            "ensembl_like_n": ensembl_n,
            "ensembl_like_fraction": float(ensembl_n / len(values)) if len(values) else 0.0,
            "candidate_gene_id_columns": "",
            "candidate_gene_name_columns": "var_names;gene_name",
            "selected_gene_id_source": "",
            "selected_gene_name_source": "var_names",
            "action_taken": "retained_unique_gene_names",
        }

    pd.DataFrame(
        [audit(adata.var_names, "var_names_HVG_scaled_matrix"), audit(adata.raw.var_names, "raw.var_names_all_gene_normalized_log")]
    ).to_csv(TABLE_DIR / "gene_identifier_audit_harmony.csv", index=False)
    examples = []
    for layer, index in [
        ("var_names_HVG_scaled_matrix", adata.var_names),
        ("raw.var_names_all_gene_normalized_log", adata.raw.var_names),
    ]:
        examples.extend(
            {"object_layer": layer, "example_order": i + 1, "var_name": value}
            for i, value in enumerate(pd.Index(index).astype(str)[:20])
        )
    pd.DataFrame(examples).to_csv(TABLE_DIR / "gene_identifier_examples_harmony.csv", index=False)
    pd.DataFrame(
        [
            {
                "step": "integration_harmony_summary",
                "input_h5ad": str(PRE_HARMONY_H5AD),
                "output_h5ad_or_object": str(CANONICAL_H5AD),
                "n_obs_before": adata.n_obs,
                "n_obs_after": adata.n_obs,
                "n_vars_before": adata.raw.n_vars,
                "n_vars_after": adata.n_vars,
                "raw_n_vars": adata.raw.n_vars,
                "HVG_count": adata.n_vars,
                "pca_shape": str(tuple(adata.obsm[PCA_BASIS].shape)),
                "harmony_shape": str(tuple(adata.obsm[INTEGRATED_BASIS].shape)),
                "batch_key": BATCH_KEY,
                "backend_package": "rapids-singlecell_GPU",
                "code_file": str(Path(__file__).resolve()),
                "random_seed": SEED,
                "total_elapsed_seconds_harmony_rerun": elapsed,
                "notes_fallbacks": "maxiter10 nonconverged; maxiter50 same-method rerun converged",
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
        f"code_files={PROJECT_ROOT / 'codes/01-celltype_integration_clustering/04-integration-harmony/build_normalized_combined_atlas.py'};{PROJECT_ROOT / 'codes/01-celltype_integration_clustering/04-integration-harmony/run_harmony.py'};{Path(__file__).resolve()}",
    ]
    for package in packages:
        lines.append(f"{package}={importlib.metadata.version(package)}")
    (TABLE_DIR / "package_versions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    forbidden_clusters = [
        column for column in adata.obs.columns
        if column.startswith("leiden") or column.startswith("louvain")
    ]
    validation = {
        "output_h5ad": str(CANONICAL_H5AD),
        "preserved_nonconverged_h5ad": str(INITIAL_COPY),
        "clean_pre_harmony_pca_h5ad": str(PRE_HARMONY_H5AD),
        "n_obs": adata.n_obs,
        "n_vars_HVG": adata.n_vars,
        "raw_n_vars_all_gene_normalized_log": adata.raw.n_vars,
        "raw_role": "all_gene_normalized_log_expression_for_DEG",
        "counts_layer_present": "counts" in adata.layers,
        "non_UCSC_raw_counts_h5ad": adata.uns.get("non_UCSC_raw_counts_h5ad", ""),
        "pca_shape": list(adata.obsm[PCA_BASIS].shape),
        "harmony_shape": list(adata.obsm[INTEGRATED_BASIS].shape),
        "harmony_converged": True,
        "initial_max_iter_harmony": INITIAL_MAX_ITER,
        "final_max_iter_harmony": INCREASED_MAX_ITER,
        "neighbor_graph_present": "neighbors" in adata.uns,
        "umap_present": "X_umap" in adata.obsm,
        "cluster_columns_present": forbidden_clusters,
        "obs_names_unique": bool(adata.obs_names.is_unique),
        "random_seed": SEED,
    }
    (TABLE_DIR / "harmony_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

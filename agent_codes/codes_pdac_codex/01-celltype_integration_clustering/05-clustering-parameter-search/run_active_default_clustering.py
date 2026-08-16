#!/usr/bin/env python3
"""Run the active skill default broad Leiden clustering without a grid."""

from __future__ import annotations

import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cpsparse
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc


SEED = 42
N_PCS = 20
N_NEIGHBORS = 20
INITIAL_RESOLUTION = 0.8
ADJUSTED_RESOLUTION = 0.4
MAX_INITIAL_CLUSTERS = 20
INTEGRATED_BASIS = "X_pca_inte"

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
SOURCE_H5AD = PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_harmony.h5ad"
H5AD_DIR = PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/05-clustering-parameter-search/selected"
OUTPUT_H5AD = H5AD_DIR / "adata_inte.h5ad"
TABLE_DIR = PROJECT_ROOT / "tables/01-celltype_integration_clustering/05-clustering-parameter-search"
FIGURE_DIR = PROJECT_ROOT / "figures/01-celltype_integration_clustering/05-clustering-parameter-search/selected"


def release_gpu_arrays(adata: sc.AnnData | None = None) -> None:
    if adata is not None:
        rsc.get.anndata_to_CPU(adata, convert_all=True)
        for key in list(adata.obsm):
            value = adata.obsm[key]
            if isinstance(value, cp.ndarray):
                adata.obsm[key] = cp.asnumpy(value)
        for key in list(adata.obsp):
            value = adata.obsp[key]
            if cpsparse.issparse(value):
                adata.obsp[key] = value.get()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def is_oom(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in message or "outofmemory" in message or "cuda_error_out_of_memory" in message


def require_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {path}")


def package_version_lines() -> list[str]:
    packages = [
        "rapids-singlecell-cu13", "cudf-cu13", "cuml-cu13", "cugraph-cu13",
        "cuvs-cu13", "cupy-cuda13x", "scanpy", "anndata", "numpy", "pandas",
        "scipy", "matplotlib",
    ]
    lines = [
        "environment_category=rapids",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        "install_manager=uv",
        f"random_seed={SEED}",
        f"code_file={Path(__file__).resolve()}",
        f"source_h5ad={SOURCE_H5AD}",
        "candidate_grid=disabled_by_active_core_skill",
    ]
    lines.extend(f"{name}={importlib.metadata.version(name)}" for name in packages)
    return lines


def cluster_key(resolution: float) -> str:
    return f"leiden_res{str(resolution).replace('.', 'p')}"


def validate_clean_harmony(adata: sc.AnnData) -> None:
    if INTEGRATED_BASIS not in adata.obsm or adata.obsm[INTEGRATED_BASIS].shape[1] < N_PCS:
        raise ValueError(f"Harmony input lacks {INTEGRATED_BASIS} with {N_PCS} dimensions")
    if adata.raw is None:
        raise ValueError("Harmony input lacks adata.raw")
    graph_fields = [key for key in adata.obsp if key in {"connectivities", "distances"}]
    cluster_fields = [key for key in adata.obs if key.startswith(("leiden", "louvain"))]
    if graph_fields or "X_umap" in adata.obsm or cluster_fields:
        raise ValueError(
            f"Harmony input is not clean: graph={graph_fields}, "
            f"umap={'X_umap' in adata.obsm}, clusters={cluster_fields}"
        )


def run_gpu(adata: sc.AnnData, resolution: float, raw_cluster_column: str) -> None:
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        n_pcs=N_PCS,
        use_rep=INTEGRATED_BASIS,
        random_state=SEED,
        algorithm="brute",
    )
    rsc.tl.umap(adata, random_state=SEED)
    rsc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=raw_cluster_column,
        random_state=SEED,
    )
    release_gpu_arrays(adata)


def run_cpu(resolution: float, raw_cluster_column: str) -> sc.AnnData:
    adata = sc.read_h5ad(SOURCE_H5AD)
    validate_clean_harmony(adata)
    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        n_pcs=N_PCS,
        use_rep=INTEGRATED_BASIS,
        random_state=SEED,
        method="umap",
    )
    sc.tl.umap(adata, random_state=SEED)
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=raw_cluster_column,
        random_state=SEED,
    )
    return adata


def execute_clean_run(resolution: float) -> tuple[sc.AnnData, dict[str, object]]:
    raw_cluster_column = cluster_key(resolution)
    adata = sc.read_h5ad(SOURCE_H5AD)
    validate_clean_harmony(adata)
    started = time.time()
    backend = "rapids-singlecell_GPU"
    error_summary = ""
    fallback_backend = ""
    clean_input_reloaded = False
    try:
        run_gpu(adata, resolution, raw_cluster_column)
    except Exception as exc:
        release_gpu_arrays(adata)
        error_summary = f"GPU:{type(exc).__name__}:{exc}"
        if not is_oom(exc):
            raise
        backend = "scanpy_CPU_after_GPU_OOM"
        fallback_backend = "scanpy_CPU"
        clean_input_reloaded = True
        adata = run_cpu(resolution, raw_cluster_column)
    return adata, {
        "resolution": resolution,
        "raw_cluster_column": raw_cluster_column,
        "backend": backend,
        "error_summary": error_summary,
        "fallback_backend": fallback_backend,
        "clean_input_reloaded": clean_input_reloaded,
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    outputs = [
        OUTPUT_H5AD,
        H5AD_DIR,
        FIGURE_DIR,
        TABLE_DIR / "selected_clustering.csv",
        TABLE_DIR / "selected_clustering_parameters.csv",
        TABLE_DIR / "selected_cluster_counts.csv",
        TABLE_DIR / "selected_clustering_validation.json",
        TABLE_DIR / "gpu_backend_capability_summary.csv",
        TABLE_DIR / "package_versions.txt",
        TABLE_DIR / "readme.txt",
    ]
    if not SOURCE_H5AD.is_file():
        raise FileNotFoundError(SOURCE_H5AD)
    for path in outputs:
        require_absent(path)

    random.seed(SEED)
    np.random.seed(SEED)
    cp.random.seed(SEED)
    started = time.time()
    adata, initial_run = execute_clean_run(INITIAL_RESOLUTION)
    initial_key = str(initial_run["raw_cluster_column"])
    initial_counts = adata.obs[initial_key].astype(str).value_counts().sort_index()
    initial_n_clusters = int(initial_counts.size)
    if initial_counts.empty:
        raise ValueError("Initial broad Leiden result is empty")

    backend_rows = [
        {
            "step": f"neighbors_umap_leiden_res{str(INITIAL_RESOLUTION).replace('.', 'p')}",
            "planned_backend": "rapids-singlecell_GPU",
            "attempted_backend": (
                "rapids-singlecell_GPU_then_scanpy_CPU"
                if initial_run["fallback_backend"] else "rapids-singlecell_GPU"
            ),
            "status": "completed_CPU_after_GPU_OOM" if initial_run["fallback_backend"] else "completed",
            "error_summary": initial_run["error_summary"],
            "fallback_backend": initial_run["fallback_backend"],
            "clean_input_reloaded": initial_run["clean_input_reloaded"],
            "final_backend_for_rerun": initial_run["backend"],
        }
    ]
    final_resolution = INITIAL_RESOLUTION
    raw_cluster_column = initial_key
    final_run = initial_run
    adjustment_reason = "none_initial_cluster_count_at_or_below_20"
    if initial_n_clusters > MAX_INITIAL_CLUSTERS:
        del adata
        release_gpu_arrays()
        adata, final_run = execute_clean_run(ADJUSTED_RESOLUTION)
        final_resolution = ADJUSTED_RESOLUTION
        raw_cluster_column = str(final_run["raw_cluster_column"])
        adjustment_reason = (
            f"initial_resolution_{INITIAL_RESOLUTION}_produced_{initial_n_clusters}_clusters_gt_"
            f"{MAX_INITIAL_CLUSTERS}; reloaded_clean_harmony_and_reran_once_at_{ADJUSTED_RESOLUTION}"
        )
        backend_rows.append(
            {
                "step": f"neighbors_umap_leiden_res{str(ADJUSTED_RESOLUTION).replace('.', 'p')}",
                "planned_backend": "rapids-singlecell_GPU",
                "attempted_backend": (
                    "rapids-singlecell_GPU_then_scanpy_CPU"
                    if final_run["fallback_backend"] else "rapids-singlecell_GPU"
                ),
                "status": "completed_CPU_after_GPU_OOM" if final_run["fallback_backend"] else "completed",
                "error_summary": final_run["error_summary"],
                "fallback_backend": final_run["fallback_backend"],
                "clean_input_reloaded": True,
                "final_backend_for_rerun": final_run["backend"],
            }
        )

    if raw_cluster_column not in adata.obs:
        raise ValueError(f"Missing raw technical cluster column: {raw_cluster_column}")
    cluster_counts = adata.obs[raw_cluster_column].astype(str).value_counts().sort_index()
    if cluster_counts.empty or cluster_counts.size > 100:
        raise ValueError(f"Abnormal accepted broad Leiden cluster count: {cluster_counts.size}")
    graph_elapsed = float(initial_run["elapsed_seconds"])
    if final_resolution != INITIAL_RESOLUTION:
        graph_elapsed += float(final_run["elapsed_seconds"])

    H5AD_DIR.parent.mkdir(parents=True, exist_ok=True)
    H5AD_DIR.mkdir(parents=True, exist_ok=False)
    FIGURE_DIR.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=False)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sc.set_figure_params(vector_friendly=True, dpi=150, figsize=(4, 4))
    sc.settings.figdir = FIGURE_DIR
    sc.pl.umap(adata, color=raw_cluster_column, show=False, save="_raw_clusters.pdf")
    sc.pl.umap(adata, color="series", show=False, save="_series_diagnostic.pdf")
    sc.pl.umap(adata, color="status", show=False, save="_status_diagnostic.pdf")
    expected_figures = [
        FIGURE_DIR / "umap_raw_clusters.pdf",
        FIGURE_DIR / "umap_series_diagnostic.pdf",
        FIGURE_DIR / "umap_status_diagnostic.pdf",
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in expected_figures):
        raise FileNotFoundError(f"Missing clustering figures: {expected_figures}")

    adata.write_h5ad(OUTPUT_H5AD, compression="gzip", compression_opts=1)
    check = sc.read_h5ad(OUTPUT_H5AD, backed="r")
    validation_errors: list[str] = []
    if check.shape != adata.shape:
        validation_errors.append("shape mismatch")
    if raw_cluster_column not in check.obs:
        validation_errors.append("raw cluster column missing")
    if "X_umap" not in check.obsm or "neighbors" not in check.uns:
        validation_errors.append("graph/UMAP missing")
    if check.raw is None or check.raw.n_vars != adata.raw.n_vars:
        validation_errors.append("raw expression snapshot missing or mismatched")
    check.file.close()
    if validation_errors:
        raise ValueError(f"Selected clustering validation failed: {validation_errors}")

    counts = cluster_counts.rename("n_cells").reset_index()
    counts.columns = ["cluster", "n_cells"]
    counts.to_csv(TABLE_DIR / "selected_cluster_counts.csv", index=False)
    parameters = {
        "candidate_label": (
            f"active-default_pcs-{N_PCS}_nn-{N_NEIGHBORS}_res-"
            f"{str(final_resolution).replace('.', 'p')}"
        ),
        "selected_h5ad": str(OUTPUT_H5AD),
        "selected_h5ad_relative": "h5ad/01-celltype_integration_clustering/05-clustering-parameter-search/selected/adata_inte.h5ad",
        "source_h5ad": str(SOURCE_H5AD),
        "final_raw_cluster_column": raw_cluster_column,
        "n_pcs": N_PCS,
        "n_neighbors": N_NEIGHBORS,
        "resolution": final_resolution,
        "initial_resolution": INITIAL_RESOLUTION,
        "initial_n_clusters": initial_n_clusters,
        "final_resolution": final_resolution,
        "final_n_clusters": int(cluster_counts.size),
        "resolution_adjustment_reason": adjustment_reason,
        "clustering_algorithm": "leiden",
        "integration_source": "Harmony",
        "selection_source": "active_core_skill_default",
        "manual_selection_note": "not_applicable_active_default_no_candidate_grid",
        "neighbor_use_rep": INTEGRATED_BASIS,
        "neighbor_algorithm": "brute",
        "umap_random_state": SEED,
        "leiden_random_state": SEED,
        "random_seed": SEED,
        "backend_package": final_run["backend"],
        "candidate_grid_run": False,
        "n_clusters": int(cluster_counts.size),
        "elapsed_seconds": time.time() - started,
        "graph_elapsed_seconds": graph_elapsed,
        "notes_fallbacks": final_run["fallback_backend"],
    }
    pd.DataFrame([parameters]).to_csv(TABLE_DIR / "selected_clustering_parameters.csv", index=False)
    pd.DataFrame([parameters]).to_csv(TABLE_DIR / "selected_clustering.csv", index=False)
    pd.DataFrame(backend_rows).to_csv(TABLE_DIR / "gpu_backend_capability_summary.csv", index=False)
    (TABLE_DIR / "package_versions.txt").write_text("\n".join(package_version_lines()) + "\n", encoding="utf-8")
    (TABLE_DIR / "readme.txt").write_text(
        "Active core-skill default broad clustering run; parameter grid disabled.\n\n"
        f"Input: {SOURCE_H5AD}\n"
        f"Initial parameters: n_pcs={N_PCS}, n_neighbors={N_NEIGHBORS}, resolution={INITIAL_RESOLUTION}, algorithm=Leiden.\n"
        f"Accepted resolution: {final_resolution}; initial clusters={initial_n_clusters}; final clusters={cluster_counts.size}.\n"
        f"Adjustment reason: {adjustment_reason}.\n"
        "Neighbors, UMAP, and Leiden were run from the clean Harmony object; QC and Harmony were not rerun.\n"
        f"Raw technical cluster column: {raw_cluster_column}; n_clusters={cluster_counts.size}.\n"
        "Biological broad labels are not created in this clustering task.\n",
        encoding="utf-8",
    )
    validation = {
        **parameters,
        "output_h5ad_readable": OUTPUT_H5AD.is_file(),
        "obs_names_unique": bool(adata.obs_names.is_unique),
        "raw_cluster_column_present": raw_cluster_column in adata.obs,
        "leiden_coarse_present": "leiden_coarse" in adata.obs,
        "neighbor_graph_present": "neighbors" in adata.uns,
        "umap_present": "X_umap" in adata.obsm,
        "raw_expression_present": adata.raw is not None,
        "validation_errors": validation_errors,
    }
    (TABLE_DIR / "selected_clustering_validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

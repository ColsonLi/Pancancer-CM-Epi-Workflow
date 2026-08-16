#!/usr/bin/env python3
"""Rerun one agent-selected epithelial subtype candidate from clean Harmony."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cpsparse
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
from anndata import AnnData


SEED = 42
INTEGRATED_BASIS = "X_pca_inte"
PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
SOURCE_H5AD = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/02-subtype-harmony/epithelial/adata_epithelial_harmony.h5ad"
)
GRID_TABLE_ROOT = (
    PROJECT_ROOT
    / "tables/02-cell_subtype_integration_clustering/03-subtype-clustering-grid/epithelial"
)
MANIFEST_PATH = GRID_TABLE_ROOT / "epithelial_clustering_grid_manifest.csv"
TABLE_ROOT = (
    PROJECT_ROOT
    / "tables/02-cell_subtype_integration_clustering/04-subtype-selected-clustering/epithelial"
)
FIGURE_DIR = (
    PROJECT_ROOT
    / "figures/02-cell_subtype_integration_clustering/04-subtype-selected-clustering/epithelial"
)
PENDING_FIGURE_DIR = (
    PROJECT_ROOT
    / "figures/02-cell_subtype_integration_clustering/04-subtype-selected-clustering/epithelial_pending_validation"
)
H5AD_DIR = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/04-subtype-selected-clustering/epithelial"
)
PENDING_H5AD_DIR = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/04-subtype-selected-clustering/epithelial_pending_validation"
)
OUTPUT_H5AD = H5AD_DIR / "adata_epi_selected_clustered.h5ad"
PENDING_H5AD = PENDING_H5AD_DIR / OUTPUT_H5AD.name
LABEL_RE = re.compile(r"^pcs-(?P<pcs>[0-9]+)_nn-(?P<nn>[0-9]+)_res-(?P<res>[0-9]+p[0-9]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--selection-note", required=True)
    parser.add_argument("--confirm-agent-selected", action="store_true")
    return parser.parse_args()


def parse_label(label: str) -> tuple[int, int, float, str]:
    match = LABEL_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"Invalid epithelial candidate label: {label}")
    n_pcs = int(match.group("pcs"))
    n_neighbors = int(match.group("nn"))
    resolution = float(match.group("res").replace("p", "."))
    token = match.group("res")
    return n_pcs, n_neighbors, resolution, token


def release_gpu_arrays(adata: sc.AnnData) -> None:
    for key in list(adata.obsm):
        if isinstance(adata.obsm[key], cp.ndarray):
            adata.obsm[key] = cp.asnumpy(adata.obsm[key])
    for key in list(adata.obsp):
        if cpsparse.issparse(adata.obsp[key]):
            adata.obsp[key] = adata.obsp[key].get()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


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
        "matplotlib",
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
    args = parse_args()
    if not args.confirm_agent_selected:
        raise PermissionError("Selected epithelial rerun requires explicit agent-selection confirmation")
    if not args.selection_note.strip():
        raise ValueError("Selection note cannot be empty")
    if not SOURCE_H5AD.is_file() or not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Epithelial Harmony source or completed grid manifest is missing")
    for path in [TABLE_ROOT, FIGURE_DIR, PENDING_FIGURE_DIR, H5AD_DIR, PENDING_H5AD_DIR]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite selected epithelial artifact: {path}")

    n_pcs, n_neighbors, resolution, token = parse_label(args.candidate_label)
    manifest = pd.read_csv(MANIFEST_PATH)
    row = manifest.loc[manifest["candidate_label"].eq(args.candidate_label)]
    if row.shape[0] != 1:
        raise ValueError("Selected epithelial candidate is absent or non-unique")
    row = row.iloc[0]
    if row["status"] != "completed" or not bool(row["completed"]):
        raise ValueError("Selected epithelial candidate is not completed")
    algorithm = str(row["algorithm"])
    raw_key = str(row["raw_cluster_column"])
    expected_key = f"{algorithm}_res{token}"
    if algorithm not in {"leiden", "louvain"} or raw_key != expected_key:
        raise ValueError("Selected manifest algorithm/raw-column contract is invalid")
    graph_table = GRID_TABLE_ROOT / str(row["graph_label"]) / "cluster_counts_by_resolution.csv"
    expected_counts = pd.read_csv(graph_table)
    expected_counts = expected_counts.loc[
        np.isclose(expected_counts["resolution"], resolution)
        & expected_counts["raw_cluster_column"].eq(raw_key),
        ["cluster", "n_cells"],
    ].copy()
    expected_counts["cluster"] = expected_counts["cluster"].astype(str)
    expected_counts = expected_counts.sort_values("cluster").reset_index(drop=True)
    if expected_counts.empty:
        raise ValueError("Selected candidate cluster counts are absent from aggregate grid table")

    random.seed(SEED)
    np.random.seed(SEED)
    cp.random.seed(SEED)
    started = time.time()
    adata = sc.read_h5ad(SOURCE_H5AD)
    if adata.raw is None or INTEGRATED_BASIS not in adata.obsm:
        raise ValueError("Clean epithelial Harmony source lacks raw or integrated basis")
    if "neighbors" in adata.uns or "X_umap" in adata.obsm:
        raise ValueError("Clean epithelial Harmony source already contains graph/UMAP")
    forbidden = [
        column for column in adata.obs if column.startswith(("leiden_res", "louvain_res"))
    ]
    if forbidden or "cell_subtype" in adata.obs:
        raise ValueError(f"Clean epithelial Harmony source already contains cluster/subtype columns: {forbidden}")

    # Match the completed grid's graph construction exactly.  The grid used a
    # minimal AnnData containing only the required obs columns and a CuPy copy
    # of X_pca_inte; building directly on the full expression AnnData can yield
    # a different GPU Leiden partition despite the same explicit seed.
    graph_obs = adata.obs[["sample", "series", "status"]].copy()
    integrated_cpu = np.asarray(adata.obsm[INTEGRATED_BASIS])
    graph_adata = AnnData(obs=graph_obs)
    graph_adata.obsm[INTEGRATED_BASIS] = cp.asarray(integrated_cpu)
    del graph_obs, integrated_cpu

    free_before, total_vram = cp.cuda.runtime.memGetInfo()
    graph_started = time.time()
    rsc.pp.neighbors(
        graph_adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=INTEGRATED_BASIS,
        random_state=SEED,
        algorithm="brute",
    )
    rsc.tl.umap(graph_adata, random_state=SEED)
    cluster_function = rsc.tl.leiden if algorithm == "leiden" else rsc.tl.louvain
    cluster_function(
        graph_adata,
        resolution=resolution,
        key_added=raw_key,
        random_state=SEED,
    )
    graph_elapsed = time.time() - graph_started
    release_gpu_arrays(graph_adata)
    adata.obs[raw_key] = graph_adata.obs[raw_key].copy()
    adata.obsm["X_umap"] = np.asarray(graph_adata.obsm["X_umap"])
    adata.obsp["distances"] = graph_adata.obsp["distances"].copy()
    adata.obsp["connectivities"] = graph_adata.obsp["connectivities"].copy()
    adata.uns["neighbors"] = graph_adata.uns["neighbors"].copy()
    del graph_adata
    actual_counts = (
        adata.obs[raw_key]
        .astype(str)
        .value_counts()
        .sort_index()
        .rename("n_cells")
        .reset_index()
    )
    actual_counts.columns = ["cluster", "n_cells"]
    exact_counts_match = actual_counts.equals(expected_counts)
    if not exact_counts_match:
        raise ValueError(
            "Clean selected rerun did not exactly reproduce the completed grid candidate"
        )
    if "cell_subtype" in adata.obs:
        raise ValueError("Selected clustering must stop before subtype annotation")
    selected_at = datetime.now().astimezone().isoformat()
    adata.uns["selected_epithelial_clustering_contract"] = {
        "candidate_label": args.candidate_label,
        "selection_source": "user_authorized_agent_led_selection",
        "selection_note": args.selection_note.strip(),
        "selected_at": selected_at,
        "clean_source_h5ad": str(SOURCE_H5AD),
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "algorithm": algorithm,
        "raw_cluster_column": raw_key,
        "random_seed": SEED,
        "graph_construction": "grid_matched_minimal_anndata_cupy_X_pca_inte",
        "candidate_grid_n_clusters": int(expected_counts.shape[0]),
        "selected_clean_rerun_n_clusters": int(actual_counts.shape[0]),
        "cluster_counts_match_candidate_exactly": bool(exact_counts_match),
        "clean_selected_rerun_is_authoritative": True,
    }

    PENDING_H5AD_DIR.mkdir(parents=True, exist_ok=False)
    PENDING_FIGURE_DIR.mkdir(parents=True, exist_ok=False)
    adata.write_h5ad(PENDING_H5AD, compression="gzip", compression_opts=1)
    pending = sc.read_h5ad(PENDING_H5AD, backed="r")
    errors = []
    if pending.shape != adata.shape:
        errors.append("shape mismatch")
    if pending.raw is None or pending.raw.n_vars != adata.raw.n_vars:
        errors.append("raw mismatch")
    if raw_key not in pending.obs or "X_umap" not in pending.obsm or "neighbors" not in pending.uns:
        errors.append("selected clustering outputs missing")
    if "cell_subtype" in pending.obs:
        errors.append("cell_subtype created prematurely")
    pending.file.close()
    del pending
    if errors:
        raise ValueError(f"Pending selected epithelial H5AD failed validation: {errors}")
    sc.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
    sc.settings.figdir = PENDING_FIGURE_DIR
    sc.pl.umap(adata, color=raw_key, show=False, save="_selected_raw_clusters.pdf")
    figure_path = PENDING_FIGURE_DIR / "umap_selected_raw_clusters.pdf"
    if not figure_path.is_file() or figure_path.stat().st_size == 0:
        raise FileNotFoundError("Selected epithelial raw-cluster UMAP missing")
    PENDING_H5AD_DIR.replace(H5AD_DIR)
    PENDING_FIGURE_DIR.replace(FIGURE_DIR)

    TABLE_ROOT.mkdir(parents=True, exist_ok=False)
    selected_record = {
        "candidate_label": args.candidate_label,
        "selected_h5ad": str(OUTPUT_H5AD),
        "final_raw_cluster_column": raw_key,
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "clustering_algorithm": algorithm,
        "integration_source": "Harmony",
        "selection_note": args.selection_note.strip(),
        "selection_source": "user_authorized_agent_led_selection",
        "selected_at": selected_at,
    }
    pd.DataFrame([selected_record]).to_csv(TABLE_ROOT / "selected_clustering.csv", index=False)
    actual_counts.to_csv(TABLE_ROOT / "selected_cluster_counts.csv", index=False)
    expected_counts.merge(
        actual_counts,
        on="cluster",
        how="outer",
        suffixes=("_candidate_grid", "_selected_clean_rerun"),
    ).to_csv(TABLE_ROOT / "selected_vs_candidate_cluster_count_comparison.csv", index=False)
    parameters = {
        **selected_record,
        "source_h5ad": str(SOURCE_H5AD),
        "neighbor_use_rep": INTEGRATED_BASIS,
        "neighbor_algorithm": "brute",
        "graph_construction": "grid_matched_minimal_anndata_cupy_X_pca_inte",
        "random_seed": SEED,
        "backend": "rapids-singlecell_GPU",
        "gpu_free_bytes_before_graph": int(free_before),
        "gpu_total_bytes": int(total_vram),
        "graph_elapsed_seconds": graph_elapsed,
        "total_elapsed_seconds": time.time() - started,
        "candidate_grid_n_clusters": int(expected_counts.shape[0]),
        "selected_clean_rerun_n_clusters": int(actual_counts.shape[0]),
        "cluster_counts_match_candidate_exactly": bool(exact_counts_match),
        "candidate_grid_outputs_preserved": True,
        "cell_subtype_created": False,
        "code_file": str(Path(__file__).resolve()),
    }
    pd.DataFrame([parameters]).to_csv(TABLE_ROOT / "selected_clustering_parameters.csv", index=False)
    pd.DataFrame(
        [
            {
                "step": "selected_epithelial_clustering_rerun",
                "planned_backend": "rapids-singlecell_GPU",
                "attempted_backend": "rapids-singlecell_GPU",
                "status": "completed",
                "error_summary": "",
                "fallback_backend": "",
                "clean_input_reloaded": True,
                "final_backend_for_rerun": "rapids-singlecell_GPU",
            }
        ]
    ).to_csv(TABLE_ROOT / "gpu_backend_capability_summary.csv", index=False)
    (TABLE_ROOT / "package_versions.txt").write_text(
        "\n".join(package_version_lines()) + "\n", encoding="utf-8"
    )
    validation = {
        **parameters,
        "n_obs": int(adata.n_obs),
        "n_vars_hvg": int(adata.n_vars),
        "raw_n_vars": int(adata.raw.n_vars),
        "neighbor_graph_present": "neighbors" in adata.uns,
        "umap_present": "X_umap" in adata.obsm,
        "raw_cluster_column_present": raw_key in adata.obs,
        "cell_subtype_present": "cell_subtype" in adata.obs,
        "output_h5ad_readable": OUTPUT_H5AD.is_file(),
        "selected_figure": str(FIGURE_DIR / figure_path.name),
    }
    (TABLE_ROOT / "selected_clustering_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

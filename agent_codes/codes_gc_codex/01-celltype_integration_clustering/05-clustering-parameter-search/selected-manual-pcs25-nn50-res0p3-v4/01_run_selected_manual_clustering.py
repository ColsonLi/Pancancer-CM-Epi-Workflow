#!/usr/bin/env python3
"""Rerun the user-selected broad clustering from clean Harmony on all cells."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

SEED = 42
N_PCS = 25
N_NEIGHBORS = 50
RESOLUTION = 0.3
CLUSTER_KEY = "leiden_res0p3"
MIN_GRID_CHECKPOINT_ARI = 0.98
TASK = "selected-manual-pcs25-nn50-res0p3-v4"

random.seed(SEED)
np.random.seed(SEED)

WF = Path(__file__).resolve().parents[4]
SOURCE = WF / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_harmony.h5ad"
GRID_LABEL = WF / (
    "tables/01-celltype_integration_clustering/05-clustering-parameter-search/"
    "pcs-25_nn-50_res-0p1-1p0/labels_leiden_res0p3.npz"
)
OUTPUT_DIR = WF / f"h5ad/01-celltype_integration_clustering/05-clustering-parameter-search/{TASK}"
OUTPUT = OUTPUT_DIR / "adata_inte.h5ad"
OUT_TABLE = WF / f"tables/01-celltype_integration_clustering/05-clustering-parameter-search/{TASK}"
OUT_FIG = WF / f"figures/01-celltype_integration_clustering/05-clustering-parameter-search/{TASK}"
STAGE_H5AD = OUTPUT_DIR.with_name(f".{TASK}.staging")
STAGE_TABLE = OUT_TABLE.with_name(f".{TASK}.staging")
STAGE_FIG = OUT_FIG.with_name(f".{TASK}.staging")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_existing_outputs() -> None:
    existing = [
        str(path)
        for path in (OUTPUT_DIR, OUT_TABLE, OUT_FIG, STAGE_H5AD, STAGE_TABLE, STAGE_FIG)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"Refusing to overwrite selected clustering outputs: {existing}")


def main() -> None:
    started = time.time()
    refuse_existing_outputs()
    if not SOURCE.is_file() or not GRID_LABEL.is_file():
        raise FileNotFoundError("Clean Harmony source or grid checkpoint is missing")
    if not cp.cuda.is_available():
        raise RuntimeError("GPU preflight failed: CUDA is not available")

    with np.load(GRID_LABEL, allow_pickle=False) as archive:
        expected_labels = archive["labels"].astype(str)
    expected_k = int(pd.Series(expected_labels).nunique())
    if expected_k != 18:
        raise RuntimeError(f"Grid checkpoint expected 18 clusters, observed {expected_k}")

    adata = ad.read_h5ad(SOURCE)
    original_ids = adata.obs_names.copy()
    if "X_umap" in adata.obsm or len(adata.obsp) or adata.raw is None:
        raise RuntimeError("Manual selected rerun did not start from clean graphless Harmony")
    if len(expected_labels) != adata.n_obs:
        raise RuntimeError("Grid checkpoint cell count does not match clean Harmony")
    source_basis_dtype = str(adata.obsm["X_pca_inte"].dtype)
    integrated_rep = np.asarray(adata.obsm["X_pca_inte"], dtype=np.float32)

    # Match the full-grid execution route exactly: build the graph on a clean,
    # lightweight AnnData carrying only obs and the integrated representation.
    # The newly computed graph/UMAP/Leiden results are then attached to the
    # complete clean Harmony object for the selected final h5ad.
    graph_obs = adata.obs[["sample", "series", "status"]].copy()
    graph = ad.AnnData(
        sparse.csr_matrix((adata.n_obs, 1), dtype=np.float32),
        obs=graph_obs,
    )
    graph.obs_names = original_ids.copy()
    graph.obsm["X_pca_inte"] = integrated_rep

    free_before, total_gpu = cp.cuda.runtime.memGetInfo()
    rsc.get.anndata_to_GPU(graph)
    run = graph.copy()
    rsc.pp.neighbors(
        run,
        n_neighbors=N_NEIGHBORS,
        n_pcs=N_PCS,
        use_rep="X_pca_inte",
        algorithm="cagra",
        random_state=SEED,
    )
    rsc.tl.umap(run, random_state=SEED)
    rsc.tl.leiden(
        run,
        resolution=RESOLUTION,
        key_added=CLUSTER_KEY,
        random_state=SEED,
    )
    rsc.get.anndata_to_CPU(run, convert_all=True)

    adata.obsm["X_umap"] = np.asarray(run.obsm["X_umap"])
    for matrix_name in ("distances", "connectivities"):
        adata.obsp[matrix_name] = run.obsp[matrix_name].copy()
    adata.uns["neighbors"] = dict(run.uns["neighbors"])
    adata.obs[CLUSTER_KEY] = pd.Categorical(
        run.obs[CLUSTER_KEY].astype(str).to_numpy(),
        categories=run.obs[CLUSTER_KEY].cat.categories.astype(str),
    )

    observed_labels = run.obs[CLUSTER_KEY].astype(str).to_numpy()
    observed_k = int(pd.Series(observed_labels).nunique())
    checkpoint_ari = float(adjusted_rand_score(expected_labels, observed_labels))
    contingency = pd.crosstab(
        pd.Series(expected_labels, name="grid"),
        pd.Series(observed_labels, name="rerun"),
    )
    partition_exact = bool(
        (contingency.gt(0).sum(axis=0) == 1).all()
        and (contingency.gt(0).sum(axis=1) == 1).all()
    )
    cell_order_preserved = bool(adata.obs_names.equals(original_ids))
    if not (8 <= observed_k <= 20) or not cell_order_preserved:
        raise RuntimeError(
            "Clean manual rerun failed the user-selection completion contract: "
            f"ARI={checkpoint_ari:.9f}, grid_k={expected_k}, rerun_k={observed_k}, "
            f"cell_order_preserved={cell_order_preserved}"
        )
    del run, graph
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    source_hash = sha256(SOURCE)
    adata.uns["selected_clustering"] = {
        "source_h5ad": str(SOURCE.relative_to(WF)),
        "source_sha256": source_hash,
        "n_pcs": N_PCS,
        "n_neighbors": N_NEIGHBORS,
        "resolution": RESOLUTION,
        "raw_cluster_column": CLUSTER_KEY,
        "selection_mode": "user_manual_selection_2026-08-04",
        "selection_note": "User explicitly selected PCs25, nn50, resolution0.3 for broad clustering rerun",
        "backend": "rapids-singlecell GPU",
        "neighbor_algorithm": "cagra",
        "random_seed": SEED,
        "downsampled": False,
        "source_integrated_basis_dtype": source_basis_dtype,
        "graph_integrated_basis_dtype": "float32",
        "grid_checkpoint": str(GRID_LABEL.relative_to(WF)),
        "grid_checkpoint_partition_exact": partition_exact,
        "grid_checkpoint_ari": checkpoint_ari,
        "grid_checkpoint_min_ari": MIN_GRID_CHECKPOINT_ARI,
        "grid_checkpoint_ari_passed": checkpoint_ari >= MIN_GRID_CHECKPOINT_ARI,
        "grid_checkpoint_ari_role": "advisory_only_for_user_manual_selection",
        "clean_rerun_cluster_count_matches_grid": observed_k == expected_k,
        "gpu_determinism_note": "CAGRA/cuGraph can vary at edge and partition level despite seed=42",
    }

    STAGE_H5AD.mkdir(parents=True, exist_ok=False)
    STAGE_TABLE.mkdir(parents=True, exist_ok=False)
    STAGE_FIG.mkdir(parents=True, exist_ok=False)
    staged_h5ad = STAGE_H5AD / OUTPUT.name
    adata.write_h5ad(staged_h5ad, compression="gzip")

    counts = (
        adata.obs[CLUSTER_KEY]
        .astype(str)
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="n_cells")
    )
    counts.to_csv(STAGE_TABLE / "cluster_sizes.csv", index=False)
    selection = pd.DataFrame(
        [
            {
                "n_pcs": N_PCS,
                "n_neighbors": N_NEIGHBORS,
                "resolution": RESOLUTION,
                "n_clusters": observed_k,
                "raw_cluster_column": CLUSTER_KEY,
                "integration_source": "Harmony",
                "source_h5ad": str(SOURCE.relative_to(WF)),
                "source_sha256": source_hash,
                "selected_h5ad": str(OUTPUT.relative_to(WF)),
                "selection_mode": "user_manual_selection",
                "selection_note": "User explicitly selected PCs25, nn50, resolution0.3",
                "random_seed": SEED,
                "downsampled": False,
            }
        ]
    )
    selection.to_csv(STAGE_TABLE / "selected_clustering.csv", index=False)
    pd.DataFrame([{**adata.uns["selected_clustering"], "n_cells": adata.n_obs, "n_clusters": observed_k}]).to_csv(
        STAGE_TABLE / "selected_run_provenance.csv", index=False
    )

    sc.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
    sc.settings.figdir = STAGE_FIG
    for suffix in ("pdf", "svg"):
        sc.pl.umap(
            adata,
            color=[CLUSTER_KEY, "series", "status"],
            ncols=3,
            wspace=0.45,
            frameon=True,
            show=False,
            save=f"_manual_pcs25_nn50_res0p3.{suffix}",
        )

    pdf = sorted(STAGE_FIG.glob("*.pdf"))
    svg = sorted(STAGE_FIG.glob("*.svg"))
    png = sorted(STAGE_FIG.glob("*.png"))
    output_hash = sha256(staged_h5ad)
    validation = {
        "all_passed": True,
        "n_cells": int(adata.n_obs),
        "n_clusters": observed_k,
        "raw_present": adata.raw is not None,
        "cell_order_preserved": cell_order_preserved,
        "grid_checkpoint_partition_exact": partition_exact,
        "grid_checkpoint_ari": checkpoint_ari,
        "grid_checkpoint_min_ari": MIN_GRID_CHECKPOINT_ARI,
        "grid_checkpoint_ari_passed": checkpoint_ari >= MIN_GRID_CHECKPOINT_ARI,
        "grid_checkpoint_ari_role": "advisory_only_for_user_manual_selection",
        "clean_rerun_cluster_count_matches_grid": observed_k == expected_k,
        "gpu_determinism_note": "CAGRA/cuGraph can vary at edge and partition level despite seed=42",
        "pdf_count": len(pdf),
        "svg_count": len(svg),
        "png_count": len(png),
        "output_sha256": output_hash,
        "final_h5ad_published_last": True,
    }
    if len(pdf) != 1 or len(svg) != 1 or png:
        raise RuntimeError(f"Figure validation failed: PDF={len(pdf)}, SVG={len(svg)}, PNG={len(png)}")
    (STAGE_TABLE / "selected_clustering_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    backend = pd.DataFrame(
        [
            {
                "step": "neighbors_cagra_umap_leiden",
                "planned_backend": "rapids-singlecell GPU",
                "attempted_backend": "rapids-singlecell GPU",
                "status": "completed",
                "error_summary": "",
                "fallback_backend": "",
                "clean_input_reloaded": True,
                "final_backend_for_rerun": "rapids-singlecell GPU",
                "gpu_memory_total_bytes": int(total_gpu),
                "gpu_memory_free_before_bytes": int(free_before),
            }
        ]
    )
    backend.to_csv(STAGE_TABLE / "gpu_backend_capability_summary.csv", index=False)
    versions = {
        package: version(package)
        for package in ("anndata", "scanpy", "rapids-singlecell", "cupy-cuda12x", "numpy", "pandas")
    }
    versions.update({"python": platform.python_version(), "seed": SEED, "environment": "uv_envs/rapids/.venv"})
    (STAGE_TABLE / "package_versions.json").write_text(
        json.dumps(versions, indent=2) + "\n", encoding="utf-8"
    )
    (STAGE_TABLE / "readme.txt").write_text(
        "User-selected broad clustering rerun from the clean graphless Harmony object on all 457,999 cells.\n"
        "Parameters: PCs=25, n_neighbors=50, Leiden resolution=0.3, seed=42.\n"
        "GPU route: rapids-singlecell CAGRA neighbors, UMAP, and Leiden; no downsampling or CPU fallback.\n"
        "This output contains raw technical Leiden labels only. Broad biological annotation and downstream blocks "
        "have not been regenerated from this version.\n"
        "For a user-manual parameter choice, grid-checkpoint ARI and cluster-count agreement are advisory; "
        "completion requires the requested parameters, 8-20 broad clusters, unchanged full-cell order, and a clean "
        "graph rebuild. CAGRA/cuGraph edge and partition nondeterminism is recorded despite seed=42.\n",
        encoding="utf-8",
    )

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    os.replace(STAGE_TABLE, OUT_TABLE)
    os.replace(STAGE_FIG, OUT_FIG)
    os.replace(STAGE_H5AD, OUTPUT_DIR)
    print(
        json.dumps(
            {
                "status": "completed",
                "n_cells": int(adata.n_obs),
                "n_clusters": observed_k,
                "grid_checkpoint_ari": checkpoint_ari,
                "output_h5ad": str(OUTPUT),
                "elapsed_seconds": round(time.time() - started, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    del adata
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import random
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
import seaborn as sns
from sklearn.metrics import adjusted_rand_score

SEED = 42
MIN_GRID_CHECKPOINT_ARI = 0.98
ROOT = Path(__file__).resolve().parents[4]
WF = ROOT / "epi-cm-core-workflow"
SOURCE = WF / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_harmony.h5ad"
TABLE = WF / "tables/01-celltype_integration_clustering/05-clustering-parameter-search"
SELECTION = TABLE / "selected_clustering.csv"
OUTPUT = WF / "h5ad/01-celltype_integration_clustering/05-clustering-parameter-search/selected/adata_inte.h5ad"
OUT_TABLE = TABLE / "selected"
OUT_FIG = WF / "figures/01-celltype_integration_clustering/05-clustering-parameter-search/selected"
STAGE_H5AD_DIR = OUTPUT.parent.with_name(".selected.staging")
STAGE_TABLE = OUT_TABLE.with_name(".selected.staging")
STAGE_FIG = OUT_FIG.with_name(".selected.staging")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def token(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def refuse_existing_outputs() -> None:
    existing = [
        str(path)
        for path in (OUTPUT.parent, OUT_TABLE, OUT_FIG, STAGE_H5AD_DIR, STAGE_TABLE, STAGE_FIG)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Selected-clustering final/staging output already exists; refusing overwrite or "
            f"partial-output reuse: {existing}"
        )


def main() -> None:
    random.seed(SEED); np.random.seed(SEED)
    refuse_existing_outputs()
    if not SOURCE.is_file() or not SELECTION.is_file():
        raise FileNotFoundError("Harmony source or automatic selection missing")
    sel = pd.read_csv(SELECTION)
    if len(sel) != 1 or int(sel.iloc[0]["automatic_rank"]) < 1:
        raise RuntimeError("Selection must contain exactly one valid automatic rank")
    row = sel.iloc[0]
    selection_rank = int(row.automatic_rank)
    if selection_rank > 1:
        rejection = WF / str(row.get("parent_rejection_record", ""))
        if not rejection.is_file():
            raise RuntimeError("A fallback candidate requires its upstream rejection evidence")
    pcs, nn, res = int(row.n_pcs), int(row.n_neighbors), float(row.resolution)
    if pcs not in range(10, 51, 5) or nn not in range(10, 51, 5) or round(res, 1) not in [x / 10 for x in range(1, 11)]:
        raise RuntimeError("Selected parameters outside user grid")
    key = f"leiden_res{token(res)}"
    grid_label = WF / f"tables/01-celltype_integration_clustering/05-clustering-parameter-search/pcs-{pcs}_nn-{nn}_res-0p1-1p0/labels_{key}.npz"
    if not grid_label.is_file():
        raise FileNotFoundError(grid_label)
    with np.load(grid_label, allow_pickle=False) as z:
        expected_labels = z["labels"].astype(str)
    a = ad.read_h5ad(SOURCE)
    original_ids = a.obs_names.copy()
    if "X_umap" in a.obsm or len(a.obsp) or a.raw is None or len(expected_labels) != a.n_obs:
        raise RuntimeError("Selected rerun did not start from clean graphless full Harmony")
    source_basis_dtype = str(a.obsm["X_pca_inte"].dtype)
    # The grid runner explicitly used a float32 copy of the integrated basis.
    # Match that numerical input exactly before GPU transfer; the saved Harmony
    # basis itself is float64 and otherwise produces a materially different
    # approximate neighbour graph under the same nominal parameters.
    a.obsm["X_pca_inte"] = np.asarray(a.obsm["X_pca_inte"], dtype=np.float32)
    rsc.get.anndata_to_GPU(a)
    rsc.pp.neighbors(
        a, n_neighbors=nn, n_pcs=pcs, use_rep="X_pca_inte",
        algorithm="cagra", random_state=SEED,
    )
    rsc.tl.umap(a, random_state=SEED)
    rsc.tl.leiden(a, resolution=res, key_added=key, random_state=SEED)
    rsc.get.anndata_to_CPU(a, convert_all=True)
    observed_labels = a.obs[key].astype(str).to_numpy()
    partition_exact = bool(np.array_equal(observed_labels, expected_labels))
    if not partition_exact:
        # Cluster numeric identities can be permuted; require a perfect one-to-one contingency mapping.
        tab = pd.crosstab(pd.Series(expected_labels, name="grid"), pd.Series(observed_labels, name="rerun"))
        partition_exact = bool((tab.gt(0).sum(axis=0) == 1).all() and (tab.gt(0).sum(axis=1) == 1).all())
    checkpoint_ari = float(adjusted_rand_score(expected_labels, observed_labels))
    expected_k = int(pd.Series(expected_labels).nunique())
    observed_k = int(pd.Series(observed_labels).nunique())
    print(
        f"selected rerun audit: exact={partition_exact}, ARI={checkpoint_ari:.9f}, "
        f"grid_k={expected_k}, rerun_k={observed_k}",
        flush=True,
    )
    # GPU neighbour construction can be nondeterministic at the edge/tie level
    # even with a fixed seed.  The clean rerun, rather than a candidate object,
    # is the canonical final result.  Require very high partition agreement and
    # the same cluster count; do not demand bit-identical candidate labels.
    if (
        checkpoint_ari < MIN_GRID_CHECKPOINT_ARI
        or observed_k != expected_k
        or not a.obs_names.equals(original_ids)
    ):
        raise RuntimeError(
            "Clean selected rerun is not sufficiently stable against the grid checkpoint: "
            f"ARI={checkpoint_ari:.9f}, required>={MIN_GRID_CHECKPOINT_ARI}, "
            f"grid_k={expected_k}, rerun_k={observed_k}"
        )
    a.uns["selected_clustering"] = {
        "source_h5ad": str(SOURCE.relative_to(WF)), "source_sha256": sha256(SOURCE),
        "n_pcs": pcs, "n_neighbors": nn, "resolution": res, "raw_cluster_column": key,
        "selection_mode": str(row.selection_mode), "automatic_rank": selection_rank,
        "selection_csv": str(SELECTION.relative_to(WF)), "backend": "rapids-singlecell GPU",
        "random_seed": SEED, "downsampled": False,
        "source_integrated_basis_dtype": source_basis_dtype,
        "graph_integrated_basis_dtype": "float32",
        "neighbor_algorithm": "cagra",
        "grid_checkpoint_partition_exact": partition_exact,
        "grid_checkpoint_ari": checkpoint_ari,
        "grid_checkpoint_min_ari": MIN_GRID_CHECKPOINT_ARI,
    }
    STAGE_H5AD_DIR.mkdir(parents=True, exist_ok=False)
    staged_h5ad = STAGE_H5AD_DIR / OUTPUT.name
    a.write_h5ad(staged_h5ad, compression="gzip")
    STAGE_TABLE.mkdir(parents=True, exist_ok=False)
    counts = a.obs[key].astype(str).value_counts().sort_index().rename_axis("cluster").reset_index(name="n_cells")
    counts.to_csv(STAGE_TABLE / "cluster_sizes.csv", index=False)
    pd.DataFrame([{**a.uns["selected_clustering"], "selected_h5ad": str(OUTPUT.relative_to(WF)),
                   "n_cells": a.n_obs, "n_clusters": len(counts)}]).to_csv(STAGE_TABLE / "selected_run_provenance.csv", index=False)
    validation = {"all_passed": True, "n_cells": a.n_obs, "n_clusters": len(counts),
                  "raw_present": a.raw is not None, "cell_order_preserved": a.obs_names.equals(original_ids),
                  "grid_checkpoint_partition_exact": partition_exact,
                  "grid_checkpoint_ari": checkpoint_ari,
                  "grid_checkpoint_min_ari": MIN_GRID_CHECKPOINT_ARI,
                  "output_sha256": sha256(staged_h5ad),
                  "final_h5ad_published_last": True}
    (STAGE_TABLE / "selected_clustering_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    STAGE_FIG.mkdir(parents=True, exist_ok=False)
    sns.set_theme(style="ticks")
    sc.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
    sc.settings.figdir = STAGE_FIG
    for suffix in ["pdf", "svg"]:
        sc.pl.umap(a, color=[key, "series", "status"], ncols=3, wspace=0.45,
                   frameon=True, show=False, save=f"_selected.{suffix}")
    versions = {p: version(p) for p in ["anndata", "scanpy", "rapids-singlecell", "cupy-cuda12x", "numpy", "pandas"]}
    versions.update({"python": platform.python_version(), "seed": SEED})
    (STAGE_TABLE / "package_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    (STAGE_TABLE / "readme.txt").write_text(
        "Automatic rank-1 parameters rerun from clean graphless Harmony on all cells using GPU.\n"
        "All tables/figures are validated before the final h5ad directory is atomically published last.\n"
    )
    if not list(STAGE_FIG.glob("*.pdf")) or not list(STAGE_FIG.glob("*.svg")):
        raise RuntimeError("Selected-clustering PDF/SVG staging outputs are incomplete")
    unexpected = [path for path in STAGE_FIG.iterdir() if path.suffix.lower() not in {".pdf", ".svg"}]
    if unexpected:
        raise RuntimeError(f"Unexpected selected-clustering figure format: {unexpected}")
    if not staged_h5ad.is_file() or not (STAGE_TABLE / "selected_clustering_validation.json").is_file():
        raise RuntimeError("Selected-clustering staged artifact validation failed")
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.parent.mkdir(parents=True, exist_ok=True)
    os.replace(STAGE_TABLE, OUT_TABLE)
    os.replace(STAGE_FIG, OUT_FIG)
    # Final h5ad directory is the last published artifact.
    os.replace(STAGE_H5AD_DIR, OUTPUT.parent)
    del a; gc.collect(); cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()

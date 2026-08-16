#!/usr/bin/env python3
"""Clean GPU rerun of the user-selected B or epithelial GC subtype candidate."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block02_common import SEED, WORKFLOW_ROOT, cell_id_digest  # noqa: E402


MODULE = "02-cell_subtype_integration_clustering"
SOURCE_VERSION = "selected-manual-pcs25-nn50-res0p3-v4"
OUTPUT_VERSION = "selected-user-override-b25-40-0p5-epi40-40-0p4-v5"
MIN_CHECKPOINT_ARI = 0.98
CHOICES = {
    "b_cells": {
        "n_pcs": 25,
        "n_neighbors": 40,
        "resolution": 0.5,
        "cell_abbrev": "b",
        "reason": "User explicitly selected B PCs=25, nn=40, resolution=0.5.",
    },
    "epithelial_cells": {
        "n_pcs": 40,
        "n_neighbors": 40,
        "resolution": 0.4,
        "cell_abbrev": "epi",
        "reason": "User explicitly selected epithelial PCs=40, nn=40, resolution=0.4.",
    },
}


def versioned_root(category: str, task: str, version: str) -> Path:
    return WORKFLOW_ROOT / category / MODULE / task / version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage", required=True, choices=sorted(CHOICES))
    return parser.parse_args()


def partial_h5ad(path: Path) -> Path:
    return path.with_name(f".{path.stem}.partial{path.suffix}")


def write_h5ad_once(adata: ad.AnnData, path: Path) -> None:
    partial = partial_h5ad(path)
    if path.exists() or partial.exists():
        raise FileExistsError(f"Refusing h5ad overwrite: {path} / {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(partial)
    os.replace(partial, path)


def write_csv_once(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing CSV overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def write_text_once(value: str, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing text overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(value)
    os.replace(partial, path)


def main() -> None:
    import cupy as cp
    import rapids_singlecell as rsc

    lineage = parse_args().lineage
    choice = CHOICES[lineage]
    source_path = (
        versioned_root("h5ad", "02-subtype-harmony", SOURCE_VERSION)
        / lineage / f"adata_{lineage}_harmony.h5ad"
    )
    manifest_path = (
        versioned_root("tables", "03-subtype-clustering-grid", SOURCE_VERSION)
        / lineage / "candidate_manifest.csv"
    )
    out_dir = versioned_root("h5ad", "04-subtype-selected-clustering", OUTPUT_VERSION) / lineage
    table_dir = versioned_root("tables", "04-subtype-selected-clustering", OUTPUT_VERSION) / lineage
    figure_dir = versioned_root("figures", "04-subtype-selected-clustering", OUTPUT_VERSION) / lineage
    output_path = out_dir / f"adata_{choice['cell_abbrev']}_selected_clustered.h5ad"
    expected = [
        output_path,
        partial_h5ad(output_path),
        table_dir / "selected_clustering_parameters.csv",
        table_dir / "selected_clustering_completion.json",
    ]
    if any(path.exists() for path in expected):
        raise FileExistsError(f"Selected output already/residually exists for {lineage}")

    manifest = pd.read_csv(manifest_path)
    candidate = manifest[
        manifest["status"].eq("completed")
        & manifest["n_pcs"].eq(choice["n_pcs"])
        & manifest["n_neighbors"].eq(choice["n_neighbors"])
        & manifest["resolution"].round(1).eq(choice["resolution"])
    ]
    if len(candidate) != 1:
        raise RuntimeError(f"Exact user-selected candidate is missing or duplicated: {lineage}")
    candidate = candidate.iloc[0]
    expected_k = int(candidate.n_clusters)
    cluster_key = f"leiden_res{choice['resolution']:.1f}".replace(".", "p")

    adata = ad.read_h5ad(source_path)
    if cell_id_digest(adata.obs_names) != str(candidate.cell_id_sha256):
        raise RuntimeError("Harmony cell IDs differ from the immutable grid candidate")
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.neighbors(
        adata,
        n_neighbors=choice["n_neighbors"],
        n_pcs=choice["n_pcs"],
        use_rep="X_pca_inte",
        key_added="selected_neighbors",
        random_state=SEED,
    )
    rsc.tl.umap(
        adata,
        neighbors_key="selected_neighbors",
        key_added="X_umap",
        random_state=SEED,
    )
    rsc.tl.leiden(
        adata,
        resolution=choice["resolution"],
        neighbors_key="selected_neighbors",
        key_added=cluster_key,
        random_state=SEED,
    )
    rsc.get.anndata_to_CPU(adata, convert_all=True)
    cp.get_default_memory_pool().free_all_blocks()

    with np.load(Path(candidate.terminal_path), allow_pickle=False) as checkpoint:
        checkpoint_labels = checkpoint["labels"]
    checkpoint_ari = float(adjusted_rand_score(
        checkpoint_labels, adata.obs[cluster_key].cat.codes.to_numpy()
    ))
    rerun_k = int(adata.obs[cluster_key].nunique())
    all_passed = bool(rerun_k == expected_k and checkpoint_ari >= MIN_CHECKPOINT_ARI)

    counts = (
        adata.obs[cluster_key].astype(str).value_counts().sort_index()
        .rename_axis("cluster").reset_index(name="n_cells")
    )
    composition_columns = [c for c in ("sample", "series", "status") if c in adata.obs]
    composition = (
        adata.obs.assign(cluster=adata.obs[cluster_key].astype(str))
        .groupby(["cluster", *composition_columns], observed=True)
        .size().rename("n_cells").reset_index()
    )
    composition["fraction_within_cluster"] = (
        composition["n_cells"] / composition.groupby("cluster")["n_cells"].transform("sum")
    )
    parameters = pd.DataFrame([{
        "lineage": lineage,
        "n_cells": int(adata.n_obs),
        "n_pcs": int(choice["n_pcs"]),
        "n_neighbors": int(choice["n_neighbors"]),
        "resolution": float(choice["resolution"]),
        "n_clusters": rerun_k,
        "grid_n_clusters": expected_k,
        "cluster_key": cluster_key,
        "selection_mode": "user_explicit_parameter_selection",
        "selection_reason": choice["reason"],
        "clean_rerun_checkpoint_ari": checkpoint_ari,
        "minimum_checkpoint_ari": MIN_CHECKPOINT_ARI,
        "all_passed": all_passed,
        "source_harmony_h5ad": str(source_path),
        "source_candidate_manifest": str(manifest_path),
        "source_terminal": str(candidate.terminal_path),
        "selected_h5ad": str(output_path),
        "backend": "rapids_singlecell_gpu",
        "random_seed": SEED,
        "downsampled": False,
    }])
    write_csv_once(counts, table_dir / "selected_cluster_counts.csv")
    write_csv_once(composition, table_dir / "selected_cluster_sample_composition.csv")
    write_csv_once(parameters, table_dir / "selected_clustering_parameters.csv")

    figure_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
    sc.settings.figdir = figure_dir
    sc.pl.umap(adata, color=cluster_key, show=False, save="_selected_raw_clusters.pdf")
    sc.pl.umap(adata, color=cluster_key, show=False, save="_selected_raw_clusters.svg")
    sc.pl.umap(
        adata, color=[cluster_key, *composition_columns], ncols=1, wspace=0.4,
        show=False, save="_selected_cluster_batch_diagnostics.pdf",
    )
    sc.pl.umap(
        adata, color=[cluster_key, *composition_columns], ncols=1, wspace=0.4,
        show=False, save="_selected_cluster_batch_diagnostics.svg",
    )
    write_h5ad_once(adata, output_path)

    completion = {
        "lineage": lineage,
        "selected_h5ad": str(output_path),
        "n_cells": int(adata.n_obs),
        "n_clusters": rerun_k,
        "grid_n_clusters": expected_k,
        "cluster_key": cluster_key,
        "clean_rerun_checkpoint_ari": checkpoint_ari,
        "minimum_checkpoint_ari": MIN_CHECKPOINT_ARI,
        "all_passed": all_passed,
        "selection_mode": "user_explicit_parameter_selection",
        "backend": "rapids_singlecell_gpu",
        "random_seed": SEED,
        "downsampled": False,
    }
    write_text_once(json.dumps(completion, indent=2) + "\n", table_dir / "selected_clustering_completion.json")
    write_text_once(
        "User-selected GC lineage clustering clean GPU rerun.\n"
        f"Source version: {SOURCE_VERSION}\nOutput version: {OUTPUT_VERSION}\n"
        "The exact user-selected candidate was used; no fallback candidate was substituted.\n",
        table_dir / "readme.txt",
    )
    write_text_once(
        f"python={platform.python_version()}\nanndata={ad.__version__}\n"
        f"scanpy={sc.__version__}\nnumpy={np.__version__}\npandas={pd.__version__}\n"
        f"rapids_singlecell={rsc.__version__}\ncupy={cp.__version__}\n"
        f"code={Path(__file__).resolve()}\nseed={SEED}\n",
        table_dir / "package_versions.txt",
    )
    print(json.dumps(completion, indent=2), flush=True)
    del adata
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()

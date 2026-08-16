#!/usr/bin/env python3
"""Clean GPU rerun of one agent-selected GC subtype clustering candidate."""

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
from block02_common import SEED, cell_id_digest, task_root  # noqa: E402


MIN_CHECKPOINT_ARI = 0.98
STRUCTURAL_WEIGHTS = {
    "graph_robustness": 0.55,
    "cluster_size_score": 0.25,
    "sample_mixing": 0.10,
    "series_mixing": 0.05,
    "status_mixing": 0.05,
}

CHOICES = {
    "b_cells": {
        "n_pcs": 50, "n_neighbors": 45, "resolution": 0.7,
        "n_clusters": 7, "cell_abbrev": "b",
        "reason": (
            "The seven-cluster plateau at resolutions 0.6-0.7 separates the detached islands "
            "and the major B-cell continuum without the overlapping central fragmentation at "
            "resolutions 0.8 and above."
        ),
    },
    "endothelial_cells": {
        "n_pcs": 35, "n_neighbors": 50, "resolution": 0.3,
        "n_clusters": 9, "cell_abbrev": "endo",
        "reason": (
            "The nine-cluster plateau at resolutions 0.3-0.4 resolves the detached vascular "
            "islands and upper/lower endothelial compartments; higher resolutions fragment the "
            "continuous main manifolds."
        ),
    },
    "epithelial_cells": {
        "n_pcs": 35, "n_neighbors": 40, "resolution": 0.6,
        "n_clusters": 14, "cell_abbrev": "epi",
        "reason": (
            "This is the highest no-fixed-K structural score at K=14 and resolves the detached "
            "epithelial islands plus the main GC epithelial states without the more diffuse "
            "overlapping splits above K=14."
        ),
    },
    "myeloid_cells": {
        "n_pcs": 50, "n_neighbors": 45, "resolution": 0.2,
        "n_clusters": 10, "cell_abbrev": "mye",
        "reason": (
            "K=10 has very high seed and neighboring-parameter stability and preserves the "
            "separate myeloid islands while avoiding the central fragmentation beginning at K=12."
        ),
    },
    "plasma_cells": {
        "n_pcs": 45, "n_neighbors": 50, "resolution": 0.3,
        "n_clusters": 5, "cell_abbrev": "plasma",
        "reason": (
            "K=5 resolves the two detached islands and three main plasma/B-like programs; K=6 "
            "and above increasingly divide the overlapping main cloud."
        ),
    },
    "stromal_cells": {
        "n_pcs": 20, "n_neighbors": 45, "resolution": 0.3,
        "n_clusters": 10, "cell_abbrev": "stromal",
        "reason": (
            "The ten-cluster plateau at resolutions 0.3-0.4 separates the visible stromal "
            "islands and branches; K=14 and above visibly fragment continuous structures."
        ),
    },
    "t_cells": {
        "n_pcs": 25, "n_neighbors": 45, "resolution": 0.6,
        "n_clusters": 10, "cell_abbrev": "t",
        "reason": (
            "The K=10 plateau at resolutions 0.6-0.7 provides detailed T/NK state resolution "
            "without the overlapping high-resolution stripes and unstable extra splits."
        ),
    },
}


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


def write_text_once(text: str, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing text overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(text)
    os.replace(partial, path)


def add_symmetric_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    lookup = frame.set_index(["n_pcs", "n_neighbors", "resolution"])
    robustness = []
    for _, row in frame.iterrows():
        pcs = int(row.n_pcs)
        nn = int(row.n_neighbors)
        res = round(float(row.resolution), 1)
        values = [
            row.get("local_seed_ari"), row.get("previous_resolution_ari"),
            row.get("previous_nn_ari"), row.get("previous_pcs_ari"),
        ]
        for key, field in (
            ((pcs, nn, round(res + 0.1, 1)), "previous_resolution_ari"),
            ((pcs, nn + 5, res), "previous_nn_ari"),
            ((pcs + 5, nn, res), "previous_pcs_ari"),
        ):
            if key in lookup.index:
                neighbor = lookup.loc[key]
                if isinstance(neighbor, pd.DataFrame):
                    neighbor = neighbor.iloc[0]
                values.append(neighbor.get(field))
        valid = [float(value) for value in values if pd.notna(value)]
        robustness.append(float(np.mean(valid)))
    frame["graph_robustness"] = robustness
    frame["structural_score_no_fixed_k"] = sum(
        frame[column].fillna(0.0) * weight
        for column, weight in STRUCTURAL_WEIGHTS.items()
    )
    return frame


def main() -> None:
    import cupy as cp
    import rapids_singlecell as rsc

    lineage = parse_args().lineage
    choice = CHOICES[lineage]
    source_path = (
        task_root("h5ad", "02-subtype-harmony") / lineage /
        f"adata_{lineage}_harmony.h5ad"
    )
    manifest_path = (
        task_root("tables", "03-subtype-clustering-grid") / lineage /
        "candidate_manifest.csv"
    )
    out_dir = task_root("h5ad", "04-subtype-selected-clustering") / lineage
    table_dir = task_root("tables", "04-subtype-selected-clustering") / lineage
    figure_dir = task_root("figures", "04-subtype-selected-clustering") / lineage
    output_path = out_dir / f"adata_{choice['cell_abbrev']}_selected_clustered.h5ad"
    parameters_path = table_dir / "selected_clustering_parameters.csv"
    evidence_path = table_dir / "candidate_selection_evidence_within_selected_k.csv"
    counts_path = table_dir / "selected_cluster_counts.csv"
    composition_path = table_dir / "selected_cluster_sample_composition.csv"
    completion_path = table_dir / "selected_clustering_completion.json"
    expected_outputs = [
        output_path, partial_h5ad(output_path), parameters_path, evidence_path,
        counts_path, composition_path, completion_path,
    ]
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"Selected output already/residually exists for {lineage}")

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["status"].eq("completed")].copy()
    manifest["resolution"] = manifest["resolution"].round(1)
    manifest = add_symmetric_robustness(manifest)
    selected_k = int(choice["n_clusters"])
    within_k = manifest[manifest["n_clusters"].eq(selected_k)].copy().sort_values(
        ["structural_score_no_fixed_k", "graph_robustness", "local_seed_ari"],
        ascending=False,
        kind="mergesort",
    )
    within_k.insert(0, "structural_rank_within_selected_k", np.arange(1, len(within_k) + 1))
    requested = within_k[
        within_k["n_pcs"].eq(choice["n_pcs"])
        & within_k["n_neighbors"].eq(choice["n_neighbors"])
        & within_k["resolution"].eq(choice["resolution"])
    ]
    if len(requested) != 1:
        raise RuntimeError(f"Chosen candidate is missing or duplicated: {lineage}")
    requested = requested.iloc[0]
    attempt_order = pd.concat(
        [requested.to_frame().T, within_k.drop(index=requested.name)],
        ignore_index=True,
    )
    rejection_rows = []
    adata = None
    selected = None
    pcs = nn = rerun_k = -1
    resolution = checkpoint_ari = float("nan")
    cluster_key = ""
    for _, candidate in attempt_order.iterrows():
        trial = ad.read_h5ad(source_path)
        if cell_id_digest(trial.obs_names) != str(candidate.cell_id_sha256):
            raise RuntimeError("Harmony cell IDs differ from the immutable grid candidate")
        trial_pcs = int(candidate.n_pcs)
        trial_nn = int(candidate.n_neighbors)
        trial_resolution = float(candidate.resolution)
        trial_key = f"leiden_res{trial_resolution:.1f}".replace(".", "p")
        rsc.get.anndata_to_GPU(trial)
        rsc.pp.neighbors(
            trial, n_neighbors=trial_nn, n_pcs=trial_pcs, use_rep="X_pca_inte",
            key_added="selected_neighbors", random_state=SEED,
        )
        rsc.tl.umap(
            trial, neighbors_key="selected_neighbors", key_added="X_umap",
            random_state=SEED,
        )
        rsc.tl.leiden(
            trial, resolution=trial_resolution, neighbors_key="selected_neighbors",
            key_added=trial_key, random_state=SEED,
        )
        rsc.get.anndata_to_CPU(trial, convert_all=True)
        cp.get_default_memory_pool().free_all_blocks()
        with np.load(Path(candidate.terminal_path), allow_pickle=False) as checkpoint:
            checkpoint_labels = checkpoint["labels"]
        trial_ari = float(adjusted_rand_score(
            checkpoint_labels, trial.obs[trial_key].cat.codes.to_numpy(),
        ))
        trial_k = int(trial.obs[trial_key].nunique())
        if trial_k == selected_k and trial_ari >= MIN_CHECKPOINT_ARI:
            adata = trial
            selected = candidate
            pcs = trial_pcs
            nn = trial_nn
            resolution = trial_resolution
            cluster_key = trial_key
            checkpoint_ari = trial_ari
            rerun_k = trial_k
            break
        rejection_rows.append({
            "lineage": lineage,
            "attempt": len(rejection_rows) + 1,
            "structural_rank_within_selected_k": int(candidate.structural_rank_within_selected_k),
            "n_pcs": trial_pcs,
            "n_neighbors": trial_nn,
            "resolution": trial_resolution,
            "grid_n_clusters": int(candidate.n_clusters),
            "rerun_n_clusters": trial_k,
            "clean_rerun_checkpoint_ari": trial_ari,
            "minimum_checkpoint_ari": MIN_CHECKPOINT_ARI,
            "rejection_reason": "cluster_count_or_checkpoint_ari_mismatch",
            "terminal_path": str(candidate.terminal_path),
        })
        del trial
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    if adata is None or selected is None:
        raise RuntimeError(f"No K={selected_k} candidate passed the clean-rerun checkpoint")

    counts = (
        adata.obs[cluster_key].astype(str).value_counts().sort_index()
        .rename_axis("cluster").reset_index(name="n_cells")
    )
    composition_columns = [column for column in ("sample", "series", "status") if column in adata.obs]
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
        "n_cells": adata.n_obs,
        "n_pcs": pcs,
        "n_neighbors": nn,
        "resolution": resolution,
        "n_clusters": rerun_k,
        "cluster_key": cluster_key,
        "selection_mode": "user_authorized_agent_selection",
        "selection_reason": choice["reason"],
        "requested_n_pcs": int(choice["n_pcs"]),
        "requested_n_neighbors": int(choice["n_neighbors"]),
        "requested_resolution": float(choice["resolution"]),
        "clean_rerun_rejections_before_acceptance": len(rejection_rows),
        "selected_k_basis": "lineage_specific_umap_plateau_and_biological_complexity",
        "fixed_target_cluster_count_used": False,
        "cluster_complexity_score_used": False,
        "structural_rank_within_selected_k": int(selected.structural_rank_within_selected_k),
        "structural_score_no_fixed_k": float(selected.structural_score_no_fixed_k),
        "graph_robustness": float(selected.graph_robustness),
        "local_seed_ari": float(selected.local_seed_ari),
        "clean_rerun_checkpoint_ari": checkpoint_ari,
        "minimum_checkpoint_ari": MIN_CHECKPOINT_ARI,
        "source_harmony_h5ad": str(source_path),
        "source_candidate_manifest": str(manifest_path),
        "source_terminal": str(selected.terminal_path),
        "selected_h5ad": str(output_path),
        "backend": "rapids_singlecell_gpu",
        "random_seed": SEED,
        "downsampled": False,
    }])

    write_csv_once(within_k, evidence_path)
    write_csv_once(counts, counts_path)
    write_csv_once(composition, composition_path)
    write_csv_once(parameters, parameters_path)
    if rejection_rows:
        write_csv_once(pd.DataFrame(rejection_rows), table_dir / "selection_rejections.csv")

    figure_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
    sc.settings.figdir = figure_dir
    sc.pl.umap(adata, color=cluster_key, show=False, save="_selected_raw_clusters.pdf")
    sc.pl.umap(adata, color=cluster_key, show=False, save="_selected_raw_clusters.svg")
    diagnostic_colors = [cluster_key, *composition_columns]
    sc.pl.umap(
        adata, color=diagnostic_colors, ncols=1, wspace=0.4, show=False,
        save="_selected_cluster_batch_diagnostics.pdf",
    )
    sc.pl.umap(
        adata, color=diagnostic_colors, ncols=1, wspace=0.4, show=False,
        save="_selected_cluster_batch_diagnostics.svg",
    )

    write_h5ad_once(adata, output_path)
    completion = {
        "lineage": lineage,
        "selected_h5ad": str(output_path),
        "n_cells": int(adata.n_obs),
        "n_clusters": rerun_k,
        "cluster_key": cluster_key,
        "clean_rerun_checkpoint_ari": checkpoint_ari,
        "all_passed": True,
        "agent_selection_authorized_by_user": True,
        "fixed_target_cluster_count_used": False,
        "cluster_complexity_score_used": False,
        "backend": "rapids_singlecell_gpu",
        "random_seed": SEED,
        "downsampled": False,
    }
    write_text_once(json.dumps(completion, indent=2) + "\n", completion_path)
    write_text_once(
        "Agent-selected GC lineage clustering clean rerun.\n"
        "The lineage-specific K was chosen from UMAP plateau and biological complexity; "
        "no score rewarded proximity to a preset cluster count.\n"
        "The exact candidate was rerun from the clean lineage Harmony h5ad and checked "
        "against its immutable grid labels.\n",
        table_dir / "readme.txt",
    )
    write_text_once(
        f"python={platform.python_version()}\n"
        f"anndata={ad.__version__}\nscanpy={sc.__version__}\n"
        f"numpy={np.__version__}\npandas={pd.__version__}\n"
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

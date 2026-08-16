#!/usr/bin/env python3
"""Clean GPU rerun of the GC lineage-specific v2 clustering choice.

The biological cluster count is selected independently for every lineage.  No
term in this script rewards proximity to a preset number of clusters.  Once a
lineage-specific cluster count has been selected from marker/fragmentation
review, the best structural candidate at that count is rerun from the clean
Harmony object and checked against the immutable grid checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

CODE_DIR = Path(__file__).resolve().parent
WORKFLOW_ROOT = CODE_DIR.parents[2]
MODULE = "02-cell_subtype_integration_clustering"
SEED = 42
MIN_CHECKPOINT_ARI = 0.999999


# The cluster counts were chosen lineage by lineage after excluding a fixed-K
# objective and reviewing stability, tiny/sample-dominated splits, and the
# marker evidence from the original selected result.  For a chosen K, the
# PCS/NN/resolution tuple is the highest structural-score candidate at that K.
CHOICES = {
    "b_cells": {
        "n_pcs": 50, "n_neighbors": 50, "resolution": 0.6, "n_clusters": 7,
        "reason": "Seven clusters retain naive/memory/GC/plasma-like structure while avoiding the weak extra splits seen at K=10.",
    },
    "endothelial_cells": {
        "n_pcs": 50, "n_neighbors": 25, "resolution": 0.4, "n_clusters": 9,
        "reason": "Nine clusters recover all six endothelial marker families with stronger graph stability than the old K=10 choice.",
    },
    "epithelial_cells": {
        "n_pcs": 30, "n_neighbors": 45, "resolution": 0.7, "n_clusters": 14,
        "reason": "Fourteen clusters increase GC epithelial state resolution without accepting the sample-specific fragmentation at the K=20 boundary.",
    },
    "mast_cells": {
        "n_pcs": 45, "n_neighbors": 30, "resolution": 0.1, "n_clusters": 4,
        "reason": "The old K=10 result had eight low-confidence mast splits; K=4 is stable and avoids unsupported fragmentation.",
    },
    "myeloid_cells": {
        "n_pcs": 45, "n_neighbors": 50, "resolution": 0.1, "n_clusters": 8,
        "reason": "K=8 is the structural stability maximum and matches the available monocyte/macrophage/DC/neutrophil marker resolution.",
    },
    "plasma_cells": {
        "n_pcs": 40, "n_neighbors": 40, "resolution": 0.4, "n_clusters": 5,
        "reason": "K=5 preserves the supported plasma/B-like programs and avoids the mostly moderate extra splits in the old K=10 result.",
    },
    "stromal_cells": {
        "n_pcs": 50, "n_neighbors": 50, "resolution": 0.3, "n_clusters": 10,
        "reason": "K=10 remains the strongest structural plateau even after completely removing the fixed-K reward and retains all stromal marker families.",
    },
    "t_cells": {
        "n_pcs": 50, "n_neighbors": 45, "resolution": 0.7, "n_clusters": 10,
        "reason": "All ten old T/NK clusters had high marker support; K=10 is retained on biological evidence, not because ten is a target.",
    },
}

STRUCTURAL_WEIGHTS = {
    "graph_robustness": 0.35,
    "cluster_size_score": 0.20,
    "sample_mixing": 0.15,
    "series_mixing": 0.10,
    "status_mixing": 0.05,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lineage", required=True, choices=sorted(CHOICES))
    return p.parse_args()


def digest(ids) -> str:
    h = hashlib.sha256()
    for value in ids:
        h.update(str(value).encode())
        h.update(b"\0")
    return h.hexdigest()


def partial(path: Path) -> Path:
    return path.with_name(f".{path.stem}.partial{path.suffix}")


def write_h5ad_once(adata: ad.AnnData, path: Path) -> None:
    tmp = partial(path)
    if path.exists() or tmp.exists():
        raise FileExistsError(f"Refusing overwrite: {path} / {tmp}")
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def write_csv_once(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def add_symmetric_robustness(df: pd.DataFrame) -> pd.DataFrame:
    lookup = df.set_index(["n_pcs", "n_neighbors", "resolution"])
    result = []
    for _, row in df.iterrows():
        pcs = int(row.n_pcs)
        nn = int(row.n_neighbors)
        res = round(float(row.resolution), 1)
        values = [row.get("local_seed_ari"), row.get("previous_resolution_ari"),
                  row.get("previous_nn_ari"), row.get("previous_pcs_ari")]
        for key, field in (
            ((pcs, nn, round(res + 0.1, 1)), "previous_resolution_ari"),
            ((pcs, nn + 5, res), "previous_nn_ari"),
            ((pcs + 5, nn, res), "previous_pcs_ari"),
        ):
            if key in lookup.index:
                nxt = lookup.loc[key]
                if isinstance(nxt, pd.DataFrame):
                    nxt = nxt.iloc[0]
                values.append(nxt.get(field))
        valid = [float(x) for x in values if pd.notna(x)]
        result.append(float(np.mean(valid)))
    df["graph_robustness"] = result
    return df


def main() -> None:
    args = parse_args()
    lineage = args.lineage
    choice = CHOICES[lineage]
    source_path = WORKFLOW_ROOT / "h5ad" / MODULE / "02-subtype-harmony" / lineage / f"adata_{lineage}_harmony.h5ad"
    manifest_path = WORKFLOW_ROOT / "tables" / MODULE / "03-subtype-clustering-grid" / lineage / "candidate_manifest.csv"
    out_dir = WORKFLOW_ROOT / "h5ad" / MODULE / "04-subtype-selected-clustering-v2" / lineage
    table_dir = WORKFLOW_ROOT / "tables" / MODULE / "04-subtype-selected-clustering-v2" / lineage
    fig_dir = WORKFLOW_ROOT / "figures" / MODULE / "04-subtype-selected-clustering-v2" / lineage
    output_path = out_dir / f"adata_{lineage}_selected_clustered_v2.h5ad"
    params_path = table_dir / "selected_clustering_parameters_v2.json"
    counts_path = table_dir / "selected_cluster_counts_v2.csv"
    composition_path = table_dir / "selected_cluster_sample_composition_v2.csv"
    ranking_path = table_dir / "candidate_ranking_within_selected_k_v2.csv"

    existing = [output_path, partial(output_path), params_path, counts_path, composition_path, ranking_path]
    if any(p.exists() for p in existing):
        raise FileExistsError(f"v2 output already/residually exists for {lineage}; refusing overwrite")

    df = pd.read_csv(manifest_path)
    df = df[(df.status == "completed") & df.n_clusters.between(2, 20)].copy()
    df["resolution"] = df.resolution.round(1)
    df = add_symmetric_robustness(df)
    denom = sum(STRUCTURAL_WEIGHTS.values())
    df["structural_score_no_fixed_k"] = sum(
        df[key].fillna(0.0) * weight for key, weight in STRUCTURAL_WEIGHTS.items()
    ) / denom
    selected_k = int(choice["n_clusters"])
    within = df[df.n_clusters == selected_k].copy().sort_values(
        ["structural_score_no_fixed_k", "graph_robustness", "cluster_size_score", "local_seed_ari",
         "n_pcs", "n_neighbors", "resolution"],
        ascending=[False, False, False, False, True, True, True], kind="mergesort",
    )
    within.insert(0, "rank_within_selected_k", np.arange(1, len(within) + 1))
    import cupy as cp
    import rapids_singlecell as rsc

    rejection_rows = []
    adata = None
    selected = None
    checkpoint_ari = float("nan")
    rerun_k = -1
    cluster_key = ""
    pcs = nn = -1
    res = float("nan")
    for _, candidate in within.iterrows():
        trial = ad.read_h5ad(source_path)
        if digest(trial.obs_names) != str(candidate.cell_id_sha256):
            raise RuntimeError("Harmony cells differ from the immutable candidate")
        pcs = int(candidate.n_pcs)
        nn = int(candidate.n_neighbors)
        res = float(candidate.resolution)
        trial_key = f"leiden_selected_v2_pcs{pcs}_nn{nn}_res{res:.1f}".replace(".", "p")
        rsc.get.anndata_to_GPU(trial)
        rsc.pp.neighbors(trial, n_neighbors=nn, n_pcs=pcs, use_rep="X_pca_inte",
                         key_added="selected_neighbors_v2", random_state=SEED)
        rsc.tl.umap(trial, neighbors_key="selected_neighbors_v2", key_added="X_umap", random_state=SEED)
        rsc.tl.leiden(trial, resolution=res, neighbors_key="selected_neighbors_v2",
                      key_added=trial_key, random_state=SEED)
        rsc.get.anndata_to_CPU(trial, convert_all=True)
        cp.get_default_memory_pool().free_all_blocks()
        with np.load(Path(candidate.terminal_path), allow_pickle=False) as z:
            checkpoint_labels = z["labels"]
        trial_ari = float(adjusted_rand_score(checkpoint_labels, trial.obs[trial_key].cat.codes.to_numpy()))
        trial_k = int(trial.obs[trial_key].nunique())
        if trial_ari >= MIN_CHECKPOINT_ARI and trial_k == selected_k:
            adata = trial
            selected = candidate
            checkpoint_ari = trial_ari
            rerun_k = trial_k
            cluster_key = trial_key
            break
        rejection_rows.append({
            "lineage": lineage,
            "rank_within_selected_k": int(candidate.rank_within_selected_k),
            "n_pcs": pcs,
            "n_neighbors": nn,
            "resolution": res,
            "grid_n_clusters": int(candidate.n_clusters),
            "rerun_n_clusters": trial_k,
            "clean_rerun_checkpoint_ari": trial_ari,
            "required_checkpoint_ari": MIN_CHECKPOINT_ARI,
            "rejection_reason": "checkpoint_ari_or_cluster_count_mismatch",
            "terminal_path": str(candidate.terminal_path),
        })
        del trial
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    if adata is None or selected is None:
        raise RuntimeError(f"No K={selected_k} candidate passed the clean-rerun checkpoint")

    counts = adata.obs[cluster_key].astype(str).value_counts().sort_index().rename_axis("cluster").reset_index(name="n_cells")
    composition = (
        adata.obs.assign(cluster=adata.obs[cluster_key].astype(str))
        .groupby(["cluster", "sample", "series", "status"], observed=True)
        .size().rename("n_cells").reset_index()
    )
    composition["fraction_within_cluster"] = composition.n_cells / composition.groupby("cluster").n_cells.transform("sum")

    table_dir.mkdir(parents=True, exist_ok=True)
    write_csv_once(within, ranking_path)
    write_csv_once(counts, counts_path)
    write_csv_once(composition, composition_path)
    if rejection_rows:
        write_csv_once(pd.DataFrame(rejection_rows), table_dir / "selection_rejections_v2.csv")

    fig_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.autoshow = False
    sc.settings.figdir = str(fig_dir)
    sc.pl.umap(adata, color=[cluster_key, "status", "series"], ncols=1, show=False, save="_selected_v2.pdf")
    sc.pl.umap(adata, color=[cluster_key, "status", "series"], ncols=1, show=False, save="_selected_v2.svg")

    payload = {
        "lineage": lineage,
        "project": "gastric_cancer",
        "n_pcs": pcs,
        "n_neighbors": nn,
        "resolution": res,
        "n_clusters": rerun_k,
        "cluster_key": cluster_key,
        "selection_method": "lineage_specific_biological_k_then_structural_rank_within_k",
        "fixed_target_cluster_count_used": False,
        "cluster_complexity_score_used": False,
        "structural_weights": STRUCTURAL_WEIGHTS,
        "structural_score_no_fixed_k": float(selected.structural_score_no_fixed_k),
        "rank_within_selected_k": int(selected.rank_within_selected_k),
        "rejected_higher_ranked_candidates": len(rejection_rows),
        "biological_selection_reason": choice["reason"],
        "clean_rerun_checkpoint_ari": checkpoint_ari,
        "required_checkpoint_ari": MIN_CHECKPOINT_ARI,
        "cell_id_sha256": digest(adata.obs_names),
        "source_harmony_h5ad": str(source_path),
        "source_candidate_manifest": str(manifest_path),
        "source_terminal": str(selected.terminal_path),
        "selected_h5ad": str(output_path),
        "backend": "rapids_singlecell_gpu",
        "random_seed": SEED,
        "downsampled": False,
    }
    tmp_json = params_path.with_name(f".{params_path.name}.partial")
    tmp_json.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp_json, params_path)
    (table_dir / "readme.txt").write_text(
        "GC subtype v2 final selection. The immutable 810-candidate grid was reused.\n"
        "No score rewards proximity to a preset cluster count. Biological K was selected per lineage,\n"
        "then PCS/NN/resolution was chosen by structural rank within that K and clean-rerun checkpointed.\n"
    )
    (table_dir / "package_versions.txt").write_text(
        f"python={platform.python_version()}\n"
        f"anndata={ad.__version__}\nscanpy={sc.__version__}\n"
        f"numpy={np.__version__}\npandas={pd.__version__}\n"
        f"rapids_singlecell={rsc.__version__}\ncupy={cp.__version__}\n"
        f"code={Path(__file__).resolve()}\nseed={SEED}\n"
    )
    write_h5ad_once(adata, output_path)
    del adata
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

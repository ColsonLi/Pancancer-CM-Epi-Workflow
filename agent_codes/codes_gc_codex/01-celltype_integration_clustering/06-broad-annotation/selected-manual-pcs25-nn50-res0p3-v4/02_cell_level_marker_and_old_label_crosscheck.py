#!/usr/bin/env python3
"""Add cell-level marker-score evidence for new broad clusters."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc

SEED = 42
TASK = "selected-manual-pcs25-nn50-res0p3-v4"
RAW_KEY = "leiden_res0p3"
random.seed(SEED)
np.random.seed(SEED)

WF = Path(__file__).resolve().parents[4]
INPUT = WF / f"h5ad/01-celltype_integration_clustering/05-clustering-parameter-search/{TASK}/adata_inte.h5ad"
TABLE = WF / f"tables/01-celltype_integration_clustering/06-broad-annotation/{TASK}"

OUTPUTS = [
    TABLE / "02_cell_level_best_marker_label_counts.csv",
    TABLE / "02_cell_level_best_marker_label_proportions.csv",
    TABLE / "02_crosscheck_completion.json",
]


def main() -> None:
    started = time.time()
    if any(path.exists() for path in OUTPUTS):
        raise FileExistsError("Refusing to overwrite crosscheck outputs")
    marker_table = pd.read_csv(TABLE / "broad_marker_panel_availability.csv")
    marker_sets = {
        row.cell_type: str(row.available_markers).split(";")
        for row in marker_table.itertuples(index=False)
    }
    adata = ad.read_h5ad(INPUT)

    rsc.get.anndata_to_GPU(adata, convert_all=True)
    for label, genes in marker_sets.items():
        rsc.tl.score_genes(
            adata,
            gene_list=genes,
            score_name=f"broad_score__{label}",
            ctrl_size=max(20, len(genes)),
            use_raw=True,
            random_state=SEED,
        )
    rsc.get.anndata_to_CPU(adata, convert_all=True)
    score_columns = [f"broad_score__{label}" for label in marker_sets]
    scores = adata.obs[score_columns].to_numpy(float)
    best_index = np.argmax(scores, axis=1)
    labels = np.asarray(list(marker_sets), dtype=object)
    best_labels = labels[best_index]
    clusters = adata.obs[RAW_KEY].astype(str).to_numpy()

    best_counts = pd.crosstab(
        pd.Series(clusters, name="cluster"),
        pd.Series(best_labels, name="best_marker_label"),
    ).sort_index(key=lambda index: index.astype(int))
    best_props = best_counts.div(best_counts.sum(axis=1), axis=0)
    best_counts.to_csv(OUTPUTS[0])
    best_props.to_csv(OUTPUTS[1])

    completion = {
        "status": "completed",
        "n_cells": int(adata.n_obs),
        "n_clusters": int(pd.Series(clusters).nunique()),
        "marker_panels": len(marker_sets),
        "score_backend": "rapids-singlecell GPU",
        "old_label_crosscheck": "not_run_old_outputs_deleted_by_user",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    OUTPUTS[2].write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(completion), flush=True)
    del adata
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()

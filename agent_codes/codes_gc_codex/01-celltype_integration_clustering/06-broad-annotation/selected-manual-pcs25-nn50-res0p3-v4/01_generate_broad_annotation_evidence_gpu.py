#!/usr/bin/env python3
"""Generate GPU raw-expression evidence for broad annotation of the new 15 clusters."""

from __future__ import annotations

import gc
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

SEED = 42
TASK = "selected-manual-pcs25-nn50-res0p3-v4"
RAW_KEY = "leiden_res0p3"
PARAMETER_TOKEN = "pcs25_nn50_res0p3"
MIN_AVAILABLE_CANONICAL_MARKERS = 2

random.seed(SEED)
np.random.seed(SEED)

WF = Path(__file__).resolve().parents[4]
INPUT = WF / (
    "h5ad/01-celltype_integration_clustering/05-clustering-parameter-search/"
    f"{TASK}/adata_inte.h5ad"
)
TABLE = WF / f"tables/01-celltype_integration_clustering/06-broad-annotation/{TASK}"
STAGE = TABLE.with_name(f".{TASK}-evidence.staging")

MARKERS = {
    "Epithelial Cells": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "MUC1", "KRT20", "INSM1", "PDX1"],
    "T Cells": ["CD3D", "CD3E", "TRBC1", "TRBC2", "CD247", "IL7R"],
    "B Cells": ["CD79A", "MS4A1", "CD37", "CD74", "HLA-DRA", "CD22"],
    "Plasma Cells": ["MZB1", "JCHAIN", "SDC1", "IGHG1", "IGHG3", "IGKC"],
    "Myeloid Cells": ["LST1", "TYROBP", "FCER1G", "CTSS", "LYZ", "AIF1"],
    "Mast Cells": ["TPSAB1", "TPSB2", "CPA3", "MS4A2", "KIT", "HDC"],
    "Endothelial Cells": ["PECAM1", "VWF", "EMCN", "KDR", "ENG", "ESAM"],
    "Stromal Cells": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "COL6A1", "ACTA2"],
}


def refuse_existing() -> None:
    existing = [str(path) for path in (TABLE, STAGE) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing broad-evidence overwrite/reuse: {existing}")


def export_rank_result(adata: ad.AnnData, key: str, out_dir: Path) -> dict[str, pd.DataFrame]:
    result = adata.uns[key]
    out_dir.mkdir(parents=True, exist_ok=False)
    frames: dict[str, pd.DataFrame] = {}
    for group in result["names"].dtype.names:
        frame = pd.DataFrame(
            {
                "gene": result["names"][group],
                "score": result["scores"][group],
                "logfoldchanges": result["logfoldchanges"][group],
                "pvals": result["pvals"][group],
                "pvals_adj": result["pvals_adj"][group],
            }
        )
        frames[str(group)] = frame
        frame.to_csv(
            out_dir / f"{group}_degs_{RAW_KEY}_{PARAMETER_TOKEN}.csv",
            index=False,
        )
    return frames


def main() -> None:
    started = time.time()
    refuse_existing()
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    if not cp.cuda.is_available():
        raise RuntimeError("GPU raw-expression evidence requires CUDA")

    adata = ad.read_h5ad(INPUT)
    if adata.raw is None or RAW_KEY not in adata.obs:
        raise RuntimeError("Selected clustering lacks adata.raw or the requested cluster column")
    clusters = sorted(adata.obs[RAW_KEY].astype(str).unique(), key=int)
    if not 8 <= len(clusters) <= 20:
        raise RuntimeError(f"Expected 8-20 broad clusters, observed {len(clusters)}")
    raw_genes = set(adata.raw.var_names.astype(str))
    marker_rows = []
    used_markers: dict[str, list[str]] = {}
    for label, configured in MARKERS.items():
        available = [gene for gene in configured if gene in raw_genes]
        if len(available) < MIN_AVAILABLE_CANONICAL_MARKERS:
            raise RuntimeError(f"Insufficient canonical markers for {label}: {available}")
        used_markers[label] = available
        marker_rows.append(
            {
                "cell_type": label,
                "configured_markers": ";".join(configured),
                "available_markers": ";".join(available),
                "n_configured": len(configured),
                "n_available": len(available),
                "minimum_required": MIN_AVAILABLE_CANONICAL_MARKERS,
                "expression_source": "input_h5ad.adata.raw normalized_log",
            }
        )

    free_before, total_gpu = cp.cuda.runtime.memGetInfo()
    rsc.get.anndata_to_GPU(adata, convert_all=True)
    for label, genes in used_markers.items():
        rsc.tl.score_genes(
            adata,
            gene_list=genes,
            score_name=f"broad_score__{label}",
            ctrl_size=max(20, len(genes)),
            use_raw=True,
            random_state=SEED,
        )
    rsc.tl.rank_genes_groups(
        adata,
        groupby=RAW_KEY,
        method="t-test",
        use_raw=True,
        n_genes=adata.raw.n_vars,
        key_added=f"degs_{RAW_KEY}",
    )
    rsc.get.anndata_to_CPU(adata, convert_all=True)
    free_after, _ = cp.cuda.runtime.memGetInfo()

    STAGE.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(marker_rows).to_csv(STAGE / "broad_marker_panel_availability.csv", index=False)
    deg_dir = STAGE / f"degs_{RAW_KEY}_{PARAMETER_TOKEN}"
    raw_degs = export_rank_result(adata, f"degs_{RAW_KEY}", deg_dir)
    score_columns = [f"broad_score__{label}" for label in MARKERS]
    score_mean = (
        adata.obs.assign(_cluster=adata.obs[RAW_KEY].astype(str))
        .groupby("_cluster", observed=True)[score_columns]
        .mean()
        .reindex(clusters)
    )
    score_mean.columns = list(MARKERS)
    score_mean.reset_index(names="cluster").to_csv(
        STAGE / "broad_marker_score_by_raw_cluster.csv", index=False
    )

    review_rows = []
    for cluster in clusters:
        deg = raw_degs[cluster]
        positive = deg.loc[pd.to_numeric(deg["logfoldchanges"], errors="coerce") > 0].copy()
        top300 = positive.head(300)["gene"].astype(str).tolist()
        row: dict[str, object] = {
            "cluster": cluster,
            "n_cells": int(adata.obs[RAW_KEY].astype(str).eq(cluster).sum()),
            "top30_positive_degs": ";".join(top300[:30]),
        }
        ordered_scores = score_mean.loc[cluster].sort_values(ascending=False)
        row["score_rank_order"] = ";".join(ordered_scores.index.astype(str))
        row["best_marker_score_label"] = str(ordered_scores.index[0])
        row["best_marker_score"] = float(ordered_scores.iloc[0])
        row["second_marker_score_label"] = str(ordered_scores.index[1])
        row["second_marker_score"] = float(ordered_scores.iloc[1])
        row["score_margin"] = float(ordered_scores.iloc[0] - ordered_scores.iloc[1])
        top_set = set(top300)
        for label, genes in used_markers.items():
            supported = [gene for gene in genes if gene in top_set]
            row[f"deg_support__{label}"] = ";".join(supported)
            row[f"n_deg_support__{label}"] = len(supported)
        review_rows.append(row)
    pd.DataFrame(review_rows).to_csv(STAGE / "broad_annotation_evidence_review.csv", index=False)

    backend = pd.DataFrame(
        [
            {
                "step": "canonical_marker_score_genes",
                "planned_backend": "rapids-singlecell GPU",
                "attempted_backend": "rapids-singlecell GPU",
                "status": "completed",
                "expression_source": "adata.raw normalized_log",
                "fallback_backend": "",
                "random_seed": SEED,
            },
            {
                "step": "raw_cluster_t_test_rank_genes_groups",
                "planned_backend": "rapids-singlecell GPU",
                "attempted_backend": "rapids-singlecell GPU",
                "status": "completed",
                "expression_source": "adata.raw normalized_log",
                "fallback_backend": "",
                "random_seed": SEED,
            },
        ]
    )
    backend.to_csv(STAGE / "01_gpu_backend_capability_summary.csv", index=False)
    completion = {
        "status": "completed",
        "n_cells": int(adata.n_obs),
        "n_raw_clusters": len(clusters),
        "n_raw_genes": int(adata.raw.n_vars),
        "n_broad_marker_panels": len(MARKERS),
        "deg_csv_count": len(raw_degs),
        "deg_rows_per_csv": int(adata.raw.n_vars),
        "score_backend": "rapids-singlecell GPU",
        "deg_backend": "rapids-singlecell GPU",
        "gpu_total_bytes": int(total_gpu),
        "gpu_free_before_bytes": int(free_before),
        "gpu_free_after_bytes": int(free_after),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (STAGE / "01_broad_annotation_evidence_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    versions = {
        package: version(package)
        for package in ("anndata", "rapids-singlecell", "cupy-cuda12x", "numpy", "pandas", "scipy")
    }
    versions.update({"python": platform.python_version(), "seed": SEED, "environment": "uv_envs/rapids/.venv"})
    (STAGE / "01_package_versions.json").write_text(
        json.dumps(versions, indent=2) + "\n", encoding="utf-8"
    )
    (STAGE / "01_evidence_readme.txt").write_text(
        "GPU broad-annotation evidence for the user-selected PCs25/NN50/res0.3 clean rerun.\n"
        f"Canonical marker scores and full t-test DEGs were recomputed from adata.raw for all {len(clusters)} new clusters.\n"
        "No old cluster-to-cell-type mapping was reused and no biological label was applied by this evidence step.\n",
        encoding="utf-8",
    )
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    os.replace(STAGE, TABLE)
    print(json.dumps(completion, ensure_ascii=False), flush=True)
    del adata
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()

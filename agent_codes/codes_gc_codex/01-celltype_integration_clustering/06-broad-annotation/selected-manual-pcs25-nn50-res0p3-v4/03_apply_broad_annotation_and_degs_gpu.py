#!/usr/bin/env python3
"""Apply evidence-reviewed broad labels and generate final GPU raw DEGs."""

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

SEED = 42
TASK = "selected-manual-pcs25-nn50-res0p3-v4"
RAW_KEY = "leiden_res0p3"
PARAMETER_TOKEN = "pcs25_nn50_res0p3"
random.seed(SEED)
np.random.seed(SEED)

WF = Path(__file__).resolve().parents[4]
INPUT = WF / f"h5ad/01-celltype_integration_clustering/05-clustering-parameter-search/{TASK}/adata_inte.h5ad"
TABLE = WF / f"tables/01-celltype_integration_clustering/06-broad-annotation/{TASK}"
OUTPUT_DIR = WF / f"h5ad/01-celltype_integration_clustering/06-broad-annotation/{TASK}"
OUTPUT = OUTPUT_DIR / "adata_anno.h5ad"
FIG = WF / f"figures/01-celltype_integration_clustering/06-broad-annotation/{TASK}"
STAGE_H5AD = OUTPUT_DIR.with_name(f".{TASK}-annotation.staging")
STAGE_TABLE = TABLE.parent / f".{TASK}-annotation.staging"
STAGE_FIG = FIG.with_name(f".{TASK}-annotation.staging")

MAPPING = {
    "0": "Myeloid Cells",
    "1": "Stromal Cells",
    "2": "Myeloid Cells",
    "3": "B Cells",
    "4": "T Cells",
    "5": "Epithelial Cells",
    "6": "T Cells",
    "7": "Myeloid Cells",
    "8": "Epithelial Cells",
    "9": "Endothelial Cells",
    "10": "Myeloid Cells",
    "11": "T Cells",
    "12": "Plasma Cells",
    "13": "Plasma Cells",
    "14": "Myeloid Cells",
}

MAPPING_REASON = {
    "0": "mixed myeloid/mast cluster; cell-level plurality Myeloid 0.587 versus Mast 0.268, with FCER1G support",
    "1": "stromal marker-score maximum and COL1A1/COL1A2/COL3A1/DCN/LUM/COL6A1 support",
    "2": "myeloid marker-score maximum, cell-level Myeloid 0.720, and FCER1G support",
    "3": "B-cell marker-score maximum and MS4A1/CD79A/CD37/CD74/HLA-DRA support",
    "4": "T-cell marker-score maximum, cell-level T 0.662, and IL7R support",
    "5": "epithelial marker-score maximum and PDX1/INSM1 support",
    "6": "T-cell marker-score maximum, cell-level T 0.498, and CD3D/IL7R support",
    "7": "myeloid marker-score maximum, cell-level Myeloid 0.711, and FCER1G support",
    "8": "epithelial marker-score maximum and EPCAM/KRT8/KRT19/KRT20 support",
    "9": "mixed endothelial/myeloid/B cluster; cell-level Endothelial plurality 0.379 and VWF/EMCN/ENG/ESAM vascular DEG support",
    "10": "mixed myeloid/epithelial cluster; cell-level Myeloid plurality 0.544 and FCER1G support",
    "11": "T-cell marker-score maximum, cell-level T 0.746, and CD3D/IL7R support",
    "12": "plasma marker-score maximum, cell-level Plasma 0.671, and MZB1/SDC1 support",
    "13": "plasma marker-score maximum, cell-level Plasma 0.966, and MZB1/SDC1 support",
    "14": "myeloid marker-score maximum, cell-level Myeloid 0.706, and FCER1G support",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_existing() -> None:
    final_table_outputs = [
        TABLE / "03_broad_annotation_mapping_agent_v1.csv",
        TABLE / "03_broad_annotation_completion.json",
        TABLE / f"degs_leiden_coarse_{PARAMETER_TOKEN}",
    ]
    existing = [
        str(path)
        for path in (OUTPUT_DIR, FIG, STAGE_H5AD, STAGE_TABLE, STAGE_FIG, *final_table_outputs)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"Refusing annotation overwrite/reuse: {existing}")


def export_rank_result(adata: ad.AnnData, key: str, out_dir: Path) -> int:
    result = adata.uns[key]
    out_dir.mkdir(parents=True, exist_ok=False)
    count = 0
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
        safe = str(group).replace(" ", "_").replace("/", "_")
        frame.to_csv(
            out_dir / f"{safe}_degs_leiden_coarse_{PARAMETER_TOKEN}.csv",
            index=False,
        )
        count += 1
    return count


def main() -> None:
    started = time.time()
    refuse_existing()
    evidence = pd.read_csv(TABLE / "broad_annotation_evidence_review.csv")
    cell_props = pd.read_csv(TABLE / "02_cell_level_best_marker_label_proportions.csv", index_col=0)
    marker_table = pd.read_csv(TABLE / "broad_marker_panel_availability.csv")
    marker_sets = {
        row.cell_type: str(row.available_markers).split(";")
        for row in marker_table.itertuples(index=False)
    }
    if set(evidence["cluster"].astype(str)) != set(MAPPING):
        raise RuntimeError("Evidence clusters do not exactly match the explicit mapping")

    adata = ad.read_h5ad(INPUT)
    clusters = adata.obs[RAW_KEY].astype(str)
    if set(clusters.unique()) != set(MAPPING) or adata.raw is None:
        raise RuntimeError("Selected clustering input violates mapping/raw contract")
    observed_labels = [label for label in marker_sets if label in set(MAPPING.values())]
    adata.obs["leiden_coarse"] = pd.Categorical(
        clusters.map(MAPPING), categories=observed_labels
    )
    adata.obs["cell_type"] = adata.obs["leiden_coarse"].copy()
    if adata.obs["leiden_coarse"].isna().any():
        raise RuntimeError("Broad mapping left missing labels")

    mapping_rows = []
    evidence_by_cluster = evidence.assign(cluster=evidence["cluster"].astype(str)).set_index("cluster")
    cell_props.index = cell_props.index.astype(str)
    for cluster in sorted(MAPPING, key=int):
        assigned = MAPPING[cluster]
        row = evidence_by_cluster.loc[cluster]
        mapping_rows.append(
            {
                "cluster": cluster,
                "leiden_coarse": assigned,
                "cell_type": assigned,
                "n_cells": int(clusters.eq(cluster).sum()),
                "best_mean_marker_score_label": row["best_marker_score_label"],
                "best_mean_marker_score": float(row["best_marker_score"]),
                "assigned_cell_level_best_fraction": float(cell_props.loc[cluster, assigned]),
                "top30_positive_degs": row["top30_positive_degs"],
                "mapping_reason": MAPPING_REASON[cluster],
                "mapping_method": "explicit evidence-reviewed canonical marker and raw DEG annotation",
                "approved_for_apply": True,
                "mapping_version": "pcs25_nn50_res0p3_v4",
            }
        )

    free_before, total_gpu = cp.cuda.runtime.memGetInfo()
    rsc.get.anndata_to_GPU(adata, convert_all=True)
    rsc.tl.rank_genes_groups(
        adata,
        groupby="leiden_coarse",
        method="t-test",
        use_raw=True,
        n_genes=adata.raw.n_vars,
        key_added="degs_leiden_coarse",
    )
    rsc.get.anndata_to_CPU(adata, convert_all=True)
    free_after, _ = cp.cuda.runtime.memGetInfo()

    STAGE_TABLE.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(mapping_rows).to_csv(
        STAGE_TABLE / "03_broad_annotation_mapping_agent_v1.csv", index=False
    )
    final_deg_dir = STAGE_TABLE / f"degs_leiden_coarse_{PARAMETER_TOKEN}"
    deg_csv_count = export_rank_result(adata, "degs_leiden_coarse", final_deg_dir)
    composition = (
        adata.obs["leiden_coarse"]
        .astype(str)
        .value_counts()
        .rename_axis("leiden_coarse")
        .reset_index(name="n_cells")
    )
    composition.to_csv(STAGE_TABLE / "03_broad_cell_type_composition.csv", index=False)
    pd.crosstab(adata.obs["leiden_coarse"], adata.obs["series"]).to_csv(
        STAGE_TABLE / "03_broad_cell_type_by_series_counts.csv"
    )
    pd.crosstab(adata.obs["leiden_coarse"], adata.obs["status"]).to_csv(
        STAGE_TABLE / "03_broad_cell_type_by_status_counts.csv"
    )
    pd.DataFrame(
        [
            {
                "cell_type": label,
                "markers": ";".join(genes),
                "n_markers": len(genes),
                "observed_in_annotation": label in observed_labels,
            }
            for label, genes in marker_sets.items()
        ]
    ).to_csv(STAGE_TABLE / "03_broad_marker_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "step": "final_leiden_coarse_t_test_rank_genes_groups",
                "planned_backend": "rapids-singlecell GPU",
                "attempted_backend": "rapids-singlecell GPU",
                "status": "completed",
                "expression_source": "adata.raw normalized_log",
                "fallback_backend": "",
                "random_seed": SEED,
                "gpu_total_bytes": int(total_gpu),
                "gpu_free_before_bytes": int(free_before),
                "gpu_free_after_bytes": int(free_after),
            }
        ]
    ).to_csv(STAGE_TABLE / "03_gpu_backend_capability_summary.csv", index=False)

    final_deg_relative = TABLE / f"degs_leiden_coarse_{PARAMETER_TOKEN}"
    raw_deg_relative = TABLE / f"degs_{RAW_KEY}_{PARAMETER_TOKEN}"
    adata.uns["broad_annotation"] = {
        "raw_cluster_column": RAW_KEY,
        "mapping_file": str((TABLE / "03_broad_annotation_mapping_agent_v1.csv").relative_to(WF)),
        "mapping_method": "explicit evidence-reviewed canonical marker and raw DEG annotation",
        "observed_labels_only": True,
        "absent_unforced_labels": ["Mast Cells"],
        "raw_cluster_deg_dir": str(raw_deg_relative.relative_to(WF)),
        "leiden_coarse_deg_dir": str(final_deg_relative.relative_to(WF)),
        "parameter_token": PARAMETER_TOKEN,
        "expression_layer_for_degs": "adata.raw normalized_log",
        "deg_backend": "rapids-singlecell GPU",
        "random_seed": SEED,
    }

    STAGE_H5AD.mkdir(parents=True, exist_ok=False)
    staged_h5ad = STAGE_H5AD / OUTPUT.name
    adata.write_h5ad(staged_h5ad, compression="gzip")

    STAGE_FIG.mkdir(parents=True, exist_ok=False)
    sc.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
    sc.settings.figdir = STAGE_FIG
    plot_markers = {label: marker_sets[label][:3] for label in observed_labels}
    for suffix in ("pdf", "svg"):
        sc.pl.umap(
            adata,
            color=[RAW_KEY, "leiden_coarse"],
            ncols=2,
            wspace=0.45,
            frameon=True,
            show=False,
            save=f"_broad_annotation_pcs25_nn50_res0p3.{suffix}",
        )
        sc.pl.dotplot(
            adata,
            var_names=plot_markers,
            groupby="leiden_coarse",
            use_raw=True,
            standard_scale="var",
            show=False,
            save=f"_broad_markers_pcs25_nn50_res0p3.{suffix}",
        )

    pdf = sorted(STAGE_FIG.glob("*.pdf"))
    svg = sorted(STAGE_FIG.glob("*.svg"))
    png = sorted(STAGE_FIG.glob("*.png"))
    completion = {
        "all_passed": True,
        "n_cells": int(adata.n_obs),
        "n_raw_clusters": len(MAPPING),
        "n_broad_labels": int(adata.obs["leiden_coarse"].nunique()),
        "broad_labels": observed_labels,
        "absent_unforced_labels": ["Mast Cells"],
        "mapping_complete": not adata.obs["leiden_coarse"].isna().any(),
        "raw_present": adata.raw is not None,
        "final_deg_csv_count": deg_csv_count,
        "final_deg_rows_per_csv": int(adata.raw.n_vars),
        "pdf_count": len(pdf),
        "svg_count": len(svg),
        "png_count": len(png),
        "output_sha256": sha256(staged_h5ad),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if deg_csv_count != len(observed_labels) or len(pdf) != 2 or len(svg) != 2 or png:
        raise RuntimeError(f"Broad annotation output audit failed: {completion}")
    (STAGE_TABLE / "03_broad_annotation_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    versions = {
        package: version(package)
        for package in ("anndata", "scanpy", "rapids-singlecell", "cupy-cuda12x", "numpy", "pandas", "scipy")
    }
    versions.update({"python": platform.python_version(), "seed": SEED, "environment": "uv_envs/rapids/.venv"})
    (STAGE_TABLE / "03_package_versions.json").write_text(
        json.dumps(versions, indent=2) + "\n", encoding="utf-8"
    )
    (STAGE_TABLE / "03_annotation_readme.txt").write_text(
        "Evidence-reviewed broad annotation for the user-selected PCs25/NN50/res0.3 clean rerun.\n"
        "All 15 raw clusters were explicitly mapped from newly computed canonical marker and adata.raw DEG evidence.\n"
        "Mast Cells were not forced because no new raw cluster had sufficient whole-cluster mast evidence.\n"
        "Final broad-label DEGs used rapids-singlecell GPU with adata.raw; figures are PDF/SVG only.\n",
        encoding="utf-8",
    )

    for source in STAGE_TABLE.iterdir():
        target = TABLE / source.name
        if target.exists():
            raise FileExistsError(target)
        os.replace(source, target)
    STAGE_TABLE.rmdir()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    os.replace(STAGE_FIG, FIG)
    os.replace(STAGE_H5AD, OUTPUT_DIR)
    print(json.dumps(completion, ensure_ascii=False), flush=True)
    del adata
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()

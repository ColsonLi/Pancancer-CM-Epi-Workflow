#!/usr/bin/env python3
"""Apply marker-supported broad labels, export round-2 DEGs, and save adata_anno."""

from __future__ import annotations

import importlib.metadata
import json
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


SEED = 42
PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
INPUT_H5AD = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/05-clustering-parameter-search/selected/adata_inte.h5ad"
)
SELECTED_TABLE = (
    PROJECT_ROOT
    / "tables/01-celltype_integration_clustering/05-clustering-parameter-search/selected_clustering.csv"
)
TABLE_ROOT = PROJECT_ROOT / "tables/01-celltype_integration_clustering/06-broad-annotation"
FIGURE_ROOT = PROJECT_ROOT / "figures/01-celltype_integration_clustering/06-broad-annotation"
PENDING_FIGURE_ROOT = (
    PROJECT_ROOT / "figures/01-celltype_integration_clustering/06-broad-annotation_pending_validation"
)
H5AD_DIR = PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/06-broad-annotation"
PENDING_H5AD_DIR = (
    PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/06-broad-annotation_pending_validation"
)
OUTPUT_H5AD = H5AD_DIR / "adata_anno.h5ad"
PENDING_H5AD = PENDING_H5AD_DIR / "adata_anno.h5ad"

ORDER = [
    "Epithelial Cells",
    "T/NK Cells",
    "Myeloid Cells",
    "B Cells",
    "Endothelial Cells",
    "Stromal Cells",
    "Mast Cells",
]

ANNOTATION_ROWS = [
    ("0", "Myeloid Cells", "TYROBP;FCER1G;AIF1;CD68;CTSS", "macrophage-rich myeloid program", "high"),
    ("1", "Epithelial Cells", "MKI67;TOP2A;STMN1 plus EPCAM;KRT19;KRT8;KRT18;MUC1", "cycling compartment with the strongest lineage-panel support for epithelial identity", "moderate"),
    ("2", "T/NK Cells", "IL7R;CD3D;CD3E;CD3G;CD2", "CD3-positive helper/memory T-cell branch of the combined T/NK lineage", "high"),
    ("3", "Endothelial Cells", "SPARCL1;RAMP2;GNG11;PLVAP;VWF;EGFL7", "vascular endothelial program", "high"),
    ("4", "Stromal Cells", "DCN;COL1A1;COL1A2;LUM;COL3A1", "fibroblast extracellular-matrix program", "high"),
    ("5", "B Cells", "MZB1;CD79A;FCRL5;TNFRSF17;CD38", "plasma-cell branch of the B-cell lineage", "high"),
    ("6", "Epithelial Cells", "CPA1;CPA2;PRSS1;CELA3A;CTRB1", "pancreatic acinar epithelial program", "high"),
    ("7", "Epithelial Cells", "KRT19;KRT8;KRT18;MUC1;EPCAM;S100P", "ductal/tumor epithelial program", "high"),
    ("8", "Stromal Cells", "CALD1;MYL9;TAGLN;ACTA2;PDGFRB", "pericyte/smooth-muscle stromal program", "high"),
    ("9", "Myeloid Cells", "S100A8;S100A9;LST1;FCER1G;LYZ", "inflammatory myeloid program", "high"),
    ("10", "B Cells", "MS4A1;CD79A;CD79B;CD74;BANK1", "conventional B-cell program", "high"),
    ("11", "Mast Cells", "TPSAB1;CPA3;KIT;MS4A2;HPGDS", "canonical mast-cell program", "high"),
    ("12", "T/NK Cells", "CD3D;CD3E;NKG7;KLRD1;GNLY;PRF1", "mixed cytotoxic T/NK compartment retained as a combined broad lineage", "high"),
    ("13", "Epithelial Cells", "CFTR;SLC4A4;CLDN10;KRT8;KRT18;EPCAM", "ductal/centroacinar epithelial program", "high"),
    ("14", "Epithelial Cells", "INS;CHGA;CHGB;SLC30A8;SCGN;PCSK1", "pancreatic endocrine epithelial program", "high"),
    ("15", "Epithelial Cells", "CELA3A;PRSS1;CPA1;CPA2;CTRB1", "pancreatic acinar epithelial program", "high"),
    ("16", "Myeloid Cells", "TYROBP;CTSS;AIF1;LST1;LYZ;FCER1G", "antigen-presenting myeloid program", "high"),
    ("17", "Myeloid Cells", "APOE;APOC1;C1QA;C1QB;C1QC;SPP1", "SPP1/APOC1 macrophage program", "high"),
]

MARKERS = {
    "Epithelial Cells": ["EPCAM", "KRT19", "KRT8"],
    "T/NK Cells": ["CD3D", "NKG7", "KLRD1"],
    "Myeloid Cells": ["LYZ", "TYROBP", "LST1"],
    "B Cells": ["CD79A", "CD74", "MS4A1"],
    "Endothelial Cells": ["PECAM1", "VWF", "EMCN"],
    "Stromal Cells": ["COL1A1", "DCN", "COL3A1"],
    "Mast Cells": ["TPSAB1", "CPA3", "KIT"],
}

EVIDENCE_MARKERS = {
    "Epithelial Cells": [
        "EPCAM", "KRT19", "KRT8", "KRT18", "MUC1", "KRT7",
        "PRSS1", "CPA1", "CELA3A", "INS", "CHGA", "SLC30A8",
    ],
    "T/NK Cells": ["CD3D", "CD3E", "CD3G", "IL7R", "NKG7", "KLRD1", "GNLY", "PRF1"],
    "Myeloid Cells": ["LYZ", "TYROBP", "LST1", "FCER1G", "AIF1", "CTSS"],
    "B Cells": ["CD79A", "MS4A1", "CD74", "CD37", "CD19", "CD79B", "MZB1"],
    "Endothelial Cells": ["PECAM1", "VWF", "EMCN", "KDR", "CLDN5", "EGFL7"],
    "Stromal Cells": ["COL1A1", "COL1A2", "DCN", "LUM", "COL3A1", "PDGFRB"],
    "Mast Cells": ["TPSAB1", "CPA3", "KIT", "MS4A2", "HPGDS", "LTC4S"],
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def package_version_lines() -> list[str]:
    packages = ["scanpy", "anndata", "numpy", "pandas", "scipy", "matplotlib"]
    code_files = [
        Path(__file__).resolve(),
        PROJECT_ROOT
        / "codes/01-celltype_integration_clustering/06-broad-annotation/01_export_raw_cluster_degs.py",
    ]
    lines = [
        "environment_category=rapids_with_scanpy_cpu_annotation",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"random_seed={SEED}",
        "code_files=" + ";".join(map(str, code_files)),
    ]
    lines.extend(f"{package}={importlib.metadata.version(package)}" for package in packages)
    return lines


def export_full_degs(
    adata: sc.AnnData,
    groupby: str,
    pending_dir: Path,
    final_dir: Path,
    file_token: str,
) -> list[dict[str, object]]:
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method="t-test",
        use_raw=True,
        n_genes=adata.raw.n_vars,
    )
    result = adata.uns["rank_genes_groups"]
    groups = [str(value) for value in adata.obs[groupby].cat.categories]
    pending_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for group in groups:
        frame = pd.DataFrame(
            {
                "gene": result["names"][group],
                "score": result["scores"][group],
                "logfoldchanges": result["logfoldchanges"][group],
                "pvals": result["pvals"][group],
                "pvals_adj": result["pvals_adj"][group],
            }
        )
        if frame.shape[0] != adata.raw.n_vars:
            raise ValueError(f"Group {group} post-annotation DEG table is not full-length")
        output_name = f"{safe_name(group)}_degs_{file_token}.csv"
        frame.to_csv(pending_dir / output_name, index=False)
        rows.append(
            {
                "group": group,
                "n_cells": int((adata.obs[groupby].astype(str) == group).sum()),
                "deg_rows": int(frame.shape[0]),
                "output_file": str(final_dir / output_name),
            }
        )
    if len(list(pending_dir.glob("*.csv"))) != len(groups):
        raise FileNotFoundError("Post-annotation DEG output set is incomplete")
    return rows


def build_marker_evidence(
    adata: sc.AnnData,
    cluster_key: str,
    label_map: dict[str, str],
) -> pd.DataFrame:
    genes = list(
        dict.fromkeys(
            gene
            for panel_genes in EVIDENCE_MARKERS.values()
            for gene in panel_genes
            if gene in adata.raw.var_names
        )
    )
    expression = adata.raw[:, genes].X
    rows = []
    cluster_labels = adata.obs[cluster_key].astype(str).to_numpy()
    for cluster in sorted(label_map, key=int):
        mask = cluster_labels == cluster
        cluster_expression = expression[mask]
        means = np.asarray(cluster_expression.mean(axis=0)).ravel()
        if sparse.issparse(cluster_expression):
            positive_fraction = np.asarray((cluster_expression > 0).mean(axis=0)).ravel()
        else:
            positive_fraction = (cluster_expression > 0).mean(axis=0)
        gene_metrics = dict(zip(genes, zip(means, positive_fraction, strict=True), strict=True))
        for panel, panel_genes in EVIDENCE_MARKERS.items():
            for gene in panel_genes:
                if gene not in gene_metrics:
                    continue
                mean_expression, fraction = gene_metrics[gene]
                rows.append(
                    {
                        "cluster": cluster,
                        "proposed_leiden_coarse": label_map[cluster],
                        "marker_panel": panel,
                        "gene": gene,
                        "mean_log_normalized_expression": float(mean_expression),
                        "pct_expressing": float(fraction * 100),
                        "n_cells": int(mask.sum()),
                        "expression_source": "adata.raw normalize_total_1e4 log1p all shared genes",
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    if not INPUT_H5AD.is_file() or not SELECTED_TABLE.is_file():
        raise FileNotFoundError("Selected H5AD or selected-clustering table is missing")
    selected = pd.read_csv(SELECTED_TABLE)
    if selected.shape[0] != 1:
        raise ValueError("Selected-clustering table must contain exactly one row")
    selected = selected.iloc[0]
    cluster_key = str(selected["final_raw_cluster_column"])
    n_pcs = int(selected["n_pcs"])
    n_neighbors = int(selected["n_neighbors"])
    resolution = float(selected["resolution"])
    resolution_text = str(resolution).replace(".", "p")
    deg_token = f"leiden_coarse_pcs{n_pcs}_nn{n_neighbors}_res{resolution_text}"
    final_round2_dir = TABLE_ROOT / f"degs_{deg_token}"
    pending_round2_dir = TABLE_ROOT / "degs_leiden_coarse_pending_validation"
    root_outputs = [
        TABLE_ROOT / "broad_annotation_candidate_table.csv",
        TABLE_ROOT / "broad_annotation_mapping.csv",
        TABLE_ROOT / "broad_annotation_marker_panel_evidence.csv",
        TABLE_ROOT / "broad_annotation_review_parameters.csv",
        TABLE_ROOT / "broad_lineage_counts.csv",
        TABLE_ROOT / "broad_lineage_sample_counts.csv",
        TABLE_ROOT / "broad_lineage_sample_fractions.csv",
        TABLE_ROOT / "02_broad_annotation_parameters.csv",
        TABLE_ROOT / "02_broad_annotation_validation.json",
        TABLE_ROOT / "broad_annotation_execution_attempts.csv",
        TABLE_ROOT / "02_package_versions.txt",
        TABLE_ROOT / "readme.txt",
    ]
    for path in [
        H5AD_DIR,
        PENDING_H5AD_DIR,
        FIGURE_ROOT,
        PENDING_FIGURE_ROOT,
        final_round2_dir,
        pending_round2_dir,
        *root_outputs,
    ]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite broad-annotation artifact: {path}")

    started = time.time()
    adata = sc.read_h5ad(INPUT_H5AD)
    if adata.raw is None:
        raise ValueError("Broad annotation requires adata.raw normalized/log expression")
    if cluster_key not in adata.obs or "X_umap" not in adata.obsm:
        raise KeyError("Selected raw clusters or selected UMAP are missing")
    if "leiden_coarse" in adata.obs or "cell_type" in adata.obs:
        raise ValueError("Selected clustering input already contains biological broad labels")
    observed_clusters = set(adata.obs[cluster_key].astype(str).unique())
    mapping_clusters = {row[0] for row in ANNOTATION_ROWS}
    if observed_clusters != mapping_clusters:
        raise ValueError(
            f"Broad annotation mapping does not exactly cover observed clusters: "
            f"observed={sorted(observed_clusters)}, mapped={sorted(mapping_clusters)}"
        )

    mapping = pd.DataFrame(
        ANNOTATION_ROWS,
        columns=["cluster", "leiden_coarse", "top_supporting_markers", "marker_rationale", "confidence"],
    )
    mapping["review_depth"] = 50
    mapping["selection_source"] = "user_authorized_agent_led_annotation_2026-08-06"
    cluster_counts = adata.obs[cluster_key].astype(str).value_counts()
    mapping["n_cells"] = mapping["cluster"].map(cluster_counts).astype(int)
    candidate = mapping.rename(columns={"leiden_coarse": "proposed_leiden_coarse"}).copy()
    candidate["needs_user_confirmation"] = False
    candidate["agent_decision_note"] = (
        "User explicitly authorized agent-led decisions; labels use current raw-cluster DEGs and canonical markers"
    )

    label_map = mapping.set_index("cluster")["leiden_coarse"].to_dict()
    marker_evidence = build_marker_evidence(adata, cluster_key, label_map)
    adata.obs["leiden_coarse"] = adata.obs[cluster_key].astype(str).map(label_map)
    if adata.obs["leiden_coarse"].isna().any():
        raise ValueError("Broad label assignment produced missing values")
    adata.obs["leiden_coarse"] = pd.Categorical(
        adata.obs["leiden_coarse"], categories=ORDER, ordered=True
    )
    adata.obs["cell_type"] = adata.obs["leiden_coarse"].copy()
    adata.obs["broad_annotation_source"] = "raw_cluster_DEG_top50_plus_canonical_markers"
    confidence_map = mapping.set_index("cluster")["confidence"].to_dict()
    adata.obs["broad_annotation_confidence"] = (
        adata.obs[cluster_key].astype(str).map(confidence_map).astype("category")
    )

    post_deg_rows = export_full_degs(
        adata,
        "leiden_coarse",
        pending_round2_dir,
        final_round2_dir,
        deg_token,
    )

    PENDING_FIGURE_ROOT.mkdir(parents=True, exist_ok=False)
    sc.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
    sc.settings.figdir = PENDING_FIGURE_ROOT
    sc.pl.umap(adata, color="leiden_coarse", show=False, save="_leiden_coarse.pdf")
    sc.pl.dotplot(
        adata,
        var_names=MARKERS,
        groupby="leiden_coarse",
        standard_scale="var",
        use_raw=True,
        show=False,
        save="all_leiden_coarse.pdf",
    )
    expected_figures = [
        PENDING_FIGURE_ROOT / "umap_leiden_coarse.pdf",
        PENDING_FIGURE_ROOT / "dotplot_all_leiden_coarse.pdf",
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in expected_figures):
        raise FileNotFoundError("Broad UMAP or marker dotplot was not written")

    counts = (
        adata.obs["leiden_coarse"].value_counts(sort=False).rename("n_cells").reset_index()
    )
    sample_counts = pd.crosstab(adata.obs["sample"], adata.obs["leiden_coarse"], dropna=False)
    sample_fractions = sample_counts.div(sample_counts.sum(axis=1), axis=0)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    sample_fractions.plot(kind="bar", stacked=True, ax=axis, width=0.9)
    axis.set_xlabel("Sample")
    axis.set_ylabel("Cell fraction")
    axis.legend(title="Broad lineage", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    figure.tight_layout()
    for suffix in ["pdf", "svg"]:
        figure.savefig(PENDING_FIGURE_ROOT / f"broad_lineage_composition_by_sample.{suffix}")
    plt.close(figure)

    adata.uns["broad_annotation_contract"] = {
        "raw_cluster_column": cluster_key,
        "annotation_column": "leiden_coarse",
        "cell_type_initialized_from": "leiden_coarse",
        "review_depth": 50,
        "marker_source": "saved full raw-cluster DEG CSVs plus canonical broad markers",
        "selection_source": "user_authorized_agent_led_annotation_2026-08-06",
        "round1_deg_use_raw": True,
        "round2_deg_use_raw": True,
        "round2_deg_directory": str(final_round2_dir),
    }

    PENDING_H5AD_DIR.mkdir(parents=True, exist_ok=False)
    adata.write_h5ad(PENDING_H5AD, compression="gzip", compression_opts=1)
    pending = sc.read_h5ad(PENDING_H5AD, backed="r")
    pending_errors = []
    if pending.shape != adata.shape:
        pending_errors.append("shape mismatch")
    if pending.raw is None or pending.raw.n_vars != adata.raw.n_vars:
        pending_errors.append("raw missing or mismatched")
    if cluster_key not in pending.obs or "leiden_coarse" not in pending.obs or "cell_type" not in pending.obs:
        pending_errors.append("required label column missing")
    if set(pending.obs["leiden_coarse"].astype(str).unique()) != set(ORDER):
        pending_errors.append("broad label vocabulary mismatch")
    if "X_umap" not in pending.obsm or "neighbors" not in pending.uns:
        pending_errors.append("selected UMAP or neighbors missing")
    pending.file.close()
    del pending
    if pending_errors:
        raise ValueError(f"Pending broad-annotated H5AD failed validation: {pending_errors}")

    PENDING_H5AD_DIR.replace(H5AD_DIR)
    PENDING_FIGURE_ROOT.replace(FIGURE_ROOT)
    pending_round2_dir.replace(final_round2_dir)

    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(TABLE_ROOT / "broad_annotation_candidate_table.csv", index=False)
    mapping.to_csv(TABLE_ROOT / "broad_annotation_mapping.csv", index=False)
    marker_evidence.to_csv(
        TABLE_ROOT / "broad_annotation_marker_panel_evidence.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "annotation_level": "broad_raw_clusters",
                "uniform_review_depth": 50,
                "expansion_needed": False,
                "evidence_source": "saved full raw-cluster DEG CSVs plus canonical broad markers",
                "agent_led_authorization": "user said 你自己选一下 on 2026-08-06",
            }
        ]
    ).to_csv(TABLE_ROOT / "broad_annotation_review_parameters.csv", index=False)
    counts.to_csv(TABLE_ROOT / "broad_lineage_counts.csv", index=False)
    sample_counts.to_csv(TABLE_ROOT / "broad_lineage_sample_counts.csv")
    sample_fractions.to_csv(TABLE_ROOT / "broad_lineage_sample_fractions.csv")
    elapsed = time.time() - started
    parameters = {
        "step": "broad_annotation_round2_deg_and_h5ad",
        "input_h5ad": str(INPUT_H5AD),
        "output_h5ad": str(OUTPUT_H5AD),
        "raw_cluster_column": cluster_key,
        "annotation_column": "leiden_coarse",
        "cell_type_initialized": True,
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "n_obs": int(adata.n_obs),
        "n_vars_hvg_X": int(adata.n_vars),
        "n_vars_raw": int(adata.raw.n_vars),
        "n_raw_clusters": len(mapping),
        "n_broad_lineages": len(ORDER),
        "review_depth": 50,
        "round2_deg_method": "t-test",
        "round2_deg_use_raw": True,
        "top_markers_per_broad_dotplot_group": 3,
        "composition_plot_manual_reason": "Scanpy has no dedicated sample-composition plotting API",
        "random_seed": SEED,
        "elapsed_seconds": elapsed,
        "code_file": str(Path(__file__).resolve()),
    }
    pd.DataFrame([parameters]).to_csv(
        TABLE_ROOT / "02_broad_annotation_parameters.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "attempt": 1,
                "status": "completed",
                "error_summary": "",
                "biological_method_changed": False,
            },
        ]
    ).to_csv(TABLE_ROOT / "broad_annotation_execution_attempts.csv", index=False)
    (TABLE_ROOT / "02_package_versions.txt").write_text(
        "\n".join(package_version_lines()) + "\n", encoding="utf-8"
    )
    (TABLE_ROOT / "readme.txt").write_text(
        "Broad annotation and two-round DEG task.\n\n"
        f"Input selected clustering: {INPUT_H5AD}\n"
        f"Raw cluster column: {cluster_key}.\n"
        "Round 1 full raw-cluster DEGs were exported by 01_export_raw_cluster_degs.py.\n"
        "The 18 raw clusters were reviewed uniformly to top 50 DEGs and checked against canonical "
        "broad-lineage and pancreas epithelial marker panels.\n"
        "CD3-positive and cytotoxic lymphocytes remain combined as T/NK Cells at the broad level; "
        "Module 02 must resolve their subtypes by independent lineage reclustering.\n"
        "Round 2 full DEGs use adata.raw normalized/log all-gene expression with Scanpy t-test.\n"
        f"Output annotated object: {OUTPUT_H5AD}\n",
        encoding="utf-8",
    )
    validation = {
        **parameters,
        "mapping_exactly_covers_raw_clusters": True,
        "lineage_order": ORDER,
        "round2_groups": post_deg_rows,
        "round2_all_files_full_length": all(row["deg_rows"] == adata.raw.n_vars for row in post_deg_rows),
        "figures": [str(FIGURE_ROOT / path.name) for path in expected_figures]
        + [
            str(FIGURE_ROOT / "broad_lineage_composition_by_sample.pdf"),
            str(FIGURE_ROOT / "broad_lineage_composition_by_sample.svg"),
        ],
        "output_h5ad_readable": OUTPUT_H5AD.is_file(),
        "raw_cluster_column_preserved": cluster_key in adata.obs,
        "leiden_coarse_present": "leiden_coarse" in adata.obs,
        "cell_type_present": "cell_type" in adata.obs,
    }
    (TABLE_ROOT / "02_broad_annotation_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

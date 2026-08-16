#!/usr/bin/env python3
"""Prepare marker-reviewed candidate subtype tables without final annotation writes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lineage_workflow import PROJECT_ROOT


# These states were curated from the first 50 rows of every saved full DEG CSV.
SPECS: dict[str, dict[str, tuple[str, str, str]]] = {
    "epithelial": {
        "0": ("acinar digestive-enzyme", "Acinar Cells", "high"),
        "1": ("S100P CEACAM6 malignant-ductal-like", "Epithelial Cells", "high"),
        "2": ("basal mesenchymal-like ductal", "Epithelial Cells", "high"),
        "3": ("T-cell-like off-lineage", "T/NK Cells", "low"),
        "4": ("proliferative cycling", "Epithelial Cells", "high"),
        "5": ("poorly resolved transcriptional", "Epithelial Cells", "low"),
        "6": ("inflammatory basal-like ductal", "Epithelial Cells", "high"),
        "7": ("acinar digestive-enzyme", "Acinar Cells", "high"),
        "8": ("acinar digestive-enzyme", "Acinar Cells", "high"),
        "9": ("secretory ion-transport ductal", "Epithelial Cells", "high"),
        "10": ("ion-transport normal-ductal-like", "Epithelial Cells", "high"),
        "11": ("REG4 intestinal-like ductal", "Epithelial Cells", "high"),
        "12": ("endocrine neuroendocrine", "Endocrine Cells", "high"),
        "13": ("secretory malignant-ductal-like", "Epithelial Cells", "high"),
    },
    "t_nk": {
        "0": ("interferon-stimulated exhausted cytotoxic T", "T Cells", "high"),
        "1": ("GZMK CD8 effector-memory", "T Cells", "high"),
        "2": ("FOXP3 IL2RA regulatory T", "T Cells", "high"),
        "3": ("CCR7 TCF7 naive central-memory T", "T Cells", "high"),
        "4": ("acinar-like off-lineage", "Acinar Cells", "low"),
        "5": ("KLRB1 CCR6 CD4 memory-like", "T Cells", "high"),
        "6": ("KLRC2 cytotoxic NK", "NK Cells", "high"),
        "7": ("ZNF683 tissue-resident cytotoxic T", "T Cells", "high"),
        "8": ("FCGR3A FGFBP2 cytotoxic NK", "NK Cells", "high"),
    },
    "myeloid": {
        "0": ("CSF3R FCGR3B neutrophil", "Myeloid Cells", "high"),
        "1": ("epithelial-like off-lineage", "Epithelial Cells", "low"),
        "2": ("acinar-like off-lineage", "Acinar Cells", "low"),
        "3": ("T-cell-like off-lineage", "T/NK Cells", "low"),
        "4": ("IL1B immediate-early inflammatory monocyte", "Myeloid Cells", "high"),
        "5": ("low-complexity antigen-presenting myeloid", "Myeloid Cells", "medium"),
        "6": ("FCN1 VCAN classical monocyte", "Myeloid Cells", "high"),
        "7": ("CD1C CLEC10A cDC2-like", "Myeloid Cells", "high"),
        "8": ("C1Q antigen-presenting macrophage", "Myeloid Cells", "high"),
        "9": ("SPP1 lipid-associated macrophage", "Myeloid Cells", "high"),
        "10": ("FCGR3A nonclassical monocyte", "Myeloid Cells", "high"),
    },
    "b": {
        "0": ("CD27 TNFRSF13B memory B", "B Cells", "high"),
        "1": ("T-cell-like off-lineage", "T/NK Cells", "low"),
        "2": ("NR4A immediate-early activated B", "B Cells", "high"),
        "3": ("TCL1A FCER2 naive B", "B Cells", "high"),
        "4": ("XBP1 secretory plasma cell", "B Cells", "high"),
        "5": ("acinar-like off-lineage", "Acinar Cells", "low"),
        "6": ("epithelial-like off-lineage", "Epithelial Cells", "low"),
        "7": ("RGS13 MEF2B germinal-center-like B", "B Cells", "high"),
        "8": ("JCHAIN PRDM1 plasma cell", "B Cells", "high"),
    },
    "endothelial": {
        "0": ("CA4 CD36 capillary with acinar-like signal", "Endothelial Cells", "medium"),
        "1": ("COL15A1 matrix capillary-like", "Endothelial Cells", "high"),
        "2": ("ACKR1 venous endothelial", "Endothelial Cells", "high"),
        "3": ("epithelial-like off-lineage", "Epithelial Cells", "low"),
        "4": ("RGS5 PDGFRB pericyte-like off-lineage", "Stromal Cells", "low"),
        "5": ("T-cell-like off-lineage", "T/NK Cells", "low"),
        "6": ("FABP4 RBP7 capillary with erythroid signal", "Endothelial Cells", "medium"),
        "7": ("SEMA3G EFNB2 GJA5 arterial endothelial", "Endothelial Cells", "high"),
        "8": ("PROX1 CCL21 lymphatic endothelial", "Endothelial Cells", "high"),
        "9": ("basement-membrane angiogenic endothelial", "Endothelial Cells", "high"),
        "10": ("myeloid-like off-lineage", "Myeloid Cells", "low"),
        "11": ("fibroblast-like off-lineage", "Stromal Cells", "low"),
        "12": ("acinar-like off-lineage", "Acinar Cells", "low"),
    },
    "stromal": {
        "0": ("T-cell-like off-lineage", "T/NK Cells", "low"),
        "1": ("low-complexity mixed fibroblast", "Stromal Cells", "low"),
        "2": ("epithelial-like off-lineage", "Epithelial Cells", "low"),
        "3": ("MYH11 ACTA2 contractile smooth-muscle pericyte", "Stromal Cells", "high"),
        "4": ("CTHRC1 COL11A1 matrix CAF-like", "Stromal Cells", "high"),
        "5": ("C7 complement inflammatory fibroblast", "Stromal Cells", "high"),
        "6": ("OGN DPT adventitial fibroblast", "Stromal Cells", "high"),
        "7": ("acinar-like off-lineage", "Acinar Cells", "low"),
        "8": ("S100B PLP1 Schwann-like", "Stromal Cells", "high"),
    },
    "mast": {
        "0": ("TPSAB1 LTC4S canonical mast", "Mast Cells", "high"),
        "1": ("NR4A NF-kB immediate-early activated mast", "Mast Cells", "high"),
        "2": ("CMA1 HPGDS inflammatory mast", "Mast Cells", "high"),
        "3": ("epithelial stromal-like mixed off-lineage", "Epithelial/Stromal Cells", "low"),
        "4": ("T-cell-like off-lineage", "T/NK Cells", "low"),
        "5": ("plasma-cell-like off-lineage", "B Cells", "low"),
        "6": ("acinar-like off-lineage", "Acinar Cells", "low"),
        "7": ("CLC FCER1A basophil-like", "Basophil/DC-like Cells", "medium"),
        "8": ("HLA-II myeloid-like off-lineage", "Myeloid Cells", "low"),
    },
}

PREFIXES = {
    "epithelial": "Epi",
    "myeloid": "Mye",
    "b": "B",
    "endothelial": "Endo",
    "stromal": "Str",
    "mast": "Mast",
}


def evidence_key(row: pd.Series) -> tuple[float, float, float, float]:
    return (
        float(np.nan_to_num(row["pvals_adj"], nan=np.inf)),
        float(np.nan_to_num(row["pvals"], nan=np.inf)),
        -float(np.nan_to_num(row["score"], nan=-np.inf)),
        -float(np.nan_to_num(row["logfoldchanges"], nan=-np.inf)),
    )


def prepare_candidate_annotation(slug: str, code_file: Path) -> None:
    if slug not in SPECS:
        raise KeyError(f"No reviewed annotation specification for {slug}")
    table_dir = (
        PROJECT_ROOT
        / f"tables/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/{slug}"
    )
    manifest_path = table_dir / "raw_cluster_deg_file_manifest.csv"
    output_csv = table_dir / "candidate_subtype_annotation.csv"
    parameter_json = table_dir / "candidate_annotation_review_parameters.json"
    for path in [output_csv, parameter_json]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite candidate annotation artifact: {path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_csv(manifest_path, dtype={"cluster": str})
    manifest["cluster"] = manifest["cluster"].astype(str)
    observed = set(manifest["cluster"])
    expected = set(SPECS[slug])
    if observed != expected:
        raise ValueError(
            f"Reviewed {slug} cluster specification mismatch: missing={sorted(observed - expected)}, "
            f"extra={sorted(expected - observed)}"
        )

    full_tables: dict[str, pd.DataFrame] = {}
    top_rows: dict[str, pd.DataFrame] = {}
    initial: dict[str, tuple[str, pd.Series]] = {}
    for row in manifest.itertuples(index=False):
        frame = pd.read_csv(row.output_csv)
        positive = frame.loc[frame["logfoldchanges"].gt(0) & frame["gene"].notna()].copy()
        if positive.empty:
            raise ValueError(f"Cluster {row.cluster} has no positive DEG")
        full_tables[row.cluster] = positive
        top_rows[row.cluster] = frame.head(50).copy()
        first = positive.iloc[0]
        initial[row.cluster] = (str(first["gene"]), first)

    gene_to_clusters: dict[str, list[str]] = {}
    for cluster, (gene, _) in initial.items():
        gene_to_clusters.setdefault(gene, []).append(cluster)

    selected_gene: dict[str, str] = {}
    rationale: dict[str, str] = {}
    reserved: set[str] = set()
    losers: list[tuple[str, str, str]] = []
    for gene, clusters in sorted(gene_to_clusters.items()):
        ranked = sorted(clusters, key=lambda cluster: evidence_key(initial[cluster][1]))
        winner = ranked[0]
        selected_gene[winner] = gene
        reserved.add(gene)
        if len(ranked) == 1:
            rationale[winner] = (
                "Selected the first positive-logfoldchange gene in the Scanpy-sorted full DEG table."
            )
        else:
            rationale[winner] = (
                f"Duplicate top gene {gene}; retained for cluster {winner} because it had the "
                "strongest adjusted-P/raw-P/score evidence among competing clusters."
            )
            for cluster in ranked[1:]:
                losers.append((cluster, gene, winner))

    for cluster, duplicate_gene, winner in losers:
        replacement = None
        for row in full_tables[cluster].itertuples(index=False):
            gene = str(row.gene)
            if gene not in reserved:
                replacement = gene
                break
        if replacement is None:
            raise ValueError(f"Could not resolve duplicate subtype gene for cluster {cluster}")
        selected_gene[cluster] = replacement
        reserved.add(replacement)
        rationale[cluster] = (
            f"Top positive DEG {duplicate_gene} was retained by stronger-evidence cluster {winner}; "
            f"selected next-ranked unused positive DEG {replacement} from this cluster's full table."
        )

    rows = []
    for record in manifest.itertuples(index=False):
        cluster = record.cluster
        state, marker_lineage, confidence = SPECS[slug][cluster]
        if slug == "t_nk":
            prefix = "NK" if marker_lineage == "NK Cells" else "T"
            prefix_rationale = (
                "Combined T/NK lineage; NK markers support NK prefix."
                if prefix == "NK"
                else "Combined T/NK lineage; T prefix retained, including off-lineage-like clusters."
            )
        else:
            prefix = PREFIXES[slug]
            prefix_rationale = (
                f"Retained current {slug} lineage prefix under the skill rule; marker evidence "
                "does not move cells between lineage h5ad objects."
            )
        top_positive = top_rows[cluster].loc[
            top_rows[cluster]["logfoldchanges"].gt(0), "gene"
        ].astype(str).head(10)
        original_top_gene = initial[cluster][0]
        gene = selected_gene[cluster]
        rows.append(
            {
                "cluster": cluster,
                "n_cells": int(record.n_cells),
                "proposed cell_subtype": f"{prefix}_{gene}",
                "functional_state": state,
                "selected_prefix": prefix,
                "most_significant_deg_gene": original_top_gene,
                "selected_subtype_gene": gene,
                "gene_selection_rationale": rationale[cluster],
                "top supporting markers": ";".join(top_positive),
                "marker-supported lineage": marker_lineage,
                "prefix rationale": prefix_rationale,
                "user_modified_marker": False,
                "confidence": confidence,
                "needs_user_confirmation": True,
                "review_depth": 50,
            }
        )
    candidate = pd.DataFrame(rows)
    candidate["cluster_sort"] = pd.to_numeric(candidate["cluster"], errors="coerce")
    candidate = candidate.sort_values(["cluster_sort", "cluster"]).drop(columns="cluster_sort")
    if candidate["proposed cell_subtype"].duplicated().any():
        raise ValueError("Candidate cell_subtype values are not unique")
    if candidate.shape[0] != len(expected):
        raise ValueError("Candidate annotation does not cover every raw cluster exactly once")
    candidate.to_csv(output_csv, index=False)
    review = {
        "lineage_slug": slug,
        "raw_cluster_count": int(candidate.shape[0]),
        "review_depth": 50,
        "review_depth_applied_equally_to_all_clusters": True,
        "source": "saved full raw-cluster DEG CSVs",
        "candidate_table": str(output_csv),
        "candidate_subtypes_unique": True,
        "all_clusters_covered_once": True,
        "final_mapping_written": False,
        "cell_subtype_written_to_h5ad": False,
        "requires_confirmation_gate": True,
        "code_file": str(code_file.resolve()),
    }
    parameter_json.write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(review, indent=2, ensure_ascii=False))

#!/usr/bin/env python3
"""Prepare an agent-led one-to-one Epi_<GENE> mapping from saved full DEGs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
TABLE_ROOT = (
    PROJECT_ROOT
    / "tables/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/epithelial"
)
RAW_VALIDATION = TABLE_ROOT / "01_raw_cluster_deg_validation.json"
MAPPING_PATH = TABLE_ROOT / "epithelial_subtype_mapping.csv"
CANDIDATE_PATH = TABLE_ROOT / "epithelial_subtype_candidate_table.csv"
REVIEW_DEPTHS = [50, 100, 200, 300]

UBIQUITOUS_EXCLUDE = {
    "MALAT1",
    "NEAT1",
    "XIST",
    "FTL",
    "FTH1",
    "TMSB10",
    "TMSB4X",
    "EEF1A1",
    "B2M",
    "ACTB",
    "GAPDH",
}


def eligible_gene(gene: str) -> bool:
    gene = str(gene)
    return not (
        gene in UBIQUITOUS_EXCLUDE
        or gene.startswith("MT-")
        or re.fullmatch(r"RP[SL][0-9A-Z.-]+", gene) is not None
        or gene.startswith("HIST")
    )


def functional_state(top_genes: list[str]) -> str:
    genes = set(top_genes)
    marker_sets = [
        ("cycling", {"MKI67", "TOP2A", "UBE2C", "CENPF", "TYMS", "PCNA", "CCNB1"}),
        ("acinar_like", {"PRSS1", "PRSS2", "AMY2A", "AMY2B", "CTRB1", "CTRB2", "CPA1"}),
        ("endocrine_like", {"INS", "IAPP", "GCG", "SST", "CHGA", "CHGB", "PCSK1"}),
        ("interferon_response", {"ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "OAS1"}),
        ("mesenchymal_EMT_like", {"VIM", "COL1A1", "COL1A2", "COL3A1", "SPARC", "FN1"}),
        ("regenerative_secretory", {"REG1A", "REG1B", "REG3A", "SPINK1", "TFF1", "TFF2"}),
        ("ductal_tumor_like", {"KRT19", "KRT17", "KRT8", "KRT18", "KRT7", "KRT6A", "MSLN", "S100P", "MUC1", "EPCAM"}),
        ("stress_response", {"FOS", "JUN", "JUNB", "HSPA1A", "HSPA1B", "DNAJB1"}),
    ]
    best_label, best_overlap = "epithelial_state_marker_defined", 0
    for label, marker_set in marker_sets:
        overlap = len(genes & marker_set)
        if overlap > best_overlap:
            best_label, best_overlap = label, overlap
    return best_label


def propose_unique_genes(
    candidates: dict[str, pd.DataFrame], review_depth: int
) -> dict[str, tuple[str, int, float]] | None:
    available = {
        cluster: frame.head(review_depth).reset_index(drop=True)
        for cluster, frame in candidates.items()
    }
    assigned: dict[str, tuple[str, int, float]] = {}
    used_genes: set[str] = set()
    while len(assigned) < len(available):
        proposals: dict[str, list[tuple[str, int, float]]] = {}
        for cluster, frame in available.items():
            if cluster in assigned:
                continue
            options = frame.loc[~frame["gene"].astype(str).isin(used_genes)]
            if options.empty:
                return None
            row = options.iloc[0]
            gene = str(row["gene"])
            original_rank = int(row["eligible_rank"])
            proposals.setdefault(gene, []).append((cluster, original_rank, float(row["score"])))
        progress = False
        for gene, contenders in proposals.items():
            winner = sorted(contenders, key=lambda item: (-item[2], int(item[0])))[0]
            cluster, rank, score = winner
            assigned[cluster] = (gene, rank, score)
            used_genes.add(gene)
            progress = True
        if not progress:
            return None
    return assigned


def main() -> None:
    if not RAW_VALIDATION.is_file():
        raise FileNotFoundError(RAW_VALIDATION)
    if MAPPING_PATH.exists() or CANDIDATE_PATH.exists():
        raise FileExistsError("Refusing to overwrite epithelial subtype mapping tables")
    validation = json.loads(RAW_VALIDATION.read_text(encoding="utf-8"))
    groups = validation.get("groups", [])
    if not groups:
        raise ValueError("Raw epithelial DEG validation lacks group files")
    candidates: dict[str, pd.DataFrame] = {}
    top50_by_cluster: dict[str, list[str]] = {}
    source_file_by_cluster: dict[str, str] = {}
    for group in groups:
        cluster = str(group["group"])
        path = Path(group["output_file"])
        frame = pd.read_csv(path)
        required = {"gene", "score", "logfoldchanges", "pvals_adj"}
        if len(frame) != 20244 or not required.issubset(frame.columns):
            raise ValueError(f"{cluster}: invalid full DEG file {path}")
        positive = frame.loc[
            frame["score"].gt(0)
            & frame["logfoldchanges"].gt(0)
            & frame["pvals_adj"].lt(0.05)
            & frame["gene"].astype(str).map(eligible_gene)
        ].copy()
        positive = positive.sort_values("score", ascending=False, kind="stable")
        positive = positive.drop_duplicates("gene", keep="first").reset_index(drop=True)
        positive["eligible_rank"] = np.arange(1, len(positive) + 1)
        if len(positive) < min(REVIEW_DEPTHS):
            raise ValueError(f"{cluster}: fewer than 50 eligible positive DEGs")
        candidates[cluster] = positive
        top50_by_cluster[cluster] = positive.head(50)["gene"].astype(str).tolist()
        source_file_by_cluster[cluster] = str(path)

    assigned = None
    review_depth = None
    for depth in REVIEW_DEPTHS:
        assigned = propose_unique_genes(candidates, depth)
        if assigned is not None:
            review_depth = depth
            break
    if assigned is None or review_depth is None:
        raise ValueError("Could not assign unique subtype-name genes at uniform depth <=300")

    mapping_rows, candidate_rows = [], []
    for cluster in sorted(candidates, key=lambda value: int(value)):
        gene, assigned_rank, score = assigned[cluster]
        top_genes = top50_by_cluster[cluster]
        state = functional_state(top_genes)
        confidence = "high" if assigned_rank <= 10 else "moderate"
        rationale = (
            f"rank {assigned_rank} highest available eligible positive adjusted-p DEG after "
            "global one-gene-per-cluster conflict resolution; stronger duplicate-gene cluster "
            "retains the gene and losing cluster advances to its next eligible DEG; mitochondrial, "
            "ribosomal, histone, and prespecified ubiquitous genes excluded"
        )
        mapping_rows.append(
            {
                "cluster": cluster,
                "cell_subtype": f"Epi_{gene}",
                "functional_state": state,
                "gene_selection_rationale": rationale,
                "most_significant_deg_gene": gene,
                "top_supporting_markers": ";".join(top_genes[:10]),
                "confidence": confidence,
                "review_depth": review_depth,
            }
        )
        candidate_rows.append(
            {
                "cluster": cluster,
                "proposed_cell_subtype": f"Epi_{gene}",
                "functional_state": state,
                "assigned_gene": gene,
                "assigned_eligible_rank": assigned_rank,
                "assigned_score": score,
                "top50_eligible_positive_degs": ";".join(top_genes),
                "raw_cluster_deg_file": source_file_by_cluster[cluster],
                "review_depth": review_depth,
                "needs_user_confirmation": False,
                "selection_source": "user_authorized_agent_led_annotation_2026-08-06",
            }
        )
    mapping = pd.DataFrame(mapping_rows)
    candidate = pd.DataFrame(candidate_rows)
    if not mapping["cell_subtype"].is_unique or len(mapping) != len(groups):
        raise RuntimeError("Generated epithelial subtype mapping is not one-to-one")
    mapping.to_csv(MAPPING_PATH, index=False)
    candidate.to_csv(CANDIDATE_PATH, index=False)
    print(
        json.dumps(
            {
                "n_clusters": len(mapping),
                "review_depth": review_depth,
                "n_high_confidence": int(mapping["confidence"].eq("high").sum()),
                "n_moderate_confidence": int(mapping["confidence"].eq("moderate").sum()),
                "mapping": mapping[["cluster", "cell_subtype", "functional_state"]].to_dict("records"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

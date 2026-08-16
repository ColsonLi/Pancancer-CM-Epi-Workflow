#!/usr/bin/env python3
"""Score broad-lineage DEG signatures and retain assigned/best-rank-consistent cells."""

from __future__ import annotations

import importlib.metadata
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import scanpy as sc


SEED = 42
PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
INPUT_H5AD = (
    PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/06-broad-annotation/adata_anno.h5ad"
)
SELECTION_TABLE = (
    PROJECT_ROOT
    / "tables/01-celltype_integration_clustering/05-clustering-parameter-search/selected_clustering.csv"
)
BROAD_TABLE_ROOT = PROJECT_ROOT / "tables/01-celltype_integration_clustering/06-broad-annotation"
TABLE_ROOT = PROJECT_ROOT / "tables/01-celltype_integration_clustering/07-score-rank-qc"
FIGURE_ROOT = PROJECT_ROOT / "figures/01-celltype_integration_clustering/07-score-rank-qc"
PENDING_FIGURE_ROOT = (
    PROJECT_ROOT / "figures/01-celltype_integration_clustering/07-score-rank-qc_pending_validation"
)
H5AD_DIR = PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/07-score-rank-qc"
PENDING_H5AD_DIR = (
    PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/07-score-rank-qc_pending_validation"
)
SCORED_H5AD = H5AD_DIR / "adata_anno_score_genes_rank.h5ad"
CONSISTENT_H5AD = H5AD_DIR / "adata_anno_score_genes_rank_consistent.h5ad"
PENDING_SCORED_H5AD = PENDING_H5AD_DIR / SCORED_H5AD.name
PENDING_CONSISTENT_H5AD = PENDING_H5AD_DIR / CONSISTENT_H5AD.name


def safe_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not token:
        raise ValueError(f"Cannot create a stable score token for {value!r}")
    return token


def package_version_lines() -> list[str]:
    packages = ["scanpy", "anndata", "numpy", "pandas", "scipy", "matplotlib"]
    lines = [
        "environment_category=rapids_with_scanpy_cpu_score_genes",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"random_seed={SEED}",
        f"code_files={Path(__file__).resolve()}",
    ]
    lines.extend(f"{package}={importlib.metadata.version(package)}" for package in packages)
    return lines


def main() -> None:
    if not INPUT_H5AD.is_file() or not SELECTION_TABLE.is_file():
        raise FileNotFoundError("Broad-annotated H5AD or clustering selection record is missing")
    for path in [H5AD_DIR, PENDING_H5AD_DIR, FIGURE_ROOT, PENDING_FIGURE_ROOT, TABLE_ROOT]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite score/rank artifact: {path}")

    selected = pd.read_csv(SELECTION_TABLE)
    if selected.shape[0] != 1:
        raise ValueError("Selected clustering record must contain exactly one row")
    selected = selected.iloc[0]
    n_pcs = int(selected["n_pcs"])
    n_neighbors = int(selected["n_neighbors"])
    resolution_text = str(float(selected["resolution"])).replace(".", "p")
    broad_deg_dir = BROAD_TABLE_ROOT / (
        f"degs_leiden_coarse_pcs{n_pcs}_nn{n_neighbors}_res{resolution_text}"
    )
    if not broad_deg_dir.is_dir():
        raise FileNotFoundError(f"Saved full broad DEG directory is missing: {broad_deg_dir}")

    started = time.time()
    adata = sc.read_h5ad(INPUT_H5AD)
    required_obs = ["leiden_coarse", "cell_type", "sample", "series", "status"]
    missing_obs = [column for column in required_obs if column not in adata.obs]
    if missing_obs:
        raise KeyError(f"Broad annotation lacks required obs columns: {missing_obs}")
    if adata.raw is None:
        raise ValueError("score_genes requires adata.raw normalized/log expression")
    if "best_rank_type_global" in adata.obs:
        raise ValueError("Input broad annotation already contains score/rank QC columns")

    labels = [str(value) for value in adata.obs["leiden_coarse"].cat.categories]
    if set(labels) != set(adata.obs["leiden_coarse"].astype(str).unique()):
        raise ValueError("Broad categorical labels include absent groups")
    raw_genes = set(map(str, adata.raw.var_names))
    mapping_rows = []
    score_to_label = {}
    rank_to_label = {}
    for label in labels:
        prefix = safe_token(label)
        score_column = f"{prefix}_score"
        rank_column = f"{prefix}_score_rank_pct"
        candidates = sorted(broad_deg_dir.glob(f"{safe_token(label).replace('_cells', '')}*.csv"))
        if not candidates:
            candidates = [
                path
                for path in broad_deg_dir.glob("*.csv")
                if path.name.startswith(re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_"))
            ]
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected one saved broad DEG file for {label!r}, found {candidates}")
        deg = pd.read_csv(candidates[0])
        genes = []
        for gene in deg["gene"].astype(str):
            if gene in raw_genes and gene not in genes:
                genes.append(gene)
            if len(genes) == 100:
                break
        if len(genes) != 100:
            raise ValueError(f"Could not select 100 usable saved-DEG genes for {label}")
        sc.tl.score_genes(
            adata,
            gene_list=genes,
            score_name=score_column,
            use_raw=True,
            random_state=SEED,
        )
        adata.obs[rank_column] = adata.obs[score_column].rank(
            ascending=False, pct=True, method="average"
        )
        score_to_label[score_column] = label
        rank_to_label[rank_column] = label
        mapping_rows.append(
            {
                "leiden_coarse": label,
                "score_column": score_column,
                "rank_percentile_column": rank_column,
                "n_genes": len(genes),
                "genes_semicolon": ";".join(genes),
                "source_full_deg_csv": str(candidates[0]),
                "gene_selection_rule": "first_100_unique_usable_genes_from_saved_full_leiden_coarse_DEG",
                "score_genes_use_raw": True,
            }
        )

    score_columns = list(score_to_label)
    rank_columns = list(rank_to_label)
    best_rank_column = adata.obs[rank_columns].idxmin(axis=1)
    highest_score_column = adata.obs[score_columns].idxmax(axis=1)
    adata.obs["best_rank_type_global"] = pd.Categorical(
        best_rank_column.map(rank_to_label), categories=labels, ordered=True
    )
    adata.obs["highest_score_type"] = pd.Categorical(
        highest_score_column.map(score_to_label), categories=labels, ordered=True
    )
    consistent = (
        adata.obs["best_rank_type_global"].astype(str)
        == adata.obs["leiden_coarse"].astype(str)
    )
    adata.obs["broad_score_rank_consistent"] = consistent
    if int(consistent.sum()) == 0 or int((~consistent).sum()) == 0:
        raise ValueError("Score/rank filter produced an empty kept or removed partition")
    kept_by_lineage = pd.crosstab(
        adata.obs["leiden_coarse"],
        adata.obs["broad_score_rank_consistent"],
        dropna=False,
    ).rename(columns={False: "removed_inconsistent", True: "kept_consistent"})
    for column in ["removed_inconsistent", "kept_consistent"]:
        if column not in kept_by_lineage:
            kept_by_lineage[column] = 0
    kept_by_lineage["total"] = kept_by_lineage[
        ["removed_inconsistent", "kept_consistent"]
    ].sum(axis=1)
    kept_by_lineage["kept_fraction"] = (
        kept_by_lineage["kept_consistent"] / kept_by_lineage["total"]
    )
    if (kept_by_lineage["kept_consistent"] == 0).any():
        raise ValueError("At least one broad lineage has no score/rank-consistent cells")

    PENDING_FIGURE_ROOT.mkdir(parents=True, exist_ok=False)
    sc.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
    sc.settings.figdir = PENDING_FIGURE_ROOT
    sc.pl.umap(
        adata,
        color=["leiden_coarse", "best_rank_type_global"],
        ncols=2,
        wspace=0.45,
        show=False,
        save="_assigned_vs_best_rank.pdf",
    )
    figure_path = PENDING_FIGURE_ROOT / "umap_assigned_vs_best_rank.pdf"
    if not figure_path.is_file() or figure_path.stat().st_size == 0:
        raise FileNotFoundError("Assigned-versus-best-rank UMAP was not written")

    adata.uns["broad_score_rank_qc_contract"] = {
        "score_gene_source": "saved full leiden_coarse DEG CSVs",
        "n_genes_per_observed_label": 100,
        "score_genes_use_raw": True,
        "rank_rule": "descending score percentile; smaller percentile is stronger",
        "best_rank_rule": "label of minimum rank percentile",
        "consistency_rule": "best_rank_type_global equals leiden_coarse",
        "downstream_default": str(CONSISTENT_H5AD),
        "unfiltered_audit_object": str(SCORED_H5AD),
    }
    filtered = adata[consistent].copy()

    PENDING_H5AD_DIR.mkdir(parents=True, exist_ok=False)
    adata.write_h5ad(PENDING_SCORED_H5AD, compression="gzip", compression_opts=1)
    filtered.write_h5ad(PENDING_CONSISTENT_H5AD, compression="gzip", compression_opts=1)
    pending_errors = []
    for path, expected_n_obs in [
        (PENDING_SCORED_H5AD, adata.n_obs),
        (PENDING_CONSISTENT_H5AD, filtered.n_obs),
    ]:
        pending = sc.read_h5ad(path, backed="r")
        if pending.n_obs != expected_n_obs or pending.n_vars != adata.n_vars:
            pending_errors.append(f"shape mismatch: {path.name}")
        if pending.raw is None or pending.raw.n_vars != adata.raw.n_vars:
            pending_errors.append(f"raw mismatch: {path.name}")
        missing = [
            column
            for column in ["leiden_coarse", "best_rank_type_global", "broad_score_rank_consistent"]
            if column not in pending.obs
        ]
        if missing:
            pending_errors.append(f"missing score/rank columns in {path.name}: {missing}")
        pending.file.close()
        del pending
    if pending_errors:
        raise ValueError(f"Pending score/rank H5AD validation failed: {pending_errors}")

    PENDING_H5AD_DIR.replace(H5AD_DIR)
    PENDING_FIGURE_ROOT.replace(FIGURE_ROOT)
    TABLE_ROOT.mkdir(parents=True, exist_ok=False)
    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(TABLE_ROOT / "score_column_gene_mapping.csv", index=False)
    core_columns = [
        "sample",
        "series",
        "status",
        "leiden_coarse",
        "best_rank_type_global",
        "highest_score_type",
        "broad_score_rank_consistent",
    ]
    per_cell = adata.obs[core_columns + score_columns + rank_columns].copy()
    per_cell.insert(0, "cell_id", per_cell.index)
    per_cell.to_csv(TABLE_ROOT / "per_cell_broad_score_rank_qc.csv", index=False)
    compact = per_cell[core_columns].copy()
    compact.insert(0, "cell_id", compact.index)
    compact.loc[consistent].to_csv(TABLE_ROOT / "consistent_cells.csv", index=False)
    compact.loc[~consistent].to_csv(TABLE_ROOT / "inconsistent_cells.csv", index=False)
    kept_by_lineage.reset_index().to_csv(
        TABLE_ROOT / "kept_removed_counts_by_leiden_coarse.csv", index=False
    )
    elapsed = time.time() - started
    parameters = {
        "step": "broad_score_genes_rank_consistency_qc",
        "input_h5ad": str(INPUT_H5AD),
        "unfiltered_output_h5ad": str(SCORED_H5AD),
        "consistent_output_h5ad": str(CONSISTENT_H5AD),
        "n_obs_input": int(adata.n_obs),
        "n_obs_consistent": int(filtered.n_obs),
        "n_obs_inconsistent": int((~consistent).sum()),
        "consistent_fraction": float(consistent.mean()),
        "n_observed_broad_labels": len(labels),
        "n_genes_per_label": 100,
        "score_genes_use_raw": True,
        "rank_ascending": False,
        "rank_percentile": True,
        "filter_rule": "best_rank_type_global_exactly_equals_leiden_coarse",
        "downstream_default_is_consistent_object": True,
        "random_seed": SEED,
        "elapsed_seconds": elapsed,
        "code_file": str(Path(__file__).resolve()),
    }
    pd.DataFrame([parameters]).to_csv(TABLE_ROOT / "score_rank_qc_parameters.csv", index=False)
    (TABLE_ROOT / "package_versions.txt").write_text(
        "\n".join(package_version_lines()) + "\n", encoding="utf-8"
    )
    (TABLE_ROOT / "readme.txt").write_text(
        "Broad score/rank consistency QC.\n\n"
        f"Input: {INPUT_H5AD}\n"
        f"Broad DEG source: {broad_deg_dir}\n"
        "For every observed leiden_coarse label, the first 100 unique usable genes from the "
        "saved full broad DEG table were scored with scanpy score_genes(use_raw=True).\n"
        "Within each score column, descending percentile ranks were calculated; the label with "
        "the smallest percentile is best_rank_type_global.\n"
        "The default downstream object retains only cells where best_rank_type_global exactly "
        "matches leiden_coarse. The full scored object is retained for audit.\n"
        f"Unfiltered output: {SCORED_H5AD}\n"
        f"Consistent-cell handoff: {CONSISTENT_H5AD}\n",
        encoding="utf-8",
    )
    validation = {
        **parameters,
        "score_columns": score_columns,
        "rank_columns": rank_columns,
        "all_labels_have_100_saved_deg_genes": bool((mapping["n_genes"] == 100).all()),
        "all_lineages_retain_cells": bool((kept_by_lineage["kept_consistent"] > 0).all()),
        "unfiltered_output_readable": SCORED_H5AD.is_file(),
        "consistent_output_readable": CONSISTENT_H5AD.is_file(),
        "figure": str(FIGURE_ROOT / figure_path.name),
    }
    (TABLE_ROOT / "score_rank_qc_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

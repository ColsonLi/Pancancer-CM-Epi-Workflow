#!/usr/bin/env python3
"""Audit all observed lineages after extraction and clean Harmony preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import scanpy as sc


PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
CONSISTENT_H5AD = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/07-score-rank-qc/adata_anno_score_genes_rank_consistent.h5ad"
)
LINEAGES = [
    ("epithelial", "Epithelial Cells"),
    ("t_nk", "T/NK Cells"),
    ("myeloid", "Myeloid Cells"),
    ("b", "B Cells"),
    ("endothelial", "Endothelial Cells"),
    ("stromal", "Stromal Cells"),
    ("mast", "Mast Cells"),
]
EXTRACTION_TABLE_ROOT = (
    PROJECT_ROOT / "tables/02-cell_subtype_integration_clustering/01-subtype-extraction"
)
HARMONY_TABLE_ROOT = (
    PROJECT_ROOT / "tables/02-cell_subtype_integration_clustering/02-subtype-harmony"
)


def main() -> None:
    outputs = [
        EXTRACTION_TABLE_ROOT / "lineage_extraction_summary.csv",
        EXTRACTION_TABLE_ROOT / "readme.txt",
        HARMONY_TABLE_ROOT / "lineage_harmony_readiness.csv",
        HARMONY_TABLE_ROOT / "module02_harmony_readiness_validation.json",
        HARMONY_TABLE_ROOT / "readme.txt",
    ]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite readiness artifact: {path}")

    consistent = sc.read_h5ad(CONSISTENT_H5AD, backed="r")
    expected_ids = set(map(str, consistent.obs_names))
    expected_n_obs = int(consistent.n_obs)
    consistent.file.close()
    observed_ids: set[str] = set()
    rows = []
    errors = []
    for slug, label in LINEAGES:
        extraction_path = (
            PROJECT_ROOT
            / f"h5ad/02-cell_subtype_integration_clustering/01-subtype-extraction/{slug}/adata_{slug}_normalized_log_all_genes.h5ad"
        )
        harmony_path = (
            PROJECT_ROOT
            / f"h5ad/02-cell_subtype_integration_clustering/02-subtype-harmony/{slug}/adata_{slug}_harmony.h5ad"
        )
        extraction = sc.read_h5ad(extraction_path, backed="r")
        harmony = sc.read_h5ad(harmony_path, backed="r")
        extraction_ids = list(map(str, extraction.obs_names))
        harmony_ids = list(map(str, harmony.obs_names))
        overlap = observed_ids.intersection(extraction_ids)
        if overlap:
            errors.append(f"cross-lineage duplicate IDs for {slug}: {len(overlap)}")
        observed_ids.update(extraction_ids)
        if extraction_ids != harmony_ids:
            errors.append(f"extraction/Harmony ID order mismatch: {slug}")
        if extraction.raw is None or extraction.raw.n_vars != 20244:
            errors.append(f"extraction raw mismatch: {slug}")
        if harmony.shape != (extraction.n_obs, 3000):
            errors.append(f"Harmony shape mismatch: {slug}")
        if harmony.raw is None or harmony.raw.n_vars != 20244:
            errors.append(f"Harmony raw mismatch: {slug}")
        technical_columns = [
            column
            for column in harmony.obs
            if str(column).startswith(("leiden_res", "louvain_res"))
        ]
        graph_present = "neighbors" in harmony.uns or "X_umap" in harmony.obsm
        if technical_columns or graph_present:
            errors.append(f"Harmony is not a clean clustering input: {slug}")
        rows.append(
            {
                "slug": slug,
                "leiden_coarse": label,
                "n_obs": int(harmony.n_obs),
                "n_samples": int(harmony.obs["sample"].nunique()),
                "n_series": int(harmony.obs["series"].nunique()),
                "n_vars_hvg": int(harmony.n_vars),
                "n_vars_raw": int(harmony.raw.n_vars),
                "pca_dims": int(harmony.obsm["X_pca"].shape[1]),
                "harmony_dims": int(harmony.obsm["X_pca_inte"].shape[1]),
                "neighbor_graph_present": "neighbors" in harmony.uns,
                "umap_present": "X_umap" in harmony.obsm,
                "technical_cluster_columns": ";".join(technical_columns),
                "extraction_h5ad": str(extraction_path),
                "harmony_h5ad": str(harmony_path),
            }
        )
        extraction.file.close()
        harmony.file.close()

    if observed_ids != expected_ids:
        errors.append(
            f"lineage union differs from consistent atlas: union={len(observed_ids)}, expected={expected_n_obs}"
        )
    pending_directories = sorted(
        str(path)
        for category in ["h5ad", "tables", "figures"]
        for path in (
            PROJECT_ROOT / category / "02-cell_subtype_integration_clustering"
        ).rglob("*_pending_validation")
    )
    if pending_directories:
        errors.append(f"pending directories remain: {pending_directories}")
    if errors:
        raise ValueError(errors)

    summary = pd.DataFrame(rows)
    EXTRACTION_TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    HARMONY_TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    summary[
        ["slug", "leiden_coarse", "n_obs", "n_samples", "n_series", "n_vars_raw", "extraction_h5ad"]
    ].to_csv(EXTRACTION_TABLE_ROOT / "lineage_extraction_summary.csv", index=False)
    summary.to_csv(HARMONY_TABLE_ROOT / "lineage_harmony_readiness.csv", index=False)
    EXTRACTION_TABLE_ROOT.joinpath("readme.txt").write_text(
        "Seven observed broad lineages were independently extracted by score/rank-consistent IDs.\n"
        "Expression comes from the audited three-series post-QC/log shared-gene atlas; UCSC is not renormalized.\n",
        encoding="utf-8",
    )
    HARMONY_TABLE_ROOT.joinpath("readme.txt").write_text(
        "Seven observed broad lineages independently completed 3000-HVG, regression, scaling, 50-PC PCA, and sample-keyed Harmony.\n"
        "Every saved Harmony object is a clean clustering input without neighbors, UMAP, Leiden, or Louvain outputs.\n",
        encoding="utf-8",
    )
    validation = {
        "status": "passed",
        "consistent_input_h5ad": str(CONSISTENT_H5AD),
        "n_lineages": len(LINEAGES),
        "expected_n_obs": expected_n_obs,
        "lineage_union_n_obs": len(observed_ids),
        "lineage_ids_disjoint": True,
        "lineage_union_matches_consistent_atlas": True,
        "all_harmony_objects_clean": True,
        "pending_directories": pending_directories,
        "rows": rows,
        "code_file": str(Path(__file__).resolve()),
    }
    (HARMONY_TABLE_ROOT / "module02_harmony_readiness_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

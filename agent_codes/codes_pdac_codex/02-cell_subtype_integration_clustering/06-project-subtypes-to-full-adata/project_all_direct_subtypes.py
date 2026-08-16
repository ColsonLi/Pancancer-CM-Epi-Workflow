#!/usr/bin/env python3
"""Project every finalized lineage subtype to the consistent atlas by exact cell ID."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc


PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
FULL_H5AD = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/07-score-rank-qc/adata_anno_score_genes_rank_consistent.h5ad"
)
H5AD_DIR = (
    PROJECT_ROOT / "h5ad/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata"
)
PENDING_H5AD_DIR = H5AD_DIR.with_name(
    "06-project-subtypes-to-full-adata_pending_validation"
)
OUTPUT_H5AD = H5AD_DIR / "adata_anno_cellsubtype.h5ad"
PENDING_H5AD = PENDING_H5AD_DIR / OUTPUT_H5AD.name
TABLE_DIR = (
    PROJECT_ROOT
    / "tables/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata/all_lineages"
)
FIGURE_DIR = (
    PROJECT_ROOT / "figures/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata"
)
PENDING_FIGURE_DIR = FIGURE_DIR.with_name(
    "06-project-subtypes-to-full-adata_pending_validation"
)

LINEAGES = [
    ("epithelial", "epi", "Epithelial Cells"),
    ("t_nk", "t_nk", "T/NK Cells"),
    ("myeloid", "mye", "Myeloid Cells"),
    ("b", "b", "B Cells"),
    ("endothelial", "endo", "Endothelial Cells"),
    ("stromal", "str", "Stromal Cells"),
    ("mast", "mast", "Mast Cells"),
]
PREFIX_TO_BROAD = {
    "Epi": "Epithelial Cells",
    "T": "T/NK Cells",
    "NK": "T/NK Cells",
    "Mye": "Myeloid Cells",
    "B": "B Cells",
    "Endo": "Endothelial Cells",
    "Str": "Stromal Cells",
    "Mast": "Mast Cells",
}


def main() -> None:
    if not FULL_H5AD.is_file():
        raise FileNotFoundError(FULL_H5AD)
    for path in [H5AD_DIR, PENDING_H5AD_DIR, TABLE_DIR, FIGURE_DIR, PENDING_FIGURE_DIR]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite all-lineage projection artifact: {path}")

    full = ad.read_h5ad(FULL_H5AD)
    required = ["leiden_coarse", "cell_type", "sample", "series", "status"]
    missing = [column for column in required if column not in full.obs]
    if missing:
        raise KeyError(f"Projection target lacks columns: {missing}")
    if not full.obs_names.is_unique or full.n_obs != 319_042:
        raise ValueError("Projection target must contain 319,042 unique cell IDs")

    cell_subtype = pd.Series(index=full.obs_names, dtype=object)
    functional_state = pd.Series(index=full.obs_names, dtype=object)
    annotation_lineage = pd.Series(index=full.obs_names, dtype=object)
    subtype_order: list[str] = []
    match_rows: list[dict] = []
    seen_ids: set[str] = set()
    for slug, abbrev, broad in LINEAGES:
        lineage_h5ad = (
            PROJECT_ROOT
            / f"h5ad/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/{slug}/adata_{abbrev}.h5ad"
        )
        mapping_path = (
            PROJECT_ROOT
            / f"tables/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/{slug}/final_subtype_mapping.csv"
        )
        if not lineage_h5ad.is_file() or not mapping_path.is_file():
            raise FileNotFoundError(f"Missing finalized {slug} H5AD or mapping")
        lineage = ad.read_h5ad(lineage_h5ad, backed="r")
        lineage_ids = pd.Index(lineage.obs_names.astype(str))
        expected_ids = pd.Index(
            full.obs_names[full.obs["leiden_coarse"].astype(str).eq(broad)].astype(str)
        )
        if set(lineage_ids) != set(expected_ids):
            lineage.file.close()
            raise ValueError(f"{slug}: finalized IDs differ from full-atlas broad lineage IDs")
        overlap = seen_ids.intersection(lineage_ids)
        if overlap:
            lineage.file.close()
            raise ValueError(f"{slug}: {len(overlap)} cell IDs overlap a prior lineage")
        seen_ids.update(lineage_ids)
        cell_subtype.loc[lineage_ids] = lineage.obs["cell_subtype"].astype(str).to_numpy()
        functional_state.loc[lineage_ids] = lineage.obs["functional_state"].astype(str).to_numpy()
        annotation_lineage.loc[lineage_ids] = slug
        mapping = pd.read_csv(mapping_path)
        subtype_order.extend(mapping["cell_subtype"].astype(str).tolist())
        match_rows.append(
            {
                "lineage_slug": slug,
                "leiden_coarse": broad,
                "source_h5ad": str(lineage_h5ad),
                "source_cells": int(lineage.n_obs),
                "matched_cells": int(len(lineage_ids)),
                "unmatched_cells": 0,
                "n_subtypes": int(mapping.shape[0]),
                "projection_key": "exact_obs_names",
            }
        )
        lineage.file.close()

    if len(seen_ids) != full.n_obs or cell_subtype.isna().any() or functional_state.isna().any():
        raise ValueError("Finalized lineage IDs do not form an exact partition of the full atlas")
    if len(subtype_order) != len(set(subtype_order)):
        raise ValueError("Projected cell_subtype names are not globally unique")
    derived_broad = cell_subtype.str.split("_", n=1).str[0].map(PREFIX_TO_BROAD)
    if derived_broad.isna().any():
        raise ValueError("A projected subtype prefix lacks a broad-lineage mapping")
    broad_conflicts = derived_broad.ne(full.obs["leiden_coarse"].astype(str))
    if broad_conflicts.any():
        raise ValueError(f"Subtype prefixes conflict with broad identities for {broad_conflicts.sum()} cells")

    full.obs["cell_type_before_subtype_projection"] = full.obs["cell_type"].copy()
    full.obs["cell_subtype"] = pd.Categorical(
        cell_subtype, categories=subtype_order, ordered=True
    )
    state_order = functional_state.drop_duplicates().astype(str).tolist()
    full.obs["functional_state"] = pd.Categorical(
        functional_state, categories=state_order, ordered=True
    )
    full.obs["cell_subtype_annotation_lineage"] = pd.Categorical(annotation_lineage)
    full.uns["all_lineage_subtype_projection_contract"] = {
        "source_full_h5ad": str(FULL_H5AD),
        "projection_key": "exact obs_names",
        "lineages": [slug for slug, _, _ in LINEAGES],
        "n_projected_cells": int(full.n_obs),
        "n_projected_subtypes": int(len(subtype_order)),
        "prefix_to_broad_mapping": PREFIX_TO_BROAD,
        "unmatched_cells": 0,
        "broad_identity_conflicts": 0,
    }
    palette = sc.pl.palettes.default_102[: len(subtype_order)]
    full.uns["cell_subtype_colors"] = palette

    PENDING_FIGURE_DIR.parent.mkdir(parents=True, exist_ok=True)
    PENDING_FIGURE_DIR.mkdir(parents=False, exist_ok=False)
    sc.set_figure_params(figsize=(6, 6), dpi=150, vector_friendly=True)
    figure = sc.pl.umap(
        full,
        color="cell_subtype",
        legend_loc="right margin",
        show=False,
        return_fig=True,
    )
    pending_figure = PENDING_FIGURE_DIR / "umap_all_lineage_cell_subtypes.pdf"
    figure.savefig(pending_figure, bbox_inches="tight")
    if pending_figure.stat().st_size == 0 or pending_figure.read_bytes()[:4] != b"%PDF":
        raise ValueError("Projected all-lineage subtype UMAP is invalid")

    PENDING_H5AD_DIR.parent.mkdir(parents=True, exist_ok=True)
    PENDING_H5AD_DIR.mkdir(parents=False, exist_ok=False)
    full.write_h5ad(PENDING_H5AD, compression="gzip", compression_opts=1)
    check = ad.read_h5ad(PENDING_H5AD, backed="r")
    errors = []
    if check.shape != full.shape:
        errors.append("shape mismatch")
    if check.raw is None or check.raw.n_vars != 20_244:
        errors.append("raw expression missing or not 20,244 genes")
    if set(check.obs["cell_subtype"].astype(str).unique()) != set(subtype_order):
        errors.append("subtype set mismatch")
    if check.obs[["cell_subtype", "functional_state"]].isna().any().any():
        errors.append("missing projected annotation")
    check.file.close()
    if errors:
        raise ValueError(f"Pending all-lineage projection validation failed: {errors}")
    PENDING_H5AD_DIR.replace(H5AD_DIR)
    PENDING_FIGURE_DIR.replace(FIGURE_DIR)

    TABLE_DIR.mkdir(parents=True, exist_ok=False)
    match_report = pd.DataFrame(match_rows)
    match_report.to_csv(TABLE_DIR / "projection_match_report.csv", index=False)
    pd.DataFrame(
        {"cell_subtype": subtype_order, "color": palette}
    ).to_csv(TABLE_DIR / "cell_subtype_palette.csv", index=False)
    counts = (
        full.obs["cell_subtype"]
        .value_counts(sort=False)
        .rename("n_cells")
        .rename_axis("cell_subtype")
        .reset_index()
    )
    counts.to_csv(TABLE_DIR / "projected_cell_subtype_counts.csv", index=False)
    validation = {
        "source_h5ad": str(FULL_H5AD),
        "output_h5ad": str(OUTPUT_H5AD),
        "n_obs_full": int(full.n_obs),
        "n_lineages": len(LINEAGES),
        "n_cell_subtypes_total": len(subtype_order),
        "all_lineage_ID_sets_exact": True,
        "all_lineage_ID_sets_disjoint": True,
        "all_cells_projected_once": True,
        "unmatched_cells": 0,
        "broad_conflicts_after_prefix_derivation": 0,
        "raw_n_vars": int(full.raw.n_vars),
        "output_h5ad_readable": OUTPUT_H5AD.is_file(),
        "figure": str(FIGURE_DIR / pending_figure.name),
        "cell_subtype_values": subtype_order,
        "validation_errors": [],
    }
    (TABLE_DIR / "projection_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    versions = [
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"anndata={importlib.metadata.version('anndata')}",
        f"scanpy={importlib.metadata.version('scanpy')}",
        f"pandas={importlib.metadata.version('pandas')}",
        f"code_file={Path(__file__).resolve()}",
    ]
    (TABLE_DIR / "package_versions.txt").write_text(
        "\n".join(versions) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

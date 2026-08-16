#!/usr/bin/env python3
"""Project epithelial subtypes to the score/rank-consistent full atlas by exact cell ID."""

from __future__ import annotations

import importlib.metadata
import json
import sys
import time
from pathlib import Path

import pandas as pd
import scanpy as sc


SEED = 42
PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
FULL_H5AD = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/07-score-rank-qc/adata_anno_score_genes_rank_consistent.h5ad"
)
EPI_H5AD = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/epithelial/adata_epi.h5ad"
)
H5AD_DIR = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata"
)
PENDING_H5AD_DIR = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata_pending_validation"
)
OUTPUT_H5AD = H5AD_DIR / "adata_anno_cellsubtype.h5ad"
PENDING_H5AD = PENDING_H5AD_DIR / OUTPUT_H5AD.name
FIGURE_DIR = (
    PROJECT_ROOT
    / "figures/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata"
)
PENDING_FIGURE_DIR = (
    PROJECT_ROOT
    / "figures/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata_pending_validation"
)
TABLE_DIR = (
    PROJECT_ROOT
    / "tables/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata/epithelial"
)

NON_EPI_BROAD_NODE_LABELS = {
    "T Cells": "T_Broad",
    "NK Cells": "NK_Broad",
    "Myeloid Cells": "Mye_Broad",
    "B Cells": "B_Broad",
    "Endothelial Cells": "Endo_Broad",
    "Stromal Cells": "S_Broad",
}
PREFIX_TO_CELL_TYPE = {
    "Epi": "Epithelial Cells",
    "T": "T Cells",
    "NK": "NK Cells",
    "Mye": "Myeloid Cells",
    "B": "B Cells",
    "Endo": "Endothelial Cells",
    "S": "Stromal Cells",
}
BROAD_ORDER = [
    "Epithelial Cells",
    "T Cells",
    "NK Cells",
    "Myeloid Cells",
    "B Cells",
    "Endothelial Cells",
    "Stromal Cells",
]


def package_version_lines() -> list[str]:
    packages = ["scanpy", "anndata", "numpy", "pandas", "scipy", "matplotlib"]
    lines = [
        "environment_category=rapids_with_scanpy_projection",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"random_seed={SEED}",
        f"code_files={Path(__file__).resolve()}",
    ]
    lines.extend(f"{package}={importlib.metadata.version(package)}" for package in packages)
    return lines


def main() -> None:
    if not FULL_H5AD.is_file() or not EPI_H5AD.is_file():
        raise FileNotFoundError("Consistent full atlas or annotated epithelial H5AD is missing")
    for path in [H5AD_DIR, PENDING_H5AD_DIR, FIGURE_DIR, PENDING_FIGURE_DIR, TABLE_DIR]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite subtype projection artifact: {path}")

    started = time.time()
    full = sc.read_h5ad(FULL_H5AD)
    epi = sc.read_h5ad(EPI_H5AD)
    required_full = ["leiden_coarse", "cell_type", "sample", "series", "status"]
    missing = [column for column in required_full if column not in full.obs]
    if missing:
        raise KeyError(f"Full consistent atlas lacks columns: {missing}")
    required_epi = ["cell_subtype", "functional_state"]
    missing = [column for column in required_epi if column not in epi.obs]
    if missing:
        raise KeyError(f"Annotated epithelial object lacks columns: {missing}")
    if not full.obs_names.is_unique or not epi.obs_names.is_unique:
        raise ValueError("Projection requires unique full and epithelial cell IDs")
    missing_epi_ids = epi.obs_names.difference(full.obs_names)
    if len(missing_epi_ids):
        raise KeyError(f"Annotated epithelial IDs absent from projection target: {len(missing_epi_ids)}")
    expected_epi_ids = full.obs_names[full.obs["leiden_coarse"].astype(str).eq("Epithelial Cells")]
    if set(epi.obs_names) != set(expected_epi_ids):
        raise ValueError("Annotated epithelial IDs do not exactly equal consistent full-atlas epithelial IDs")

    full.obs["cell_type_before_epithelial_projection"] = full.obs["cell_type"].copy()
    if "cell_subtype" in full.obs:
        full.obs["cell_subtype_before_epithelial_projection"] = full.obs["cell_subtype"].copy()
    cell_subtype = full.obs["leiden_coarse"].astype(str).map(NON_EPI_BROAD_NODE_LABELS)
    cell_subtype.loc[epi.obs_names] = epi.obs["cell_subtype"].astype(str)
    if cell_subtype.isna().any():
        missing_labels = sorted(full.obs.loc[cell_subtype.isna(), "leiden_coarse"].astype(str).unique())
        raise ValueError(f"No projection/node label rule for broad labels: {missing_labels}")
    epi_order = [str(value) for value in epi.obs["cell_subtype"].cat.categories]
    non_epi_order = [NON_EPI_BROAD_NODE_LABELS[label] for label in BROAD_ORDER if label != "Epithelial Cells"]
    subtype_order = epi_order + non_epi_order
    full.obs["cell_subtype"] = pd.Categorical(cell_subtype, categories=subtype_order, ordered=True)
    functional_state = pd.Series(
        "broad_lineage_resolution_not_subclustered",
        index=full.obs_names,
        dtype=object,
    )
    functional_state.loc[epi.obs_names] = epi.obs["functional_state"].astype(str)
    full.obs["functional_state"] = functional_state.astype("category")
    annotation_source = pd.Series(
        "non_epithelial_broad_lineage_retained_as_CM_node_resolution",
        index=full.obs_names,
        dtype=object,
    )
    annotation_source.loc[epi.obs_names] = "epithelial_subtype_exact_ID_projection"
    full.obs["cell_subtype_annotation_source"] = annotation_source.astype("category")
    derived_cell_type = full.obs["cell_subtype"].astype(str).str.split("_", n=1).str[0].map(PREFIX_TO_CELL_TYPE)
    if derived_cell_type.isna().any():
        raise ValueError("Subtype-prefix to cell_type mapping produced missing values")
    full.obs["cell_type"] = pd.Categorical(derived_cell_type, categories=BROAD_ORDER, ordered=True)
    conflicts = full.obs["cell_type"].astype(str) != full.obs["leiden_coarse"].astype(str)
    if conflicts.any():
        raise ValueError("Projected subtype prefixes conflict with retained broad identities")

    broad_colors = ["#D55E00", "#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#E69F00", "#999999"]
    subtype_palette = sc.pl.palettes.default_102[: len(subtype_order)]
    full.uns["leiden_coarse_colors"] = broad_colors
    full.uns["cell_type_colors"] = broad_colors
    full.uns["cell_subtype_colors"] = subtype_palette
    epi.uns["cell_subtype_colors"] = subtype_palette[: len(epi_order)]
    full.uns["epithelial_subtype_projection_contract"] = {
        "projection_source_h5ad": str(EPI_H5AD),
        "projection_target_h5ad": str(FULL_H5AD),
        "projection_key": "exact obs_names",
        "n_projected_epithelial_cells": int(epi.n_obs),
        "unmatched_epithelial_cells": 0,
        "non_epithelial_resolution": "validated broad lineage retained as one CM node label per broad class",
        "non_epithelial_node_label_mapping": NON_EPI_BROAD_NODE_LABELS,
        "prefix_to_cell_type_mapping": PREFIX_TO_CELL_TYPE,
    }

    PENDING_FIGURE_DIR.mkdir(parents=True, exist_ok=False)
    sc.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
    sc.settings.figdir = PENDING_FIGURE_DIR
    sc.pl.umap(
        full,
        color=["leiden_coarse", "cell_type"],
        ncols=2,
        wspace=0.45,
        show=False,
        save="_leiden_coarse_vs_projected_cell_type.pdf",
    )
    sc.pl.umap(full, color="cell_subtype", show=False, save="_projected_cell_subtype.pdf")
    pending_epi_figure_dir = PENDING_FIGURE_DIR / "epithelial"
    pending_epi_figure_dir.mkdir(parents=True, exist_ok=False)
    sc.settings.figdir = pending_epi_figure_dir
    sc.pl.umap(epi, color="cell_subtype", show=False, save="_projected_cell_subtype.pdf")
    expected_figures = [
        PENDING_FIGURE_DIR / "umap_leiden_coarse_vs_projected_cell_type.pdf",
        PENDING_FIGURE_DIR / "umap_projected_cell_subtype.pdf",
        pending_epi_figure_dir / "umap_projected_cell_subtype.pdf",
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in expected_figures):
        raise FileNotFoundError("Required full/lineage projection UMAP is missing")

    PENDING_H5AD_DIR.mkdir(parents=True, exist_ok=False)
    full.write_h5ad(PENDING_H5AD, compression="gzip", compression_opts=1)
    pending = sc.read_h5ad(PENDING_H5AD, backed="r")
    errors = []
    if pending.shape != full.shape:
        errors.append("shape mismatch")
    if pending.raw is None or pending.raw.n_vars != full.raw.n_vars:
        errors.append("raw mismatch")
    for column in ["leiden_coarse", "cell_type", "cell_subtype", "functional_state"]:
        if column not in pending.obs:
            errors.append(f"missing column {column}")
    pending.file.close()
    del pending
    if errors:
        raise ValueError(f"Pending projected full H5AD failed validation: {errors}")
    PENDING_H5AD_DIR.replace(H5AD_DIR)
    PENDING_FIGURE_DIR.replace(FIGURE_DIR)

    TABLE_DIR.mkdir(parents=True, exist_ok=False)
    match_report = pd.DataFrame(
        [
            {
                "lineage": "Epithelial Cells",
                "source_h5ad": str(EPI_H5AD),
                "target_h5ad": str(FULL_H5AD),
                "output_h5ad": str(OUTPUT_H5AD),
                "source_cells": int(epi.n_obs),
                "matched_cells": int(epi.n_obs),
                "unmatched_cells": 0,
                "duplicate_source_ids": 0,
                "projection_key": "exact_obs_names",
                "non_epithelial_resolution_note": "one validated broad-lineage CM node label per non-epithelial broad class; no non-epithelial subtype reclustering",
            }
        ]
    )
    match_report.to_csv(TABLE_DIR / "projection_match_report.csv", index=False)
    pd.DataFrame(
        {"leiden_coarse": BROAD_ORDER, "color": broad_colors}
    ).to_csv(TABLE_DIR / "broad_palette.csv", index=False)
    pd.DataFrame(
        {"cell_subtype": subtype_order, "color": subtype_palette}
    ).to_csv(TABLE_DIR / "cell_subtype_palette.csv", index=False)
    pd.DataFrame(
        [{"broad_label": key, "non_epithelial_node_label": value} for key, value in NON_EPI_BROAD_NODE_LABELS.items()]
    ).to_csv(TABLE_DIR / "non_epithelial_broad_node_label_mapping.csv", index=False)
    elapsed = time.time() - started
    parameters = {
        "step": "project_epithelial_subtypes_to_consistent_full_atlas",
        "source_h5ad": str(EPI_H5AD),
        "target_h5ad": str(FULL_H5AD),
        "output_h5ad": str(OUTPUT_H5AD),
        "n_obs_full": int(full.n_obs),
        "n_obs_epithelial_projected": int(epi.n_obs),
        "n_cell_subtypes_total": len(subtype_order),
        "n_epithelial_subtypes": len(epi_order),
        "n_non_epithelial_broad_node_labels": len(non_epi_order),
        "projection_key": "exact_obs_names",
        "unmatched_cells": 0,
        "broad_conflicts_after_prefix_derivation": 0,
        "random_seed": SEED,
        "elapsed_seconds": elapsed,
        "code_file": str(Path(__file__).resolve()),
    }
    pd.DataFrame([parameters]).to_csv(TABLE_DIR / "projection_parameters.csv", index=False)
    (TABLE_DIR / "package_versions.txt").write_text(
        "\n".join(package_version_lines()) + "\n", encoding="utf-8"
    )
    validation = {
        **parameters,
        "epithelial_ID_set_exact": True,
        "cell_subtype_values": subtype_order,
        "cell_type_values": BROAD_ORDER,
        "output_h5ad_readable": OUTPUT_H5AD.is_file(),
        "figures": [str(FIGURE_DIR / path.relative_to(PENDING_FIGURE_DIR)) for path in expected_figures],
    }
    (TABLE_DIR / "projection_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the DEG-marker pseudobulk reference for canonical PDAC Tangram."""

from __future__ import annotations

import importlib.metadata
import json
import random
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


SEED = 42
MAX_MARKERS_PER_SUBTYPE = 100
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
REFERENCE_ATLAS = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata/adata_anno_cellsubtype.h5ad"
)
SPATIAL_AUDIT = (
    PROJECT_ROOT
    / "tables/04-spatial-validation-optional/02-build-spatial-h5ad/spatial_h5ad_build_audit.csv"
)
SPATIAL_CLINICAL = (
    PROJECT_ROOT
    / "tables/04-spatial-validation-optional/01-input-staging/spatial_input_manifest.csv"
)
CM_LOADING = (
    PROJECT_ROOT
    / "tables/03-epi-cm-discovery/01-cm-lineage-analysis/02-balanced-joint-nmf/loading_df_cell_subtype_by_CM_fraction.csv"
)
OUT_H5AD = (
    PROJECT_ROOT
    / "h5ad/04-spatial-validation-optional/03-tangram-reference/adata_cell_subtype_pseudobulk_deg_markers.h5ad"
)
TABLE_DIR = PROJECT_ROOT / "tables/04-spatial-validation-optional/03-tangram-reference"

LINEAGES = ["epithelial", "t_nk", "myeloid", "b", "endothelial", "stromal", "mast"]


def lineage_annotation_paths(slug: str) -> tuple[Path, Path]:
    root = (
        PROJECT_ROOT
        / f"tables/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/{slug}"
    )
    return root / "final_subtype_mapping.csv", root / "raw_cluster_deg_file_manifest.csv"


def require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Tangram reference artifact: {path}")


def select_markers(
    subtype: str,
    deg_path: Path,
    source_resolution: str,
    raw_genes: set[str],
) -> list[dict]:
    frame = pd.read_csv(deg_path)
    required = {"gene", "score", "logfoldchanges", "pvals_adj"}
    if not required.issubset(frame.columns):
        raise KeyError(f"{deg_path} lacks DEG columns: {sorted(required - set(frame.columns))}")
    if len(frame) != len(raw_genes):
        # Both post-annotation sources are required to be complete all-gene exports.
        raise ValueError(
            f"{subtype}: expected {len(raw_genes):,} full DEG rows, observed {len(frame):,}"
        )
    selected = frame.loc[
        frame["pvals_adj"].lt(0.05)
        & frame["logfoldchanges"].gt(0)
        & frame["gene"].astype(str).isin(raw_genes)
    ].copy()
    selected = selected.sort_values("score", ascending=False, kind="stable")
    selected = selected.drop_duplicates("gene", keep="first").head(MAX_MARKERS_PER_SUBTYPE)
    if selected.empty:
        raise ValueError(f"{subtype}: no eligible positive adjusted-p DEG marker")
    rows = []
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        rows.append(
            {
                "cell_subtype": subtype,
                "marker_rank": rank,
                "gene": str(row.gene),
                "score": float(row.score),
                "logfoldchanges": float(row.logfoldchanges),
                "pvals_adj": float(row.pvals_adj),
                "deg_source_resolution": source_resolution,
                "deg_source_file": str(deg_path),
            }
        )
    return rows


def main() -> None:
    lineage_inputs = [path for slug in LINEAGES for path in lineage_annotation_paths(slug)]
    required = [REFERENCE_ATLAS, SPATIAL_AUDIT, SPATIAL_CLINICAL, CM_LOADING, *lineage_inputs]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Tangram reference inputs are missing: {missing}")
    require_new(OUT_H5AD)
    require_new(TABLE_DIR)
    OUT_H5AD.parent.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=False)

    atlas = sc.read_h5ad(REFERENCE_ATLAS)
    if atlas.raw is None or "cell_subtype" not in atlas.obs:
        raise ValueError("Projected atlas lacks adata.raw or cell_subtype")
    if atlas.raw.n_vars != 20244:
        raise ValueError(f"Expected 20,244 reference raw genes, observed {atlas.raw.n_vars}")
    raw_genes = set(atlas.raw.var_names.astype(str))
    subtype_counts = atlas.obs["cell_subtype"].astype(str).value_counts()

    marker_rows = []
    source_rows = []
    expected_subtypes = []
    for slug in LINEAGES:
        mapping_path, manifest_path = lineage_annotation_paths(slug)
        mapping = pd.read_csv(mapping_path, dtype={"cluster": str})
        manifest = pd.read_csv(manifest_path, dtype={"cluster": str})
        if mapping["cluster"].duplicated().any() or manifest["cluster"].duplicated().any():
            raise ValueError(f"{slug}: duplicate cluster in mapping or DEG manifest")
        joined = mapping[["cluster", "cell_subtype"]].merge(
            manifest[["cluster", "output_csv", "n_deg_rows"]],
            on="cluster",
            how="inner",
            validate="one_to_one",
        )
        if len(joined) != len(mapping) or not joined["n_deg_rows"].eq(20_244).all():
            raise ValueError(f"{slug}: final mapping and full DEG manifest do not match")
        for row in joined.itertuples(index=False):
            subtype = str(row.cell_subtype)
            deg_path = Path(row.output_csv)
            if not deg_path.is_file():
                raise FileNotFoundError(deg_path)
            expected_subtypes.append(subtype)
            marker_rows.extend(
                select_markers(
                    subtype,
                    deg_path,
                    "Block02 final direct-clustering raw cluster DEG mapped one-to-one to cell_subtype",
                    raw_genes,
                )
            )
            source_rows.append(
                {
                    "lineage_slug": slug,
                    "cluster": str(row.cluster),
                    "cell_subtype": subtype,
                    "deg_source_file": str(deg_path),
                    "deg_rows": int(row.n_deg_rows),
                    "resolution_note": "Block02 finalized direct-clustering subtype",
                }
            )
    markers = pd.DataFrame(marker_rows)
    if len(expected_subtypes) != len(set(expected_subtypes)):
        raise ValueError("Final cell_subtype names are not globally unique")
    epi_subtypes = [subtype for subtype in expected_subtypes if subtype.startswith("Epi_")]
    non_epi_subtypes = [subtype for subtype in expected_subtypes if not subtype.startswith("Epi_")]
    if set(markers["cell_subtype"]) != set(expected_subtypes):
        raise ValueError("Marker table does not exactly cover all finalized subtypes")
    if set(expected_subtypes) != set(subtype_counts.index):
        raise ValueError(
            f"Projected atlas lacks reference cells for: {sorted(set(expected_subtypes) - set(subtype_counts.index))}"
        )
    marker_genes = pd.Index(markers["gene"].drop_duplicates().astype(str))
    gene_indices = atlas.raw.var_names.astype(str).get_indexer(marker_genes)
    if (gene_indices < 0).any():
        raise RuntimeError("Selected marker disappeared from reference raw genes")

    pseudobulk = np.zeros((len(expected_subtypes), len(marker_genes)), dtype=np.float32)
    cell_rows = []
    raw_X = atlas.raw.X
    for row_index, subtype in enumerate(expected_subtypes):
        mask = atlas.obs["cell_subtype"].astype(str).eq(subtype).to_numpy()
        n_cells = int(mask.sum())
        if n_cells == 0:
            raise ValueError(f"No reference cells for {subtype}")
        values = raw_X[mask][:, gene_indices]
        mean = np.asarray(values.mean(axis=0)).ravel().astype(np.float32)
        pseudobulk[row_index, :] = mean
        cell_rows.append(
            {
                "cell_subtype": subtype,
                "n_reference_cells": n_cells,
                "compartment": "epithelial" if subtype.startswith("Epi_") else "non_epithelial",
                "reference_resolution": (
                    "Block02 finalized direct-clustering subtype"
                ),
            }
        )
    obs = pd.DataFrame(cell_rows).set_index("cell_subtype", drop=False)
    var = pd.DataFrame(index=marker_genes)
    var["selected_as_deg_marker"] = True
    reference = ad.AnnData(X=pseudobulk, obs=obs, var=var)
    reference.uns["pseudobulk_contract"] = {
        "source_h5ad": str(REFERENCE_ATLAS),
        "expression_source": "adata.raw normalized/log all-gene matrix",
        "aggregation": "arithmetic mean across every retained reference cell per cell_subtype; no downsampling",
        "marker_filter": "pvals_adj<0.05 and logfoldchanges>0, descending score, max 100 per subtype",
        "non_epithelial_resolution": "Block02 finalized direct-clustering subtypes",
    }
    reference.write_h5ad(OUT_H5AD, compression="gzip")

    markers.to_csv(TABLE_DIR / "tangram_deg_marker_selection.csv", index=False)
    pd.DataFrame({"gene": marker_genes}).to_csv(
        TABLE_DIR / "tangram_union_deg_marker_genes.csv", index=False
    )
    pd.DataFrame({"cell_subtype": epi_subtypes}).to_csv(
        TABLE_DIR / "epithelial_subtypes.csv", index=False
    )
    pd.DataFrame(source_rows).to_csv(
        TABLE_DIR / "all_subtype_marker_sources.csv", index=False
    )
    pd.DataFrame(cell_rows).to_csv(TABLE_DIR / "pseudobulk_reference_manifest.csv", index=False)

    loading = pd.read_csv(CM_LOADING, index_col=0)
    if set(loading.index.astype(str)) != set(non_epi_subtypes):
        raise ValueError("Canonical loading rows do not exactly match projected non-epi subtypes")
    loading.to_csv(TABLE_DIR / "cm_loading_fraction_non_epi_node_by_CM.csv")

    spatial_audit = pd.read_csv(SPATIAL_AUDIT)
    clinical = pd.read_csv(SPATIAL_CLINICAL)
    clinical = clinical.drop_duplicates("sample").set_index("sample")
    spatial_manifest = spatial_audit.copy()
    for column in ["sample_label", "patient_id", "status", "ptnm", "clinical_stage"]:
        spatial_manifest[column] = spatial_manifest["sample"].map(clinical[column])
    if spatial_manifest["sample"].nunique() != 14 or not spatial_manifest["status"].eq("tumor").all():
        raise ValueError("Expected 14 eligible tumor spatial samples")
    spatial_manifest.to_csv(TABLE_DIR / "spatial_sample_scope_manifest.csv", index=False)

    marker_counts = markers.groupby("cell_subtype").size()
    validation = {
        "reference_h5ad": str(OUT_H5AD),
        "reference_shape": [int(reference.n_obs), int(reference.n_vars)],
        "n_reference_subtypes": int(reference.n_obs),
        "n_epithelial_subtypes": len(epi_subtypes),
        "n_non_epithelial_subtypes": len(non_epi_subtypes),
        "n_union_marker_genes": int(len(marker_genes)),
        "min_markers_per_subtype": int(marker_counts.min()),
        "max_markers_per_subtype": int(marker_counts.max()),
        "all_reference_cells_used": int(sum(row["n_reference_cells"] for row in cell_rows))
        == int(atlas.n_obs),
        "reference_raw_genes": int(atlas.raw.n_vars),
        "spatial_samples": int(spatial_manifest["sample"].nunique()),
        "cm_loading_source": str(CM_LOADING),
        "validation_passed": True,
    }
    (TABLE_DIR / "tangram_reference_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    versions = [
        "environment_category=rapids_reference_preparation",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"scanpy={importlib.metadata.version('scanpy')}",
        f"anndata={importlib.metadata.version('anndata')}",
        f"numpy={importlib.metadata.version('numpy')}",
        f"pandas={importlib.metadata.version('pandas')}",
        f"scipy={importlib.metadata.version('scipy')}",
        f"random_seed={SEED}",
        f"code_file={Path(__file__).resolve()}",
    ]
    (TABLE_DIR / "package_versions.txt").write_text("\n".join(versions) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()

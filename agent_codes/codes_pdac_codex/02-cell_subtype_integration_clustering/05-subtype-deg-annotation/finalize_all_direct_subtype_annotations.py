#!/usr/bin/env python3
"""Finalize all reviewed direct-clustering subtype candidates after user authorization."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc


PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
AUTHORIZATION_SOURCE = (
    "user_authorized_agent_selection_and_requested_completion_through_Block4"
)
LINEAGES = [
    {
        "slug": "epithelial",
        "abbrev": "epi",
        "broad": "Epithelial Cells",
        "prefixes": {"Epi"},
    },
    {
        "slug": "t_nk",
        "abbrev": "t_nk",
        "broad": "T/NK Cells",
        "prefixes": {"T", "NK"},
    },
    {
        "slug": "myeloid",
        "abbrev": "mye",
        "broad": "Myeloid Cells",
        "prefixes": {"Mye"},
    },
    {"slug": "b", "abbrev": "b", "broad": "B Cells", "prefixes": {"B"}},
    {
        "slug": "endothelial",
        "abbrev": "endo",
        "broad": "Endothelial Cells",
        "prefixes": {"Endo"},
    },
    {
        "slug": "stromal",
        "abbrev": "str",
        "broad": "Stromal Cells",
        "prefixes": {"Str"},
    },
    {
        "slug": "mast",
        "abbrev": "mast",
        "broad": "Mast Cells",
        "prefixes": {"Mast"},
    },
]


def version_lines() -> list[str]:
    packages = ["scanpy", "anndata", "numpy", "pandas", "scipy", "matplotlib"]
    lines = [
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"authorization_source={AUTHORIZATION_SOURCE}",
        f"code_file={Path(__file__).resolve()}",
    ]
    lines.extend(f"{package}={importlib.metadata.version(package)}" for package in packages)
    return lines


def finalize_lineage(config: dict) -> dict:
    slug = str(config["slug"])
    abbrev = str(config["abbrev"])
    broad = str(config["broad"])
    allowed_prefixes = set(config["prefixes"])
    cluster_h5ad = (
        PROJECT_ROOT
        / f"h5ad/02-cell_subtype_integration_clustering/03-subtype-clustering/{slug}/adata_{abbrev}_selected_clustered.h5ad"
    )
    table_dir = (
        PROJECT_ROOT
        / f"tables/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/{slug}"
    )
    candidate_path = table_dir / "candidate_subtype_annotation.csv"
    h5ad_dir = (
        PROJECT_ROOT
        / f"h5ad/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/{slug}"
    )
    pending_h5ad_dir = h5ad_dir.with_name(f"{slug}_pending_validation")
    output_h5ad = h5ad_dir / f"adata_{abbrev}.h5ad"
    pending_h5ad = pending_h5ad_dir / output_h5ad.name
    figure_dir = (
        PROJECT_ROOT
        / f"figures/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/{slug}"
    )
    pending_figure_dir = figure_dir.with_name(f"{slug}_pending_validation")
    final_mapping_path = table_dir / "final_subtype_mapping.csv"
    counts_path = table_dir / "final_subtype_counts.csv"
    parameters_path = table_dir / "final_subtype_annotation_parameters.csv"
    validation_path = table_dir / "final_subtype_annotation_validation.json"
    versions_path = table_dir / "final_subtype_annotation_package_versions.txt"

    required = [cluster_h5ad, candidate_path]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{slug}: missing inputs {missing}")
    for path in [
        h5ad_dir,
        pending_h5ad_dir,
        figure_dir,
        pending_figure_dir,
        final_mapping_path,
        counts_path,
        parameters_path,
        validation_path,
        versions_path,
    ]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite final subtype artifact: {path}")

    candidate = pd.read_csv(candidate_path, dtype={"cluster": str})
    required_columns = {
        "cluster",
        "proposed cell_subtype",
        "functional_state",
        "selected_prefix",
        "confidence",
        "needs_user_confirmation",
    }
    missing_columns = required_columns.difference(candidate.columns)
    if missing_columns:
        raise KeyError(f"{slug}: candidate table lacks {sorted(missing_columns)}")
    if candidate["cluster"].duplicated().any():
        raise ValueError(f"{slug}: candidate clusters are not unique")
    if candidate["proposed cell_subtype"].duplicated().any():
        raise ValueError(f"{slug}: candidate subtype names are not unique")
    if set(candidate["selected_prefix"].astype(str)) - allowed_prefixes:
        raise ValueError(f"{slug}: candidate subtype prefix is outside {sorted(allowed_prefixes)}")

    mapping = candidate.rename(columns={"proposed cell_subtype": "cell_subtype"}).copy()
    mapping["confirmed_by_user_instruction"] = True
    mapping["confirmation_source"] = AUTHORIZATION_SOURCE
    mapping["needs_user_confirmation"] = False

    adata = sc.read_h5ad(cluster_h5ad)
    contract = adata.uns.get("lineage_direct_clustering_contract", {})
    cluster_key = str(contract.get("raw_cluster_column", ""))
    if not cluster_key or cluster_key not in adata.obs:
        raise KeyError(f"{slug}: raw cluster column is missing")
    if "cell_subtype" in adata.obs or "functional_state" in adata.obs:
        raise ValueError(f"{slug}: clustering input is already biologically annotated")
    observed = set(adata.obs[cluster_key].astype(str).unique())
    if observed != set(mapping["cluster"]):
        raise ValueError(
            f"{slug}: final mapping does not cover raw clusters exactly; "
            f"observed={sorted(observed)}, mapped={sorted(set(mapping['cluster']))}"
        )

    label_map = mapping.set_index("cluster")["cell_subtype"].astype(str).to_dict()
    state_map = mapping.set_index("cluster")["functional_state"].astype(str).to_dict()
    cluster_values = adata.obs[cluster_key].astype(str)
    subtype_order = mapping["cell_subtype"].astype(str).tolist()
    state_order = mapping["functional_state"].astype(str).drop_duplicates().tolist()
    adata.obs["cell_subtype"] = pd.Categorical(
        cluster_values.map(label_map), categories=subtype_order, ordered=True
    )
    adata.obs["functional_state"] = pd.Categorical(
        cluster_values.map(state_map), categories=state_order, ordered=True
    )
    if adata.obs[["cell_subtype", "functional_state"]].isna().any().any():
        raise ValueError(f"{slug}: subtype mapping generated missing labels")
    adata.uns["final_subtype_annotation_contract"] = {
        "source_cluster_h5ad": str(cluster_h5ad),
        "candidate_table": str(candidate_path),
        "final_mapping": str(final_mapping_path),
        "raw_cluster_column": cluster_key,
        "target_leiden_coarse": broad,
        "mapping_key": "raw numeric cluster label",
        "authorization_source": AUTHORIZATION_SOURCE,
        "candidate_review_depth": int(mapping["review_depth"].min()),
        "all_clusters_mapped_once": True,
    }

    pending_h5ad_dir.parent.mkdir(parents=True, exist_ok=True)
    pending_figure_dir.parent.mkdir(parents=True, exist_ok=True)
    pending_h5ad_dir.mkdir(parents=False, exist_ok=False)
    pending_figure_dir.mkdir(parents=False, exist_ok=False)
    sc.set_figure_params(figsize=(4, 4), dpi=150, vector_friendly=True)
    figure = sc.pl.umap(
        adata,
        color="cell_subtype",
        legend_loc="right margin",
        show=False,
        return_fig=True,
    )
    pending_figure = pending_figure_dir / "umap_cell_subtype.pdf"
    figure.savefig(pending_figure, bbox_inches="tight")
    if pending_figure.stat().st_size == 0 or pending_figure.read_bytes()[:4] != b"%PDF":
        raise ValueError(f"{slug}: invalid subtype UMAP PDF")
    adata.write_h5ad(pending_h5ad, compression="gzip", compression_opts=1)

    check = sc.read_h5ad(pending_h5ad, backed="r")
    errors = []
    if check.shape != adata.shape:
        errors.append("shape mismatch")
    if check.raw is None or check.raw.n_vars != 20_244:
        errors.append("raw expression missing or not 20,244 genes")
    for column in [cluster_key, "cell_subtype", "functional_state"]:
        if column not in check.obs:
            errors.append(f"missing obs column {column}")
    if set(check.obs["cell_subtype"].astype(str).unique()) != set(subtype_order):
        errors.append("subtype values differ from final mapping")
    check.file.close()
    if errors:
        raise ValueError(f"{slug}: pending final subtype H5AD validation failed: {errors}")
    pending_h5ad_dir.replace(h5ad_dir)
    pending_figure_dir.replace(figure_dir)

    mapping.to_csv(final_mapping_path, index=False)
    counts = (
        adata.obs["cell_subtype"]
        .value_counts(sort=False)
        .rename("n_cells")
        .rename_axis("cell_subtype")
        .reset_index()
    )
    counts.to_csv(counts_path, index=False)
    parameters = {
        "lineage_slug": slug,
        "target_leiden_coarse": broad,
        "input_h5ad": str(cluster_h5ad),
        "output_h5ad": str(output_h5ad),
        "raw_cluster_column": cluster_key,
        "resolution": float(contract["accepted_resolution"]),
        "n_pcs": int(contract["n_pcs"]),
        "n_neighbors": int(contract["n_neighbors"]),
        "n_cells": int(adata.n_obs),
        "n_subtypes": int(len(mapping)),
        "authorization_source": AUTHORIZATION_SOURCE,
        "code_file": str(Path(__file__).resolve()),
    }
    pd.DataFrame([parameters]).to_csv(parameters_path, index=False)
    versions_path.write_text("\n".join(version_lines()) + "\n", encoding="utf-8")
    validation = {
        **parameters,
        "output_h5ad_readable": output_h5ad.is_file(),
        "all_raw_clusters_mapped_once": True,
        "subtype_names_unique": bool(mapping["cell_subtype"].is_unique),
        "cell_count_preserved": int(counts["n_cells"].sum()) == int(adata.n_obs),
        "raw_expression_present": adata.raw is not None,
        "raw_n_vars": int(adata.raw.n_vars),
        "umap_present": "X_umap" in adata.obsm,
        "figure": str(figure_dir / pending_figure.name),
        "validation_errors": [],
    }
    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return validation


def main() -> None:
    results = [finalize_lineage(config) for config in LINEAGES]
    summary = {
        "lineages_finalized": len(results),
        "cells_finalized": sum(int(result["n_cells"]) for result in results),
        "subtypes_finalized": sum(int(result["n_subtypes"]) for result in results),
        "authorization_source": AUTHORIZATION_SOURCE,
        "all_validation_passed": all(not result["validation_errors"] for result in results),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

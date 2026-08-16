#!/usr/bin/env python3
"""Apply agent-authorized epithelial subtype labels and export post-annotation DEGs."""

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
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/04-subtype-selected-clustering/epithelial/adata_epi_selected_clustered.h5ad"
)
SELECTED_TABLE = (
    PROJECT_ROOT
    / "tables/02-cell_subtype_integration_clustering/04-subtype-selected-clustering/epithelial/selected_clustering.csv"
)
TABLE_ROOT = (
    PROJECT_ROOT
    / "tables/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/epithelial"
)
MAPPING_PATH = TABLE_ROOT / "epithelial_subtype_mapping.csv"
CANDIDATE_PATH = TABLE_ROOT / "epithelial_subtype_candidate_table.csv"
FIGURE_DIR = (
    PROJECT_ROOT
    / "figures/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/epithelial"
)
PENDING_FIGURE_DIR = (
    PROJECT_ROOT
    / "figures/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/epithelial_pending_validation"
)
H5AD_DIR = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/epithelial"
)
PENDING_H5AD_DIR = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/epithelial_pending_validation"
)
OUTPUT_H5AD = H5AD_DIR / "adata_epi.h5ad"
PENDING_H5AD = PENDING_H5AD_DIR / OUTPUT_H5AD.name


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def package_version_lines() -> list[str]:
    packages = ["scanpy", "anndata", "numpy", "pandas", "scipy", "matplotlib"]
    code_files = [
        PROJECT_ROOT
        / "codes/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/epithelial/01_export_raw_subtype_cluster_degs.py",
        Path(__file__).resolve(),
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


def main() -> None:
    for path in [INPUT_H5AD, SELECTED_TABLE, MAPPING_PATH, CANDIDATE_PATH]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in [FIGURE_DIR, PENDING_FIGURE_DIR, H5AD_DIR, PENDING_H5AD_DIR]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite epithelial subtype artifact: {path}")

    selected = pd.read_csv(SELECTED_TABLE)
    if selected.shape[0] != 1:
        raise ValueError("Selected epithelial clustering table must contain one row")
    selected = selected.iloc[0]
    cluster_key = str(selected["final_raw_cluster_column"])
    n_pcs = int(selected["n_pcs"])
    n_neighbors = int(selected["n_neighbors"])
    resolution = float(selected["resolution"])
    resolution_text = str(resolution).replace(".", "p")
    deg_token = f"cell_subtype_pcs{n_pcs}_nn{n_neighbors}_res{resolution_text}"
    final_deg_dir = TABLE_ROOT / f"degs_{deg_token}"
    pending_deg_dir = TABLE_ROOT / "degs_cell_subtype_pending_validation"
    root_outputs = [
        TABLE_ROOT / "02_subtype_annotation_parameters.csv",
        TABLE_ROOT / "02_subtype_annotation_validation.json",
        TABLE_ROOT / "epithelial_subtype_counts.csv",
        TABLE_ROOT / "epithelial_subtype_counts_by_sample.csv",
        TABLE_ROOT / "epithelial_subtype_review_parameters.csv",
    ]
    for path in [final_deg_dir, pending_deg_dir, *root_outputs]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite epithelial subtype artifact: {path}")

    mapping = pd.read_csv(MAPPING_PATH, dtype={"cluster": str})
    candidate = pd.read_csv(CANDIDATE_PATH, dtype={"cluster": str})
    required_mapping = {
        "cluster",
        "cell_subtype",
        "functional_state",
        "gene_selection_rationale",
        "most_significant_deg_gene",
        "top_supporting_markers",
        "confidence",
        "review_depth",
    }
    missing = required_mapping.difference(mapping.columns)
    if missing:
        raise KeyError(f"Final epithelial subtype mapping lacks columns: {missing}")
    if mapping["cluster"].duplicated().any() or mapping["cell_subtype"].duplicated().any():
        raise ValueError("Final epithelial mapping must be one-to-one and subtype names unique")
    if not mapping["cell_subtype"].astype(str).str.startswith("Epi_").all():
        raise ValueError("Every epithelial subtype must use the Epi_ prefix")
    if not candidate["needs_user_confirmation"].eq(False).all():
        raise ValueError("Agent-authorized candidate table still requests user confirmation")

    started = time.time()
    adata = sc.read_h5ad(INPUT_H5AD)
    if adata.raw is None or cluster_key not in adata.obs or "X_umap" not in adata.obsm:
        raise ValueError("Selected epithelial input lacks raw/cluster/UMAP state")
    if "cell_subtype" in adata.obs:
        raise ValueError("Selected epithelial input is already subtype annotated")
    observed = set(adata.obs[cluster_key].astype(str).unique())
    if observed != set(mapping["cluster"]):
        raise ValueError("Final epithelial mapping does not exactly cover selected raw clusters")
    map_label = mapping.set_index("cluster")["cell_subtype"].to_dict()
    map_state = mapping.set_index("cluster")["functional_state"].to_dict()
    map_confidence = mapping.set_index("cluster")["confidence"].to_dict()
    adata.obs["cell_subtype"] = pd.Categorical(
        adata.obs[cluster_key].astype(str).map(map_label),
        categories=mapping["cell_subtype"].tolist(),
        ordered=True,
    )
    adata.obs["functional_state"] = (
        adata.obs[cluster_key].astype(str).map(map_state).astype("category")
    )
    adata.obs["subtype_annotation_confidence"] = (
        adata.obs[cluster_key].astype(str).map(map_confidence).astype("category")
    )
    adata.obs["cell_type"] = pd.Categorical(
        ["Epithelial Cells"] * adata.n_obs, categories=["Epithelial Cells"]
    )
    adata.obs["subtype_annotation_source"] = "saved_full_raw_cluster_DEGs_agent_authorized"

    sc.tl.rank_genes_groups(
        adata,
        groupby="cell_subtype",
        method="t-test",
        use_raw=True,
        n_genes=adata.raw.n_vars,
    )
    result = adata.uns["rank_genes_groups"]
    pending_deg_dir.mkdir(parents=True, exist_ok=False)
    deg_rows = []
    for subtype in mapping["cell_subtype"]:
        frame = pd.DataFrame(
            {
                "gene": result["names"][subtype],
                "score": result["scores"][subtype],
                "logfoldchanges": result["logfoldchanges"][subtype],
                "pvals": result["pvals"][subtype],
                "pvals_adj": result["pvals_adj"][subtype],
            }
        )
        if frame.shape[0] != adata.raw.n_vars:
            raise ValueError(f"Post-annotation subtype DEG table is not full-length: {subtype}")
        output_name = f"{safe_name(subtype)}_degs_{deg_token}.csv"
        frame.to_csv(pending_deg_dir / output_name, index=False)
        deg_rows.append(
            {
                "cell_subtype": subtype,
                "n_cells": int((adata.obs["cell_subtype"].astype(str) == subtype).sum()),
                "deg_rows": int(frame.shape[0]),
                "output_file": str(final_deg_dir / output_name),
            }
        )
    if len(list(pending_deg_dir.glob("*.csv"))) != mapping.shape[0]:
        raise FileNotFoundError("Post-annotation epithelial subtype DEG output set is incomplete")

    marker_dict = {
        row.cell_subtype: [row.most_significant_deg_gene]
        for row in mapping.itertuples(index=False)
    }
    raw_genes = set(map(str, adata.raw.var_names))
    missing_markers = [gene for genes in marker_dict.values() for gene in genes if gene not in raw_genes]
    if missing_markers:
        raise KeyError(f"Subtype name genes absent from adata.raw: {missing_markers}")
    PENDING_FIGURE_DIR.mkdir(parents=True, exist_ok=False)
    sc.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
    sc.settings.figdir = PENDING_FIGURE_DIR
    sc.pl.umap(adata, color="cell_subtype", show=False, save="_cell_subtype.pdf")
    sc.pl.dotplot(
        adata,
        var_names=marker_dict,
        groupby="cell_subtype",
        standard_scale="var",
        use_raw=True,
        show=False,
        save="cell_subtype_name_genes.pdf",
    )
    expected_figures = [
        PENDING_FIGURE_DIR / "umap_cell_subtype.pdf",
        PENDING_FIGURE_DIR / "dotplot_cell_subtype_name_genes.pdf",
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in expected_figures):
        raise FileNotFoundError("Epithelial subtype UMAP or dotplot is missing")

    adata.uns["epithelial_subtype_annotation_contract"] = {
        "raw_cluster_column": cluster_key,
        "annotation_column": "cell_subtype",
        "prefix": "Epi",
        "mapping_source": str(MAPPING_PATH),
        "candidate_source": str(CANDIDATE_PATH),
        "selection_source": "user_authorized_agent_led_annotation",
        "review_depth": int(mapping["review_depth"].iloc[0]),
        "round1_deg_use_raw": True,
        "round2_deg_use_raw": True,
        "post_annotation_deg_directory": str(final_deg_dir),
    }
    PENDING_H5AD_DIR.mkdir(parents=True, exist_ok=False)
    adata.write_h5ad(PENDING_H5AD, compression="gzip", compression_opts=1)
    pending = sc.read_h5ad(PENDING_H5AD, backed="r")
    errors = []
    if pending.shape != adata.shape:
        errors.append("shape mismatch")
    if pending.raw is None or pending.raw.n_vars != adata.raw.n_vars:
        errors.append("raw mismatch")
    if cluster_key not in pending.obs or "cell_subtype" not in pending.obs:
        errors.append("raw cluster or subtype column missing")
    if set(pending.obs["cell_subtype"].astype(str).unique()) != set(mapping["cell_subtype"]):
        errors.append("subtype vocabulary mismatch")
    pending.file.close()
    del pending
    if errors:
        raise ValueError(f"Pending annotated epithelial H5AD failed validation: {errors}")
    PENDING_H5AD_DIR.replace(H5AD_DIR)
    PENDING_FIGURE_DIR.replace(FIGURE_DIR)
    pending_deg_dir.replace(final_deg_dir)

    counts = adata.obs["cell_subtype"].value_counts(sort=False).rename("n_cells").reset_index()
    by_sample = pd.crosstab(adata.obs["sample"], adata.obs["cell_subtype"], dropna=False)
    counts.to_csv(TABLE_ROOT / "epithelial_subtype_counts.csv", index=False)
    by_sample.to_csv(TABLE_ROOT / "epithelial_subtype_counts_by_sample.csv")
    review_depth = int(mapping["review_depth"].iloc[0])
    pd.DataFrame(
        [
            {
                "annotation_level": "epithelial_subtype_selected_raw_clusters",
                "uniform_review_depth": review_depth,
                "selection_source": "user_authorized_agent_led_annotation_2026-08-06",
                "prefix_source": "documented_default_Epi",
            }
        ]
    ).to_csv(TABLE_ROOT / "epithelial_subtype_review_parameters.csv", index=False)
    elapsed = time.time() - started
    parameters = {
        "step": "epithelial_subtype_annotation_and_round2_deg",
        "input_h5ad": str(INPUT_H5AD),
        "output_h5ad": str(OUTPUT_H5AD),
        "raw_cluster_column": cluster_key,
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "n_obs": int(adata.n_obs),
        "n_vars_hvg": int(adata.n_vars),
        "n_vars_raw": int(adata.raw.n_vars),
        "n_subtypes": int(mapping.shape[0]),
        "review_depth": review_depth,
        "round2_deg_method": "t-test",
        "round2_deg_use_raw": True,
        "cell_subtype_unique_per_raw_cluster": True,
        "random_seed": SEED,
        "elapsed_seconds": elapsed,
        "code_file": str(Path(__file__).resolve()),
    }
    pd.DataFrame([parameters]).to_csv(TABLE_ROOT / "02_subtype_annotation_parameters.csv", index=False)
    (TABLE_ROOT / "package_versions.txt").write_text(
        "\n".join(package_version_lines()) + "\n", encoding="utf-8"
    )
    validation = {
        **parameters,
        "mapping_exactly_covers_raw_clusters": True,
        "subtype_names_unique": bool(mapping["cell_subtype"].is_unique),
        "all_subtypes_Epi_prefixed": True,
        "round2_groups": deg_rows,
        "round2_all_files_full_length": all(row["deg_rows"] == adata.raw.n_vars for row in deg_rows),
        "output_h5ad_readable": OUTPUT_H5AD.is_file(),
        "figures": [str(FIGURE_DIR / path.name) for path in expected_figures],
    }
    (TABLE_ROOT / "02_subtype_annotation_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

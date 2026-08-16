#!/usr/bin/env python3
"""Export complete pre-annotation epithelial subtype-cluster DEG tables."""

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
PENDING_DIR = TABLE_ROOT / "degs_raw_clusters_pending_validation"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def package_version_lines() -> list[str]:
    packages = ["scanpy", "anndata", "numpy", "pandas", "scipy"]
    lines = [
        "environment_category=rapids_with_scanpy_cpu_deg",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"random_seed={SEED}",
        f"code_files={Path(__file__).resolve()}",
    ]
    lines.extend(f"{package}={importlib.metadata.version(package)}" for package in packages)
    return lines


def main() -> None:
    if not INPUT_H5AD.is_file() or not SELECTED_TABLE.is_file():
        raise FileNotFoundError("Selected epithelial H5AD or selection table is missing")
    selected = pd.read_csv(SELECTED_TABLE)
    if selected.shape[0] != 1:
        raise ValueError("Selected epithelial clustering table must contain one row")
    selected = selected.iloc[0]
    cluster_key = str(selected["final_raw_cluster_column"])
    n_pcs = int(selected["n_pcs"])
    n_neighbors = int(selected["n_neighbors"])
    resolution_text = str(float(selected["resolution"])).replace(".", "p")
    deg_token = f"{cluster_key}_pcs{n_pcs}_nn{n_neighbors}_res{resolution_text}"
    final_dir = TABLE_ROOT / f"degs_{deg_token}"
    outputs = [
        final_dir,
        PENDING_DIR,
        TABLE_ROOT / "01_raw_cluster_deg_parameters.csv",
        TABLE_ROOT / "01_raw_cluster_deg_validation.json",
    ]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite epithelial DEG artifact: {path}")

    started = time.time()
    adata = sc.read_h5ad(INPUT_H5AD)
    if adata.raw is None:
        raise ValueError("Selected epithelial H5AD lacks all-gene normalized/log raw")
    if cluster_key not in adata.obs or "X_umap" not in adata.obsm:
        raise KeyError("Selected epithelial raw cluster column or UMAP is missing")
    if "cell_subtype" in adata.obs:
        raise ValueError("Pre-annotation DEG input already contains cell_subtype")
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category").cat.remove_unused_categories()
    groups = [str(value) for value in adata.obs[cluster_key].cat.categories]
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        method="t-test",
        use_raw=True,
        n_genes=adata.raw.n_vars,
    )
    result = adata.uns["rank_genes_groups"]
    PENDING_DIR.mkdir(parents=True, exist_ok=False)
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
            raise ValueError(f"Epithelial raw cluster {group} DEG table is not full-length")
        output_name = f"{safe_name(group)}_degs_{deg_token}.csv"
        frame.to_csv(PENDING_DIR / output_name, index=False)
        rows.append(
            {
                "group": group,
                "n_cells": int((adata.obs[cluster_key].astype(str) == group).sum()),
                "deg_rows": int(frame.shape[0]),
                "output_file": str(final_dir / output_name),
            }
        )
    if len(list(PENDING_DIR.glob("*.csv"))) != len(groups):
        raise FileNotFoundError("Epithelial raw-cluster DEG file set is incomplete")
    PENDING_DIR.replace(final_dir)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    elapsed = time.time() - started
    parameters = {
        "step": "epithelial_round1_raw_cluster_full_deg_export",
        "input_h5ad": str(INPUT_H5AD),
        "groupby": cluster_key,
        "method": "t-test",
        "use_raw": True,
        "n_obs": int(adata.n_obs),
        "n_vars_hvg_X": int(adata.n_vars),
        "n_vars_raw": int(adata.raw.n_vars),
        "n_groups": len(groups),
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "resolution": float(selected["resolution"]),
        "full_length_per_group": True,
        "combined_deg_csv_created": False,
        "top_only_deg_csv_created": False,
        "backend": "scanpy_CPU",
        "random_seed": SEED,
        "elapsed_seconds": elapsed,
        "code_file": str(Path(__file__).resolve()),
        "output_directory": str(final_dir),
    }
    pd.DataFrame([parameters]).to_csv(TABLE_ROOT / "01_raw_cluster_deg_parameters.csv", index=False)
    (TABLE_ROOT / "package_versions.txt").write_text(
        "\n".join(package_version_lines()) + "\n", encoding="utf-8"
    )
    validation = {
        **parameters,
        "groups": rows,
        "all_files_exist": all(Path(row["output_file"]).is_file() for row in rows),
        "all_files_full_length": all(row["deg_rows"] == adata.raw.n_vars for row in rows),
        "canonical_columns": ["gene", "score", "logfoldchanges", "pvals", "pvals_adj"],
        "cell_subtype_absent_during_round1": "cell_subtype" not in adata.obs,
    }
    (TABLE_ROOT / "01_raw_cluster_deg_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

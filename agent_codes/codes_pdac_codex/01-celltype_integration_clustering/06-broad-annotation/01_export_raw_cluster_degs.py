#!/usr/bin/env python3
"""Export complete raw-cluster DEG tables from the formal selected atlas."""

from __future__ import annotations

import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


SEED = 42
PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
INPUT_H5AD = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/05-clustering-parameter-search/selected/adata_inte.h5ad"
)
SELECTED_TABLE = (
    PROJECT_ROOT
    / "tables/01-celltype_integration_clustering/05-clustering-parameter-search/selected_clustering.csv"
)
TABLE_ROOT = PROJECT_ROOT / "tables/01-celltype_integration_clustering/06-broad-annotation"
PENDING_DEG_DIR = TABLE_ROOT / "degs_raw_clusters_pending_validation"
PARAMETER_TABLE = TABLE_ROOT / "01_raw_cluster_deg_parameters.csv"
VALIDATION_PATH = TABLE_ROOT / "01_raw_cluster_deg_validation.json"
PACKAGE_VERSIONS = TABLE_ROOT / "package_versions.txt"


def safe_group_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def package_version_lines(code_file: Path) -> list[str]:
    packages = ["scanpy", "anndata", "numpy", "pandas", "scipy"]
    lines = [
        "environment_category=rapids_with_scanpy_cpu_deg",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"random_seed={SEED}",
        f"code_files={code_file.resolve()}",
    ]
    lines.extend(f"{package}={importlib.metadata.version(package)}" for package in packages)
    return lines


def main() -> None:
    if not INPUT_H5AD.is_file() or not SELECTED_TABLE.is_file():
        raise FileNotFoundError("Formal selected H5AD or selected-clustering record is missing")
    selection = pd.read_csv(SELECTED_TABLE)
    if selection.shape[0] != 1:
        raise ValueError("Selected-clustering record must have exactly one row")
    selected = selection.iloc[0]
    cluster_key = str(selected["final_raw_cluster_column"])
    n_pcs = int(selected["n_pcs"])
    n_neighbors = int(selected["n_neighbors"])
    resolution_text = str(selected["resolution"]).replace(".", "p")
    final_deg_dir = TABLE_ROOT / (
        f"degs_{cluster_key}_pcs{n_pcs}_nn{n_neighbors}_res{resolution_text}"
    )
    for path in [final_deg_dir, PENDING_DEG_DIR, PARAMETER_TABLE, VALIDATION_PATH]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite broad-annotation artifact: {path}")

    started = time.time()
    adata = sc.read_h5ad(INPUT_H5AD)
    if adata.raw is None:
        raise ValueError("Selected H5AD lacks adata.raw normalized/log all-gene expression")
    if cluster_key not in adata.obs:
        raise KeyError(f"Selected H5AD lacks raw cluster column {cluster_key!r}")
    if "leiden_coarse" in adata.obs:
        raise ValueError("Selected clustering input already contains leiden_coarse")
    if not adata.obs_names.is_unique:
        raise ValueError("Selected H5AD obs_names are not unique")
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category").cat.remove_unused_categories()
    groups = [str(value) for value in adata.obs[cluster_key].cat.categories]
    if len(groups) < 2:
        raise ValueError("Raw-cluster DEG requires at least two non-empty clusters")

    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        method="t-test",
        use_raw=True,
        n_genes=adata.raw.n_vars,
    )
    result = adata.uns["rank_genes_groups"]
    required = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]
    missing = [key for key in required if key not in result]
    if missing:
        raise KeyError(f"rank_genes_groups result lacks fields: {missing}")

    PENDING_DEG_DIR.mkdir(parents=True, exist_ok=False)
    output_rows = []
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
            raise ValueError(f"Group {group} DEG row count is not full-length")
        output_path = PENDING_DEG_DIR / (
            f"{safe_group_name(group)}_degs_{cluster_key}_pcs{n_pcs}_nn{n_neighbors}_res{resolution_text}.csv"
        )
        frame.to_csv(output_path, index=False)
        output_rows.append(
            {
                "group": group,
                "n_cells": int((adata.obs[cluster_key].astype(str) == group).sum()),
                "deg_rows": int(frame.shape[0]),
                "output_file": str(final_deg_dir / output_path.name),
            }
        )

    written = sorted(PENDING_DEG_DIR.glob("*.csv"))
    if len(written) != len(groups) or any(path.stat().st_size == 0 for path in written):
        raise FileNotFoundError("Complete raw-cluster DEG file set was not written")
    PENDING_DEG_DIR.replace(final_deg_dir)

    elapsed = time.time() - started
    parameters = {
        "step": "round1_raw_cluster_full_deg_export",
        "input_h5ad": str(INPUT_H5AD),
        "groupby": cluster_key,
        "method": "t-test",
        "use_raw": True,
        "raw_expression_contract": "normalize_total_1e4_log1p_all_shared_genes",
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
        "output_directory": str(final_deg_dir),
    }
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([parameters]).to_csv(PARAMETER_TABLE, index=False)
    PACKAGE_VERSIONS.write_text(
        "\n".join(package_version_lines(Path(__file__))) + "\n", encoding="utf-8"
    )
    validation = {
        **parameters,
        "groups": output_rows,
        "all_files_exist": all(Path(row["output_file"]).is_file() for row in output_rows),
        "all_files_full_length": all(row["deg_rows"] == adata.raw.n_vars for row in output_rows),
        "canonical_columns": ["gene", "score", "logfoldchanges", "pvals", "pvals_adj"],
        "leiden_coarse_absent_during_round1": "leiden_coarse" not in adata.obs,
    }
    VALIDATION_PATH.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

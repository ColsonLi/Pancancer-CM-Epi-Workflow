#!/usr/bin/env python3
"""Profile raw-count QC distributions before choosing explicit PDAC thresholds."""

from __future__ import annotations

import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
INPUT_H5AD = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/02-merge-metadata/adata_merge.h5ad"
)
TABLE_DIR = PROJECT_ROOT / "tables/01-celltype_integration_clustering/03-qc/01-qc-distribution-profile"
FIGURE_DIR = PROJECT_ROOT / "figures/01-celltype_integration_clustering/03-qc/01-qc-distribution-profile"

QUANTILES = [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99, 0.995, 0.999, 1.0]
QC_METRICS = ["n_genes_by_counts", "total_counts", "pct_counts_MT", "pct_counts_RIBO"]


def require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def metric_quantiles(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = [((), frame)] if not group_columns else frame.groupby(group_columns, observed=True)
    for keys, subset in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_values = dict(zip(group_columns, keys, strict=True))
        for metric in QC_METRICS:
            values = subset[metric].astype(float)
            quantile_values = values.quantile(QUANTILES)
            for quantile, value in quantile_values.items():
                rows.append(
                    {
                        **group_values,
                        "metric": metric,
                        "quantile": quantile,
                        "value": float(value),
                        "n_cells": int(subset.shape[0]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    if not INPUT_H5AD.is_file():
        raise FileNotFoundError(INPUT_H5AD)
    outputs = [
        TABLE_DIR / "qc_distribution_quantiles_global.csv",
        TABLE_DIR / "qc_distribution_quantiles_by_series.csv",
        TABLE_DIR / "qc_distribution_quantiles_by_sample.csv",
        TABLE_DIR / "sample_qc_summary.csv",
        TABLE_DIR / "gene_detection_quantiles.csv",
        TABLE_DIR / "profile_parameters.csv",
        TABLE_DIR / "profile_validation.json",
    ]
    for path in outputs:
        require_new(path)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(INPUT_H5AD)
    if not sparse.issparse(adata.X):
        raise TypeError("Raw-count atlas X must be sparse")
    if "counts" not in adata.layers:
        raise ValueError("Raw-count atlas lacks layers['counts']")
    if adata.X.shape != adata.layers["counts"].shape:
        raise ValueError("X and layers['counts'] shapes differ")
    if np.count_nonzero(adata.X.data != np.floor(adata.X.data)):
        raise ValueError("Raw-count atlas X is not integer-valued")

    adata.var["MT"] = adata.var_names.astype(str).str.startswith("MT-")
    adata.var["RIBO"] = adata.var_names.astype(str).str.startswith("RPS")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["MT", "RIBO"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    metric_quantiles(adata.obs, []).to_csv(outputs[0], index=False)
    metric_quantiles(adata.obs, ["series"]).to_csv(outputs[1], index=False)
    metric_quantiles(adata.obs, ["series", "sample"]).to_csv(outputs[2], index=False)
    sample_summary = (
        adata.obs.groupby(["series", "sample", "status"], observed=True)
        .agg(
            n_cells=("sample", "size"),
            median_n_genes=("n_genes_by_counts", "median"),
            median_total_counts=("total_counts", "median"),
            median_pct_mt=("pct_counts_MT", "median"),
            median_pct_ribo=("pct_counts_RIBO", "median"),
            p99_n_genes=("n_genes_by_counts", lambda values: values.quantile(0.99)),
            p99_pct_mt=("pct_counts_MT", lambda values: values.quantile(0.99)),
        )
        .reset_index()
    )
    sample_summary.to_csv(outputs[3], index=False)

    gene_cells = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    gene_counts = np.asarray(adata.X.sum(axis=0)).ravel()
    gene_quantiles = pd.DataFrame(
        {
            "quantile": QUANTILES,
            "n_cells_by_counts": np.quantile(gene_cells, QUANTILES),
            "total_counts": np.quantile(gene_counts, QUANTILES),
            "n_genes": adata.n_vars,
        }
    )
    gene_quantiles.to_csv(outputs[4], index=False)

    sc.set_figure_params(figsize=(3, 3), dpi=150)
    sc.settings.figdir = FIGURE_DIR
    sc.pl.violin(
        adata,
        keys=QC_METRICS,
        groupby="series",
        jitter=0,
        multi_panel=True,
        show=False,
        save="_pre_qc_by_series.pdf",
    )
    sc.pl.violin(
        adata,
        keys=QC_METRICS,
        groupby="status",
        jitter=0,
        multi_panel=True,
        show=False,
        save="_pre_qc_by_status.pdf",
    )

    pd.DataFrame(
        [
            {
                "step": "03-qc-distribution-profile",
                "input_h5ad": str(INPUT_H5AD),
                "output_h5ad_or_object": "none_read_only_profile",
                "n_obs_before": adata.n_obs,
                "n_obs_after": adata.n_obs,
                "n_vars_before": adata.n_vars,
                "n_vars_after": adata.n_vars,
                "qc_vars": "MT;RIBO",
                "mt_prefix": "MT-",
                "ribo_prefix": "RPS",
                "percent_top": "None",
                "backend_package": f"scanpy={sc.__version__};anndata={ad.__version__}",
                "code_file": str(Path(__file__).resolve()),
                "random_seed": SEED,
                "notes": "No thresholds applied; distributions support agent threshold selection",
            }
        ]
    ).to_csv(outputs[5], index=False)
    validation = {
        "input_h5ad": str(INPUT_H5AD),
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "n_samples": int(adata.obs["sample"].nunique()),
        "n_series": int(adata.obs["series"].nunique()),
        "mt_gene_count": int(adata.var["MT"].sum()),
        "ribo_gene_count": int(adata.var["RIBO"].sum()),
        "thresholds_applied": False,
        "random_seed": SEED,
    }
    outputs[6].write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()

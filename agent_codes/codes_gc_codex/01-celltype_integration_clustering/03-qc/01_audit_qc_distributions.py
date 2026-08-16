#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[4]
INPUT = ROOT / "epi-cm-core-workflow/h5ad/01-celltype_integration_clustering/02-merge-metadata/adata_merge.h5ad"
TABLE_DIR = ROOT / "epi-cm-core-workflow/tables/01-celltype_integration_clustering/03-qc"
FIGURE_DIR = ROOT / "epi-cm-core-workflow/figures/01-celltype_integration_clustering/03-qc"


def write_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_figure_new(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".svg"]:
        path = stem.with_suffix(suffix)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
        fig.savefig(path, bbox_inches="tight")


adata = ad.read_h5ad(INPUT)
adata.X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
symbols = adata.var_names.astype(str)
mt_mask = symbols.str.startswith("MT-")
ribo_mask = symbols.str.startswith("RPS")

total_counts = np.asarray(adata.X.sum(axis=1)).ravel()
n_genes = np.diff(adata.X.indptr)
mt_counts = np.asarray(adata.X[:, mt_mask].sum(axis=1)).ravel()
ribo_counts = np.asarray(adata.X[:, ribo_mask].sum(axis=1)).ravel()
pct_mt = np.divide(mt_counts, total_counts, out=np.zeros_like(mt_counts, dtype=float), where=total_counts > 0) * 100
pct_ribo = np.divide(ribo_counts, total_counts, out=np.zeros_like(ribo_counts, dtype=float), where=total_counts > 0) * 100

metrics = pd.DataFrame(
    {
        "n_genes_by_counts": n_genes,
        "total_counts": total_counts,
        "total_counts_MT": mt_counts,
        "pct_counts_MT": pct_mt,
        "total_counts_RIBO": ribo_counts,
        "pct_counts_RIBO": pct_ribo,
        "sample": adata.obs["sample"].astype(str).to_numpy(),
        "series": adata.obs["series"].astype(str).to_numpy(),
        "status": adata.obs["status"].astype(str).to_numpy(),
    },
    index=adata.obs_names,
)

quantiles = [0, 0.001, 0.005, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995, 0.999, 1]
metric_names = ["n_genes_by_counts", "total_counts", "pct_counts_MT", "pct_counts_RIBO"]


def quantile_table(group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(group_columns, observed=True, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        prefix = dict(zip(group_columns, keys))
        for metric in metric_names:
            values = group[metric].to_numpy()
            q_values = np.quantile(values, quantiles)
            for q, value in zip(quantiles, q_values):
                rows.append(
                    {
                        **prefix,
                        "n_cells": len(group),
                        "metric": metric,
                        "quantile": q,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


write_new(TABLE_DIR / "01_qc_quantiles_by_sample.csv", quantile_table(["series", "sample", "status"]))
write_new(TABLE_DIR / "01_qc_quantiles_by_series_status.csv", quantile_table(["series", "status"]))

cell_candidates = []
for threshold in [50, 100, 200, 300, 500, 750, 1000]:
    cell_candidates.append(
        {
            "filter": "min_genes",
            "threshold": threshold,
            "n_cells_removed": int((n_genes < threshold).sum()),
            "fraction_removed": float((n_genes < threshold).mean()),
        }
    )
for threshold in [4000, 5000, 6000, 7000, 8000, 10000, 12000]:
    cell_candidates.append(
        {
            "filter": "max_genes",
            "threshold": threshold,
            "n_cells_removed": int((n_genes > threshold).sum()),
            "fraction_removed": float((n_genes > threshold).mean()),
        }
    )
for threshold in [10, 15, 20, 25, 30, 40, 50]:
    cell_candidates.append(
        {
            "filter": "max_pct_mt",
            "threshold": threshold,
            "n_cells_removed": int((pct_mt > threshold).sum()),
            "fraction_removed": float((pct_mt > threshold).mean()),
        }
    )
write_new(TABLE_DIR / "01_cell_threshold_candidate_retention.csv", pd.DataFrame(cell_candidates))

combined_rows = []
for min_genes in [200, 300, 500]:
    for max_genes in [6000, 7000, 8000]:
        for max_pct_mt in [20, 25, 30, 40]:
            keep = (n_genes >= min_genes) & (n_genes <= max_genes) & (pct_mt <= max_pct_mt)
            combined_rows.append(
                {
                    "min_genes_final": min_genes,
                    "max_genes_final": max_genes,
                    "max_pct_mt": max_pct_mt,
                    "n_cells_retained_before_doublet_filter": int(keep.sum()),
                    "n_cells_removed_before_doublet_filter": int((~keep).sum()),
                    "fraction_retained_before_doublet_filter": float(keep.mean()),
                }
            )
write_new(TABLE_DIR / "01_combined_cell_threshold_grid.csv", pd.DataFrame(combined_rows))

x_csc = adata.X.tocsc()
gene_n_cells = np.diff(x_csc.indptr)
gene_total_counts = np.asarray(x_csc.sum(axis=0)).ravel()
gene_rows = []
for min_cells in [1, 3, 5, 10, 20, 50, 100]:
    for min_counts in [1, 3, 5, 10, 20, 50, 100]:
        keep = (gene_n_cells >= min_cells) & (gene_total_counts >= min_counts)
        gene_rows.append(
            {
                "min_cells_gene": min_cells,
                "min_counts_gene": min_counts,
                "n_genes_retained_before_cell_qc": int(keep.sum()),
                "n_genes_removed_before_cell_qc": int((~keep).sum()),
            }
        )
write_new(TABLE_DIR / "01_gene_threshold_grid.csv", pd.DataFrame(gene_rows))
write_new(
    TABLE_DIR / "01_gene_detection_summary.csv",
    pd.DataFrame(
        {
            "gene_symbol": adata.var_names,
            "n_cells_by_counts": gene_n_cells,
            "total_counts": gene_total_counts,
            "is_MT": mt_mask,
            "is_RPS": ribo_mask,
        }
    ),
)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
plot_specs = [
    (np.log10(n_genes + 1), "log10(n_genes_by_counts + 1)"),
    (np.log10(total_counts + 1), "log10(total_counts + 1)"),
    (pct_mt, "pct_counts_MT"),
    (pct_ribo, "pct_counts_RIBO"),
]
for ax, (values, label) in zip(axes.ravel(), plot_specs):
    ax.hist(values, bins=120, color="#4C78A8", edgecolor="none")
    ax.set_xlabel(label)
    ax.set_ylabel("Cells")
fig.suptitle(f"Pre-QC distributions: {adata.n_obs:,} cells, {adata.n_vars:,} shared genes")
fig.tight_layout()
save_figure_new(fig, FIGURE_DIR / "01_qc_metric_distributions")
plt.close(fig)

series_order = list(dict.fromkeys(metrics["series"]))
fig, axes = plt.subplots(len(series_order), 3, figsize=(10, 2.1 * len(series_order)), squeeze=False)
for row, series in enumerate(series_order):
    mask = metrics["series"].eq(series).to_numpy()
    values = [np.log10(n_genes[mask] + 1), np.log10(total_counts[mask] + 1), pct_mt[mask]]
    labels = ["log10 genes", "log10 counts", "pct MT"]
    for ax, array, label in zip(axes[row], values, labels):
        ax.hist(array, bins=80, color="#72B7B2", edgecolor="none")
        ax.set_xlabel(label)
        ax.set_ylabel(series if label == "log10 genes" else "")
fig.suptitle("Pre-QC distributions by series")
fig.tight_layout()
save_figure_new(fig, FIGURE_DIR / "01_qc_metric_distributions_by_series")
plt.close(fig)

print(
    {
        "shape": adata.shape,
        "median_n_genes": float(np.median(n_genes)),
        "median_total_counts": float(np.median(total_counts)),
        "median_pct_mt": float(np.median(pct_mt)),
        "mt_genes": int(mt_mask.sum()),
        "rps_genes": int(ribo_mask.sum()),
    }
)

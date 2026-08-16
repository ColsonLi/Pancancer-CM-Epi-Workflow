#!/usr/bin/env python3
"""Plot the existing broad-annotation UMAP colored by source series."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SEED = 42
WORKFLOW_ROOT = Path("/mnt/disk18t/lr_xcy/riku/crc_codex/epi-cm-core-workflow")
STEP = "01-celltype_integration_clustering/06-broad-annotation"
INPUT_H5AD = WORKFLOW_ROOT / "h5ad" / STEP / "adata_anno.h5ad"
FIGURE_DIR = WORKFLOW_ROOT / "figures" / STEP
OUTPUT_PDF = FIGURE_DIR / "umap_series.pdf"
OUTPUT_SVG = FIGURE_DIR / "umap_series.svg"

# Matplotlib's qualitative palettes provide 40 deterministic categorical colors.
SERIES_COLORS = [
    *[matplotlib.colors.to_hex(c) for c in plt.get_cmap("tab20").colors],
    *[matplotlib.colors.to_hex(c) for c in plt.get_cmap("tab20b").colors],
]


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(INPUT_H5AD, backed="r")
    try:
        if "X_umap" not in adata.obsm:
            raise KeyError(f"{INPUT_H5AD}: missing obsm['X_umap']")
        if "series" not in adata.obs:
            raise KeyError(f"{INPUT_H5AD}: missing obs['series']")

        coords = np.asarray(adata.obsm["X_umap"][:, :2], dtype=np.float32)
        series = adata.obs["series"].astype("string").fillna("Missing").to_numpy()
    finally:
        adata.file.close()

    labels = sorted(pd.unique(series).tolist())
    if len(labels) > len(SERIES_COLORS):
        raise ValueError(
            f"Found {len(labels)} series but only {len(SERIES_COLORS)} colors are defined"
        )

    palette = dict(zip(labels, SERIES_COLORS[: len(labels)], strict=True))
    point_colors = np.asarray([palette[label] for label in series])

    # Shuffle drawing order so no source series is systematically placed on top.
    draw_order = np.random.default_rng(SEED).permutation(coords.shape[0])

    fig, ax = plt.subplots(figsize=(12.0, 8.5))
    ax.scatter(
        coords[draw_order, 0],
        coords[draw_order, 1],
        c=point_colors[draw_order],
        s=0.45,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title("CRC single-cell UMAP colored by series", fontsize=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()

    counts = pd.Series(series).value_counts()
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=palette[label],
            markeredgecolor="none",
            markersize=5.5,
            label=f"{label} (n={int(counts[label]):,})",
        )
        for label in labels
    ]
    ax.legend(
        handles=handles,
        title=f"series ({len(labels)})",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=7.0,
        title_fontsize=8.0,
        handletextpad=0.4,
        labelspacing=0.38,
        borderaxespad=0,
    )

    fig.savefig(OUTPUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_SVG, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"cells={coords.shape[0]}")
    print(f"series={len(labels)}")
    print(OUTPUT_PDF)
    print(OUTPUT_SVG)


if __name__ == "__main__":
    main()

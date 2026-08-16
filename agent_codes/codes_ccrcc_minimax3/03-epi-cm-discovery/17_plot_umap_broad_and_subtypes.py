"""ccRCC UMAP colored by 66 cell_subtypes + broad cell_type labels overlaid + legend.

Output:
  ccrcc_umap_broad_and_subtypes.{pdf,png}

  - Cells colored by individual cell_subtype (66 distinct hues via colorsys)
  - Broad cell_type labels overlaid at median centroid with rounded bg boxes
  - Bottom legend lists every cell_subtype name with its dot, 5 columns

Style adapted from:
  /mnt/disk18t/lr_xcy/riku/crc_codex/cytoscape_umap_cm_epi_overview/codes/02_plot_tumor_singlecell_spatial_cm_epi.py
  -> draw_subtype_umap, overlay_broad_labels, build_subtype_palette
"""
from __future__ import annotations

import colorsys
import warnings
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
ROOT  = Path("/mnt/disk18t/lr_xcy/riku/ccrcc-cmepi_minimax3")
ATLAS = ROOT / "epi-cm-core-workflow/h5ad/02-cell_subtype_integration_clustering/06-ccrcc-subtypes-to-full-adata/adata_anno_cellsubtype.h5ad"
OUT   = ROOT
N_COLS = 5  # legend columns

# Broad cell-type palette (used as basis for subtype derivation)
PALETTE_CT = {
    "Epi":  "#d62728", "Endo": "#1f77b4", "S":   "#9467bd",
    "T":    "#2ca02c", "NK":   "#17becf", "Mye": "#ff7f0e",
    "B":    "#e377c2", "Mast": "#8c564b",
}
SHORT_LABEL = {"Epi": "Epi", "Endo": "Endo", "S": "S", "T": "T",
               "NK": "NK", "Mye": "Mye", "B": "B", "Mast": "Mast"}

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 11, "axes.labelsize": 9,
})


# ---------------------------------------------------------------------------
# Helpers (mirroring reference behavior)
# ---------------------------------------------------------------------------
def readable_text_color(hex_color: str) -> str:
    rgb = mpl.colors.to_rgb(hex_color)
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "black" if luminance > 0.62 else "white"


def build_subtype_palette(mapping: pd.DataFrame, broad_palette: dict) -> dict:
    """Derive a unique color per cell_subtype from its parent cell_type."""
    palette = {}
    for broad, group in mapping.groupby("cell_type", sort=True):
        subtypes = group["cell_subtype"].tolist()
        base_rgb = mpl.colors.to_rgb(broad_palette[broad])
        base_h, base_l, base_s = colorsys.rgb_to_hls(*base_rgb)
        for index, subtype in enumerate(subtypes):
            fraction = 0.5 if len(subtypes) == 1 else index / (len(subtypes) - 1)
            hue = (base_h + (fraction - 0.5) * 0.12) % 1.0
            lightness = float(np.clip(0.38 + 0.34 * fraction, 0.30, 0.78))
            saturation = float(np.clip(max(base_s, 0.58) * (0.92 - 0.18 * fraction), 0.48, 0.95))
            palette[subtype] = mpl.colors.to_hex(colorsys.hls_to_rgb(hue, lightness, saturation))
    return palette


def load_data():
    a = ad.read_h5ad(ATLAS, backed="r")
    try:
        coords = np.asarray(a.obsm["X_umap"][:, :2], dtype=np.float32)
        subtype = a.obs["cell_subtype"].astype(str)
        broad   = a.obs["cell_type"].astype(str)
    finally:
        a.file.close()

    mapping = (pd.DataFrame({"cell_subtype": subtype, "cell_type": broad})
               .drop_duplicates()
               .sort_values(["cell_type", "cell_subtype"], kind="stable")
               .reset_index(drop=True))
    conflicts = mapping.groupby("cell_subtype")["cell_type"].nunique()
    if int((conflicts > 1).sum()):
        raise ValueError("subtype maps to multiple cell_type")
    return coords, subtype, broad, mapping


def compute_centroids(coords, broad):
    rows = []
    for ct in sorted(broad.unique()):
        m = broad.eq(ct).to_numpy()
        xy = coords[m]
        rows.append({"cell_type": ct,
                     "n_cells": int(m.sum()),
                     "umap1_median": float(np.median(xy[:, 0])),
                     "umap2_median": float(np.median(xy[:, 1]))})
    return pd.DataFrame(rows)


def draw_subtype_umap(ax, coords, subtype_labels, mapping, palette):
    counts = subtype_labels.value_counts()
    # draw largest first so smaller sit on top
    for st in counts.sort_values(ascending=False).index:
        m = subtype_labels.eq(st).to_numpy()
        ax.scatter(coords[m, 0], coords[m, 1], s=0.30, c=palette[st],
                   alpha=0.66, linewidths=0, rasterized=True)


def overlay_broad_labels(ax, centroids):
    for row in centroids.itertuples(index=False):
        color = PALETTE_CT[row.cell_type]
        ax.text(row.umap1_median, row.umap2_median,
                SHORT_LABEL.get(row.cell_type, row.cell_type),
                ha="center", va="center", fontsize=7.4, fontweight="bold",
                color=readable_text_color(color),
                bbox={"boxstyle": "round,pad=0.24,rounding_size=0.12",
                      "facecolor": color, "edgecolor": "white",
                      "linewidth": 0.8, "alpha": 0.96},
                zorder=20)


def main():
    coords, subtype, broad, mapping = load_data()
    palette = build_subtype_palette(mapping, PALETTE_CT)
    centroids = compute_centroids(coords, broad)
    print(f"[info] cells={len(subtype)} | subtypes={subtype.nunique()} "
          f"| broad={broad.nunique()}", flush=True)

    # --- Figure: UMAP on top, legend at bottom ---
    fig = plt.figure(figsize=(9.2, 10.5))
    umap_ax = fig.add_axes([0.14, 0.31, 0.72, 0.62])
    draw_subtype_umap(umap_ax, coords, subtype, mapping, palette)
    overlay_broad_labels(umap_ax, centroids)

    umap_ax.set_xlabel("UMAP1"); umap_ax.set_ylabel("UMAP2")
    umap_ax.set_title(f"Cell subtypes ({subtype.nunique()}) with broad cell-type annotations",
                      loc="left", fontweight="bold")
    umap_ax.set_box_aspect(1); umap_ax.grid(False)
    for sp in umap_ax.spines.values():
        sp.set_color("#4d4d4d"); sp.set_linewidth(0.7)

    # legend: 66 subtype names with colored dots, 5 columns
    handles = [Line2D([0], [0], marker="o", linestyle="none",
                     markerfacecolor=palette[row.cell_subtype],
                     markeredgecolor="none", markersize=4,
                     label=row.cell_subtype)
               for row in mapping.itertuples(index=False)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.018),
               ncol=N_COLS, frameon=False, fontsize=5.4,
               handletextpad=0.25, columnspacing=0.75, labelspacing=0.42)

    fig.suptitle("ccRCC single-cell UMAP", y=0.98, fontsize=13, fontweight="bold")

    stem = "ccrcc_umap_broad_and_subtypes"
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[done] {stem} written", flush=True)


if __name__ == "__main__":
    main()
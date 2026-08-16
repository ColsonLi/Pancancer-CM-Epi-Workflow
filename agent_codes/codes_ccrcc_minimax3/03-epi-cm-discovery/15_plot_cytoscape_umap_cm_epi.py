"""ccRCC overview figure following the CRC reference style.

Outputs (saved under /mnt/disk18t/lr_xcy/riku/ccrcc-cmepi_minimax3/):
  ccrcc_overall_umap_compact_tumor_cm_epi_network.{pdf,png}
      Panel A: full ccrCC atlas UMAP, cell-type labels at centroids (no kNN edges)
      Panel B: compact bipartite network of tumor CM <-> Epi subtype
               (Spearman q<0.05, edges colored by sign, width by |rho|)

Style adapted from:
  /mnt/disk18t/lr_xcy/riku/crc_codex/cytoscape_umap_cm_epi_overview/codes/01_plot_cytoscape_umap_cm_epi.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/mnt/disk18t/lr_xcy/riku/ccrcc-cmepi_minimax3")
ATLAS = ROOT / "epi-cm-core-workflow/h5ad/02-cell_subtype_integration_clustering/06-ccrcc-subtypes-to-full-adata/adata_anno_cellsubtype.h5ad"
ASSOC_DIR = ROOT / "epi-cm-core-workflow/tables/03-epi-cm-discovery/association-spearman"
CM_CLASS = ROOT / "epi-cm-core-workflow/tables/03-epi-cm-discovery/joint_module_classification.csv"

OUT_DIR = ROOT
CONTEXT = "tumor"
Q_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 11, "axes.labelsize": 9,
})

# Broad cell-type palette (used for UMAP)
PALETTE_CT = {
    "Epi":  "#d62728", "Endo": "#1f77b4", "S":   "#9467bd",
    "T":    "#2ca02c", "NK":   "#17becf", "Mye": "#ff7f0e",
    "B":    "#e377c2", "Mast": "#8c564b",
}

# CM palette (matches CRC reference)
PALETTE_CM = {"tCM": "#d73027", "normalCM": "#4575b4", "sharedCM": "#8c8c8c"}

# Programmatic palette for Epi subtypes (11 subtypes; tab10 extended)
PALETTE_EPI = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def readable_text_color(hex_color: str) -> str:
    rgb = mpl.colors.to_rgb(hex_color)
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "black" if luminance > 0.62 else "white"


def load_umap():
    """Load UMAP coords + cell_type labels; compute centroids per cell_type."""
    a = ad.read_h5ad(ATLAS, backed="r")
    try:
        coords = np.asarray(a.obsm["X_umap"][:, :2], dtype=np.float32)
        labels = a.obs["cell_type"].astype(str)
    finally:
        a.file.close()

    rows = []
    for ct in sorted(labels.unique()):
        mask = labels.eq(ct).to_numpy()
        xy = coords[mask]
        rows.append({
            "cell_type": ct,
            "n_cells": int(mask.sum()),
            "umap1_median": float(np.median(xy[:, 0])),
            "umap2_median": float(np.median(xy[:, 1])),
        })
    return coords, labels, pd.DataFrame(rows)


def load_network():
    """Load Spearman rho + q matrices and CM classification; build nodes/edges."""
    rho = pd.read_csv(ASSOC_DIR / f"balanced_joint_cm_epi_cm_association_{CONTEXT}_rho_matrix.csv", index_col=0)
    q   = pd.read_csv(ASSOC_DIR / f"balanced_joint_cm_epi_cm_association_{CONTEXT}_q_matrix.csv", index_col=0)

    cm_class = pd.read_csv(CM_CLASS).sort_values("global_order", kind="stable")
    cm_order = cm_class["CM"].astype(str).tolist()
    cm_label = dict(zip(cm_class["CM"], cm_class["class"]))
    if set(cm_order) != set(rho.columns.astype(str)):
        raise ValueError("CM classification vs rho matrix mismatch")

    epi_order = rho.index.astype(str).tolist()
    epi_palette = dict(zip(epi_order, [PALETTE_EPI[i % len(PALETTE_EPI)] for i in range(len(epi_order))]))

    cm_y = np.linspace(0.94, 0.06, len(cm_order))
    epi_y = np.linspace(0.91, 0.09, len(epi_order))
    nodes = []
    for cm, y in zip(cm_order, cm_y):
        cls = cm_label[cm]
        nodes.append({"id": cm, "node_type": "CM", "class": cls,
                      "color": PALETTE_CM[cls], "x": 0.18, "y": float(y)})
    for epi, y in zip(epi_order, epi_y):
        nodes.append({"id": epi, "label": epi.replace("Epi_", ""), "node_type": "Epithelial subtype",
                      "class": "Epi", "color": epi_palette[epi], "x": 0.82, "y": float(y)})
    nodes = pd.DataFrame(nodes)

    edges = []
    for epi in epi_order:
        for cm in cm_order:
            r = float(rho.loc[epi, cm]); qv = float(q.loc[epi, cm])
            if np.isfinite(r) and np.isfinite(qv) and qv < Q_THRESHOLD:
                edges.append({"source": cm, "target": epi,
                              "rho": r, "q_value": qv,
                              "direction": "positive" if r > 0 else "negative",
                              "edge_color": "#b2182b" if r > 0 else "#2166ac",
                              "edge_width": 0.45 + 4.0 * abs(r)})
    edges = pd.DataFrame(edges)
    if edges.empty:
        raise ValueError(f"No significant {CONTEXT}-context CM-Epi edges at q<{Q_THRESHOLD}")
    return nodes, edges


def draw_umap(ax, coords, labels, centroids):
    """UMAP scatter with text labels at centroids."""
    short = {"Epi": "Epi", "Endo": "Endo", "S": "S", "T": "T",
             "NK": "NK", "Mye": "Mye", "B": "B", "Mast": "Mast"}

    order = centroids.sort_values("n_cells", ascending=False)["cell_type"].tolist()
    for ct in order:
        m = labels.eq(ct).to_numpy()
        ax.scatter(coords[m, 0], coords[m, 1], s=0.30, c=PALETTE_CT[ct],
                   alpha=0.62, linewidths=0, rasterized=True, label=ct)

    for row in centroids.itertuples(index=False):
        ax.text(row.umap1_median, row.umap2_median,
                short.get(row.cell_type, row.cell_type),
                ha="center", va="center", fontsize=7.2, fontweight="bold",
                color=readable_text_color(PALETTE_CT[row.cell_type]),
                bbox={"boxstyle": "round,pad=0.24,rounding_size=0.12",
                      "facecolor": PALETTE_CT[row.cell_type],
                      "edgecolor": "white", "linewidth": 0.8, "alpha": 0.96},
                zorder=20)

    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("A  Full ccRCC atlas UMAP", loc="left", fontweight="bold")
    ax.set_box_aspect(1)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color("#4d4d4d"); sp.set_linewidth(0.7)


def draw_network(ax, nodes, edges):
    """Bipartite cytoscape-style network (CM <-> Epi)."""
    positions = nodes.set_index("id")[["x", "y"]].to_dict("index")
    cm_idx = {n: i for i, n in enumerate(nodes.loc[nodes.node_type == "CM", "id"])}
    epi_idx = {n: i for i, n in enumerate(nodes.loc[nodes.node_type == "Epithelial subtype", "id"])}

    # curved edges, sorted by q_value (most sig on top)
    for row in edges.sort_values("q_value", ascending=False).itertuples(index=False):
        s = positions[row.source]; e = positions[row.target]
        delta = cm_idx[row.source] / max(len(cm_idx) - 1, 1) - epi_idx[row.target] / max(len(epi_idx) - 1, 1)
        rad = float(np.clip(delta * 0.10, -0.09, 0.09))
        ax.add_patch(FancyArrowPatch(
            (s["x"] + 0.105, s["y"]), (e["x"] - 0.115, e["y"]),
            arrowstyle="-", connectionstyle=f"arc3,rad={rad}",
            color=row.edge_color, linewidth=row.edge_width,
            alpha=0.22 + 0.50 * min(abs(row.rho) / 0.75, 1.0), zorder=1))

    for row in nodes.itertuples(index=False):
        is_epi = row.node_type == "Epithelial subtype"
        w = 0.23 if is_epi else 0.21; h = 0.043
        ax.add_patch(FancyBboxPatch(
            (row.x - w/2, row.y - h/2), w, h,
            boxstyle="round,pad=0.007,rounding_size=0.012",
            facecolor=row.color, edgecolor="white", linewidth=1.0, zorder=5))
        # NB: must use pd.notna() — `nan or fallback` is True (NaN is truthy),
        # so the simple `or` fallback silently renders "nan" text for CM rows.
        lbl = getattr(row, "label", None)
        text = lbl if (lbl is not None and pd.notna(lbl)) else row.id
        ax.text(row.x, row.y, text,
                ha="center", va="center", fontsize=7.1, fontweight="bold",
                color=readable_text_color(row.color), zorder=6)

    ax.text(0.18, 0.995, "CM programs", ha="center", va="top", fontsize=8.5, fontweight="bold")
    ax.text(0.82, 0.995, "Epithelial subtypes", ha="center", va="top", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box"); ax.axis("off")
    ax.set_title(f"B  Compact {CONTEXT} CM–Epi network", loc="left", fontweight="bold", pad=8)
    ax.text(0.5, -0.005, f"Spearman associations; q < {Q_THRESHOLD:g}; {len(edges)} edges",
            ha="center", va="top", fontsize=7.3, transform=ax.transAxes)

    legend_items = [
        Line2D([0], [0], color="#b2182b", lw=2.1, label="Positive rho"),
        Line2D([0], [0], color="#2166ac", lw=2.1, label="Negative rho"),
        Patch(facecolor="#d73027", edgecolor="white", label="Tumor CM"),
        Patch(facecolor="#4575b4", edgecolor="white", label="Normal CM"),
        Patch(facecolor="#8c8c8c", edgecolor="white", label="Shared CM"),
    ]
    ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=7, handlelength=2.0, columnspacing=1.1)


def main():
    coords, labels, centroids = load_umap()
    nodes, edges = load_network()
    print(f"[info] {len(labels)} cells | {len(centroids)} cell types | "
          f"{len(nodes)} nodes | {len(edges)} edges (q<{Q_THRESHOLD})", flush=True)

    fig = plt.figure(figsize=(15.0, 7.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.12, 0.88], wspace=0.08)
    draw_umap(fig.add_subplot(grid[0, 0]), coords, labels, centroids)
    draw_network(fig.add_subplot(grid[0, 1]), nodes, edges)
    fig.suptitle("ccRCC cellular landscape and CM–epithelial associations",
                 fontsize=13, fontweight="bold")

    stem = "ccrcc_overall_umap_compact_tumor_cm_epi_network"
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[done] {stem} written", flush=True)


if __name__ == "__main__":
    main()
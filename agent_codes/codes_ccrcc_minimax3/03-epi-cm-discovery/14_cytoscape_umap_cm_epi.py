"""Block 03 add-on: cytoscape-style UMAP + spatial + compact CM-Epi heatmap.

Outputs (saved under /mnt/disk18t/lr_xcy/riku/ccrcc-cmepi_minimax3/):
  umap_cytoscape_style.{pdf,png}                — UMAP + kNN edges, by cell_type
  umap_cytoscape_style_by_status.{pdf,png}      — UMAP + kNN edges, by status
  spatial_cytoscape_style.{pdf,png}             — 14 spatial samples grid, kNN edges on tissue
  cm_epi_compact_heatmap.{pdf,png}              — 3-panel CM × Epi rho heatmap

Size-control knobs (kept small to keep PDF/PNG reasonable):
  - single-cell: 15k cells, k=10   -> 150k edges
  - spatial: 6 neighbors per spot, dominant cell_type only (66 -> 8)
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.collections import LineCollection
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/mnt/disk18t/lr_xcy/riku/ccrcc-cmepi_minimax3")
ATLAS = ROOT / "epi-cm-core-workflow/h5ad/02-cell_subtype_integration_clustering/06-ccrcc-subtypes-to-full-adata/adata_anno_cellsubtype.h5ad"
SP_DIR = ROOT / "epi-cm-core-workflow/h5ad/04-spatial-validation-optional/01-tangram-spatial-mapping"
ASSOC_DIR = ROOT / "epi-cm-core-workflow/tables/03-epi-cm-discovery/association-spearman"

OUT_DIR = ROOT

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------
PALETTE_CT = {
    "Epi":  "#d62728", "Endo": "#1f77b4", "S":   "#9467bd",
    "T":    "#2ca02c", "NK":   "#17becf", "Mye": "#ff7f0e",
    "B":    "#e377c2", "Mast": "#8c564b",
}
PALETTE_STATUS = {"tumor": "#c0392b", "normal-like": "#2980b9"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def subsample(a: sc.AnnData, n: int, seed: int = 42) -> sc.AnnData:
    rng = np.random.default_rng(seed)
    n = min(n, a.n_obs)
    idx = np.sort(rng.choice(a.n_obs, size=n, replace=False))
    return a[idx].copy()


def knn_edges(draw_coords: np.ndarray, knn_coords: np.ndarray, k: int):
    """kNN in `knn_coords` space; edges drawn in `draw_coords` space."""
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    nn.fit(knn_coords)
    _, idx = nn.kneighbors(knn_coords)
    edges = []
    for i in range(idx.shape[0]):
        for j in idx[i, 1:]:
            edges.append(np.asarray([draw_coords[i], draw_coords[j]]))
    return edges


def draw_cytoscape(ax, umap, edges, color_by, palette, title,
                   pt_size=2.0, edge_alpha=0.05, edge_lw=0.15,
                   show_legend=True, legend_loc="center left"):
    if edges:
        seg = LineCollection(edges, linewidths=edge_lw, alpha=edge_alpha,
                             colors="#888888")
        ax.add_collection(seg)
    cats = pd.Series(color_by).astype("category")
    color_arr = np.asarray(color_by)
    for i, cat in enumerate(cats.cat.categories):
        m = (color_arr == cat)
        ax.scatter(umap[m, 0], umap[m, 1],
                   c=palette[i % len(palette)] if isinstance(palette, list)
                     else palette.get(cat, "#777777"),
                   s=pt_size, linewidths=0, label=cat, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    if show_legend:
        ax.legend(loc=legend_loc, bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=8, markerscale=3)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {name}.{{pdf,png}}", flush=True)


# ---------------------------------------------------------------------------
# Figure 1: cytoscape-style UMAP (single-cell)
# ---------------------------------------------------------------------------
def fig_cytoscape_umap():
    print("[sc] loading atlas (subsample 15k, k=10)…", flush=True)
    a = sc.read_h5ad(ATLAS)
    a = subsample(a, n=15000, seed=42)
    umap = a.obsm["X_umap"]
    pca = a.obsm["X_pca_inte"]
    edges = knn_edges(umap, pca, k=10)
    print(f"[sc] cells={a.n_obs} edges={len(edges):,}", flush=True)

    # by cell_type
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_cytoscape(ax, umap, edges,
                   color_by=a.obs["cell_type"].astype(str).values,
                   palette=[PALETTE_CT[c] for c in sorted(a.obs["cell_type"].unique())],
                   title=f"scRNA UMAP (kNN overlay, n={a.n_obs:,}, k=10)")
    save(fig, "umap_cytoscape_style")

    # by status
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_cytoscape(ax, umap, edges,
                   color_by=a.obs["status"].astype(str).values,
                   palette=PALETTE_STATUS,
                   title=f"scRNA UMAP by status (n={a.n_obs:,})")
    save(fig, "umap_cytoscape_style_by_status")


# ---------------------------------------------------------------------------
# Figure 2: cytoscape-style spatial (14 samples grid)
# ---------------------------------------------------------------------------
def dominant_celltype(ct_pred: pd.DataFrame) -> pd.Series:
    """argmax of 66-subtype probs -> 8 cell_types (first token before '_')."""
    idx = ct_pred.values.argmax(axis=1)
    sub = ct_pred.columns[idx]
    return pd.Series([s.split("_", 1)[0] for s in sub], index=ct_pred.index)


def fig_spatial_cytoscape():
    files = sorted(SP_DIR.glob("*_tangram_mapped.h5ad"))
    print(f"[sp] {len(files)} samples", flush=True)

    # 5 cols x 3 rows for 14 samples (last cell empty)
    nrows, ncols = 3, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
    axes_flat = axes.flatten()

    for ax, f in zip(axes_flat, files):
        a = sc.read_h5ad(f)
        # filter in_tissue (binary 1/2 in Visium: 1=in, 0=out)
        keep = a.obs["in_tissue"].astype(int).values > 0
        a = a[keep].copy()
        spatial = a.obsm["spatial"]
        ct_pred = a.obsm["tangram_ct_pred"]
        ct8 = dominant_celltype(ct_pred).values
        # smaller k for dense Visium hex grid
        edges = knn_edges(spatial, spatial, k=6)

        seg = LineCollection(edges, linewidths=0.15, alpha=0.05, colors="#666")
        ax.add_collection(seg)
        for cat in np.unique(ct8):
            m = ct8 == cat
            ax.scatter(spatial[m, 0], spatial[m, 1],
                       c=PALETTE_CT.get(cat, "#777"),
                       s=0.6, linewidths=0, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        sample = a.obs["sample"].iloc[0] if "sample" in a.obs.columns else f.stem.split("_")[0]
        ax.set_title(sample, fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")

    # hide unused axes
    for ax in axes_flat[len(files):]:
        ax.axis("off")

    # shared legend
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=c, markeredgecolor="none",
                          markersize=6, label=k)
               for k, c in PALETTE_CT.items()]
    fig.legend(handles=handles, loc="lower center", ncol=8,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Tangram-mapped spatial cell-type composition "
                 f"(n={len(files)} samples, 6-NN overlay)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save(fig, "spatial_cytoscape_style")


# ---------------------------------------------------------------------------
# Figure 3: compact CM-Epi heatmap
# ---------------------------------------------------------------------------
def fig_cm_epi_compact():
    print("[cm] loading CM-Epi matrices…", flush=True)
    panels = []
    for tag, rho_f, q_f in [
        ("tumor",       "balanced_joint_cm_epi_cm_association_tumor_rho_matrix.csv",
                        "balanced_joint_cm_epi_cm_association_tumor_q_matrix.csv"),
        ("normal-like", "balanced_joint_cm_epi_cm_association_normal-like_rho_matrix.csv",
                        "balanced_joint_cm_epi_cm_association_normal-like_q_matrix.csv"),
        ("overall",     "balanced_joint_cm_epi_cm_association_overall_rho_matrix.csv",
                        "balanced_joint_cm_epi_cm_association_overall_q_matrix.csv"),
    ]:
        rho = pd.read_csv(ASSOC_DIR / rho_f, index_col=0)
        q   = pd.read_csv(ASSOC_DIR / q_f,   index_col=0)
        col_order = sorted(rho.columns, key=lambda c: (c[0], int(c.split("_")[1][2:])))
        rho, q = rho[col_order].sort_index(), q[col_order].loc[rho.index]
        panels.append((tag, rho, q))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"wspace": 0.35})
    vmax = max(np.nanmax(np.abs(p[1].values)) for p in panels)
    vmin = -vmax

    for ax, (tag, rho, q) in zip(axes, panels):
        im = ax.imshow(rho.values, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(rho.shape[1]))
        ax.set_xticklabels(rho.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(rho.shape[0]))
        ax.set_yticklabels(rho.index, fontsize=8)
        ax.set_title(f"{tag} (n={rho.shape[0]} Epi × {rho.shape[1]} CM)", fontsize=11)
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                if np.isfinite(q.iat[i, j]) and q.iat[i, j] < 0.05:
                    ax.text(j, i, "*", ha="center", va="center",
                            fontsize=6, color="black")
        ax.set_xlabel("CM modules")
    axes[0].set_ylabel("Epi subtype")

    cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                        fraction=0.02, pad=0.02, shrink=0.8)
    cbar.set_label("Spearman ρ (Epi_subtype_freq vs CM_activity)")
    fig.suptitle("CM × Epi-subtype associations (asterisk = BH q < 0.05)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "cm_epi_compact_heatmap")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    fig_cytoscape_umap()
    fig_spatial_cytoscape()
    fig_cm_epi_compact()
    print("\n[done] all figures written to", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
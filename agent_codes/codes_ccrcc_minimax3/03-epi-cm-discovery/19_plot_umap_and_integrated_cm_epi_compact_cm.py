"""ccRCC combined UMAP + 3-column bipartite (compact-CM variant).

Output:
  ccrcc_umap_and_tumor_singlecell_spatial_cm_epi_cytoscape_compact_cm.{pdf,png}

Compact-CM layout tweaks vs script 18:
  - CM box is a long horizontal rectangle (vertical separator splits CM name
    from member subtypes area)
  - Member subtypes arranged in 2 rows x 5 cols (vs 3 rows x 3 cols)
  - Larger member dots and text for better readability
  - Larger CM box height and font size
"""
from __future__ import annotations

import colorsys
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
CM_CLASS  = ROOT / "epi-cm-core-workflow/tables/03-epi-cm-discovery/joint_module_classification.csv"
H_LOADING = ROOT / "epi-cm-core-workflow/tables/03-epi-cm-discovery/h_df_loading_cell_subtype_by_CM_robust.csv"
SP_PATH   = ROOT / "epi-cm-core-workflow/tables/04-spatial-validation-optional/01-tangram-spatial-mapping/percentile_quadrant_fisher_sample_stouffer_tumor_only.csv"

OUT_DIR = ROOT
Q_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 11, "axes.labelsize": 9,
})

PALETTE_CT = {
    "Epi":  "#d62728", "Endo": "#1f77b4", "S":   "#9467bd",
    "T":    "#2ca02c", "NK":   "#17becf", "Mye": "#ff7f0e",
    "B":    "#e377c2", "Mast": "#8c564b",
}
SHORT_LABEL = {"Epi": "Epi", "Endo": "Endo", "S": "S", "T": "T",
               "NK": "NK", "Mye": "Mye", "B": "B", "Mast": "Mast"}
PALETTE_CM = {"tCM": "#d73027", "normalCM": "#4575b4", "sharedCM": "#8c8c8c"}

EDGE_STYLE = {
    "both_significant_concordant_positive":   ("#b2182b", "solid", 0.62),
    "both_significant_concordant_negative":   ("#2166ac", "solid", 0.62),
    "both_significant_opposite_direction":    ("#7b3294", "solid", 0.58),
    "spatial_only_positive":                  ("#ef8a62", (0, (4, 2)), 0.22),
    "spatial_only_negative":                  ("#67a9cf", (0, (4, 2)), 0.22),
    "single_cell_only_positive":              ("#8b0000", (0, (1, 1)), 0.78),
    "single_cell_only_negative":              ("#08306b", (0, (1, 1)), 0.78),
}

# ---- Layout constants (compact-CM style) ----
CM_BOX_LEFT       = 0.012
CM_BOX_RIGHT      = 0.555    # CM boxes are wide horizontal bars
CM_BOX_HEIGHT     = 0.052
CM_TEXT_X         = 0.052
CM_SEPARATOR_X    = 0.105
MEMBER_LEFT       = 0.125
MEMBER_RIGHT      = 0.535
EPI_BOX_LEFT      = 0.805
EPI_BOX_RIGHT     = 0.992
CM_MEMBER_Y_OFFSET = 0.0112
CM_MEMBER_DOT_SIZE = 16.0
CM_MEMBER_TEXT_X_OFFSET = 0.0055
CM_MEMBER_FONT_SIZE   = 5.0
CM_BORDER_WIDTH    = 1.4
CM_FONT_SIZE      = 7.2
CM_HEADER_FONT_SIZE = 8.4
N_LEGEND_COLS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def readable_text_color(hex_color: str) -> str:
    rgb = mpl.colors.to_rgb(hex_color)
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "black" if luminance > 0.62 else "white"


def build_subtype_palette(mapping: pd.DataFrame, broad_palette: dict) -> dict:
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_umap_data():
    a = ad.read_h5ad(ATLAS, backed="r")
    try:
        coords = np.asarray(a.obsm["X_umap"][:, :2], dtype=np.float32)
        subtype = a.obs["cell_subtype"].astype(str)
        broad = a.obs["cell_type"].astype(str)
    finally:
        a.file.close()
    mapping = (pd.DataFrame({"cell_subtype": subtype, "cell_type": broad})
               .drop_duplicates()
               .sort_values(["cell_type", "cell_subtype"], kind="stable")
               .reset_index(drop=True))
    return coords, subtype, broad, mapping


def compute_centroids(coords, broad):
    rows = []
    for ct in sorted(broad.unique()):
        m = broad.eq(ct).to_numpy()
        xy = coords[m]
        rows.append({"cell_type": ct, "n_cells": int(m.sum()),
                     "umap1_median": float(np.median(xy[:, 0])),
                     "umap2_median": float(np.median(xy[:, 1]))})
    return pd.DataFrame(rows)


def load_single_cell_edges():
    rho = pd.read_csv(ASSOC_DIR / "balanced_joint_cm_epi_cm_association_tumor_rho_matrix.csv", index_col=0)
    q = pd.read_csv(ASSOC_DIR / "balanced_joint_cm_epi_cm_association_tumor_q_matrix.csv", index_col=0)
    edges = []
    for epi in rho.index:
        for cm in rho.columns:
            r = float(rho.loc[epi, cm]); qv = float(q.loc[epi, cm])
            if np.isfinite(r) and np.isfinite(qv) and qv < Q_THRESHOLD:
                edges.append({"source": cm, "target": epi, "rho": r, "q_value": qv,
                              "effect": r, "modality": "single-cell",
                              "direction": "positive" if r > 0 else "negative"})
    return pd.DataFrame(edges)


def load_spatial_edges(nodes):
    df = pd.read_csv(SP_PATH)
    df["combined_signed_z"] = pd.to_numeric(df["combined_signed_z"], errors="coerce")
    df["q_value_bh"] = pd.to_numeric(df["q_value_bh"], errors="coerce")
    df = df[df["combined_signed_z"].notna() & df["q_value_bh"].notna()
            & df["q_value_bh"].lt(Q_THRESHOLD)].copy()
    valid = set(nodes["id"])
    df = df[df["CM"].isin(valid) & df["epi_subtype"].isin(valid)].copy()
    df = df.rename(columns={"CM": "source", "epi_subtype": "target",
                            "q_value_bh": "q_value", "combined_signed_z": "effect"})
    df["modality"] = "spatial"
    df["rho"] = np.nan
    df["direction"] = np.where(df["effect"] > 0, "positive", "negative")
    return df[["source", "target", "rho", "q_value", "effect", "modality", "direction"]]


def integrate_edges(sc_edges, sp_edges):
    sc = sc_edges[["source", "target", "rho", "q_value"]].rename(
        columns={"rho": "sc_rho", "q_value": "sc_q"})
    sp = sp_edges[["source", "target", "effect", "q_value"]].rename(
        columns={"effect": "sp_z", "q_value": "sp_q"})
    df = sc.merge(sp, on=["source", "target"], how="outer", validate="one_to_one")
    df["sc_sig"] = df["sc_q"].notna()
    df["sp_sig"] = df["sp_q"].notna()
    df["sp_dir"] = np.where(df["sp_z"].gt(0), "positive", np.where(df["sp_z"].lt(0), "negative", "ns"))
    df.loc[df["sp_z"].isna(), "sp_dir"] = "ns"
    df["sc_dir"] = np.where(df["sc_rho"].gt(0), "positive", np.where(df["sc_rho"].lt(0), "negative", "ns"))
    df.loc[df["sc_rho"].isna(), "sc_dir"] = "ns"

    both = df["sc_sig"] & df["sp_sig"]
    same = df["sc_dir"].eq(df["sp_dir"])
    df["evidence_class"] = ""
    df.loc[both & same & df["sc_rho"].gt(0), "evidence_class"] = "both_significant_concordant_positive"
    df.loc[both & same & df["sc_rho"].lt(0), "evidence_class"] = "both_significant_concordant_negative"
    df.loc[both & ~same, "evidence_class"] = "both_significant_opposite_direction"
    df.loc[~df["sc_sig"] & df["sp_sig"] & df["sp_z"].gt(0), "evidence_class"] = "spatial_only_positive"
    df.loc[~df["sc_sig"] & df["sp_sig"] & df["sp_z"].lt(0), "evidence_class"] = "spatial_only_negative"
    df.loc[df["sc_sig"] & ~df["sp_sig"] & df["sc_rho"].gt(0), "evidence_class"] = "single_cell_only_positive"
    df.loc[df["sc_sig"] & ~df["sp_sig"] & df["sc_rho"].lt(0), "evidence_class"] = "single_cell_only_negative"
    if df["evidence_class"].eq("").any():
        raise ValueError("Unclassified integrated CM-Epi edge")

    df["edge_color"] = df["evidence_class"].map(lambda x: EDGE_STYLE[x][0])
    df["edge_linestyle"] = df["evidence_class"].map(lambda x: EDGE_STYLE[x][1])
    df["edge_alpha"] = df["evidence_class"].map(lambda x: EDGE_STYLE[x][2])

    sc_strength = df["sc_rho"].abs().div(0.75).clip(upper=1).fillna(0)
    sp_scale = float(sp_edges["effect"].abs().quantile(0.95)) if not sp_edges.empty else 1.0
    sp_scale = sp_scale if sp_scale > 0 else 1.0
    sp_strength = df["sp_z"].abs().div(sp_scale).clip(upper=1).fillna(0)
    df["edge_width"] = 0.35 + 2.7 * np.where(both, (sc_strength + sp_strength) / 2,
                                              np.maximum(sc_strength, sp_strength))
    return df.sort_values(["evidence_class", "source", "target"], kind="stable")


def build_nodes(subtype_palette):
    """Build CM (and Epi) node info; attach top-N member subtypes per CM."""
    cm_class = pd.read_csv(CM_CLASS).sort_values("global_order", kind="stable")
    cm_order = cm_class["CM"].astype(str).tolist()
    cm_label = dict(zip(cm_class["CM"], cm_class["class"]))

    rho = pd.read_csv(ASSOC_DIR / "balanced_joint_cm_epi_cm_association_tumor_rho_matrix.csv", index_col=0)
    epi_order = rho.index.astype(str).tolist()

    # top-N member subtypes per CM from H_df (robust)
    h = pd.read_csv(H_LOADING, index_col=0)
    members = {}
    for cm in cm_order:
        if cm in h.columns:
            top = h[cm].sort_values(ascending=False).head(10).index.tolist()
            members[cm] = top
        else:
            members[cm] = []

    cm_y = np.linspace(0.94, 0.06, len(cm_order))
    epi_y = np.linspace(0.91, 0.09, len(epi_order))
    nodes = []
    for cm, y in zip(cm_order, cm_y):
        nodes.append({"id": cm, "node_type": "CM", "class": cm_label[cm],
                      "color": PALETTE_CM[cm_label[cm]], "x": CM_BOX_LEFT, "y": float(y),
                      "members": members[cm]})
    for epi, y in zip(epi_order, epi_y):
        nodes.append({"id": epi, "node_type": "Epithelial subtype", "class": "Epi",
                      "color": subtype_palette.get(epi, "#cccccc"), "x": EPI_BOX_LEFT, "y": float(y),
                      "members": []})
    return pd.DataFrame(nodes)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_subtype_umap(ax, coords, subtype, palette):
    counts = subtype.value_counts()
    for st in counts.sort_values(ascending=False).index:
        m = subtype.eq(st).to_numpy()
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


def draw_cm_member_labels(ax, member_rows, y):
    """Lay out member subtypes in 2 rows x N columns inside the CM box."""
    n_columns = int(np.ceil(len(member_rows) / 2))
    slot_width = (MEMBER_RIGHT - MEMBER_LEFT) / max(n_columns, 1)
    for index, member in enumerate(member_rows):
        row_index = 0 if index < n_columns else 1
        column_index = index if row_index == 0 else index - n_columns
        x = MEMBER_LEFT + column_index * slot_width
        y_member = y + (CM_MEMBER_Y_OFFSET if row_index == 0 else -CM_MEMBER_Y_OFFSET)
        ax.scatter([x], [y_member], s=CM_MEMBER_DOT_SIZE, c=[member.color],
                   edgecolors="none", zorder=8)
        ax.text(x + CM_MEMBER_TEXT_X_OFFSET, y_member, member.label,
                ha="left", va="center", fontsize=CM_MEMBER_FONT_SIZE,
                color="#202020", zorder=8)


def draw_network_with_cm_members(ax, nodes, edges, subtype_palette):
    cm_nodes = nodes[nodes.node_type == "CM"]
    epi_nodes = nodes[nodes.node_type == "Epithelial subtype"]
    cm_index = {n: i for i, n in enumerate(cm_nodes["id"])}
    epi_index = {n: i for i, n in enumerate(epi_nodes["id"])}

    # ---- edges (under nodes) ----
    for row in edges.sort_values("edge_width", ascending=True).itertuples(index=False):
        s = nodes.set_index("id").loc[row.source]
        e = nodes.set_index("id").loc[row.target]
        delta = cm_index[row.source] / max(len(cm_index) - 1, 1) - epi_index[row.target] / max(len(epi_index) - 1, 1)
        curvature = float(np.clip(delta * 0.10, -0.09, 0.09))
        ax.add_patch(FancyArrowPatch(
            (CM_BOX_RIGHT, s.y), (EPI_BOX_LEFT, e.y),
            arrowstyle="-", connectionstyle=f"arc3,rad={curvature}",
            color=row.edge_color, linewidth=row.edge_width,
            alpha=row.edge_alpha, linestyle=row.edge_linestyle, zorder=1))

    # ---- CM long horizontal bars with embedded member subtypes ----
    for row in cm_nodes.itertuples(index=False):
        # outer border (transparent fill, CM class color)
        ax.add_patch(FancyBboxPatch(
            (CM_BOX_LEFT, row.y - CM_BOX_HEIGHT / 2),
            CM_BOX_RIGHT - CM_BOX_LEFT, CM_BOX_HEIGHT,
            boxstyle="round,pad=0.004,rounding_size=0.009",
            facecolor="none", edgecolor=row.color, linewidth=CM_BORDER_WIDTH, zorder=5))
        # vertical separator inside box
        ax.plot([CM_SEPARATOR_X, CM_SEPARATOR_X],
                [row.y - CM_BOX_HEIGHT / 2 + 0.004, row.y + CM_BOX_HEIGHT / 2 - 0.004],
                color=row.color, linewidth=0.75, alpha=0.72, zorder=6)
        # CM name
        ax.text(CM_TEXT_X, row.y, row.id, ha="center", va="center",
                fontsize=CM_FONT_SIZE, fontweight="bold", color=row.color, zorder=7)
        # member subtypes to the right of the separator
        member_rows = [type("M", (), {"label": st, "color": subtype_palette.get(st, "#888888")})()
                       for st in row.members]
        draw_cm_member_labels(ax, member_rows, row.y)

    # ---- Epi nodes ----
    for row in epi_nodes.itertuples(index=False):
        ax.add_patch(FancyBboxPatch(
            (EPI_BOX_LEFT, row.y - 0.0215),
            EPI_BOX_RIGHT - EPI_BOX_LEFT, 0.043,
            boxstyle="round,pad=0.004,rounding_size=0.009",
            facecolor=row.color, edgecolor="white", linewidth=0.9, zorder=5))
        ax.text((EPI_BOX_LEFT + EPI_BOX_RIGHT) / 2, row.y, row.id,
                ha="center", va="center", fontsize=5.2, fontweight="bold",
                color=readable_text_color(row.color), zorder=6)

    # column headers
    ax.text(CM_TEXT_X, 0.992, "CM", ha="center", va="top",
            fontsize=CM_HEADER_FONT_SIZE, fontweight="bold")
    ax.text((MEMBER_LEFT + MEMBER_RIGHT) / 2, 0.992,
            "CM member subtypes (colors match UMAP)",
            ha="center", va="top", fontsize=CM_HEADER_FONT_SIZE, fontweight="bold")
    ax.text((EPI_BOX_LEFT + EPI_BOX_RIGHT) / 2, 0.992, "Epithelial subtypes",
            ha="center", va="top", fontsize=7.2, fontweight="bold")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box"); ax.axis("off")
    ax.set_title("B  Integrated tumor CM–epithelial evidence",
                 loc="left", fontweight="bold", pad=8)
    ax.text(0.70, -0.006,
            f"single-cell Spearman and spatial Fisher-to-Stouffer; q < {Q_THRESHOLD:g}; {len(edges)} edges",
            ha="center", va="top", fontsize=6.6, transform=ax.transAxes)


def main():
    coords, subtype, broad, mapping = load_umap_data()
    palette = build_subtype_palette(mapping, PALETTE_CT)
    centroids = compute_centroids(coords, broad)
    nodes = build_nodes(palette)

    sc_edges = load_single_cell_edges()
    sp_edges = load_spatial_edges(nodes)
    integrated = integrate_edges(sc_edges, sp_edges)
    print(f"[info] cells={len(subtype)} | subtypes={subtype.nunique()} "
          f"| sc={len(sc_edges)} | sp={len(sp_edges)} | integrated={len(integrated)}", flush=True)

    fig = plt.figure(figsize=(20.0, 11.5))
    # Panel A: UMAP
    umap_ax = fig.add_axes([0.035, 0.34, 0.355, 0.59])
    draw_subtype_umap(umap_ax, coords, subtype, palette)
    overlay_broad_labels(umap_ax, centroids)
    umap_ax.set_xlabel("UMAP1"); umap_ax.set_ylabel("UMAP2")
    umap_ax.set_title(f"A  Cell subtypes ({subtype.nunique()}) with broad cell-type annotations",
                      loc="left", fontweight="bold")
    umap_ax.set_box_aspect(1); umap_ax.grid(False)
    for sp in umap_ax.spines.values():
        sp.set_color("#4d4d4d"); sp.set_linewidth(0.7)

    # Panel A bottom legend
    handles = [Line2D([0], [0], marker="o", linestyle="none",
                     markerfacecolor=palette[r.cell_subtype],
                     markeredgecolor="none", markersize=4, label=r.cell_subtype)
               for r in mapping.itertuples(index=False)]
    umap_legend_ax = fig.add_axes([0.01, 0.015, 0.405, 0.285])
    umap_legend_ax.axis("off")
    umap_legend_ax.legend(handles=handles, loc="center", ncol=N_LEGEND_COLS,
                          frameon=False, fontsize=5.0,
                          handletextpad=0.25, columnspacing=0.72, labelspacing=0.38)

    # Panel B: 3-column bipartite (compact-CM)
    bi_ax = fig.add_axes([0.415, 0.19, 0.565, 0.70])
    draw_network_with_cm_members(bi_ax, nodes, integrated, palette)

    # Common title
    fig.suptitle("ccRCC single-cell landscape and tumor CM–epithelial associations",
                 y=0.985, fontsize=14, fontweight="bold")

    # Panel B legend
    legend_items = [
        Line2D([0], [0], color="#b2182b", lw=2.0, label="Both: positive"),
        Line2D([0], [0], color="#2166ac", lw=2.0, label="Both: negative"),
        Line2D([0], [0], color="#7b3294", lw=2.0, label="Both: opposite direction"),
        Line2D([0], [0], color="#ef8a62", lw=1.7, ls="--", label="Spatial only (+)"),
        Line2D([0], [0], color="#67a9cf", lw=1.7, ls="--", label="Spatial only (-)"),
        Line2D([0], [0], color="#08306b", lw=1.7, ls=":", label="Single-cell only (-)"),
        Patch(facecolor="none", edgecolor="#d73027", linewidth=1.4, label="Tumor CM"),
        Patch(facecolor="none", edgecolor="#4575b4", linewidth=1.4, label="Normal CM"),
        Patch(facecolor="none", edgecolor="#8c8c8c", linewidth=1.4, label="Shared CM"),
    ]
    fig.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.70, 0.012),
               ncol=3, frameon=False, fontsize=6.7, handlelength=2.0, columnspacing=1.05)

    stem = "ccrcc_umap_and_tumor_singlecell_spatial_cm_epi_cytoscape_compact_cm"
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[done] {stem} written", flush=True)


if __name__ == "__main__":
    main()
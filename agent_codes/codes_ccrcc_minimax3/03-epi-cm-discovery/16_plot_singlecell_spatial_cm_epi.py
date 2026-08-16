"""ccRCC integrated single-cell + spatial bipartite CM-Epi network.

Output:
  ccrcc_tumor_singlecell_spatial_cm_epi_cytoscape.{pdf,png}

Integrates tumor-context single-cell Spearman (CM x Epi subtype) with
spatial Fisher-to-Stouffer associations, classifying each CM-Epi pair:
  - both concordant positive  -> solid red
  - both concordant negative  -> solid blue
  - both opposite direction   -> solid purple
  - spatial only positive     -> dashed light orange
  - spatial only negative     -> dashed light blue
  - single-cell only positive -> dotted dark red
  - single-cell only negative -> dotted dark blue

Style adapted from:
  /mnt/disk18t/lr_xcy/riku/crc_codex/cytoscape_umap_cm_epi_overview/codes/02_plot_tumor_singlecell_spatial_cm_epi.py
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
ASSOC_DIR = ROOT / "epi-cm-core-workflow/tables/03-epi-cm-discovery/association-spearman"
CM_CLASS  = ROOT / "epi-cm-core-workflow/tables/03-epi-cm-discovery/joint_module_classification.csv"
SP_PATH   = ROOT / "epi-cm-core-workflow/tables/04-spatial-validation-optional/01-tangram-spatial-mapping/percentile_quadrant_fisher_sample_stouffer_tumor_only.csv"

OUT_DIR = ROOT
Q_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 10.5, "axes.labelsize": 9,
})

PALETTE_CM = {"tCM": "#d73027", "normalCM": "#4575b4", "sharedCM": "#8c8c8c"}
PALETTE_EPI = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8",
]

EDGE_STYLE = {
    "both_significant_concordant_positive":   ("#b2182b", "solid", 0.62),
    "both_significant_concordant_negative":   ("#2166ac", "solid", 0.62),
    "both_significant_opposite_direction":    ("#7b3294", "solid", 0.58),
    "spatial_only_positive":                  ("#ef8a62", (0, (4, 2)), 0.22),
    "spatial_only_negative":                  ("#67a9cf", (0, (4, 2)), 0.22),
    "single_cell_only_positive":              ("#8b0000", (0, (1, 1)), 0.78),
    "single_cell_only_negative":              ("#08306b", (0, (1, 1)), 0.78),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def readable_text_color(hex_color: str) -> str:
    rgb = mpl.colors.to_rgb(hex_color)
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "black" if luminance > 0.62 else "white"


def load_single_cell_edges():
    rho = pd.read_csv(ASSOC_DIR / "balanced_joint_cm_epi_cm_association_tumor_rho_matrix.csv", index_col=0)
    q   = pd.read_csv(ASSOC_DIR / "balanced_joint_cm_epi_cm_association_tumor_q_matrix.csv", index_col=0)
    edges = []
    for epi in rho.index:
        for cm in rho.columns:
            r = float(rho.loc[epi, cm]); qv = float(q.loc[epi, cm])
            if np.isfinite(r) and np.isfinite(qv) and qv < Q_THRESHOLD:
                edges.append({"source": cm, "target": epi,
                              "rho": r, "q_value": qv,
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
    df["sc_dir"] = np.where(df["sc_rho"].gt(0), "positive",
                            np.where(df["sc_rho"].lt(0), "negative", "ns"))
    df.loc[df["sc_rho"].isna(), "sc_dir"] = "ns"
    df["sp_dir"] = np.where(df["sp_z"].gt(0), "positive",
                            np.where(df["sp_z"].lt(0), "negative", "ns"))
    df.loc[df["sp_z"].isna(), "sp_dir"] = "ns"

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

    df["edge_color"]     = df["evidence_class"].map(lambda x: EDGE_STYLE[x][0])
    df["edge_linestyle"] = df["evidence_class"].map(lambda x: EDGE_STYLE[x][1])
    df["edge_alpha"]     = df["evidence_class"].map(lambda x: EDGE_STYLE[x][2])

    sc_strength = df["sc_rho"].abs().div(0.75).clip(upper=1).fillna(0)
    sp_scale = float(sp_edges["effect"].abs().quantile(0.95)) if not sp_edges.empty else 1.0
    sp_scale = sp_scale if sp_scale > 0 else 1.0
    sp_strength = df["sp_z"].abs().div(sp_scale).clip(upper=1).fillna(0)
    df["edge_width"] = 0.35 + 2.7 * np.where(both, (sc_strength + sp_strength) / 2,
                                              np.maximum(sc_strength, sp_strength))
    return df.sort_values(["evidence_class", "source", "target"], kind="stable")


def build_nodes():
    cm_class = pd.read_csv(CM_CLASS).sort_values("global_order", kind="stable")
    cm_order = cm_class["CM"].astype(str).tolist()
    cm_label = dict(zip(cm_class["CM"], cm_class["class"]))

    rho = pd.read_csv(ASSOC_DIR / "balanced_joint_cm_epi_cm_association_tumor_rho_matrix.csv", index_col=0)
    epi_order = rho.index.astype(str).tolist()
    epi_palette = dict(zip(epi_order, [PALETTE_EPI[i % len(PALETTE_EPI)] for i in range(len(epi_order))]))

    cm_y = np.linspace(0.94, 0.06, len(cm_order))
    epi_y = np.linspace(0.91, 0.09, len(epi_order))
    nodes = []
    for cm, y in zip(cm_order, cm_y):
        nodes.append({"id": cm, "node_type": "CM", "class": cm_label[cm],
                      "color": PALETTE_CM[cm_label[cm]], "x": 0.18, "y": float(y)})
    for epi, y in zip(epi_order, epi_y):
        nodes.append({"id": epi, "node_type": "Epithelial subtype", "class": "Epi",
                      "color": epi_palette[epi], "x": 0.82, "y": float(y)})
    return pd.DataFrame(nodes)


def draw_network(ax, nodes, edges, title, subtitle, panel_letter=""):
    positions = nodes.set_index("id")[["x", "y"]].to_dict("index")
    cm_idx = {n: i for i, n in enumerate(nodes.loc[nodes.node_type == "CM", "id"])}
    epi_idx = {n: i for i, n in enumerate(nodes.loc[nodes.node_type == "Epithelial subtype", "id"])}

    # weak edges first, strong on top: sort by width ascending so thick edges sit on top
    for row in edges.sort_values("edge_width", ascending=True).itertuples(index=False):
        s = positions[row.source]; e = positions[row.target]
        delta = cm_idx[row.source] / max(len(cm_idx) - 1, 1) - epi_idx[row.target] / max(len(epi_idx) - 1, 1)
        rad = float(np.clip(delta * 0.10, -0.09, 0.09))
        ax.add_patch(FancyArrowPatch(
            (s["x"] + 0.105, s["y"]), (e["x"] - 0.145, e["y"]),
            arrowstyle="-", connectionstyle=f"arc3,rad={rad}",
            color=row.edge_color, linewidth=row.edge_width,
            alpha=row.edge_alpha, linestyle=row.edge_linestyle, zorder=1))

    for row in nodes.itertuples(index=False):
        is_epi = row.node_type == "Epithelial subtype"
        w = 0.29 if is_epi else 0.21; h = 0.043
        ax.add_patch(FancyBboxPatch(
            (row.x - w/2, row.y - h/2), w, h,
            boxstyle="round,pad=0.007,rounding_size=0.012",
            facecolor=row.color, edgecolor="white", linewidth=1.0, zorder=5))
        ax.text(row.x, row.y, row.id, ha="center", va="center",
                fontsize=6.5, fontweight="bold",
                color=readable_text_color(row.color), zorder=6)

    ax.text(0.18, 0.992, "CM programs", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(0.82, 0.992, "Epithelial subtypes", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box"); ax.axis("off")
    ax.set_title(f"{panel_letter}  {title}", loc="left", fontweight="bold", pad=8)
    ax.text(0.5, -0.005, f"{subtitle}; q < {Q_THRESHOLD:g}; {len(edges)} edges",
            ha="center", va="top", fontsize=6.8, transform=ax.transAxes)


def main():
    nodes = build_nodes()
    sc_edges = load_single_cell_edges()
    sp_edges = load_spatial_edges(nodes)
    integrated = integrate_edges(sc_edges, sp_edges)
    print(f"[info] nodes={len(nodes)} | sc={len(sc_edges)} | sp={len(sp_edges)} "
          f"| integrated={len(integrated)}", flush=True)

    fig = plt.figure(figsize=(8.6, 9.2))
    ax = fig.add_axes([0.06, 0.16, 0.88, 0.76])
    draw_network(ax, nodes, integrated,
                 "Integrated tumor CM–Epi evidence",
                 "single-cell Spearman and spatial Fisher-to-Stouffer")

    legend_items = [
        Line2D([0], [0], color="#b2182b", lw=2.0, label="Both: positive"),
        Line2D([0], [0], color="#2166ac", lw=2.0, label="Both: negative"),
        Line2D([0], [0], color="#7b3294", lw=2.0, label="Both: opposite direction"),
        Line2D([0], [0], color="#ef8a62", lw=1.7, ls="--", label="Spatial only (+)"),
        Line2D([0], [0], color="#67a9cf", lw=1.7, ls="--", label="Spatial only (-)"),
        Line2D([0], [0], color="#08306b", lw=1.7, ls=":", label="Single-cell only (-)"),
        Patch(facecolor="#d73027", edgecolor="white", label="Tumor CM"),
        Patch(facecolor="#4575b4", edgecolor="white", label="Normal CM"),
        Patch(facecolor="#8c8c8c", edgecolor="white", label="Shared CM"),
    ]
    fig.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=3, frameon=False, fontsize=7, handlelength=2.0, columnspacing=1.1)
    fig.suptitle("ccRCC tumor CM–epithelial associations",
                 y=0.975, fontsize=13, fontweight="bold")

    stem = "ccrcc_tumor_singlecell_spatial_cm_epi_cytoscape"
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[done] {stem} written", flush=True)


if __name__ == "__main__":
    main()
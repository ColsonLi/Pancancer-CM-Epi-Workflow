#!/usr/bin/env python3
"""Visualization-only full-atlas CRC Figure 7-style UMAPs.

Reuses the saved full-atlas coordinates, never the epithelial-only manifold.
All writes are new joint-atlas outputs in this directory; upstream files and
the complementary epithelial-only figures are not modified.
"""

from pathlib import Path
import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/opencode/crc_joint_atlas_matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

import h5py
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.colors import to_rgba
from PIL import Image


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
ATLAS = ROOT.parent / "full_atlas_reintegration_20261107"
DATA = OUT / "source_data"
DPI = 450
INK = "#232A30"
MUTED = "#69757E"
LIGHT = "#E8ECEF"
UNASSIGNED_COLOR = "#D9DEE2"
PURPLE = "#8062A3"
WARM = "#D26843"
COOL = "#3676A1"
JUN = "CRCcalQ2_hvg3000_r0.4_c08"
VIM = "CRCcalQ2_hvg3000_r0.4_c03"
CM = "CRCcal_CM08"
UNASSIGNED = "unassigned/mixed"
NORMAL = "Normal epithelial reference"
CELL_TABLE = DATA / "CRC_joint_atlas_per_cell_plot_data.csv.gz"

LINEAGE_PALETTE = {
    "Epithelial": "#A9B2BF", "T_NK": "#58968A", "Myeloid": "#C28661",
    "Stromal": "#A4A364", "Endothelial": "#6B9BB8", "B": "#7A995A",
    "Plasma": "#B989A5", "Mast": "#C0A168", "Schwann": "#8F86AB",
    UNASSIGNED: UNASSIGNED_COLOR,
}
CLUSTER_PALETTE = {
    "c00": "#9DACBA", "c01": "#B8AE98", "c02": "#CDBEBC",
    "c03": COOL, "c04": "#A1B5A3", "c05": "#AA9DB6",
    "c06": "#88A7AB", "c07": "#BDA58E", "c08": WARM,
    "c09": "#A5A6B6", "c10": "#C8BDA6", "c11": "#96B4B0",
    "c12": "#B99EAE", "c13": "#B6C5A6", "c14": "#C9D0D5",
}
ANNOTATION_PALETTE = {
    **CLUSTER_PALETTE,
    NORMAL: "#BBC4CA",
    **{k: v for k, v in LINEAGE_PALETTE.items() if k != "Epithelial"},
    "Cycling": "#9F788E",
}
DISPLAY_LINEAGE = {"T_NK": "T / NK", UNASSIGNED: "Unassigned/mixed"}
LINEAGE_ALIASES = {"Mye": "Myeloid", "Endo": "Endothelial"}
PURPLE_RAMP = ["#5A4178", PURPLE, "#9B80B8", "#AF9AC6", "#C0B1D1"]
PANEL_NAMES = {
    "A": "broad_lineage", "B": "calibrated_annotation",
    "C": "frozen_pair", "D": "CM1_like_context",
}


def setup_style():
    for path in (Path.home() / ".local/share/fonts/codex_trajectory_20260813").glob("Arial*.TTF"):
        font_manager.fontManager.addfont(str(path))
    font = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "Liberation Sans"
    plt.rcParams.update({
        "font.family": font, "font.size": 8.0,
        "axes.labelsize": 7.5, "axes.linewidth": 0.5,
        "axes.edgecolor": "#ABB5BC", "axes.labelcolor": MUTED,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": DPI,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.default": "regular",
    })
    return font


def csv_labels(path, usecols=None):
    frame = pd.read_csv(path, usecols=usecols, index_col="cell_id", low_memory=False)
    assert frame.index.is_unique, str(path)
    return frame


def prepare_data():
    DATA.mkdir(exist_ok=True)
    atlas = csv_labels(ATLAS / "atlas/per_cell_labels.csv.gz", [
        "cell_id", "sample_id", "study_id", "donor_id_v1", "status",
        "broad_lineage_v1", "original_cell_subtype", "in_expression_fit_reference",
    ])
    final = csv_labels(ATLAS / "all_cell_final_labels.csv.gz", [
        "cell_id", "broad_lineage_v1", "atlas_cluster_v1", "new_subtype_v1",
    ])
    assert atlas.index.equals(final.index)
    assert atlas.broad_lineage_v1.equals(final.broad_lineage_v1)
    with h5py.File(ATLAS / "atlas/all_cells_expression_integrated.h5ad", "r") as handle:
        ids = pd.Index(handle["obs/cell_id"].asstr()[:], name="cell_id")
        xy_h5ad = handle["obsm/X_umap"][:]
    with np.load(ATLAS / "atlas/all_cell_representations.npz", allow_pickle=True) as rep:
        xy = rep["X_umap"]
    assert ids.equals(atlas.index) and np.array_equal(xy_h5ad, xy)
    assert xy.shape == (643655, 2) and np.isfinite(xy).all()

    epithelial = csv_labels(ROOT / "epithelial/labels_hvg3000_r0.4.csv.gz")
    nonepi = csv_labels(ROOT / "data/non_epithelial_per_cell_labels.csv.gz")
    normal = csv_labels(ROOT / "data/normal_epithelial_reference_labels.csv.gz", ["cell_id", "status"])
    assert (len(epithelial), len(nonepi), len(normal)) == (76511, 397134, 127657)
    assert nonepi.subtype_id.nunique() == 65 and normal.status.eq("normal").all()
    for frame in [epithelial, nonepi, normal]:
        assert len(frame.index.difference(atlas.index)) == 0
    assert len(epithelial.index.intersection(nonepi.index)) == 0
    assert len(epithelial.index.intersection(normal.index)) == 0
    assert len(nonepi.index.intersection(normal.index)) == 0
    covered = atlas.index.isin(epithelial.index.union(nonepi.index).union(normal.index))
    assert covered.sum() == 601302 and (~covered).sum() == 42353
    selected_pair = json.loads((ROOT / "selected_pair.json").read_text())
    assert selected_pair["pair"] == {"Epi_JUN": JUN, "Epi_VIM": VIM}

    cells = pd.DataFrame(xy, columns=["UMAP1", "UMAP2"], index=atlas.index)
    cells[["sample_id", "study_id", "status"]] = atlas[["sample_id", "study_id", "status"]]
    cells["donor_id"] = atlas.donor_id_v1
    # A is the already saved atlas broad lineage, but excluded/mixed cells
    # must not be silently assigned on the strength of that earlier label.
    cells["atlas_broad_lineage_original"] = atlas.broad_lineage_v1
    cells["broad_lineage"] = atlas.broad_lineage_v1.where(covered, UNASSIGNED)
    cells["calibrated_label"] = UNASSIGNED
    cells["annotation_source"] = UNASSIGNED
    cells["calibrated_non_epithelial_lineage"] = ""
    cells["panel_B_group"] = UNASSIGNED
    cells["atlas_cluster_original"] = final.atlas_cluster_v1
    cells["atlas_subtype_original"] = final.new_subtype_v1
    cells["original_cell_subtype"] = atlas.original_cell_subtype
    cells["in_atlas_expression_fit_reference"] = atlas.in_expression_fit_reference
    cells.loc[epithelial.index, "calibrated_label"] = epithelial.cluster_id
    cells.loc[epithelial.index, "annotation_source"] = "Calibrated tumor epithelium"
    cells.loc[epithelial.index, "panel_B_group"] = epithelial.cluster_id.str.rsplit("_", n=1).str[-1]
    cells.loc[normal.index, "calibrated_label"] = NORMAL
    cells.loc[normal.index, "annotation_source"] = NORMAL
    cells.loc[normal.index, "panel_B_group"] = NORMAL
    cells.loc[nonepi.index, "calibrated_label"] = nonepi.subtype_id
    cells.loc[nonepi.index, "annotation_source"] = "Calibrated non-epithelium"
    cells.loc[nonepi.index, "calibrated_non_epithelial_lineage"] = nonepi.lineage
    cells.loc[nonepi.index, "panel_B_group"] = nonepi.lineage.replace(LINEAGE_ALIASES)
    cells["is_JUN_pole"] = cells.calibrated_label.eq(JUN)
    cells["is_VIM_pole"] = cells.calibrated_label.eq(VIM)
    assert (int(cells.is_JUN_pole.sum()), int(cells.is_VIM_pole.sum())) == (4855, 7748)
    assert set(cells.broad_lineage).issubset(LINEAGE_PALETTE)
    assert set(cells.panel_B_group).issubset(ANNOTATION_PALETTE)

    h = pd.read_csv(ROOT / "CM/H_selected.csv", index_col=0, float_precision="round_trip")
    evidence = pd.read_csv(ROOT / "CM/CM1_composition_marker_evidence.csv", index_col=0, float_precision="round_trip")
    selection = json.loads((ROOT / "CM/selected_CM.json").read_text())
    assert selection["CM1_like"] == CM and selection["K"] == 14
    raw_h = h.loc[CM]
    assert len(raw_h) == 65 and np.isclose(raw_h.sum(), 1.0)
    assert set(raw_h.index) == set(nonepi.subtype_id.unique())
    np.testing.assert_allclose(raw_h.loc[evidence.index], evidence.H_share, rtol=1e-13, atol=1e-15)
    loading = evidence.copy()
    loading.insert(0, "CM", CM)
    loading.insert(1, "H_selected_value", raw_h.loc[loading.index])
    observed = nonepi.groupby("subtype_id").agg(n_cells=("sample_id", "size"), n_samples=("sample_id", "nunique"))
    np.testing.assert_array_equal(loading.n_cells, observed.loc[loading.index, "n_cells"])
    np.testing.assert_array_equal(loading.n_samples, observed.loc[loading.index, "n_samples"])
    loading = loading.sort_values("H_share", ascending=False, kind="stable")
    loading["H_loading_rank"] = np.arange(1, len(loading) + 1)
    loading["highlighted_in_D"] = loading.H_loading_rank.le(5)
    loading["panel_D_color"] = LIGHT
    top_ids = loading.index[:5]
    loading.loc[top_ids, "panel_D_color"] = PURPLE_RAMP
    tnk_share = loading.loc[loading.lineage.eq("T_NK"), "H_share"].sum()
    assert np.isclose(tnk_share, selection["CM1_T_NK_H_share"], rtol=1e-13)
    cells["CM1_H_loading_share"] = cells.calibrated_label.map(loading.H_share)
    cells["is_CM1_top5_subtype"] = cells.calibrated_label.isin(top_ids)
    assert int(cells.is_CM1_top5_subtype.sum()) == 68508

    # Save every cell, including the unassigned remainder, with coordinates that
    # round-trip exactly to the saved float32 representation.
    cells.to_csv(CELL_TABLE, index_label="cell_id", float_format="%.17g", compression={"method": "gzip", "compresslevel": 6})
    loading.to_csv(DATA / "CRC_joint_subtype_to_CM1_loading.csv", index_label="subtype_id", float_format="%.17g")
    counts_a = group_counts(cells, "broad_lineage", list(LINEAGE_PALETTE))
    counts_a["plot_color"] = counts_a.index.map(LINEAGE_PALETTE)
    counts_a.to_csv(DATA / "CRC_joint_lineage_counts.csv", index_label="broad_lineage")
    counts_b = group_counts(cells, "panel_B_group", list(ANNOTATION_PALETTE))
    counts_b["plot_color"] = counts_b.index.map(ANNOTATION_PALETTE)
    counts_b.to_csv(DATA / "CRC_joint_panel_B_annotation_counts.csv", index_label="panel_B_group")
    cluster_counts = cells.loc[cells.annotation_source.eq("Calibrated tumor epithelium")].groupby("calibrated_label").agg(
        n_cells=("sample_id", "size"), n_samples=("sample_id", "nunique"),
        n_patients=("donor_id", "nunique"), n_studies=("study_id", "nunique"),
    )
    cluster_counts["short_cluster_id"] = cluster_counts.index.str.rsplit("_", n=1).str[-1]
    cluster_counts["plot_color"] = cluster_counts.short_cluster_id.map(CLUSTER_PALETTE)
    original_support = pd.read_csv(ROOT / "epithelial/support_hvg3000_r0.4.csv", index_col=0)
    np.testing.assert_array_equal(cluster_counts[["n_cells", "n_samples", "n_patients", "n_studies"]],
                                  original_support.loc[cluster_counts.index, ["n_cells", "n_samples", "n_patients", "n_studies"]])
    cluster_counts.to_csv(DATA / "CRC_joint_cluster_counts.csv", index_label="cluster_id")
    provenance_counts = cells.groupby("annotation_source", sort=False).size().rename("n_cells").to_frame()
    provenance_counts["fraction_of_atlas"] = provenance_counts.n_cells / len(cells)
    provenance_counts.to_csv(DATA / "CRC_joint_coverage_counts.csv", index_label="annotation_source")
    # Preserve, rather than resolve, the source distinction between atlas broad
    # lineage and calibrated lineage (notably the existing Cycling category).
    pd.crosstab(cells.loc[nonepi.index, "calibrated_non_epithelial_lineage"],
                cells.loc[nonepi.index, "atlas_broad_lineage_original"]).to_csv(
                    DATA / "CRC_joint_non_epithelial_lineage_crosswalk.csv")
    print(f"Prepared {len(cells):,} cells; {covered.sum():,} covered; {(~covered).sum():,} unassigned/mixed", flush=True)
    return cells, loading


def group_counts(cells, column, order):
    result = cells.groupby(column).agg(n_cells=("sample_id", "size"), n_samples=("sample_id", "nunique"))
    result = result.reindex([x for x in order if x in result.index])
    result["fraction_of_atlas"] = result.n_cells / len(cells)
    assert result.n_cells.sum() == 643655
    return result


def rect(fig, left, top, width, height):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, 1 - (top + height) / fh, width / fw, height / fh])


def text(fig, x, y, label, **kwargs):
    fw, fh = fig.get_size_inches()
    return fig.text(x / fw, 1 - y / fh, label, va=kwargs.pop("va", "top"), **kwargs)


def header(fig, region, letter, title, subtitle):
    left, top, _, _ = region
    text(fig, left, top, letter, fontsize=14, fontweight="bold")
    text(fig, left + 0.29, top + 0.024, title, fontsize=10.3, fontweight="bold")
    text(fig, left + 0.29, top + 0.31, subtitle, fontsize=7.35, color=MUTED)


def style_umap(ax, limits, inset=False):
    lo, hi = limits
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    if not inset:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("UMAP 1", labelpad=4)
        ax.set_ylabel("UMAP 2", labelpad=4)
    for spine in ax.spines.values():
        spine.set_color("#ABB5BC")
        spine.set_linewidth(0.5)


def scatter(ax, context, mask=None, colors=None, color=None, size=0.22, alpha=0.9):
    order = context["order"]
    if mask is not None:
        order = order[np.asarray(mask)[order]]
    xy = context["xy"][order]
    kwargs = {"s": size, "alpha": alpha, "linewidths": 0, "edgecolors": "none", "rasterized": True, "zorder": 1}
    if colors is not None:
        kwargs["c"] = np.asarray(colors)[order]
    else:
        kwargs["color"] = color
    ax.scatter(xy[:, 0], xy[:, 1], **kwargs)
    return len(order)


def anchor(xy):
    median = np.median(xy, axis=0)
    return xy[np.argmin(np.sum((xy - median) ** 2, axis=1))]


def centroid_label(ax, context, panel, group, mask, label=None, color=MUTED, fontsize=8.2):
    point = anchor(context["xy"][np.asarray(mask)])
    artist = ax.text(*point, label or group, ha="center", va="center", fontsize=fontsize,
                     color=color, fontweight="bold", zorder=5)
    artist.set_path_effects([patheffects.withStroke(linewidth=2.1, foreground="white")])
    context["centroids"].append({"panel": panel, "group": group, "UMAP1": float(point[0]), "UMAP2": float(point[1]),
                                 "display_label": label or group, "centroid_definition": "observed cell nearest coordinate-wise median"})
    return point


def swatch(fig, x, y, color, size=4.8):
    fw, fh = fig.get_size_inches()
    fig.add_artist(plt.Line2D([x / fw], [1 - y / fh], transform=fig.transFigure, marker="o",
                             markersize=size, markeredgewidth=0, color=color, linestyle="none"))


def panel_a(fig, region, context):
    left, top, width, height = region
    cells = context["cells"]
    header(fig, region, "A", "One joint atlas, all cell lineages", "643,655 cells · saved broad lineages for covered cells; remainder kept grey")
    ax = rect(fig, left + 0.23, top + 0.67, 4.96, 4.57)
    style_umap(ax, context["limits"])
    n = scatter(ax, context, colors=context["colors_A"], size=0.20)
    assert n == 643655
    for lineage in list(LINEAGE_PALETTE)[:-1]:
        centroid_label(ax, context, "A", lineage, cells.broad_lineage.eq(lineage), DISPLAY_LINEAGE.get(lineage, lineage))
    sx, right = left + 5.47, left + width - 0.02
    text(fig, sx, top + 0.69, "Broad lineage", fontsize=7.8, color=MUTED)
    text(fig, right, top + 0.69, "n cells", fontsize=7.8, color=MUTED, ha="right")
    counts = context["counts_A"]
    for i, lineage in enumerate(LINEAGE_PALETTE):
        y = top + 1.06 + 0.275 * i
        swatch(fig, sx + 0.025, y, LINEAGE_PALETTE[lineage])
        text(fig, sx + 0.13, y, DISPLAY_LINEAGE.get(lineage, lineage), fontsize=7.4, va="center")
        text(fig, right, y, f"{int(counts.loc[lineage, 'n_cells']):,}", fontsize=7.4, ha="right", va="center")
    text(fig, sx + 0.13, top + 3.90, "Total", fontsize=8.0, fontweight="bold")
    text(fig, right, top + 3.90, "643,655", fontsize=8.0, ha="right", fontweight="bold")
    text(fig, sx, top + 4.40, "93.4%", fontsize=21, color=INK, fontweight="bold")
    text(fig, sx + 0.93, top + 4.46, "calibrated / reference\ncoverage", fontsize=7.25, color=MUTED)
    text(fig, sx, top + 4.94, "601,302 covered cells\n42,353 unassigned/mixed", fontsize=7.2, color=MUTED)
    text(fig, left + 0.29, top + height - 0.03,
         "Epithelial: 76,511 tumor + 127,657 normal reference · non-epithelial: 397,134", fontsize=7.0, color=MUTED)
    return {"main_cells": n}


def panel_b(fig, region, context):
    left, top, width, height = region
    cells = context["cells"]
    header(fig, region, "B", "Calibrated annotation on the same atlas", "15 tumor epithelial clusters + normal reference + 65 non-epithelial subtypes")
    ax = rect(fig, left + 0.23, top + 0.67, 4.96, 4.57)
    style_umap(ax, context["limits"])
    tumor = cells.annotation_source.eq("Calibrated tumor epithelium")
    n = scatter(ax, context, mask=~tumor, colors=context["colors_B"], size=0.20)
    n += scatter(ax, context, mask=tumor, colors=context["colors_B"], size=0.26, alpha=0.95)
    assert n == 643655
    # Tumor clusters and normal reference overlap in the coarse joint atlas.
    # The exact cluster key is therefore used instead of 15 colliding labels.
    centroid_label(ax, context, "B", "Epithelial", cells.broad_lineage.eq("Epithelial"), "Epithelium")
    for lineage in ["T_NK", "Myeloid", "Stromal", "Endothelial", "B", "Plasma", "Mast", "Schwann", "Cycling"]:
        centroid_label(ax, context, "B", lineage, cells.panel_B_group.eq(lineage), DISPLAY_LINEAGE.get(lineage, lineage), fontsize=8.0)
    sx, right = left + 5.47, left + width - 0.02
    text(fig, sx, top + 0.69, "Tumor epithelium", fontsize=8.0, fontweight="bold")
    text(fig, sx, top + 0.92, "15 clusters · n = 76,511", fontsize=7.1, color=MUTED)
    counts = context["counts_B"]
    for i, cluster in enumerate(CLUSTER_PALETTE):
        col, row = divmod(i, 8)
        x = sx + col * 1.00
        y = top + 1.25 + row * 0.195
        swatch(fig, x + 0.02, y, CLUSTER_PALETTE[cluster], size=4.1)
        text(fig, x + 0.11, y, cluster, fontsize=6.95, va="center")
        text(fig, x + 0.88, y, f"{int(counts.loc[cluster, 'n_cells']):,}", fontsize=6.95, va="center", ha="right")
    y = top + 2.97
    swatch(fig, sx + 0.02, y, ANNOTATION_PALETTE[NORMAL])
    text(fig, sx + 0.13, y, "Normal epi. ref.", fontsize=7.2, va="center")
    text(fig, right, y, "127,657", fontsize=7.2, ha="right", va="center")
    text(fig, sx, top + 3.25, "Non-epithelial · 65 subtypes", fontsize=7.6, fontweight="bold")
    lineages = ["T_NK", "Myeloid", "Stromal", "Endothelial", "B", "Plasma", "Mast", "Schwann", "Cycling"]
    for i, lineage in enumerate(lineages):
        y = top + 3.59 + i * 0.185
        swatch(fig, sx + 0.02, y, ANNOTATION_PALETTE[lineage], size=4.1)
        text(fig, sx + 0.13, y, DISPLAY_LINEAGE.get(lineage, lineage), fontsize=7.05, va="center")
        text(fig, right, y, f"{int(counts.loc[lineage, 'n_cells']):,}", fontsize=7.05, va="center", ha="right")
    y = top + 5.31
    swatch(fig, sx + 0.02, y, UNASSIGNED_COLOR, size=4.1)
    text(fig, sx + 0.13, y, "Unassigned/mixed", fontsize=7.05, va="center")
    text(fig, right, y, "42,353", fontsize=7.05, va="center", ha="right")
    text(fig, left + 0.29, top + height - 0.03,
         "Fine subtypes aggregated only for display; source Cycling category is retained. Grey: 42,353 unassigned/mixed.",
         fontsize=6.85, color=MUTED)
    return {"main_cells": n}


def panel_c(fig, region, context):
    left, top, width, height = region
    cells = context["cells"]
    header(fig, region, "C", "Frozen epithelial pair in atlas context", "Only c08 / c03 highlighted · local zooms reuse the identical joint coordinates")
    ax = rect(fig, left + 0.23, top + 0.67, 4.96, 4.57)
    style_umap(ax, context["limits"])
    selected = cells.is_JUN_pole | cells.is_VIM_pole
    n = scatter(ax, context, mask=~selected, color=LIGHT, size=0.16, alpha=0.75)
    n += scatter(ax, context, mask=selected, colors=context["colors_C"], size=0.31, alpha=0.95)
    assert n == 643655
    specs = [
        ("is_JUN_pole", "Epi_JUN-like (c08)", JUN, WARM, (-5.6, 12.2)),
        ("is_VIM_pole", "Epi_VIM-like (c03)", VIM, COOL, (-7.8, 9.6)),
    ]
    insets = []
    for i, (column, label, cluster, color, label_at) in enumerate(specs):
        mask = cells[column].to_numpy()
        point = anchor(context["xy"][mask])
        ax.annotate(label, xy=point, xytext=label_at, color=color, fontsize=7.9, fontweight="bold",
                    ha="left", va="center", zorder=6,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.96},
                    arrowprops={"arrowstyle": "-", "lw": 0.7, "color": color, "shrinkA": 3, "shrinkB": 2})
        context["centroids"].append({"panel": "C", "group": cluster, "UMAP1": float(point[0]), "UMAP2": float(point[1]),
                                     "display_label": label, "centroid_definition": "observed cell nearest coordinate-wise median"})
        sx = left + 5.47
        sy = top + 0.69 + i * 2.32
        text(fig, sx, sy, label, fontsize=8.1, color=color, fontweight="bold")
        crop = context["crops"][cluster]
        in_crop = ((context["xy"] >= crop[0]) & (context["xy"] <= crop[1])).all(axis=1)
        n_shown = int((mask & in_crop).sum())
        text(fig, sx, sy + 0.23, f"n = {int(mask.sum()):,} · {n_shown:,} in crop", fontsize=6.9, color=MUTED)
        iax = rect(fig, sx, sy + 0.48, 1.94, 1.74)
        style_umap(iax, crop, inset=True)
        # Cropping is display-only. Every full-atlas cell is present above.
        scatter(iax, context, mask=in_crop & ~mask, color=LIGHT, size=0.27, alpha=0.8)
        scatter(iax, context, mask=in_crop & mask, color=color, size=0.47, alpha=0.95)
        insets.append({"cluster_id": cluster, "total_state_cells": int(mask.sum()), "highlighted_cells_in_crop": n_shown,
                       "all_cells_in_crop": int(in_crop.sum()), "UMAP_min": crop[0].tolist(), "UMAP_max": crop[1].tolist()})
    text(fig, left + 0.29, top + height - 0.03,
         "c08: 4,855 · c03: 7,748 · other atlas cells: 631,052. Relative affinity names, not validated conserved identities.",
         fontsize=6.85, color=MUTED)
    return {"main_cells": n, "highlighted_cells": 12603, "background_cells": 631052, "zoom_insets": insets}


def panel_d(fig, region, context):
    left, top, width, height = region
    cells, loading = context["cells"], context["loading"]
    header(fig, region, "D", "CM1-like lymphocyte module context", "CRCcal_CM08 · fixed K = 14 · top five H-loading subtype memberships highlighted")
    ax = rect(fig, left + 0.23, top + 0.67, 4.96, 4.57)
    style_umap(ax, context["limits"])
    selected = cells.is_CM1_top5_subtype
    n = scatter(ax, context, mask=~selected, color=LIGHT, size=0.16, alpha=0.75)
    n += scatter(ax, context, mask=selected, colors=context["colors_D"], size=0.24, alpha=0.92)
    assert n == 643655
    for rank, subtype in enumerate(loading.index[:5], 1):
        centroid_label(ax, context, "D", subtype, cells.calibrated_label.eq(subtype), str(rank), color=INK, fontsize=9.0)
    sx, right = left + 5.47, left + width - 0.02
    text(fig, sx, top + 0.69, CM, fontsize=8.6, color=PURPLE, fontweight="bold")
    text(fig, sx, top + 1.01, "90.26%", fontsize=24, color=PURPLE, fontweight="bold")
    text(fig, sx, top + 1.49, "T / NK H-loading share = 0.9026", fontsize=7.0, color=MUTED)
    text(fig, sx, top + 1.94, "Top subtype", fontsize=7.3, color=MUTED)
    text(fig, right, top + 1.94, "H share", fontsize=7.3, color=MUTED, ha="right")
    for i, (subtype, row) in enumerate(loading.iloc[:5].iterrows()):
        y = top + 2.30 + i * 0.49
        swatch(fig, sx + 0.025, y + 0.005, row.panel_D_color, size=5.2)
        text(fig, sx + 0.12, y, f"{i + 1}  {subtype}", fontsize=7.0)
        text(fig, right, y + 0.20, f"{row.H_share:.2%}", fontsize=7.4, ha="right", color=PURPLE)
        markers = row.top10_markers.split(";")[:2]
        text(fig, sx + 0.12, y + 0.20, " / ".join(markers), fontsize=6.7, color=MUTED)
    text(fig, sx, top + 4.95, "Top five: 68,508 cells\n67.43% of module H loading", fontsize=7.05, color=MUTED)
    text(fig, left + 0.29, top + height - 0.03,
         "Subtype membership, not per-cell module activity. Other atlas cells: 575,147; all 42,353 unassigned/mixed stay grey.",
         fontsize=6.85, color=MUTED)
    return {"main_cells": n, "highlighted_cells": 68508, "background_cells": 575147,
            "T_NK_H_loading_share": context["frozen_T_NK_H_share"],
            "top5_H_loading_share": float(loading.iloc[:5].H_share.sum())}


PANELS = {"A": panel_a, "B": panel_b, "C": panel_c, "D": panel_d}


def make_context(cells, loading):
    xy = cells[["UMAP1", "UMAP2"]].to_numpy()
    limits = (xy.min(axis=0) - 0.85, xy.max(axis=0) + 0.85)
    context = {
        "cells": cells, "loading": loading, "xy": xy, "limits": limits,
        "order": np.random.default_rng(42).permutation(len(cells)), "centroids": [], "crops": {},
        "counts_A": group_counts(cells, "broad_lineage", list(LINEAGE_PALETTE)),
        "counts_B": group_counts(cells, "panel_B_group", list(ANNOTATION_PALETTE)),
        "frozen_T_NK_H_share": json.loads((ROOT / "CM/selected_CM.json").read_text())["CM1_T_NK_H_share"],
    }
    context["colors_A"] = np.array([to_rgba(LINEAGE_PALETTE[x]) for x in cells.broad_lineage], dtype=np.float32)
    context["colors_B"] = np.array([to_rgba(ANNOTATION_PALETTE[x]) for x in cells.panel_B_group], dtype=np.float32)
    context["colors_C"] = np.tile(to_rgba(LIGHT), (len(cells), 1)).astype(np.float32)
    context["colors_C"][cells.is_JUN_pole] = to_rgba(WARM)
    context["colors_C"][cells.is_VIM_pole] = to_rgba(COOL)
    context["colors_D"] = np.array([to_rgba(loading.panel_D_color.get(x, LIGHT)) for x in cells.calibrated_label], dtype=np.float32)
    for cluster in [JUN, VIM]:
        cloud = xy[cells.calibrated_label.eq(cluster)]
        center = np.median(cloud, axis=0)
        quantiles = np.quantile(cloud, [0.01, 0.99], axis=0)
        half_span = float(np.max(np.abs(quantiles - center)) + 0.40)
        context["crops"][cluster] = (center - half_span, center + half_span)
    return context


def save(fig, stem, preview=False):
    paths = []
    dpi = 160 if preview else DPI
    for suffix in ["png"] if preview else ["png", "pdf", "svg"]:
        path = OUT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    with Image.open(OUT / f"{stem}.png") as image:
        dims = list(image.size)
        metadata_dpi = list(image.info.get("dpi", ()))
    print(f"Rendered {stem}: {dims} px; {metadata_dpi} DPI", flush=True)
    return {"files": paths, "png_pixels": dims, "dpi": dpi, "png_metadata_dpi": metadata_dpi}


def write_notes(context, font, outputs, panel_metadata):
    cells, loading = context["cells"], context["loading"]
    a_counts = context["counts_A"].n_cells
    b_counts = context["counts_B"].n_cells
    top5 = loading.iloc[:5]
    metadata = {
        "n_atlas_cells": len(cells), "n_samples": int(cells.sample_id.nunique()), "n_studies": int(cells.study_id.nunique()),
        "n_calibrated_or_reference": 601302, "n_unassigned_mixed": 42353,
        "embedding_npz": str(ATLAS / "atlas/all_cell_representations.npz"),
        "embedding_h5ad": str(ATLAS / "atlas/all_cells_expression_integrated.h5ad"),
        "embedding_key": "X_umap", "coordinates_exactly_equal_between_inputs": True,
        "integration_recomputed": False, "labels_changed": False, "upstream_inputs_modified": False,
        "embedding_method": "Saved sample/original-major-lineage-balanced 60,000-cell expression reference; HVG3000/PCA40; Harmony technical series; 30-NN local Harmony mapping and barycentric UMAP projection for 583,655 other cells; seed42",
        "fit_n": 60000, "mapped_n": 583655,
        "panel_A_rule": "Saved atlas broad_lineage_v1 for the calibrated/reference union; 42,353 other cells overridden for display only to unassigned/mixed; original atlas lineage preserved separately",
        "panel_B_rule": "Unchanged tumor epithelial cluster IDs; normal reference pooled; 65 unchanged non-epithelial subtype IDs aggregated by their source lineage for display; Mye/Endo expanded to Myeloid/Endothelial display names; Cycling retained separately",
        "frozen_pair": {"JUN": JUN, "VIM": VIM}, "panel_C_frozen_pair_only": True,
        "CM": CM, "K": 14, "top5_subtypes": top5.index.tolist(),
        "T_NK_H_loading_share": context["frozen_T_NK_H_share"],
        "top5_H_loading_share": float(top5.H_share.sum()),
        "n_CM1_top5_cells": int(cells.is_CM1_top5_subtype.sum()),
        "panels": panel_metadata, "A_lineage_counts": {k: int(v) for k, v in a_counts.items()},
        "B_annotation_counts": {k: int(v) for k, v in b_counts.items()},
        "font": font, "dense_scatter_rasterized": True, "dense_scatter_dpi": DPI,
        "text_axes_and_leaders_vector": True, "plot_order_seed": 42,
        "palette": {"ink": INK, "muted": MUTED, "module": PURPLE, "JUN": WARM, "VIM": COOL,
                    "lineages": LINEAGE_PALETTE, "annotation": ANNOTATION_PALETTE, "top5_purple_ramp": PURPLE_RAMP},
        "outputs": outputs,
    }
    (OUT / "CRC_joint_atlas_render_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    centroids = pd.DataFrame(context["centroids"]).drop_duplicates(["panel", "group"])
    centroids.to_csv(DATA / "CRC_joint_centroid_label_anchors.csv", index=False, float_format="%.17g")
    table_a = "\n".join(f"| {DISPLAY_LINEAGE.get(k, k)} | {int(v):,} |" for k, v in a_counts.items())
    table_b = "\n".join(f"| {k} | {int(v):,} |" for k, v in b_counts.items())
    table_d = "\n".join(f"| {i + 1} | `{sid}` | {int(row.n_cells):,} | {row.H_share:.10f} |" for i, (sid, row) in enumerate(top5.iterrows()))
    readme = f"""# CRC full-atlas joint UMAP — Figure 7 style

## Deliverables

- `CRC_joint_atlas_UMAP_combined.{{png,pdf,svg}}`: four-panel **full-atlas** figure.
- `CRC_joint_atlas_UMAP_panel_[A-D]_*.{{png,pdf,svg}}`: standalone panels.
- **450 DPI**, {font}, white background, no grid, thin axes; ink `{INK}`, muted text `{MUTED}`, CM purple `{PURPLE}`, frozen JUN warm `{WARM}` and VIM cool `{COOL}`. Dense scatter layers are rasterized at 450 DPI in PDF/SVG; text, axes, centroid labels and leaders remain vector/editable.
- `CRC_joint_atlas_render_metadata.json`: exact rendering/count/provenance details, including zoom bounds and cropped-cell counts.
- `VISUAL_REVIEW_JOINT_UMAP.md`, `CRC_joint_atlas_numeric_verification.json` and `CRC_joint_atlas_format_verification.json`: final visual, source-table and output-format checks.
- The existing `CRC_epithelial_UMAP_*` files remain complementary detail panels and are not overwritten.

## Embedding provenance — reused, not recomputed

All **643,655 cells / 316 samples / 23 studies** use the existing joint atlas coordinates (paths relative to the `crc_codex/` workspace root):

1. `full_atlas_reintegration_20261107/atlas/all_cell_representations.npz:X_umap`.
2. Identical `obsm/X_umap` in `full_atlas_reintegration_20261107/atlas/all_cells_expression_integrated.h5ad`.

All cell IDs and their order were matched exactly to `atlas/per_cell_labels.csv.gz` and `all_cell_final_labels.csv.gz`; the two embedding arrays are **exactly equal**. Coordinates are not rotated, stretched, recomputed, or replaced by the epithelial-only UMAP. All main axes have identical limits and equal coordinate aspect. No expression matrix was normalized or reanalyzed.

Per `PRESPECIFIED_METHODS.md`, `atlas/completion.json`, `atlas/mapping_QC_summary.json` and `expression_engine.py`, the pre-existing embedding used a sample/original-major-lineage-balanced **60,000-cell fitted expression reference**, sample-aware Seurat **HVG3000**, scaling, **PCA40**, **Harmony on technical `series`** (theta2/lambda1, up to20 Harmony iterations), 30 neighbors, Leiden0.6/seed42 and fitted UMAP (seed42, 200 iterations). The other **583,655 cells** were mapped by PCA transformation, local Harmony-displacement propagation, 30-NN weighted cluster vote and barycentric fitted-UMAP projection. It is **not a new UMAP fit on all 643,655 cells**. Mapping vote confidence is internal assignment confidence, not independent biological validation. Visualization does not rerun any integration, mapping, clustering or CM fitting.

## Coverage and unassigned/mixed handling

The three provided populations are disjoint, fully present in the joint embedding:

| Population | Cells |
|---|---:|
| Calibrated tumor epithelial clusters | 76,511 |
| Calibrated non-epithelial cells / 65 fine subtypes | 397,134 |
| Normal epithelial reference | 127,657 |
| Calibrated/reference union | **601,302 (93.4%)** |
| **Unassigned/mixed remainder** | **42,353 (6.6%)** |
| **Full atlas in every main panel** | **643,655** |

The **42,353 remainder cells are never dropped and never assigned to a calibrated cluster or lineage**. Their display lineage, calibrated label and B group are `unassigned/mixed`, and they are light grey in A/B and part of the very-light-grey background in C/D. Their *earlier* atlas lineage/subtype is retained only in explicitly named provenance columns; this does not grant calibrated membership. They are labelled as unassigned/mixed (n=42,353) in the figure. There is no biological-identity reclassification and no source-label edit.

Read-only calibrated membership inputs (relative to `calibrated_single_cluster_20261107/`): `epithelial/labels_hvg3000_r0.4.csv.gz`, `data/non_epithelial_per_cell_labels.csv.gz`, and `data/normal_epithelial_reference_labels.csv.gz`. `selected_pair.json` and the epithelial support table are independently checked. No frozen membership is updated.

## Panel A — broad-lineage joint atlas

All **643,655** cells. For the 601,302-cell covered union, A uses the **existing atlas `broad_lineage_v1`**, not a newly inferred lineage; the 42,353 remainder is unassigned/mixed. The color key also serves as the exact lineage count table. Centroid labels are observed cells nearest coordinate-wise medians; no per-cell text.

| Display broad lineage | Cells |
|---|---:|
{table_a}

## Panel B — full calibrated annotation, display aggregation only

All **643,655** cells: all 15 unchanged tumor epithelial clusters (**76,511**), normal epithelial reference in one muted color (**127,657**), the **397,134** calibrated non-epithelial cells aggregated by their original source `lineage`, and **42,353** unassigned/mixed. Fine subtype IDs are preserved in the source table. The display expands `Mye` to Myeloid and `Endo` to Endothelial only. **Cycling remains a separate source category (2,718 cells)**; it is not silently assigned to T/NK or any new lineage.

A and B intentionally use their different existing source annotation levels; their non-epithelial lineage totals need not coincide. `CRC_joint_non_epithelial_lineage_crosswalk.csv` makes this explicit. A retains the saved atlas broad assignments for these cells; B retains calibrated source lineage. Tumor epithelial cells are drawn after the normal-reference/background layer so their calibrated colors remain visible; both layers retain every cell. Overlapping epithelial cluster centroids are not crowded into 15 unreadable text labels on the coarse joint atlas: exact c00–c14 identities/counts are in the compact categorical key, and the shared epithelial region is labelled once at its centroid. The epithelial-only figure remains the complementary fine-structure view.

| Display group | Cells |
|---|---:|
{table_b}

## Panel C — frozen pair only

**Panel C highlights the frozen pair only.** No pooling, re-selection, score threshold or new state assignment:

- **Epi_JUN-like (c08): `{JUN}`**, **4,855 cells**, warm `{WARM}`.
- **Epi_VIM-like (c03): `{VIM}`**, **7,748 cells**, cool `{COOL}`.
- **Other atlas cells: 631,052**, very light grey, including all 42,353 unassigned/mixed cells.
- **Main-panel total: 643,655**. Each cell is plotted once in the main UMAP; the two side zooms are additional views of the same cells, not additional observations. Each zoom highlights only its named frozen state.

Insets are square local crops, centered at the coordinate-wise median of their named state with half-span = maximum distance to its axis-wise 1st/99th percentile bounds +0.4 UMAP units. Crop bounds and the exact number of cells shown are recorded in the metadata and printed above each inset. No coordinates are refitted or magnified anisotropically. The JUN/VIM names indicate **relative frozen reference affinity, not proof of validated conserved biological identities**; upstream scientific conclusions are unchanged.

## Panel D — non-epithelial CM1-like context

All **643,655** cells: **68,508** cells belonging to the five highest H-loading subtypes of the unchanged **`{CM}`**, **K=14**, are highlighted by a purple ramp; **575,147** other atlas cells are very light grey. Selection is descending frozen subtype H loading only, not an epithelial association or outcome. Centroid numbers match subtype IDs in the side key.

The unchanged T/NK H-loading share is **0.9025697482418819** (display 0.9026 /90.26%). The top five contribute **{top5.H_share.sum():.15f}** of module H loading (display67.43%). These are **subtype-level NMF H loadings, not cell-level module activities**; colored cells merely belong to the selected subtypes.

| Rank | Original subtype ID | Cells | H-loading share |
|---|---|---:|---:|
{table_d}

H values are copied from `CM/H_selected.csv` and checked against `CM/CM1_composition_marker_evidence.csv` and `CM/selected_CM.json`. Full precision is retained in source data. The marker pairs in D are the first two existing `top10_markers`, not new subtype names or new differential-expression results.

## Source data

All filenames below are in `source_data/` and prefixed `CRC_joint_` to avoid replacing the epithelial-detail sources.

- `CRC_joint_atlas_per_cell_plot_data.csv.gz`: **every cell**, with `cell_id, UMAP1, UMAP2, broad_lineage, calibrated_label, is_JUN_pole, is_VIM_pole`; additionally sample/study/donor/status, annotation source, unchanged original atlas broad lineage/cluster/subtype, original subtype, calibrated source non-epithelial lineage, B display group, fit-reference membership, subtype-level CM H-loading share (blank outside calibrated non-epithelium), and D highlight flag. Coordinates have lossless float32 round-trip precision.
- `CRC_joint_lineage_counts.csv`: exact A lineage cell/sample counts, full-atlas fractions and colors.
- `CRC_joint_cluster_counts.csv`: unchanged 15 tumor epithelial clusters, original full IDs, cell/sample/patient/study counts and colors; checked against frozen support table.
- `CRC_joint_panel_B_annotation_counts.csv`: all B display categories and exact counts.
- `CRC_joint_coverage_counts.csv`: disjoint provenance-population counts.
- `CRC_joint_non_epithelial_lineage_crosswalk.csv`: calibrated source lineage versus pre-existing atlas broad lineage; no resolutions imposed.
- `CRC_joint_subtype_to_CM1_loading.csv`: all65 unchanged subtype IDs and their original lineage, frozen H value/share, cell/sample counts, marker evidence, loading rank, D highlight flag and color.
- `CRC_joint_centroid_label_anchors.csv`: actual observed-cell robust centroid anchors used for labels.

## Reproduce

```bash
/mnt/disk18t/lr_xcy/riku/go_bp/pathway_analysis/cellsubtype_degs/.micromamba/envs/codex_lr_scanpy311/bin/python render_crc_joint_atlas_fig7style.py
```

For layout-only iteration, `--render-only` reuses this figure's exported cell/loading tables; `--only combined C` restricts outputs. `--preview` writes a separate160-DPI layout preview only and never substitutes for the requested450-DPI final files. Source labels, the frozen pair, CM values and statistics are read-only inputs throughout.
"""
    (OUT / "README_JOINT_UMAP.md").write_text(readme)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true", help="Reuse exported joint-atlas source tables for layout-only iteration")
    parser.add_argument("--only", nargs="+", choices=["combined", "A", "B", "C", "D"])
    parser.add_argument("--preview", action="store_true", help="Write a separate 160-DPI combined layout preview")
    args = parser.parse_args()
    font = setup_style()
    if args.render_only:
        cells = pd.read_csv(CELL_TABLE, index_col="cell_id", float_precision="round_trip", low_memory=False)
        loading = pd.read_csv(DATA / "CRC_joint_subtype_to_CM1_loading.csv", index_col="subtype_id", float_precision="round_trip")
    else:
        cells, loading = prepare_data()
    context = make_context(cells, loading)
    targets = set(args.only or ["combined", "A", "B", "C", "D"])
    metadata_path = OUT / "CRC_joint_atlas_render_metadata.json"
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    outputs = previous.get("outputs", {})
    panel_metadata = previous.get("panels", {})
    if "combined" in targets or args.preview:
        fig = plt.figure(figsize=(16.0, 13.1))
        text(fig, 0.35, 0.20, "CRC | full-atlas joint UMAP", fontsize=16.0, fontweight="bold")
        text(fig, 0.35, 0.62, "643,655 cells · 316 samples · 23 studies · epithelial + non-epithelial cells together", fontsize=9.0, color=MUTED)
        regions = {"A": (0.35, 1.10, 7.42, 5.52), "B": (8.19, 1.10, 7.42, 5.52),
                   "C": (0.35, 6.97, 7.42, 5.52), "D": (8.19, 6.97, 7.42, 5.52)}
        for letter, region in regions.items():
            panel_metadata[letter] = PANELS[letter](fig, region, context)
        text(fig, 0.35, 12.80,
             "Saved full-atlas Harmony / UMAP coordinates; no reintegration, reclustering or relabeling. All main panels retain unassigned/mixed (n = 42,353).",
             fontsize=7.6, color=MUTED)
        stem = "CRC_joint_atlas_layout_preview" if args.preview else "CRC_joint_atlas_UMAP_combined"
        result = save(fig, stem, preview=args.preview)
        if args.preview:
            return
        outputs["combined"] = result
    for letter in "ABCD":
        if letter not in targets:
            continue
        fig = plt.figure(figsize=(8.12, 6.24))
        panel_metadata[letter] = PANELS[letter](fig, (0.35, 0.30, 7.42, 5.52), context)
        outputs[letter] = save(fig, f"CRC_joint_atlas_UMAP_panel_{letter}_{PANEL_NAMES[letter]}")
    write_notes(context, font, outputs, panel_metadata)
    print("Wrote joint-atlas source data, provenance and exact panel counts.", flush=True)


if __name__ == "__main__":
    main()

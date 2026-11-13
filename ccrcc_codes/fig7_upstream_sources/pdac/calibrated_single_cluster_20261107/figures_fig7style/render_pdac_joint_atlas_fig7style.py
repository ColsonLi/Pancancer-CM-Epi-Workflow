#!/usr/bin/env python3
"""Visualization only: the complete saved PDAC atlas, with calibrated overlays."""

from pathlib import Path
import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/opencode/pdac_joint_fig7style_matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

import h5py
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.colors import LinearSegmentedColormap, to_hex
from PIL import Image


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
ATLAS = ROOT.parent / "full_atlas_reintegration_20261107"
DATA = OUT / "source_data"
CONFIG = "A_HVG3000_sample_r0.4"
DPI = 450
INK, MUTED = "#232A30", "#69757E"
WARM, COOL, PURPLE = "#D26843", "#3676A1", "#8062A3"
GREY, VERY_LIGHT = "#D7DDE1", "#E7EBEE"
UNRESOLVED = "unresolved/mixed"
NORMAL = "Normal epithelial reference"
LINEAGES = ["Epithelial", "T_NK", "Myeloid", "Stromal", "Endothelial", "B", "Plasma", "Mast", UNRESOLVED]
LINEAGE_COLORS = {
    "Epithelial": "#A5AFBF", "T_NK": PURPLE, "Myeloid": "#B9A06E",
    "Stromal": "#7D9F95", "Endothelial": "#6D9FB7", "B": "#B87D97",
    "Plasma": "#C898B4", "Mast": "#A48870", UNRESOLVED: GREY,
}
EPI_COLORS = {
    "c0": "#9DACBA", "c1": "#B8AE98", "c2": COOL, "c3": "#CDBEBC",
    "c4": "#A1B5A3", "c5": "#AA9DB6", "c6": "#88A7AB", "c7": WARM,
    "c8": "#BDA58E", "c9": "#A5A6B6", "c10": "#C8BDA6", "c11": "#96B4B0",
}
NORMAL_COLOR = "#A6B7AD"
CELL_FILE = DATA / "PDAC_joint_atlas_per_cell_plot_data.csv.gz"
REGIONS = [(0.35, 1.06, 6.55, 5.50), (7.27, 1.06, 6.55, 5.50),
           (0.35, 7.05, 6.55, 5.50), (7.27, 7.05, 6.55, 5.50)]


def setup_style():
    for path in (Path.home() / ".local/share/fonts/codex_trajectory_20260813").glob("Arial*.TTF"):
        font_manager.fontManager.addfont(str(path))
    font = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "Liberation Sans"
    plt.rcParams.update({
        "font.family": font, "font.size": 7.5,
        "axes.labelsize": 7.2, "axes.titlesize": 9.2, "axes.linewidth": 0.55,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": DPI, "axes.grid": False, "mathtext.default": "regular",
    })
    return font


def prepare_data():
    DATA.mkdir(exist_ok=True)
    atlas = pd.read_csv(ATLAS / "tables/full_atlas_new_labels.csv.gz", low_memory=False,
                        usecols=["merged_cell_id", "sample", "study", "donor", "status", "broad_lineage", "new_subtype"])
    atlas = atlas.set_index("merged_cell_id")
    epi = pd.read_csv(ROOT / f"labels/{CONFIG}.csv.gz").set_index("cell_id")
    non = pd.read_csv(ROOT / "labels/non_epithelial_54_subtypes.csv.gz").set_index("cell_id")
    normal = pd.read_csv(ROOT / "labels/normal_epithelial_reference.csv.gz",
                         usecols=["cell_id", "reference_population", "status"]).set_index("cell_id")
    with h5py.File(ATLAS / "objects/full_atlas_verified_embeddings.h5ad") as h:
        index_key = h["obs"].attrs["_index"]
        ids = pd.Index(h["obs"][index_key].asstr()[:], name="cell_id")
        xy = h["obsm/X_umap"][:]
        provenance = {key: h["uns"][key].asstr()[()] for key in h["uns"]}
    assert len(ids) == 392388 and ids.is_unique and atlas.index.is_unique
    assert ids.difference(atlas.index).empty and np.isfinite(xy).all()
    assert len(epi) == 55357 and len(non) == 211981 and len(normal) == 34897
    for frame in [epi, non, normal]:
        assert frame.index.is_unique and frame.index.difference(ids).empty
    assert epi.index.intersection(non.index).empty
    assert epi.index.intersection(normal.index).empty
    assert non.index.intersection(normal.index).empty
    pair = json.loads((ROOT / "tables/selected_single_cluster_pair.json").read_text())
    assert pair["configuration"] == CONFIG and pair["JUN"] == "c7" and pair["VIM"] == "c2"

    cells = atlas.reindex(ids).rename(columns={"sample": "sample_id", "broad_lineage": "atlas_broad_lineage",
                                             "new_subtype": "atlas_new_subtype"})
    cells[["UMAP1", "UMAP2"]] = xy
    cells["broad_lineage"] = UNRESOLVED
    cells["calibrated_label"] = UNRESOLVED
    cells["calibrated_lineage_original"] = ""
    cells["calibration_population"] = "outside_calibrated_annotation"
    cells["cluster_id"] = ""
    cells["non_epi_subtype"] = ""
    cells["normal_reference_population"] = ""
    cells.loc[epi.index, "broad_lineage"] = "Epithelial"
    cells.loc[epi.index, "calibrated_label"] = epi.cluster_id
    cells.loc[epi.index, "cluster_id"] = epi.cluster_id
    cells.loc[epi.index, "calibrated_lineage_original"] = "Epithelial"
    cells.loc[epi.index, "calibration_population"] = "tumor_associated_ductal"
    cells.loc[normal.index, "broad_lineage"] = "Epithelial"
    cells.loc[normal.index, "calibrated_label"] = NORMAL
    cells.loc[normal.index, "normal_reference_population"] = normal.reference_population
    cells.loc[normal.index, "calibrated_lineage_original"] = "Epithelial"
    cells.loc[normal.index, "calibration_population"] = "normal_epithelial_reference"
    cells.loc[non.index, "broad_lineage"] = non.lineage.replace({"TNK": "T_NK"})
    cells.loc[non.index, "calibrated_label"] = non.non_epi_subtype
    cells.loc[non.index, "non_epi_subtype"] = non.non_epi_subtype
    cells.loc[non.index, "calibrated_lineage_original"] = non.lineage
    cells.loc[non.index, "calibration_population"] = "calibrated_non_epithelial"
    cells["is_calibrated"] = cells.calibration_population.ne("outside_calibrated_annotation")
    cells["is_JUN_pole"] = cells.cluster_id.eq("c7")
    cells["is_VIM_pole"] = cells.cluster_id.eq("c2")
    assert cells.is_JUN_pole.sum() == 3245 and cells.is_VIM_pole.sum() == 5227
    assert cells.is_calibrated.sum() == 302235 and (~cells.is_calibrated).sum() == 90153
    assert cells.loc[epi.index, "sample_id"].equals(epi.sample_id.rename("sample_id"))
    cells["panel_B_group"] = cells.broad_lineage
    cells.loc[epi.index, "panel_B_group"] = epi.cluster_id
    cells.loc[normal.index, "panel_B_group"] = NORMAL

    lineages = cells.groupby("broad_lineage").size().reindex(LINEAGES, fill_value=0).rename("n_cells").to_frame()
    lineages["percent_atlas"] = lineages.n_cells / len(cells) * 100
    lineages["color"] = [LINEAGE_COLORS[x] for x in lineages.index]
    lineages["definition"] = "Supplied calibrated lineage; tumor ductal + normal reference combined as Epithelial"
    lineages.loc[UNRESOLVED, "definition"] = "All cells absent from all three calibrated cell-ID sets; no cluster inferred"
    lineages.loc["Plasma", "definition"] = "No explicit Plasma label in calibrated inputs; B_MZB1 remains supplied B lineage"
    lineages.to_csv(DATA / "PDAC_joint_atlas_lineage_counts.csv", index_label="broad_lineage", float_format="%.17g")
    cells.groupby("atlas_broad_lineage").size().rename("n_cells").to_csv(DATA / "PDAC_joint_atlas_archived_lineage_counts.csv")
    cells.loc[~cells.is_calibrated].groupby("atlas_broad_lineage").size().rename("n_cells").to_csv(
        DATA / "PDAC_joint_atlas_unresolved_display_original_lineages.csv")
    pd.crosstab(cells.atlas_broad_lineage, cells.broad_lineage).to_csv(
        DATA / "PDAC_joint_atlas_original_vs_display_lineage_crosstab.csv")

    clusters = cells.groupby(["calibration_population", "calibrated_label", "broad_lineage"], observed=True).agg(
        n_cells=("sample_id", "size"), n_samples=("sample_id", "nunique"),
        n_studies=("study", "nunique"), n_donor_keys=("donor", "nunique"),
        UMAP1_mean=("UMAP1", "mean"), UMAP2_mean=("UMAP2", "mean"),
        UMAP1_median=("UMAP1", "median"), UMAP2_median=("UMAP2", "median"),
    ).reset_index()
    clusters.to_csv(DATA / "PDAC_joint_atlas_cluster_counts.csv", index=False, float_format="%.17g")

    contract = json.loads((ROOT / "tables/cm_calibration_contract.json").read_text())
    assert contract["selected_K"] == 11 and contract["CM1"] == "calCM02"
    H = pd.read_csv(ROOT / "tables/cm_selected_H.csv", index_col=0, float_precision="round_trip").loc["calCM02"]
    F = pd.read_csv(ROOT / "tables/cm_selected_H_fraction.csv", index_col=0, float_precision="round_trip").loc["calCM02"]
    composition = pd.read_csv(ROOT / "tables/cm_selected_lineage_composition.csv", index_col=0,
                              float_precision="round_trip").loc["calCM02"]
    assert set(H.index) == set(non.non_epi_subtype.unique()) and len(H) == 54
    np.testing.assert_allclose(F.values, (H / H.sum()).values, atol=1e-14)
    non_lineages = non.groupby("non_epi_subtype").lineage.agg(lambda s: s.unique()[0])
    cm = pd.DataFrame({"H_loading": H, "H_fraction": F})
    cm.index.name = "non_epi_subtype"
    cm["lineage_original"] = non_lineages
    cm["display_lineage"] = cm.lineage_original.replace({"TNK": "T_NK"})
    cm["n_cells"] = non.non_epi_subtype.value_counts()
    cm["module"] = "calCM02"
    cm["selected_K"] = 11
    cm = cm.sort_values("H_loading", ascending=False, kind="stable")
    cm["H_loading_rank"] = np.arange(1, len(cm) + 1)
    cm["highlighted_in_panel_D"] = cm.H_loading_rank.le(5)
    cm["panel_D_color"] = VERY_LIGHT
    ramp = LinearSegmentedColormap.from_list("CM1_purple", ["#C2B4D4", PURPLE, "#60447B"])
    for subtype, row in cm.loc[cm.highlighted_in_panel_D].iterrows():
        t = (row.H_loading - cm.iloc[4].H_loading) / (cm.iloc[0].H_loading - cm.iloc[4].H_loading)
        cm.loc[subtype, "panel_D_color"] = to_hex(ramp(0.12 + 0.82 * t))
    tnk_share = float(composition["TNK"])
    np.testing.assert_allclose(cm.loc[cm.lineage_original.eq("TNK"), "H_fraction"].sum(), tnk_share, atol=1e-14)
    np.testing.assert_allclose(tnk_share, contract["CM1_TNK_loading_fraction"], atol=1e-14)
    assert cm.head(5).index.tolist() == ["T_IL7R", "T_CTLA4", "T_GZMK", "T_IFI6", "T_TPT1"]
    cells["panel_D_is_top5_subtype"] = cells.non_epi_subtype.isin(cm.head(5).index)
    cm.to_csv(DATA / "PDAC_joint_atlas_subtype_to_CM1_loading.csv", float_format="%.17g")

    columns = ["UMAP1", "UMAP2", "broad_lineage", "calibrated_label", "is_JUN_pole", "is_VIM_pole",
               "calibration_population", "is_calibrated", "cluster_id", "non_epi_subtype", "normal_reference_population",
               "calibrated_lineage_original", "atlas_broad_lineage", "atlas_new_subtype", "sample_id", "study", "donor", "status",
               "panel_B_group", "panel_D_is_top5_subtype"]
    cells[columns].to_csv(CELL_FILE, index_label="cell_id", float_format="%.17g")
    summary = {
        "n_cells": len(cells), "n_samples": int(cells.sample_id.nunique()),
        "n_studies": int(cells.study.nunique()), "n_donor_keys": int(cells.donor.nunique()),
        "n_calibrated": int(cells.is_calibrated.sum()), "n_grey_outside_calibration": int((~cells.is_calibrated).sum()),
        "tnk_H_fraction": tnk_share, "embedding_h5ad_uns": provenance,
        "frozen_pair": {"JUN": "c7", "VIM": "c2", "JUN_n": 3245, "VIM_n": 5227},
    }
    (DATA / "PDAC_joint_atlas_plot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return cells, lineages, clusters, cm, summary


def load_data():
    return (
        pd.read_csv(CELL_FILE, index_col="cell_id", float_precision="round_trip", keep_default_na=False),
        pd.read_csv(DATA / "PDAC_joint_atlas_lineage_counts.csv", index_col="broad_lineage"),
        pd.read_csv(DATA / "PDAC_joint_atlas_cluster_counts.csv", float_precision="round_trip"),
        pd.read_csv(DATA / "PDAC_joint_atlas_subtype_to_CM1_loading.csv", index_col="non_epi_subtype", float_precision="round_trip"),
        json.loads((DATA / "PDAC_joint_atlas_plot_summary.json").read_text()),
    )


def rect(fig, left, top, width, height):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, 1 - (top + height) / fh, width / fw, height / fh])


def text(fig, x, y, label, **kwargs):
    fw, fh = fig.get_size_inches()
    return fig.text(x / fw, 1 - y / fh, label, va=kwargs.pop("va", "top"), **kwargs)


def header(fig, region, letter, title, subtitle):
    left, top, _, _ = region
    text(fig, left, top, letter, fontsize=13, fontweight="bold")
    text(fig, left + 0.29, top + 0.025, title, fontsize=10.3, fontweight="bold")
    text(fig, left + 0.29, top + 0.30, subtitle, fontsize=7.8, color=MUTED)


def limits(cells):
    xy = cells[["UMAP1", "UMAP2"]].to_numpy()
    return xy.min(0) - [0.95, 0.95], xy.max(0) + [0.95, 0.95]


def style_umap(ax, bounds, labels=True, inset=False):
    lo, hi = bounds
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xticks=[], yticks=[])
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ABB5BC")
    ax.spines[["left", "bottom"]].set_linewidth(0.45 if inset else 0.55)
    if labels:
        ax.set_xlabel("UMAP 1", labelpad=3, color=MUTED)
        ax.set_ylabel("UMAP 2", labelpad=3, color=MUTED)


def cloud(ax, cells, colors, size=0.34, alpha=0.88, shuffle=True):
    if len(cells) == 0:
        return
    order = np.random.default_rng(42).permutation(len(cells)) if shuffle else np.arange(len(cells))
    xy = cells[["UMAP1", "UMAP2"]].to_numpy()[order]
    if not isinstance(colors, str):
        colors = np.asarray(colors)[order]
    return ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=size, alpha=alpha,
                      edgecolors="none", linewidths=0, rasterized=True)


def centroid(cells):
    return cells[["UMAP1", "UMAP2"]].mean().to_numpy()


def label(ax, value, anchor, target=None, color=INK, fs=7.4, ha="center"):
    target = anchor if target is None else target
    t = ax.annotate(value, xy=anchor, xytext=target, ha=ha, va="center", fontsize=fs,
                    color=color, fontweight="bold", zorder=7,
                    arrowprops=None if np.allclose(anchor, target) else {
                        "arrowstyle": "-", "color": color, "lw": 0.48, "shrinkA": 2, "shrinkB": 1.5},
                    annotation_clip=False)
    t.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground="white")])
    return t


def dot(fig, x, y, color, size=15):
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([x / fw, 1 - (y + 0.07) / fh, 0.10 / fw, 0.14 / fh])
    ax.scatter([0.5], [0.5], s=size, color=color, edgecolors="none")
    ax.set_axis_off()


def add_broad_centroid_labels(ax, cells, panel="A"):
    records = []
    display = {"T_NK": "T_NK", "Myeloid": "Myeloid", "Stromal": "Stromal", "Endothelial": "Endothelial",
               "B": "B", "Mast": "Mast"}
    for group, value in display.items():
        subset = cells.loc[cells.broad_lineage.eq(group)]
        anchor = centroid(subset)
        label(ax, value, anchor, fs=7.1)
        records.append({"panel": panel, "label": value, "centroid_UMAP1": anchor[0], "centroid_UMAP2": anchor[1],
                        "label_UMAP1": anchor[0], "label_UMAP2": anchor[1]})
    if panel == "A":
        subset = cells.loc[cells.broad_lineage.eq("Epithelial")]
        anchor, target = centroid(subset), (12.0, -0.3)
        label(ax, "Epithelial", anchor, target=target, fs=7.1)
        records.append({"panel": panel, "label": "Epithelial", "centroid_UMAP1": anchor[0], "centroid_UMAP2": anchor[1],
                        "label_UMAP1": target[0], "label_UMAP2": target[1]})
    else:
        normal = cells.loc[cells.calibrated_label.eq(NORMAL)]
        epi = cells.loc[cells.calibration_population.eq("tumor_associated_ductal")]
        label(ax, "Normal epi.", centroid(normal), target=(15.0, 4.8), fs=6.8)
        label(ax, "Tumor c0–c11", centroid(epi), target=(12.2, -8.0), fs=6.8)
        for value, subset, target in [("Normal epi.", normal, (15.0, 4.8)), ("Tumor c0–c11", epi, (12.2, -8.0))]:
            anchor = centroid(subset)
            records.append({"panel": panel, "label": value, "centroid_UMAP1": anchor[0], "centroid_UMAP2": anchor[1],
                            "label_UMAP1": target[0], "label_UMAP2": target[1]})
    return records


def panel_a(fig, region, cells, lineages, clusters, cm, summary):
    left, top, width, height = region
    header(fig, region, "A", "Joint atlas · calibrated broad lineages", "392,388 cells · 125 samples · all cells retained")
    ax = rect(fig, left + 0.19, top + 0.66, 4.25, 4.40)
    style_umap(ax, limits(cells))
    cloud(ax, cells.loc[~cells.is_calibrated], GREY, size=0.30, alpha=0.72)
    resolved = cells.loc[cells.is_calibrated]
    cloud(ax, resolved, resolved.broad_lineage.map(LINEAGE_COLORS))
    records = add_broad_centroid_labels(ax, cells)
    lx, right = left + 4.63, left + width - 0.02
    text(fig, lx, top + 0.73, "Lineage", fontsize=7.6, color=MUTED)
    text(fig, right, top + 0.73, "n cells", fontsize=7.6, color=MUTED, ha="right")
    for i, group in enumerate(LINEAGES):
        y = top + 1.04 + i * 0.265
        dot(fig, lx - 0.015, y + 0.045, LINEAGE_COLORS[group])
        text(fig, lx + 0.13, y, group, fontsize=7.4 if group != UNRESOLVED else 6.9)
        text(fig, right, y, f"{int(lineages.loc[group, 'n_cells']):,}", fontsize=7.4, ha="right")
    text(fig, lx, top + 3.68, "Calibrated coverage", fontsize=7.5, fontweight="bold")
    text(fig, lx, top + 3.90, "302,235 cells · 77.0%", fontsize=8.0, color=PURPLE, fontweight="bold")
    text(fig, lx, top + 4.26, "Grey: all cells outside calibration.\nNo new cluster assignment.\nPlasma: no explicit input label;\nB_MZB1 retains B lineage.",
         fontsize=6.7, color=MUTED, linespacing=1.5)
    text(fig, left + 0.29, top + height - 0.07, "unresolved/mixed (n=90,153) includes uncalibrated profiles; archived labels are retained in source data.",
         fontsize=6.8, color=MUTED)
    return records


def ductal_centroid_inset(fig, left, top, cells):
    text(fig, left, top, "Ductal cluster centroids", fontsize=6.8, color=MUTED)
    ax = rect(fig, left, top + 0.18, 1.85, 1.20)
    bounds = (np.array([1.8, -6.15]), np.array([10.9, 0.60]))
    style_umap(ax, bounds, labels=False, inset=True)
    label_positions = {
        "c0": (6.0, -5.60), "c1": (2.75, -3.3), "c2": (5.65, -0.6),
        "c3": (10.10, -3.1), "c4": (10.0, -0.7), "c5": (10.1, -4.4),
        "c6": (9.0, -1.25), "c7": (7.45, 0.0), "c8": (4.3, -5.65),
        "c9": (8.9, -5.55), "c10": (4.55, -0.6), "c11": (4.7, -4.5),
    }
    records = []
    for group in EPI_COLORS:
        anchor = centroid(cells.loc[cells.cluster_id.eq(group)])
        ax.scatter(*anchor, s=9, color=EPI_COLORS[group], edgecolors="white", linewidths=0.3, zorder=6)
        label(ax, group, anchor, label_positions[group], color=WARM if group == "c7" else COOL if group == "c2" else INK, fs=6.2)
        records.append({"panel": "B_centroid_inset", "label": group, "centroid_UMAP1": anchor[0], "centroid_UMAP2": anchor[1],
                        "label_UMAP1": label_positions[group][0], "label_UMAP2": label_positions[group][1]})
    return records


def panel_b(fig, region, cells, lineages, clusters, cm, summary):
    left, top, width, height = region
    header(fig, region, "B", "Full calibrated annotation", "12 ductal clusters · normal reference · 54 non-epithelial subtypes")
    ax = rect(fig, left + 0.19, top + 0.66, 4.25, 4.40)
    style_umap(ax, limits(cells))
    cloud(ax, cells.loc[~cells.is_calibrated], GREY, size=0.30, alpha=0.72)
    colors = {**LINEAGE_COLORS, **EPI_COLORS, NORMAL: NORMAL_COLOR}
    resolved = cells.loc[cells.is_calibrated]
    cloud(ax, resolved, resolved.panel_B_group.map(colors))
    records = add_broad_centroid_labels(ax, cells, panel="B")
    lx, right = left + 4.63, left + width - 0.02
    text(fig, lx, top + 0.73, "Tumor ductal · 55,357", fontsize=7.4, fontweight="bold")
    counts = cells.loc[cells.cluster_id.ne(""), "cluster_id"].value_counts()
    for i, group in enumerate(EPI_COLORS):
        col, row = i // 6, i % 6
        x, y = lx + col * 0.99, top + 1.02 + row * 0.20
        dot(fig, x - 0.02, y + 0.04, EPI_COLORS[group], size=12)
        text(fig, x + 0.10, y, group, fontsize=6.6)
        text(fig, x + 0.90, y, f"{int(counts[group]):,}", fontsize=6.6, ha="right")
    text(fig, lx, top + 2.27, "Reference + non-epithelium", fontsize=7.0, fontweight="bold")
    rows = [("Normal epi.", 34897, NORMAL_COLOR)]
    rows += [(g, int(lineages.loc[g, "n_cells"]), LINEAGE_COLORS[g]) for g in ["T_NK", "Myeloid", "Stromal", "Endothelial", "B", "Mast"]]
    rows += [("unresolved/mixed", 90153, GREY)]
    for i, (value, n, color) in enumerate(rows):
        y = top + 2.50 + i * 0.17
        dot(fig, lx - 0.02, y + 0.045, color, size=12)
        text(fig, lx + 0.11, y, value, fontsize=6.7)
        text(fig, right, y, f"{n:,}", fontsize=6.7, ha="right")
    records += ductal_centroid_inset(fig, lx, top + 3.92, cells)
    text(fig, left + 0.29, top + height - 0.07, "Non-epithelial colors aggregate the 54 unchanged subtypes by supplied lineage. Inset: centroids only.",
         fontsize=6.8, color=MUTED)
    return records


def panel_c(fig, region, cells, lineages, clusters, cm, summary):
    left, top, width, height = region
    header(fig, region, "C", "Frozen pair on the joint atlas", "Only c7 and c2 highlighted · 383,916 other cells in very light grey")
    ax = rect(fig, left + 0.19, top + 0.66, 4.25, 4.40)
    style_umap(ax, limits(cells))
    pair_mask = cells.is_JUN_pole | cells.is_VIM_pole
    cloud(ax, cells.loc[~pair_mask], VERY_LIGHT, size=0.28, alpha=0.66)
    poles = cells.loc[pair_mask]
    cloud(ax, poles, np.where(poles.is_JUN_pole, WARM, COOL), size=0.52, alpha=0.92)
    lx = left + 4.63
    records = []
    for key, title, color, offset, role, genes in [
        ("is_JUN_pole", "Epi_JUN-like (c7)", WARM, 0.72, "positive pole", "JUN · ATF3\nHSPA1A · GADD45B"),
        ("is_VIM_pole", "Epi_VIM-like (c2)", COOL, 3.00, "negative reference", "S100A11 · S100A6\nLCN2 · MMP7"),
    ]:
        state = cells.loc[cells[key]]
        center = state[["UMAP1", "UMAP2"]].median().to_numpy()
        half = np.array([4.50, 3.375])
        lo, hi = center - half, center + half
        in_window = cells.UMAP1.between(lo[0], hi[0]) & cells.UMAP2.between(lo[1], hi[1])
        text(fig, lx, top + offset, title, fontsize=7.6, color=color, fontweight="bold")
        text(fig, lx, top + offset + 0.22, f"n={len(state):,} · {role}", fontsize=6.8, color=MUTED)
        iax = rect(fig, lx, top + offset + 0.47, 1.85, 1.39)
        style_umap(iax, (lo, hi), labels=False, inset=True)
        context = cells.loc[in_window & ~cells[key]]
        selected = cells.loc[in_window & cells[key]]
        cloud(iax, context, VERY_LIGHT, size=0.35, alpha=0.70)
        cloud(iax, selected, color, size=0.80, alpha=0.96)
        text(fig, lx, top + offset + 1.94, genes, fontsize=6.6, color=MUTED, linespacing=1.25)
        records.append({"state": title, "n_full_atlas": len(state), "n_highlighted_in_zoom": len(selected),
                        "n_background_in_zoom": len(context), "center_UMAP1": center[0], "center_UMAP2": center[1],
                        "UMAP1_min": lo[0], "UMAP1_max": hi[0], "UMAP2_min": lo[1], "UMAP2_max": hi[1],
                        "center_definition": "coordinate-wise median of the frozen state", "same_zoom_scale": True})
    text(fig, left + 0.29, top + height - 0.07, "Frozen c7/c2 membership only; no pooled arms or re-selection. c2 is a negative reference, not a canonical EMT claim.",
         fontsize=6.8, color=MUTED)
    return records


def panel_d(fig, region, cells, lineages, clusters, cm, summary):
    left, top, width, height = region
    top5 = cm.head(5)
    n_top = int(cells.panel_D_is_top5_subtype.sum())
    header(fig, region, "D", "CM1-like non-epithelial context", f"calCM02 · top 5 H-loading subtypes · {n_top:,} highlighted cells")
    ax = rect(fig, left + 0.19, top + 0.66, 4.25, 4.40)
    style_umap(ax, limits(cells))
    cloud(ax, cells.loc[~cells.panel_D_is_top5_subtype], VERY_LIGHT, size=0.28, alpha=0.66)
    highlighted = cells.loc[cells.panel_D_is_top5_subtype]
    cloud(ax, highlighted, highlighted.non_epi_subtype.map(top5.panel_D_color), size=0.40, alpha=0.92)
    positions = {"T_IL7R": (-4.8, 3.8), "T_CTLA4": (-12.9, -3.3), "T_GZMK": (-5.3, 10.4),
                 "T_IFI6": (-13.2, 13.0), "T_TPT1": (-4.0, -1.4)}
    records = []
    for subtype in top5.index:
        anchor = centroid(cells.loc[cells.non_epi_subtype.eq(subtype)])
        label(ax, subtype, anchor, positions[subtype], color=PURPLE, fs=7.0)
        records.append({"panel": "D", "label": subtype, "centroid_UMAP1": anchor[0], "centroid_UMAP2": anchor[1],
                        "label_UMAP1": positions[subtype][0], "label_UMAP2": positions[subtype][1]})
    lx, right = left + 4.63, left + width - 0.02
    text(fig, lx, top + 0.77, "calCM02", fontsize=13.0, color=PURPLE, fontweight="bold")
    text(fig, right, top + 0.84, "K = 11", fontsize=7.4, color=MUTED, ha="right")
    text(fig, lx, top + 1.23, f"{100 * summary['tnk_H_fraction']:.2f}%", fontsize=17, color=PURPLE, fontweight="bold")
    text(fig, lx, top + 1.61, "T/NK share of total H loading", fontsize=7.3, color=MUTED)
    text(fig, lx, top + 2.10, "Top subtype", fontsize=7.3, color=MUTED)
    text(fig, right, top + 2.10, "H share", fontsize=7.3, color=MUTED, ha="right")
    for i, (subtype, row) in enumerate(top5.iterrows()):
        y = top + 2.45 + i * 0.46
        dot(fig, lx - 0.02, y + 0.045, row.panel_D_color, size=19)
        text(fig, lx + 0.14, y, subtype, fontsize=7.5, fontweight="bold")
        text(fig, right, y, f"{100 * row.H_fraction:.2f}%", fontsize=7.5, ha="right")
        text(fig, lx + 0.14, y + 0.18, f"{int(row.n_cells):,} cells", fontsize=6.8, color=MUTED)
    text(fig, lx, top + 4.92, f"Top 5: {100 * top5.H_fraction.sum():.2f}% of H", fontsize=7.1, color=PURPLE)
    text(fig, left + 0.29, top + height - 0.07, "Subtype-level CM composition, not a per-cell activity score. All other 329,108 atlas cells remain grey.",
         fontsize=6.8, color=MUTED)
    return records


def save(fig, stem):
    fig.canvas.draw()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=DPI, facecolor="white")
    plt.close(fig)
    with Image.open(OUT / f"{stem}.png") as image:
        dims, dpi = list(image.size), list(image.info.get("dpi", []))
    print(f"Rendered {stem}: {dims} px, {dpi} DPI", flush=True)
    return {"files": [f"{stem}.{x}" for x in ["png", "pdf", "svg"]], "png_pixels": dims, "dpi": DPI}


def write_notes(data, font, outputs, zooms):
    cells, lineages, clusters, cm, summary = data
    remainder = cells.loc[~cells.is_calibrated].atlas_broad_lineage.value_counts()
    remainder_md = "\n".join(f"| `{group}` | {int(n):,} |" for group, n in remainder.items())
    lineage_md = "\n".join(f"| {group} | {int(row.n_cells):,} |" for group, row in lineages.iterrows())
    cluster_counts = cells.loc[cells.cluster_id.ne(""), "cluster_id"].value_counts()
    cluster_md = ", ".join(f"{c}={int(cluster_counts[c]):,}" for c in EPI_COLORS)
    cm_md = "\n".join(f"| {subtype} | {row.H_loading:.17g} | {100 * row.H_fraction:.9f}% | {int(row.n_cells):,} |"
                      for subtype, row in cm.head(5).iterrows())
    metadata = {
        **summary, "font": font, "configuration": CONFIG, "DPI": DPI,
        "embedding_source": str(ATLAS / "objects/full_atlas_verified_embeddings.h5ad"), "embedding_key": "obsm/X_umap",
        "atlas_cell_identifier_column": "merged_cell_id", "coordinates_reused_unchanged": True,
        "all_cells_plotted_no_downsampling": True, "new_integration_or_clustering": False, "upstream_inputs_modified": False,
        "panel_A": {"n_cells": 392388, "colored_calibrated_cells": 302235, "grey_unresolved_display": 90153,
                    "lineage_definition": "calibrated lineages; TNK displayed as T_NK; epithelial tumor/reference combined",
                    "Plasma": "0 explicit labels; no B_MZB1 reassignment"},
        "panel_B": {"n_cells": 392388, "tumor_ductal": 55357, "n_tumor_clusters": 12,
                    "normal_reference": 34897, "non_epithelial": 211981, "n_non_epithelial_subtypes": 54,
                    "grey_unresolved_display": 90153, "label_definition": "centroids only; mean coordinates in inset, not moved cells"},
        "panel_C": {"n_cells": 392388, "JUN_c7": 3245, "VIM_c2": 5227, "other_grey": 383916,
                    "frozen_pair_only": True, "colors": {"c7": WARM, "c2": COOL}, "zoom_windows": zooms},
        "panel_D": {"n_cells": 392388, "highlighted_cells": 63280, "other_grey": 329108,
                    "module": "calCM02", "K": 11, "TNK_H_fraction": summary["tnk_H_fraction"],
                    "top_5_subtypes": cm.head(5).index.tolist(), "per_cell_activity_inferred": False},
        "rendering": "450 DPI rasterized dense scatters; vector fonts, axes, centroid labels and leader lines in PDF/SVG",
        "palette": {"lineages": LINEAGE_COLORS, "ductal_clusters": EPI_COLORS, "normal_reference": NORMAL_COLOR,
                    "warm": WARM, "cool": COOL, "purple": PURPLE, "muted": MUTED, "ink": INK},
        "outputs": outputs,
    }
    (OUT / "PDAC_joint_atlas_render_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    notes = f"""# PDAC full-atlas joint UMAP — Figure 7 style

## Delivered files

- `PDAC_joint_atlas_UMAP_combined.{{png,pdf,svg}}`: the full A–D figure.
- `PDAC_joint_atlas_UMAP_panel_[A-D]_*.{{png,pdf,svg}}`: four standalone panels.
- PNGs and raster point layers in PDF/SVG are **450 DPI**. Text, axes, centroid marks and leaders remain vectors in PDF/SVG; SVG text is editable. {font}; white background, no grid, thin axes; ink `{INK}`, muted `{MUTED}`, warm `{WARM}`, cool `{COOL}`, purple `{PURPLE}`.
- The earlier `PDAC_epithelial_UMAP_*` files are complementary epithelial-only detail panels; they were not replaced or modified.

## Embedding provenance — no integration rerun

**All 392,388 cells / {summary['n_samples']} samples / {summary['n_studies']} studies / {summary['n_donor_keys']} donor keys** are plotted in every main UMAP, without downsampling. All four panels have identical coordinate limits and orientation. Cell identities and unchanged coordinates are read directly, using h5py, from:

`{ATLAS / 'objects/full_atlas_verified_embeddings.h5ad'}`

- Cell ID: `obs/merged_cell_id` (the label CSV identifier is also **`merged_cell_id`**, not `cell_id`). Exported as `cell_id` for convenience.
- Coordinates: **`obsm/X_umap`**, with matching 392,388-cell order. Stored all-cell PCA and sample-Harmony coordinates are not recalculated or substituted with epithelial coordinates.
- Object `X` intentionally has shape **392,388 × 0**; no expression is needed or inferred for this visualization.
- Original all-cell source: `{summary['embedding_h5ad_uns']['source_full_atlas']}`.
- Linked expression: `{summary['embedding_h5ad_uns']['source_expression']}`.
- Saved provenance: “{summary['embedding_h5ad_uns']['reuse']}”.

This is the archived coherent **all-cell fit before the broad marker-rank filter**, not a projection or a concatenation of independently fitted lineage UMAPs. No integration, dimensional reduction, reference selection, reclustering, gene-score fit, CM fit or statistical test is run here. Upstream labels and statistics are read-only inputs.

## Calibrated coverage and explicit unresolved/mixed handling

The three supplied calibrated cell-ID sets are mutually exclusive and completely compatible with the embedding:

| Population | Cells |
|---|---:|
| Tumor-associated ductal, `{CONFIG}` | 55,357 |
| Calibrated non-epithelial, 54 subtypes | 211,981 |
| Normal epithelial reference | 34,897 |
| **Calibrated total** | **302,235 ({100 * summary['n_calibrated'] / summary['n_cells']:.9f}% of atlas; displayed 77.0%)** |
| **Outside all three calibrated tables** | **90,153** |
| **Complete atlas** | **392,388** |

**All 90,153 cells outside calibration are displayed in light grey as “unresolved/mixed (n=90,153)”. They are never dropped or assigned to an epithelial cluster, non-epithelial subtype, or frozen pole.** This is a display/coverage status, **not a claim that all those cells carry the archived `Unresolved_mixed` label**. Their original atlas labels are retained in `atlas_broad_lineage` and `atlas_new_subtype` in the per-cell source table; `cluster_id` and `non_epi_subtype` remain empty for these cells.

The full atlas label table's explicit category is exactly **`Unresolved_mixed`**: **15,539 cells across the entire atlas**, of which **5,852 are among the 90,153 cells outside calibration**. The complete original category breakdown of the grey remainder is:

| Original `broad_lineage` category | Cells outside calibration |
|---|---:|
{remainder_md}

The other 9,687 atlas-`Unresolved_mixed` cells have supplied calibrated annotations; those calibrated inputs are preserved, not overruled by the separate atlas review. Original-vs-display crosstabs are exported. `broad_lineage` in the plotting table means **calibrated display lineage**, while `atlas_broad_lineage` is the unmodified full-atlas label. `TNK` is displayed as `T_NK` only typographically.

No input has a separate **Plasma** lineage. The requested lineage count table includes **Plasma=0** as an explicit absent category; **B_MZB1 remains B**, exactly as supplied. No plasma reassignment is invented. “Normal epithelial reference” denotes the supplied acinar/normal-dominated reference population, **not necessarily normal-source tissue**: the frozen reference contains 24,526 normal-like-source and 10,371 tumor-source cells.

## Panels and exact counts

### A — complete joint UMAP by calibrated broad lineage

Every atlas cell is plotted: **302,235 colored calibrated cells + 90,153 grey cells = 392,388**. Epithelial combines frozen tumor ductal and normal reference; non-epithelial cells use their supplied lineage. Thus these are calibrated-coverage counts, not a relabeling of the original atlas broad-lineage count table.

| Display lineage | n cells |
|---|---:|
{lineage_md}

### B — complete calibrated annotation

All **392,388** cells: **55,357 tumor ductal cells / 12 frozen clusters**, **34,897 reference cells** in one muted color, **211,981 non-epithelial cells / 54 fine subtypes** aggregated by their supplied six lineages, and **90,153 unresolved/mixed display cells** in grey. The full 54-subtype identities remain in the per-cell and cluster count tables. No reclassification or new subtype merge is performed.

Tumor cluster counts: **{cluster_md}**. Only centroid labels are used. The small ductal inset shows **arithmetic-mean cluster centroids in the unchanged joint coordinate system**, with displaced text/leader lines for readability, not a second UMAP. The full all-cell manifold does not imply that the independently calibrated epithelial clusters form separate islands. Display text may move, never cell coordinates.

### C — **frozen pair only** on the complete joint UMAP

Only **Epi_JUN-like (c7), 3,245 cells**, warm `{WARM}`, and **Epi_VIM-like (c2), 5,227 cells**, cool `{COOL}`, are highlighted. **All other 383,916 cells** are very light grey, including other epithelial clusters, normal reference, all non-epithelial cells, and all unresolved/mixed profiles. There are no pooled arms, substituted cluster labels, new selections, expression scores, or expanded pole memberships. **c2 is the frozen negative reference pole, not a canonical EMT/mesenchymal identity claim.** Marker text is descriptive frozen-pair context, not a newly computed expression statistic.

The two equal-scale small zooms are centered on each state's coordinate-wise median (**±4.5 UMAP1, ±3.375 UMAP2**). Each zoom highlights **only its own frozen state**; all other local cells are grey. Clipped zoom windows do not remove any cells from the main full-atlas plot. Exact per-inset visible/background counts and limits are in `PDAC_joint_atlas_frozen_pair_zoom_windows.csv`.

### D — CM1-like subtype context, not a per-cell CM score

All **392,388** cells: the five non-epithelial subtypes with highest **saved calCM02 H loadings**, **63,280 cells**, in a purple ramp; **329,108 other cells** in very light grey. The module and rank are frozen: **calCM02, K=11**. The saved T/NK H fraction is **{summary['tnk_H_fraction']:.17g} = 68.609030932%**, displayed **68.61%**. This is the **share of total module H loading**, not a T-cell count percentage or per-cell activity estimate. Ranking/color mapping is display only; no module is refit.

| Highlighted subtype | Saved raw H | Share of total H | Cells |
|---|---:|---:|---:|
{cm_md}

All five happen to be T subtypes. They carry **{100 * cm.head(5).H_fraction.sum():.9f}%** of total calCM02 H; the **68.61%** T/NK share additionally includes other supplied T/NK subtypes. Exact full-precision values for all 54 subtypes are exported from `cm_selected_H.csv` / `cm_selected_H_fraction.csv`, with lineage composition checked against `cm_selected_lineage_composition.csv`.

## Source data and reproduction

In `source_data/`, all newly delivered joint-atlas files start **`PDAC_joint_atlas_`**:

- `per_cell_plot_data.csv.gz`: **392,388 rows**, cell ID, round-trip-precision UMAP1/2, calibrated display broad lineage/label, frozen `is_JUN_pole` / `is_VIM_pole`, population/coverage flags, unchanged epithelial cluster/non-epithelial subtype/reference identity, original atlas labels, sample/study/donor/status, and panel grouping flags.
- `lineage_counts.csv`: complete panel A count table including explicitly absent Plasma=0.
- `cluster_counts.csv`: all **68** labels: 12 ductal clusters + one reference + 54 non-epithelial subtypes + the unresolved display population; counts, support and mean/median coordinates. Unresolved is a display status, not a fitted cluster.
- `subtype_to_CM1_loading.csv`: all **54** unchanged subtypes, original and display lineages, exact saved raw H and H fractions, counts, rank, highlighted flag and purple colors.
- `archived_lineage_counts.csv`, `unresolved_display_original_lineages.csv`, `original_vs_display_lineage_crosstab.csv`: explicit original-label/coverage distinctions.
- `centroid_label_positions.csv`, `frozen_pair_zoom_windows.csv`: display-only annotation/zoom coordinates.
- `plot_summary.json` and `../PDAC_joint_atlas_render_metadata.json`: provenance and panel/render counts.

```bash
{ROOT.parent}/epi-cm-core-workflow/uv_envs/main/.venv/bin/python {OUT}/render_pdac_joint_atlas_fig7style.py
```

Use `--render-only` for layout-only rerendering from the exported tables. `--preview` writes a low-DPI draft to `/tmp/opencode/PDAC_joint_atlas_UMAP_layout_preview.png`; the delivered files always use 450 DPI. Rendered PNG inspection is recorded in `PDAC_joint_atlas_visual_inspection.json` after opening the actual images.
"""
    (OUT / "README_JOINT_UMAP.md").write_text(notes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    font = setup_style()
    data = load_data() if args.render_only else prepare_data()
    fig = plt.figure(figsize=(14.2, 13.3))
    text(fig, 0.35, 0.18, "PDAC | one joint atlas, calibrated epithelial and non-epithelial states", fontsize=15.0, fontweight="bold")
    text(fig, 0.35, 0.52, "392,388 cells · 125 samples · saved all-cell PCA / sample Harmony / UMAP · no reintegration or downsampling",
         fontsize=9.0, color=MUTED)
    a_labels = panel_a(fig, REGIONS[0], *data)
    b_labels = panel_b(fig, REGIONS[1], *data)
    zooms = panel_c(fig, REGIONS[2], *data)
    d_labels = panel_d(fig, REGIONS[3], *data)
    text(fig, 0.35, 12.97, "Saved all-cell coordinates; calibrated labels unchanged. Grey coverage status never supplies a new cluster assignment. Frozen c7/c2 and calCM02 are visualization overlays only.",
         fontsize=7.3, color=MUTED)
    if args.preview:
        path = Path("/tmp/opencode/PDAC_joint_atlas_UMAP_layout_preview.png")
        fig.savefig(path, dpi=170, facecolor="white")
        plt.close(fig)
        print(f"Preview: {path}", flush=True)
        return
    outputs = {"combined": save(fig, "PDAC_joint_atlas_UMAP_combined")}
    for letter, suffix, fn in [
        ("A", "broad_lineages", panel_a), ("B", "calibrated_annotation", panel_b),
        ("C", "frozen_pair", panel_c), ("D", "CM1_context", panel_d),
    ]:
        fig = plt.figure(figsize=(7.10, 6.40))
        fn(fig, (0.27, 0.22, 6.55, 5.50), *data)
        text(fig, 0.56, 6.04, "PDAC · 392,388 cells · unchanged joint all-cell UMAP · all cells plotted", fontsize=7.0, color=MUTED)
        outputs[letter] = save(fig, f"PDAC_joint_atlas_UMAP_panel_{letter}_{suffix}")
    pd.DataFrame(a_labels + b_labels + d_labels).to_csv(DATA / "PDAC_joint_atlas_centroid_label_positions.csv", index=False, float_format="%.17g")
    pd.DataFrame(zooms).to_csv(DATA / "PDAC_joint_atlas_frozen_pair_zoom_windows.csv", index=False, float_format="%.17g")
    write_notes(data, font, outputs, zooms)
    print(json.dumps({"output_directory": str(OUT), "n_cells": len(data[0]), "font": font,
                      "lineage_counts": data[1].n_cells.to_dict(), "panel_D_highlighted": 63280}, indent=2), flush=True)


if __name__ == "__main__":
    main()

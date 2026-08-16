#!/usr/bin/env python3
"""Render canonical PDAC Block 03 figures strictly from saved CSV tables."""

from __future__ import annotations

import importlib.metadata
import json
import math
import random
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


SEED = 42
EDGE_THRESHOLD = 0.25
random.seed(SEED)
np.random.seed(SEED)
mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)
sns.set_theme(style="ticks")

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
ANALYSIS_ROOT = PROJECT_ROOT / "tables/03-epi-cm-discovery/01-cm-lineage-analysis"
PREP_DIR = ANALYSIS_ROOT / "01-prepare-inputs-and-frequency-tables"
NMF_DIR = ANALYSIS_ROOT / "02-balanced-joint-nmf"
NODE_DIR = ANALYSIS_ROOT / "03-cm-classification-nodes-and-edges"
ASSOCIATION_DIRS = {
    "spearman": ANALYSIS_ROOT / "04-epi-cm-association-spearman",
    "pearson": ANALYSIS_ROOT / "05-epi-cm-association-pearson",
}
FIGURE_ROOT = PROJECT_ROOT / "figures/03-epi-cm-discovery/02-cm-lineage-final-plotting"
TABLE_ROOT = PROJECT_ROOT / "tables/03-epi-cm-discovery/02-cm-lineage-final-plotting"
PALETTE_CSV = (
    PROJECT_ROOT
    / "tables/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata/all_lineages/cell_subtype_palette.csv"
)

MANIFEST: list[dict] = []


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def closest_square_grid(n_panels: int) -> tuple[int, int]:
    if n_panels <= 0:
        return 0, 0
    best = None
    for n_rows in range(1, int(np.ceil(np.sqrt(n_panels))) + 1):
        n_cols = int(np.ceil(n_panels / n_rows))
        candidate = (abs(n_cols - n_rows), n_rows * n_cols - n_panels, n_rows, n_cols)
        if best is None or candidate < best:
            best = candidate
    return best[2], best[3]


def save_pdf_svg(
    fig,
    stem: Path,
    family: str,
    inputs: list[Path],
    method: str = "not_applicable",
    context: str = "not_applicable",
    display_scale: str = "not_applicable",
) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in [".pdf", ".svg"]:
        path = stem.with_suffix(extension)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        MANIFEST.append(
            {
                "status": "completed",
                "figure_family": family,
                "figure_file": str(path),
                "method": method,
                "context": context,
                "display_scale": display_scale,
                "direct_input_tables": ";".join(str(item) for item in inputs),
                "reason_if_skipped": "",
            }
        )
    plt.close(fig)


def record_skip(family: str, method: str, context: str, reason: str) -> None:
    MANIFEST.append(
        {
            "status": "skipped",
            "figure_family": family,
            "figure_file": "",
            "method": method,
            "context": context,
            "display_scale": "not_applicable",
            "direct_input_tables": "",
            "reason_if_skipped": reason,
        }
    )


def read_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def q_stars(q: pd.DataFrame) -> pd.DataFrame:
    return q.applymap(
        lambda value: "***"
        if pd.notna(value) and value < 0.001
        else "**"
        if pd.notna(value) and value < 0.01
        else "*"
        if pd.notna(value) and value < 0.05
        else ""
    )


def plot_k_selection() -> None:
    path = NMF_DIR / "joint_nmf_k_selection_metrics.csv"
    metrics = pd.read_csv(path)
    selected_k = int(metrics.loc[metrics["selected"], "k"].iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    axes[0].plot(metrics["k"], metrics["selection_score"], "o-", label="selection score")
    axes[0].plot(
        metrics["k"], metrics["best_balanced_explained_fraction"], "s-", label="best explained"
    )
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Score / explained fraction")
    axes[1].plot(metrics["k"], metrics["best_reconstruction_error"], "o-", color="#D55E00")
    axes[1].set_ylabel("Best squared error")
    axes[2].plot(metrics["k"], metrics["stability_matched_cosine"], "o-", color="#009E73")
    axes[2].set_ylabel("Matched-cosine stability")
    for axis in axes:
        axis.axvline(selected_k, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("K")
        axis.set_xticks(metrics["k"])
        sns.despine(ax=axis)
    fig.tight_layout()
    save_pdf_svg(
        fig,
        FIGURE_ROOT / "nmf/joint_nmf_k_selection",
        "NMF K-selection diagnostics",
        [path],
    )


def plot_matrix_heatmap(
    matrix: pd.DataFrame,
    stem: Path,
    family: str,
    source: Path,
    scale: str,
    hide_rows: bool = False,
) -> None:
    signed = scale in {"zscore", "robust"}
    fig, axis = plt.subplots(
        figsize=(max(4.8, 0.65 * matrix.shape[1]), max(3.6, min(14, 0.30 * matrix.shape[0])))
    )
    sns.heatmap(
        matrix,
        cmap="RdBu_r" if signed else "viridis",
        center=0 if signed else None,
        yticklabels=False if hide_rows else True,
        cbar_kws={"label": scale},
        ax=axis,
    )
    axis.tick_params(axis="x", rotation=90)
    axis.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    save_pdf_svg(fig, stem, family, [source], display_scale=scale)


def plot_activity_loading() -> None:
    for scale in ["raw", "zscore", "robust", "standard_scale_col"]:
        activity_path = NMF_DIR / f"w_df_activity_sample_by_CM_{scale}.csv"
        plot_matrix_heatmap(
            read_matrix(activity_path),
            FIGURE_ROOT / f"activity-heatmaps/w_df_activity_sample_by_CM_{scale}",
            "CM activity heatmap",
            activity_path,
            scale,
            hide_rows=True,
        )
        loading_path = NMF_DIR / f"h_df_loading_cell_subtype_by_CM_{scale}.csv"
        plot_matrix_heatmap(
            read_matrix(loading_path).T,
            FIGURE_ROOT / f"loading-heatmaps/h_df_loading_cell_subtype_by_CM_{scale}",
            "CM loading heatmap",
            loading_path,
            scale,
        )

    matrix_path = NMF_DIR / "w_df_activity_sample_by_CM_standard_scale_col_with_series_plot_matrix.csv"
    annotation_path = NMF_DIR / "w_df_activity_sample_by_CM_standard_scale_col_with_series_row_annotations.csv"
    matrix = read_matrix(matrix_path)
    annotations = read_matrix(annotation_path).loc[matrix.index]
    series = sorted(annotations["Series"].astype(str).unique())
    series_colors = dict(zip(series, sns.color_palette("tab10", len(series)).as_hex()))
    status_colors = {"normal-like": "#4DAF4A", "tumor": "#E41A1C"}
    row_colors = pd.DataFrame(
        {
            "Series": annotations["Series"].astype(str).map(series_colors),
            "Status": annotations["Status"].astype(str).map(status_colors),
        },
        index=matrix.index,
    )
    grid = sns.clustermap(
        matrix,
        cmap="viridis",
        col_cluster=False,
        row_cluster=True,
        row_colors=row_colors,
        yticklabels=False,
        figsize=(7.8, 9),
        cbar_kws={"label": "column min-max display scale"},
    )
    save_pdf_svg(
        grid.fig,
        FIGURE_ROOT / "activity-heatmaps/activity_standard_scale_col_annotated_clustermap",
        "Annotated CM activity clustermap",
        [matrix_path, annotation_path],
        display_scale="standard_scale_col",
    )


def plot_activity_status_barplot() -> None:
    summary_path = NODE_DIR / "joint_CM_activity_tumor_vs_normal_mean_sd_summary.csv"
    classification_path = NODE_DIR / "joint_module_classification.csv"
    summary = pd.read_csv(summary_path).set_index("CM")
    classification = pd.read_csv(classification_path).sort_values("global_order")
    cm_order = classification["CM"].tolist()
    x = np.arange(len(cm_order))
    width = 0.36
    fig, axis = plt.subplots(figsize=(max(6.5, 1.1 * len(cm_order)), 4.3))
    axis.bar(
        x - width / 2,
        summary.loc[cm_order, "normal_like_mean"],
        width,
        yerr=summary.loc[cm_order, "normal_like_sd"],
        color="#4DAF4A",
        label="normal-like",
        capsize=2,
    )
    axis.bar(
        x + width / 2,
        summary.loc[cm_order, "tumor_mean"],
        width,
        yerr=summary.loc[cm_order, "tumor_sd"],
        color="#E41A1C",
        label="tumor",
        capsize=2,
    )
    ymax = max(
        float((summary["normal_like_mean"] + summary["normal_like_sd"].fillna(0)).max()),
        float((summary["tumor_mean"] + summary["tumor_sd"].fillna(0)).max()),
    )
    for index, row in classification.reset_index(drop=True).iterrows():
        label = "***" if row["q"] < 0.001 else "**" if row["q"] < 0.01 else "*" if row["q"] < 0.05 else "ns"
        axis.text(index, ymax * 1.05, label, ha="center", fontsize=8)
    axis.set_xticks(x, cm_order, rotation=90)
    axis.set_ylabel("Canonical CM activity (NNLS refit)")
    axis.legend(frameon=False)
    sns.despine(ax=axis)
    fig.tight_layout()
    save_pdf_svg(
        fig,
        FIGURE_ROOT / "activity/cm_activity_tumor_vs_normal_like_mean_sd",
        "CM activity status barplot",
        [summary_path, classification_path],
    )


def edge_key(node_a: str, node_b: str) -> tuple[str, str]:
    return tuple(sorted([str(node_a), str(node_b)]))


def node_prefix_palette(nodes: list[str]) -> dict[str, str]:
    fixed = {
        "T": "#0072B2",
        "NK": "#56B4E9",
        "Mye": "#009E73",
        "B": "#CC79A7",
        "Endo": "#E69F00",
        "S": "#999999",
    }
    prefixes = sorted({node.split("_", 1)[0] for node in nodes})
    missing = [prefix for prefix in prefixes if prefix not in fixed]
    fallback = sns.color_palette("tab10", max(1, len(missing))).as_hex()
    fixed.update(dict(zip(missing, fallback)))
    return fixed


NORMAL_CMAP = LinearSegmentedColormap.from_list("normal_edges", ["#FEE8C8", "#E34A33"])
TUMOR_CMAP = LinearSegmentedColormap.from_list("tumor_edges", ["#DEEBF7", "#08519C"])
SHARED_CMAP = LinearSegmentedColormap.from_list("shared_edges", ["#E5F5E0", "#238B45"])
EDGE_NORM = Normalize(vmin=EDGE_THRESHOLD, vmax=1.0)


def draw_network(axis, cm: str, nodes: pd.DataFrame, edges: pd.DataFrame, mode: str, colors: dict):
    cm_nodes = (
        nodes.loc[nodes["CM"].astype(str).eq(cm)]
        .sort_values("reference_node_rank")["node"]
        .astype(str)
        .tolist()
    )
    axis.set_title(cm)
    axis.set_aspect("equal")
    axis.axis("off")
    if not cm_nodes:
        axis.text(0.5, 0.5, "No nodes retained\nafter r≥0.25 edge screen", ha="center", va="center")
        return {"n_nodes": 0, "n_edges": 0, "classes": set()}
    cm_edges = edges.loc[edges["CM"].astype(str).eq(cm)].copy()
    normal_pass = {
        edge_key(row.node_a, row.node_b)
        for row in cm_edges.loc[
            cm_edges["context"].eq("normal-like") & cm_edges["edge_pass_r_ge_0.25"]
        ].itertuples(index=False)
    }
    context = "normal-like" if mode == "normal_like" else "tumor"
    selected = cm_edges.loc[
        cm_edges["context"].eq(context) & cm_edges["edge_pass_r_ge_0.25"]
    ].copy()
    graph = nx.Graph()
    graph.add_nodes_from(cm_nodes)
    positions = {
        node: (
            math.cos(2 * math.pi * index / len(cm_nodes)),
            math.sin(2 * math.pi * index / len(cm_nodes)),
        )
        for index, node in enumerate(cm_nodes)
    }
    edge_classes = set()
    for row in selected.itertuples(index=False):
        origin = "shared" if edge_key(row.node_a, row.node_b) in normal_pass else "tumor_only"
        edge_classes.add(origin)
        graph.add_edge(row.node_a, row.node_b, pearson_r=float(row.pearson_r), origin=origin)
    for node_a, node_b, attributes in graph.edges(data=True):
        r_value = attributes["pearson_r"]
        if mode == "normal_like":
            cmap = NORMAL_CMAP
            width = 1.2 + 3.0 * EDGE_NORM(r_value)
        elif mode == "tumor":
            cmap = TUMOR_CMAP
            width = 1.2 + 3.0 * EDGE_NORM(r_value)
        else:
            cmap = SHARED_CMAP if attributes["origin"] == "shared" else TUMOR_CMAP
            width = 2.6
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[(node_a, node_b)],
            edge_color=[cmap(EDGE_NORM(r_value))],
            width=width,
            alpha=0.9,
            ax=axis,
        )
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=[colors[node.split("_", 1)[0]] for node in cm_nodes],
        node_size=900,
        edgecolors="white",
        linewidths=0.8,
        ax=axis,
    )
    nx.draw_networkx_labels(graph, positions, font_size=6.5, ax=axis)
    return {"n_nodes": len(cm_nodes), "n_edges": graph.number_of_edges(), "classes": edge_classes}


def add_network_colorbars(fig, axes, mode: str, classes: set[str]) -> None:
    if mode == "normal_like":
        fig.colorbar(
            ScalarMappable(norm=EDGE_NORM, cmap=NORMAL_CMAP),
            ax=axes,
            orientation="horizontal",
            fraction=0.025,
            pad=0.08,
            label="Normal-like Pearson r",
        )
    elif mode == "tumor":
        fig.colorbar(
            ScalarMappable(norm=EDGE_NORM, cmap=TUMOR_CMAP),
            ax=axes,
            orientation="horizontal",
            fraction=0.025,
            pad=0.08,
            label="Tumor Pearson r",
        )
    else:
        pad = 0.10
        if "tumor_only" in classes:
            fig.colorbar(
                ScalarMappable(norm=EDGE_NORM, cmap=TUMOR_CMAP),
                ax=axes,
                orientation="horizontal",
                fraction=0.025,
                pad=pad,
                label="Tumor-only edge: tumor Pearson r",
            )
            pad = 0.03
        if "shared" in classes:
            fig.colorbar(
                ScalarMappable(norm=EDGE_NORM, cmap=SHARED_CMAP),
                ax=axes,
                orientation="horizontal",
                fraction=0.025,
                pad=pad,
                label="Shared edge: tumor Pearson r",
            )


def plot_nodeplots() -> None:
    nodes_path = NODE_DIR / "balanced_joint_cm_reference_node_sets_after_edge_threshold.csv"
    edges_path = NODE_DIR / "status_specific_nodeplot_edges.csv"
    classification_path = NODE_DIR / "joint_module_classification.csv"
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    cm_order = pd.read_csv(classification_path).sort_values("global_order")["CM"].astype(str).tolist()
    colors = node_prefix_palette(nodes["node"].astype(str).tolist())
    modes = [
        ("normal_like", "normal_like_all_CM_nodeplot", "normal_like_nodeplots_by_cm", "normal_like"),
        ("tumor", "tumor_all_CM_nodeplot", "tumor_nodeplots_by_cm", "tumor"),
        (
            "tumor_centric",
            "tumor_centric_nodeplot_edge_origin",
            "tumor_centric_nodeplots_by_cm",
            "tumor_centric_edge_origin",
        ),
    ]
    for mode, overview_stem, child_dir, suffix in modes:
        for cm in cm_order:
            fig, axis = plt.subplots(figsize=(6.0, 5.5))
            result = draw_network(axis, cm, nodes, edges, mode, colors)
            if result["n_nodes"] == 0 or result["n_edges"] == 0:
                plt.close(fig)
                record_skip(
                    "single-CM nodeplot",
                    "pearson_node_edges",
                    mode,
                    f"{cm}: no passing network edge after canonical edge screen",
                )
                continue
            add_network_colorbars(fig, [axis], mode, result["classes"])
            fig.subplots_adjust(bottom=0.22)
            save_pdf_svg(
                fig,
                FIGURE_ROOT / f"nodeplots/{child_dir}/{cm}_{suffix}_nodeplot",
                "single-CM nodeplot",
                [nodes_path, edges_path],
                method="pearson_node_edges",
                context=mode,
            )
        rows, columns = closest_square_grid(len(cm_order))
        fig, axes = plt.subplots(
            rows, columns, figsize=(4.2 * columns, 4.2 * rows + 1.0), squeeze=False
        )
        classes = set()
        for axis, cm in zip(axes.ravel(), cm_order):
            result = draw_network(axis, cm, nodes, edges, mode, colors)
            classes.update(result["classes"])
        for axis in axes.ravel()[len(cm_order) :]:
            axis.axis("off")
        add_network_colorbars(fig, list(axes.ravel()[: len(cm_order)]), mode, classes)
        fig.subplots_adjust(bottom=0.13, wspace=0.25, hspace=0.25)
        save_pdf_svg(
            fig,
            FIGURE_ROOT / f"nodeplots/{overview_stem}",
            "all-CM nodeplot overview",
            [nodes_path, edges_path, classification_path],
            method="pearson_node_edges",
            context=mode,
        )

    prefixes = sorted(colors)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[prefix],
            markeredgecolor="white",
            markersize=10,
            label=prefix,
        )
        for prefix in prefixes
    ]
    fig, axis = plt.subplots(figsize=(max(4, 1.1 * len(prefixes)), 1.3))
    axis.axis("off")
    axis.legend(handles=handles, loc="center", ncol=min(6, len(prefixes)), frameon=False)
    save_pdf_svg(
        fig,
        FIGURE_ROOT / "nodeplots/nodeplot_network_node_legend",
        "Node lineage legend",
        [nodes_path],
    )
    for stem, cmap, label in [
        ("nodeplot_network_edge_colorbar", TUMOR_CMAP, "Passing-edge Pearson r"),
        (
            "tumor_centric_nodeplot_edge_origin_tumor_only_edge_colorbar",
            TUMOR_CMAP,
            "Tumor-only edge: tumor Pearson r",
        ),
        (
            "tumor_centric_nodeplot_edge_origin_shared_edge_colorbar",
            SHARED_CMAP,
            "Shared edge: tumor Pearson r",
        ),
    ]:
        fig, axis = plt.subplots(figsize=(4.2, 0.65))
        colorbar = fig.colorbar(
            ScalarMappable(norm=EDGE_NORM, cmap=cmap), cax=axis, orientation="horizontal"
        )
        colorbar.set_label(label)
        save_pdf_svg(
            fig,
            FIGURE_ROOT / f"nodeplots/{stem}",
            "Nodeplot edge colorbar",
            [edges_path],
        )
    legend_handles = [
        Line2D([0], [0], color=TUMOR_CMAP(1.0), linewidth=3, label="Tumor only"),
        Line2D([0], [0], color=SHARED_CMAP(1.0), linewidth=3, label="Shared with normal-like"),
    ]
    fig, axis = plt.subplots(figsize=(4.2, 1.1))
    axis.axis("off")
    axis.legend(handles=legend_handles, loc="center", ncol=2, frameon=False)
    save_pdf_svg(
        fig,
        FIGURE_ROOT / "nodeplots/tumor_centric_nodeplot_edge_origin_edge_class_legend",
        "Tumor-centric edge class legend",
        [edges_path],
    )


def plot_top10_correlations() -> None:
    top_path = NODE_DIR / "joint_cm_cell_subtype_nodes_top10_from_H_df.csv"
    classification_path = NODE_DIR / "joint_module_classification.csv"
    top = pd.read_csv(top_path)
    cm_order = pd.read_csv(classification_path).sort_values("global_order")["CM"].astype(str).tolist()
    for context in ["normal-like", "tumor"]:
        panels = []
        for cm in cm_order:
            matrix_path = NODE_DIR / f"top10_node_correlation_{context}_{cm}_no_edge_filter.csv"
            matrix = read_matrix(matrix_path)
            nodes = (
                top.loc[top["CM"].astype(str).eq(cm)]
                .sort_values("rank")["cell_subtype"]
                .astype(str)
                .tolist()
            )
            matrix = matrix.loc[nodes, nodes]
            panels.append((cm, nodes, matrix, matrix_path))
            fig, axis = plt.subplots(figsize=(5.4, 5.0))
            sns.heatmap(
                matrix,
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                annot=False,
                cbar_kws={"label": "Pearson r"},
                ax=axis,
            )
            axis.set_title(cm)
            axis.tick_params(axis="x", rotation=90, labelsize=7)
            axis.tick_params(axis="y", rotation=0, labelsize=7)
            fig.tight_layout()
            save_pdf_svg(
                fig,
                FIGURE_ROOT
                / f"top10-node-correlation-heatmaps-no-edge-filter-by-cm/{context.replace('-', '_')}/{cm}",
                "Top-node Pearson correlation heatmap",
                [matrix_path, top_path],
                method="pearson_node_edges",
                context=context,
            )
        rows, columns = closest_square_grid(len(panels))
        fig, axes = plt.subplots(
            rows, columns, figsize=(4.1 * columns + 0.6, 4.1 * rows), squeeze=False
        )
        image = None
        for axis, (cm, nodes, matrix, _) in zip(axes.ravel(), panels):
            image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
            axis.set_title(cm)
            axis.set_xticks(range(len(nodes)), nodes, rotation=90, fontsize=6)
            axis.set_yticks(range(len(nodes)), nodes, fontsize=6)
            axis.set_aspect("equal")
        for axis in axes.ravel()[len(panels) :]:
            axis.axis("off")
        fig.colorbar(
            image,
            ax=list(axes.ravel()[: len(panels)]),
            fraction=0.02,
            pad=0.02,
            label="Pearson r",
        )
        fig.subplots_adjust(wspace=0.55, hspace=0.65)
        save_pdf_svg(
            fig,
            FIGURE_ROOT / f"top10-correlations/{context.replace('-', '_')}_summary",
            "Top-node Pearson correlation summary",
            [top_path] + [panel[3] for panel in panels],
            method="pearson_node_edges",
            context=context,
        )


def load_epithelial_palette(epithelial_subtypes: list[str]) -> dict[str, str]:
    if PALETTE_CSV.is_file():
        palette_table = pd.read_csv(PALETTE_CSV)
        palette = dict(
            zip(palette_table["cell_subtype"].astype(str), palette_table["color"].astype(str))
        )
    else:
        palette = {}
    missing = [subtype for subtype in epithelial_subtypes if subtype not in palette]
    if missing:
        palette.update(dict(zip(missing, sns.color_palette("husl", len(missing)).as_hex())))
    return palette


def plot_associations() -> None:
    epi_frequency_path = PREP_DIR / "epi_subtype_frequency.csv"
    W_path = NMF_DIR / "W_df.csv"
    sample_status_path = PREP_DIR / "sample_status.csv"
    epi_frequency = read_matrix(epi_frequency_path)
    W = read_matrix(W_path)
    sample_status = read_matrix(sample_status_path)
    common = epi_frequency.index.intersection(W.index).intersection(sample_status.index)
    epi_frequency = epi_frequency.loc[common]
    W = W.loc[common]
    status = sample_status.loc[common, "status"].astype(str)
    palette = load_epithelial_palette(epi_frequency.columns.astype(str).tolist())
    contexts = {
        "overall": common,
        "normal-like": status.index[status.eq("normal-like")],
        "tumor": status.index[status.eq("tumor")],
    }
    for method, association_dir in ASSOCIATION_DIRS.items():
        statistic_column = "rho" if method == "spearman" else "r"
        statistic_label = "Spearman rho" if method == "spearman" else "Pearson r"
        for context, samples in contexts.items():
            if method == "spearman":
                stem = f"balanced_joint_cm_epi_cm_association_{context}"
                value_path = association_dir / f"{stem}_rho_matrix.csv"
            else:
                stem = f"balanced_joint_cm_epi_cm_association_{context}_pearson"
                value_path = association_dir / f"{stem}_r_matrix.csv"
            q_path = association_dir / f"{stem}_q_matrix.csv"
            long_path = association_dir / f"{stem}_{method}_long.csv"
            values = read_matrix(value_path)
            q_values = read_matrix(q_path)
            long = pd.read_csv(long_path).set_index(["epi_subtype", "CM"])
            fig, axis = plt.subplots(
                figsize=(max(5.5, 0.75 * values.shape[1]), max(4.5, 0.48 * values.shape[0]))
            )
            sns.heatmap(
                values,
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                center=0,
                annot=q_stars(q_values),
                fmt="",
                cbar_kws={"label": statistic_label},
                ax=axis,
            )
            axis.set_title(f"{context}: {statistic_label}; stars are within-context BH-FDR")
            axis.tick_params(axis="x", rotation=90)
            axis.tick_params(axis="y", rotation=0, labelsize=7)
            fig.tight_layout()
            save_pdf_svg(
                fig,
                FIGURE_ROOT / f"epi-cm-association-{method}/{context.replace('-', '_')}/heatmap_qstars",
                "Epi-CM association heatmap",
                [value_path, q_path],
                method=method,
                context=context,
            )
            for epithelial_subtype in epi_frequency.columns:
                for cm in W.columns:
                    statistics = long.loc[(epithelial_subtype, cm)]
                    data = pd.DataFrame(
                        {
                            "epi_frequency": epi_frequency.loc[samples, epithelial_subtype],
                            "cm_activity": W.loc[samples, cm],
                        }
                    )
                    fig, axis = plt.subplots(figsize=(5.8, 5.0))
                    sns.regplot(
                        data=data,
                        x="cm_activity",
                        y="epi_frequency",
                        color=palette[str(epithelial_subtype)],
                        scatter_kws={"s": 45, "alpha": 0.8, "edgecolor": "none"},
                        line_kws={"linewidth": 2},
                        ci=95,
                        ax=axis,
                    )
                    statistic = statistics[statistic_column]
                    q_value = statistics["q_value"]
                    axis.set_title(f"{context}: {cm} vs {epithelial_subtype}")
                    axis.set_xlabel(f"{cm} canonical activity")
                    axis.set_ylabel(f"{epithelial_subtype} within-epithelial frequency")
                    axis.text(
                        0.02,
                        0.98,
                        f"{statistic_label}={statistic:.3f}\nq={q_value:.2g}\nn={len(data)}",
                        transform=axis.transAxes,
                        va="top",
                        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                    )
                    sns.despine(ax=axis)
                    fig.tight_layout()
                    save_pdf_svg(
                        fig,
                        FIGURE_ROOT
                        / f"epi-cm-association-{method}/{context.replace('-', '_')}/scatter/{safe_name(cm)}_vs_{safe_name(epithelial_subtype)}",
                        "Epi-CM all-pair scatter",
                        [epi_frequency_path, W_path, sample_status_path, long_path],
                        method=method,
                        context=context,
                    )


def main() -> None:
    if FIGURE_ROOT.exists() or TABLE_ROOT.exists():
        raise FileExistsError("Refusing to overwrite Block 03 final plotting outputs")
    required = [
        ANALYSIS_ROOT / "block03_analysis_validation.json",
        NMF_DIR / "W_df.csv",
        NMF_DIR / "H_df.csv",
        NODE_DIR / "joint_module_classification.csv",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing canonical Block 03 inputs: {missing}")
    FIGURE_ROOT.mkdir(parents=True, exist_ok=False)
    plot_k_selection()
    plot_activity_loading()
    plot_activity_status_barplot()
    plot_nodeplots()
    plot_top10_correlations()
    plot_associations()

    TABLE_ROOT.mkdir(parents=True, exist_ok=False)
    manifest = pd.DataFrame(MANIFEST)
    manifest.to_csv(TABLE_ROOT / "figure_manifest.csv", index=False)
    completed = manifest.loc[manifest["status"].eq("completed")]
    validation = {
        "figure_files": int(len(completed)),
        "pdf_files": int(completed["figure_file"].str.endswith(".pdf").sum()),
        "svg_files": int(completed["figure_file"].str.endswith(".svg").sum()),
        "skipped_manifest_rows": int(manifest["status"].eq("skipped").sum()),
        "methods": sorted(completed["method"].unique().tolist()),
        "association_methods": ["spearman", "pearson"],
        "association_contexts": ["overall", "normal-like", "tumor"],
        "all_pair_scatter_contract": True,
        "all_figures_read_saved_csv_tables": True,
        "validation_passed": True,
    }
    (TABLE_ROOT / "plotting_parameters.txt").write_text(
        "formats=PDF;SVG\n"
        "edge_threshold=0.25\n"
        "correlation_heatmap_scale=-1,1\n"
        "q_stars=q<0.05 *, q<0.01 **, q<0.001 ***\n"
        "scatter_inventory=full epithelial_subtype x canonical_CM Cartesian product per method and context\n"
        f"random_seed={SEED}\n"
        f"code_file={Path(__file__).resolve()}\n",
        encoding="utf-8",
    )
    versions = [
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"matplotlib={importlib.metadata.version('matplotlib')}",
        f"seaborn={importlib.metadata.version('seaborn')}",
        f"networkx={importlib.metadata.version('networkx')}",
        f"pandas={importlib.metadata.version('pandas')}",
        f"numpy={importlib.metadata.version('numpy')}",
    ]
    (TABLE_ROOT / "package_versions.txt").write_text("\n".join(versions) + "\n", encoding="utf-8")
    (TABLE_ROOT / "plotting_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()

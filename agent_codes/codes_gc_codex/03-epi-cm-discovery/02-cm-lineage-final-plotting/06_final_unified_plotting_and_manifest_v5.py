#!/usr/bin/env python3
"""Single canonical plotting route for the current GC CM analysis."""

from __future__ import annotations

import json
import math
import os
import platform
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
})
sns.set_theme(style="ticks")

CODE_DIR = Path(__file__).resolve().parent
WF = CODE_DIR.parents[2]
BLOCK = "03-epi-cm-discovery"
ANALYSIS_TAG = "agent-led-current-selection-v5-cm-analysis-r1"
PLOT_TAG = "agent-led-current-selection-v5-cm-plotting-r2"
NMF_TAG = "agent-led-current-selection-v5-nmf-r1"
DISPLAY_TAG = "agent-led-current-selection-v5-cm-analysis-display-r2"
SEED = 42
EDGE_THRESHOLD = 0.25
NMF = WF / "tables" / BLOCK / "01-cm-lineage-analysis" / "02_balanced_joint_nmf" / NMF_TAG
NODES = WF / "tables" / BLOCK / "01-cm-lineage-analysis" / "03_cm_classification_nodes_and_edges" / ANALYSIS_TAG
ASSOC = WF / "tables" / BLOCK / "01-cm-lineage-analysis" / "04_epi_cm_association_spearman" / ANALYSIS_TAG
DISPLAY = WF / "tables" / BLOCK / "01-cm-lineage-analysis" / "05_prepare_display_matrices" / DISPLAY_TAG
PALETTE = WF / "tables" / "02-cell_subtype_integration_clustering" / "06-project-subtypes-to-full-adata" / "agent-led-current-selection-v5" / "subtype_palette.csv"
FIG = WF / "figures" / BLOCK / "02-cm-lineage-final-plotting" / PLOT_TAG
TABLE = WF / "tables" / BLOCK / "02-cm-lineage-final-plotting" / PLOT_TAG
MANIFEST_ROWS = []


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def save_pdf_svg(fig, stem: Path, family: str, inputs: list[Path], scale: str = "not_applicable"):
    stem.parent.mkdir(parents=True, exist_ok=True)
    files = [stem.with_suffix(".pdf"), stem.with_suffix(".svg")]
    for path in files:
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        MANIFEST_ROWS.append({
            "figure_family": family, "figure_file": str(path),
            "direct_input_tables": ";".join(map(str, inputs)),
            "plotting_block": "06_final_unified_plotting_and_manifest_v5.py",
            "display_scale": scale,
        })
    plt.close(fig)
    return files


def load_matrix(path: Path, index_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if index_col not in frame and "index" in frame:
        frame = frame.rename(columns={"index": index_col})
    if index_col not in frame:
        raise RuntimeError(f"{path} lacks {index_col}")
    return frame.set_index(index_col)


def closest_square(n: int):
    best = None
    for rows in range(1, int(np.ceil(np.sqrt(n))) + 1):
        cols = int(np.ceil(n / rows))
        candidate = (abs(cols - rows), rows * cols - n, rows, cols)
        if best is None or candidate < best:
            best = candidate
    return best[2], best[3]


def plot_k_selection():
    path = NMF / "joint_nmf_k_selection_metrics.csv"
    frame = pd.read_csv(path)
    selected_k = int(frame.loc[frame.selected, "k"].iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    axes[0].plot(frame.k, frame.selection_score, marker="o", label="Selection score")
    axes[0].plot(frame.k, frame.best_balanced_explained_fraction, marker="s", label="Best explained fraction")
    axes[0].legend(fontsize=7)
    axes[0].set_ylabel("Score / fraction")
    axes[1].plot(frame.k, frame.best_reconstruction_error, marker="o", color="#d95f02")
    axes[1].set_ylabel("Best squared reconstruction error")
    axes[2].plot(frame.k, frame.stability_matched_cosine, marker="o", color="#1b9e77")
    axes[2].set_ylabel("Matched-cosine stability")
    for ax in axes:
        ax.axvline(selected_k, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("K")
        ax.set_xticks(frame.k)
        ax.tick_params(axis="x", rotation=90)
        sns.despine(ax=ax)
    fig.tight_layout()
    save_pdf_svg(fig, FIG / "nmf" / "joint_nmf_k_selection", "K-selection diagnostics", [path])


def plot_heatmap(frame: pd.DataFrame, stem: Path, family: str, source: Path, scale: str, hide_rows=False):
    signed = scale in {"zscore", "robust"}
    fig, ax = plt.subplots(figsize=(max(5.2, 0.48 * frame.shape[1]), max(3.5, min(15, 0.22 * frame.shape[0]))))
    sns.heatmap(
        frame, cmap="RdBu_r" if signed else "viridis", center=0 if signed else None,
        yticklabels=False if hide_rows else True, xticklabels=True,
        cbar_kws={"label": scale}, ax=ax,
    )
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    save_pdf_svg(fig, stem, family, [source], scale)


def plot_activity_and_loading_heatmaps():
    activity_specs = [
        ("w_df_activity_sample_by_CM_raw.csv", "raw"),
        ("w_df_activity_sample_by_CM_zscore.csv", "zscore"),
        ("w_df_activity_sample_by_CM_robust.csv", "robust"),
        ("w_df_activity_sample_by_CM_standard_scale_col.csv", "standard_scale_col"),
    ]
    for filename, scale in activity_specs:
        path = DISPLAY / filename
        frame = load_matrix(path, "sample")
        plot_heatmap(frame, FIG / "activity_heatmaps" / filename.removesuffix(".csv"), "CM activity heatmap", path, scale, hide_rows=True)
    loading_specs = [
        ("h_df_loading_cell_subtype_by_CM_raw.csv", "raw"),
        ("h_df_loading_cell_subtype_by_CM_zscore.csv", "zscore"),
        ("h_df_loading_cell_subtype_by_CM_robust.csv", "robust"),
        ("h_df_loading_cell_subtype_by_CM_standard_scale_col.csv", "standard_scale_col"),
    ]
    for filename, scale in loading_specs:
        path = DISPLAY / filename
        frame = load_matrix(path, "cell_subtype")
        plot_heatmap(frame, FIG / "loading_heatmaps" / filename.removesuffix(".csv"), "CM loading heatmap", path, scale)

    matrix_path = DISPLAY / "w_df_activity_sample_by_CM_standard_scale_col_with_series_plot_matrix.csv"
    annotation_path = DISPLAY / "w_df_activity_sample_by_CM_standard_scale_col_with_series_row_annotations.csv"
    matrix = load_matrix(matrix_path, "sample")
    annotations = load_matrix(annotation_path, "sample").loc[matrix.index]
    grid = sns.clustermap(matrix, cmap="viridis", col_cluster=False, row_cluster=True, yticklabels=False, figsize=(7, 9), cbar_kws={"label": "standard_scale_col"})
    save_pdf_svg(grid.fig, FIG / "activity_heatmaps" / "w_df_activity_sample_activity_per_CM_standard_scale_col_clustermap", "CM activity clustermap", [matrix_path], "standard_scale_col")
    series_colors = dict(zip(sorted(annotations.Series.unique()), sns.color_palette("tab10", annotations.Series.nunique()).as_hex()))
    status_colors = {"normal-like": "#4daf4a", "tumor": "#e41a1c"}
    row_colors = pd.DataFrame({"Series": annotations.Series.map(series_colors), "Status": annotations.Status.map(status_colors)}, index=matrix.index)
    grid = sns.clustermap(matrix, cmap="viridis", col_cluster=False, row_cluster=True, row_colors=row_colors, yticklabels=False, figsize=(7.6, 9), cbar_kws={"label": "standard_scale_col"})
    save_pdf_svg(grid.fig, FIG / "activity_heatmaps" / "w_df_activity_sample_activity_per_CM_standard_scale_col_with_series_clustermap", "Annotated CM activity clustermap", [matrix_path, annotation_path], "standard_scale_col")


def plot_activity_barplot():
    summary_path = NMF / "joint_CM_activity_tumor_vs_normal_mean_sd_summary.csv"
    class_path = NMF / "joint_module_classification.csv"
    summary = pd.read_csv(summary_path)
    classification = pd.read_csv(class_path).sort_values("global_order")
    cm_order = classification.CM.tolist()
    status_order = ["normal-like", "tumor"]
    x = np.arange(len(cm_order)); width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = {"normal-like": "#4daf4a", "tumor": "#e41a1c"}
    for offset, status in zip((-width / 2, width / 2), status_order):
        sub = summary[summary.status.eq(status)].set_index("CM").reindex(cm_order)
        ax.bar(x + offset, sub["mean"], width, yerr=sub["std"], capsize=2, color=colors[status], label=status)
    stars = classification.q.map(lambda q: "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "ns")
    ymax = summary.groupby("CM").apply(lambda d: float((d["mean"] + d["std"].fillna(0)).max()), include_groups=False).reindex(cm_order)
    for i, (label, y) in enumerate(zip(stars, ymax)):
        if label != "ns":
            ax.text(i, y * 1.05, label, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, cm_order, rotation=90)
    ax.set_ylabel("CM activity (mean ± SD)")
    ax.legend(frameon=False)
    sns.despine(ax=ax); fig.tight_layout()
    save_pdf_svg(fig, FIG / "activity" / "activity_df_tumor_vs_normal_mean_sd_barplot", "CM activity status barplot", [summary_path, class_path])


def network_inputs():
    nodes_path = NODES / "balanced_joint_cm_reference_node_sets_after_edge_threshold.csv"
    edges_path = NODES / "status_specific_nodeplot_edges.csv"
    return pd.read_csv(nodes_path), pd.read_csv(edges_path), nodes_path, edges_path


def draw_cm_network(ax, cm, nodes, edges, mode, prefix_colors):
    cm_nodes = nodes.loc[nodes.CM.eq(cm)].sort_values("reference_node_rank").node.astype(str).tolist()
    cm_edges = edges.loc[edges.CM.eq(cm)].copy()
    normal_pass = {tuple(sorted((r.node_a, r.node_b))) for r in cm_edges.loc[cm_edges.context.eq("normal-like") & cm_edges["edge_pass_r_ge_0.25"]].itertuples()}
    if mode == "normal_like":
        chosen = cm_edges.loc[cm_edges.context.eq("normal-like") & cm_edges["edge_pass_r_ge_0.25"]].copy()
    else:
        chosen = cm_edges.loc[cm_edges.context.eq("tumor") & cm_edges["edge_pass_r_ge_0.25"]].copy()
    graph = nx.Graph(); graph.add_nodes_from(cm_nodes)
    pos = {node: (np.cos(2*np.pi*i/max(1,len(cm_nodes))), np.sin(2*np.pi*i/max(1,len(cm_nodes)))) for i,node in enumerate(cm_nodes)}
    for row in chosen.itertuples():
        origin = "shared" if tuple(sorted((row.node_a, row.node_b))) in normal_pass else "tumor_only"
        graph.add_edge(row.node_a, row.node_b, r=float(row.pearson_r), origin=origin)
    norm = Normalize(EDGE_THRESHOLD, 1.0)
    for a,b,data in graph.edges(data=True):
        if mode == "normal_like": cmap = mpl.colormaps["Reds"]
        elif mode == "tumor": cmap = mpl.colormaps["Blues"]
        else: cmap = mpl.colormaps["Reds"] if data["origin"] == "shared" else mpl.colormaps["Blues"]
        width = 2.6 if mode == "tumor_centric" else 1.4 + 3.2 * norm(data["r"])
        nx.draw_networkx_edges(graph, pos, edgelist=[(a,b)], edge_color=[cmap(norm(data["r"]))], width=width, alpha=.88, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=[prefix_colors[n.split("_",1)[0]] for n in cm_nodes], node_size=850, edgecolors="white", linewidths=.7, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=6.3, ax=ax)
    ax.set_title(cm); ax.set_aspect("equal"); ax.axis("off")
    return chosen


def add_network_bars(fig, axes, mode):
    norm = Normalize(EDGE_THRESHOLD, 1.0)
    if mode == "normal_like":
        fig.colorbar(ScalarMappable(norm=norm, cmap="Reds"), ax=axes, orientation="horizontal", fraction=.025, pad=.06, label="Normal-like Pearson r")
    elif mode == "tumor":
        fig.colorbar(ScalarMappable(norm=norm, cmap="Blues"), ax=axes, orientation="horizontal", fraction=.025, pad=.06, label="Tumor Pearson r")
    else:
        fig.colorbar(ScalarMappable(norm=norm, cmap="Blues"), ax=axes, orientation="horizontal", fraction=.025, pad=.09, label="Tumor-only edge r")
        fig.colorbar(ScalarMappable(norm=norm, cmap="Reds"), ax=axes, orientation="horizontal", fraction=.025, pad=.02, label="Shared edge r in tumor")


def plot_nodeplots():
    nodes, edges, nodes_path, edges_path = network_inputs()
    cm_order = nodes.CM.drop_duplicates().tolist()
    prefixes = sorted({x.split("_",1)[0] for x in nodes.node.astype(str)})
    colors = sns.color_palette("tab10", len(prefixes)).as_hex(); prefix_colors = dict(zip(prefixes, colors))
    for mode, overview, subdir, suffix in [
        ("normal_like", "normal_like_all_CM_nodeplot", "normal_like_nodeplots_by_cm", "_normal_like_nodeplot"),
        ("tumor", "tumor_all_CM_nodeplot", "tumor_nodeplots_by_cm", "_tumor_nodeplot"),
        ("tumor_centric", "tumor_centric_nodeplot_edge_origin", "tumor_centric_nodeplots_by_cm", "_tumor_centric_nodeplot_edge_origin"),
    ]:
        for cm in cm_order:
            fig, ax = plt.subplots(figsize=(5.4 if mode != "tumor_centric" else 6.6, 5.4))
            draw_cm_network(ax, cm, nodes, edges, mode, prefix_colors)
            add_network_bars(fig, [ax], mode)
            fig.subplots_adjust(bottom=.22)
            save_pdf_svg(fig, FIG / "nodeplots" / subdir / f"{cm}{suffix}", f"{mode} nodeplot", [nodes_path, edges_path])
        rows, cols = closest_square(len(cm_order))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4.0, rows*4.0+1), squeeze=False)
        for ax, cm in zip(axes.ravel(), cm_order): draw_cm_network(ax, cm, nodes, edges, mode, prefix_colors)
        for ax in axes.ravel()[len(cm_order):]: ax.axis("off")
        add_network_bars(fig, list(axes.ravel()[:len(cm_order)]), mode)
        fig.subplots_adjust(bottom=.12, wspace=.25, hspace=.25)
        save_pdf_svg(fig, FIG / "nodeplots" / overview, f"{mode} all-CM nodeplot", [nodes_path, edges_path])

    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=prefix_colors[p],markeredgecolor='white',markersize=10,label=p) for p in prefixes]
    fig,ax=plt.subplots(figsize=(max(4,len(prefixes)*1.2),1.3)); ax.axis('off'); ax.legend(handles=handles,loc='center',ncol=min(5,len(prefixes)),frameon=False)
    save_pdf_svg(fig, FIG/"nodeplots"/"nodeplot_network_node_legend", "Node prefix legend", [nodes_path])
    fig,ax=plt.subplots(figsize=(4,.6)); ScalarMappable(norm=Normalize(EDGE_THRESHOLD,1),cmap='viridis'); cb=fig.colorbar(ScalarMappable(norm=Normalize(EDGE_THRESHOLD,1),cmap='viridis'),cax=ax,orientation='horizontal'); cb.set_label('Passing-edge Pearson r')
    save_pdf_svg(fig, FIG/"nodeplots"/"nodeplot_network_edge_colorbar", "Nodeplot edge colorbar", [edges_path])
    for stem,cmap,label in [("tumor_centric_nodeplot_edge_origin_tumor_only_edge_colorbar","Blues","Tumor-only edge r"),("tumor_centric_nodeplot_edge_origin_shared_edge_colorbar","Reds","Shared edge r in tumor")]:
        fig,ax=plt.subplots(figsize=(4,.6)); cb=fig.colorbar(ScalarMappable(norm=Normalize(EDGE_THRESHOLD,1),cmap=cmap),cax=ax,orientation='horizontal'); cb.set_label(label); save_pdf_svg(fig,FIG/"nodeplots"/stem,"Tumor-centric edge colorbar",[edges_path])
    handles=[Line2D([0],[0],color=mpl.colormaps['Blues'](1),lw=3,label='Tumor only'),Line2D([0],[0],color=mpl.colormaps['Reds'](1),lw=3,label='Shared with normal-like')]
    fig,ax=plt.subplots(figsize=(4,1.1)); ax.axis('off'); ax.legend(handles=handles,loc='center',ncol=2,frameon=False); save_pdf_svg(fig,FIG/"nodeplots"/"tumor_centric_nodeplot_edge_origin_edge_class_legend","Tumor-centric edge class legend",[edges_path])


def plot_top10_correlations():
    top_path = NODES / "joint_cm_cell_subtype_nodes_top10_from_H_df.csv"
    top = pd.read_csv(top_path); cm_order = top.CM.drop_duplicates().tolist()
    for context in ("normal-like", "tumor"):
        matrices=[]
        for cm in cm_order:
            path=NODES/"top10_node_correlation_matrices"/context/f"{cm}_top10_pearson_correlation.csv"
            matrix=load_matrix(path,"node"); nodes=top.loc[top.CM.eq(cm)].sort_values('rank').cell_subtype.tolist(); matrix=matrix.loc[nodes,nodes]
            matrices.append((cm,nodes,matrix,path))
            fig,ax=plt.subplots(figsize=(5.4,5.0)); sns.heatmap(matrix,cmap='RdBu_r',vmin=-1,vmax=1,center=0,square=True,annot=False,cbar_kws={'label':'Pearson r'},ax=ax); ax.set_title(cm); ax.tick_params(axis='x',rotation=90,labelsize=6); ax.tick_params(axis='y',labelsize=6); fig.tight_layout()
            save_pdf_svg(fig,FIG/"top10_node_correlation_heatmaps_no_edge_filter_by_cm"/context.replace('-','_')/f"{cm}_{context.replace('-','_')}_top10_node_correlation_heatmap_no_edge_filter","Top10 node correlation heatmap",[path,top_path])
        rows,cols=closest_square(len(matrices)); fig,axes=plt.subplots(rows,cols,figsize=(cols*4.1+1,rows*4.1),squeeze=False)
        image=None
        for ax,(cm,nodes,matrix,path) in zip(axes.ravel(),matrices): image=ax.imshow(matrix,cmap='RdBu_r',vmin=-1,vmax=1); ax.set_title(cm); ax.set_xticks(range(len(nodes)),nodes,rotation=90,fontsize=5.5); ax.set_yticks(range(len(nodes)),nodes,fontsize=5.5); ax.set_aspect('equal')
        for ax in axes.ravel()[len(matrices):]: ax.axis('off')
        fig.colorbar(image,ax=list(axes.ravel()[:len(matrices)]),fraction=.02,pad=.02,label='Pearson r'); fig.subplots_adjust(wspace=.55,hspace=.65)
        save_pdf_svg(fig,FIG/"top10_correlations"/f"{context.replace('-','_')}_top10_node_correlation_heatmap_no_edge_filter","Top10 node correlation summary",[top_path,NODES/"top10_node_correlation_matrices"])


def plot_epi_cm_heatmaps_and_scatters():
    palette_df=pd.read_csv(PALETTE); palette=dict(zip(palette_df.cell_subtype.astype(str),palette_df.color.astype(str)))
    source_path=ASSOC/"balanced_joint_cm_epi_cm_association_spearman_all_pair_scatter_source.csv"; source=pd.read_csv(source_path)
    for context in ("normal-like","tumor"):
        stem=f"balanced_joint_cm_epi_cm_association_{context}"
        rho_path=ASSOC/f"{stem}_rho_matrix.csv"; q_path=ASSOC/f"{stem}_q_matrix.csv"
        rho=load_matrix(rho_path,"epithelial_subtype"); q=load_matrix(q_path,"epithelial_subtype")
        stars=q.map(lambda x:'***' if x<.001 else '**' if x<.01 else '*' if x<.05 else 'ns')
        fig,ax=plt.subplots(figsize=(8.2,6.8)); sns.heatmap(rho,cmap='RdBu_r',vmin=-1,vmax=1,center=0,annot=stars,fmt='',cbar_kws={'label':'Spearman rho'},ax=ax); ax.tick_params(axis='x',rotation=90); ax.tick_params(axis='y',rotation=0,labelsize=7); fig.tight_layout()
        save_pdf_svg(fig,FIG/"epi_cm_association_spearman"/f"{stem}_heatmap_qstars","Spearman Epi-CM heatmap",[rho_path,q_path])
        subset=source[source.context.eq(context)]
        for (epi,cm),data in subset.groupby(["epithelial_subtype","CM"],sort=False):
            first=data.iloc[0]; fig,ax=plt.subplots(figsize=(6,5)); sns.regplot(data=data,x='cm_activity',y='epi_frequency',color=palette[epi],scatter_kws={'s':60,'alpha':.85,'edgecolor':'none'},line_kws={'lw':2},ci=95,ax=ax); ax.set_xlabel(f'{cm} score'); ax.set_ylabel(f'{epi} fraction'); ax.set_ylim(0,1); ax.set_title(f"{'Tumor samples' if context=='tumor' else 'Normal-like samples'}: {cm} vs {epi}"); ax.text(.02,.98,f"Spearman rho={first.rho:.3f}, q={first.q:.1e}\nn={len(data)}",transform=ax.transAxes,va='top',ha='left',fontsize=10,bbox=dict(boxstyle='round,pad=.3',fc='white',ec='gray',alpha=.8)); sns.despine(ax=ax); fig.tight_layout()
            save_pdf_svg(fig,FIG/"epi_cm_association_spearman"/"scatter"/context.replace('-','_')/f"scatter_{safe(cm)}_vs_{safe(epi)}","Spearman Epi-CM scatter",[source_path,PALETTE])


def main():
    completion=TABLE/"plotting_completion.json"
    if completion.exists(): raise FileExistsError(completion)
    plot_k_selection(); plot_activity_and_loading_heatmaps(); plot_activity_barplot(); plot_nodeplots(); plot_top10_correlations(); plot_epi_cm_heatmaps_and_scatters()
    manifest=pd.DataFrame(MANIFEST_ROWS)
    TABLE.mkdir(parents=True,exist_ok=True)
    manifest.to_csv(TABLE/"figure_manifest.csv",index=False)
    result={"n_figure_files":len(manifest),"n_pdf":int(manifest.figure_file.str.endswith('.pdf').sum()),"n_svg":int(manifest.figure_file.str.endswith('.svg').sum()),"n_figure_families":int(manifest.figure_family.nunique()),"pearson_branch_run":False,"spearman_all_pairs_plotted":True}
    completion.write_text(json.dumps(result,indent=2)+"\n")
    (TABLE/"plotting_parameters.csv").write_text("plotting_block,formats,scatter_style,node_edge_threshold,seed\n06_final_unified_plotting_and_manifest_v5.py,PDF;SVG,Spearman regplot ci95,0.25,42\n")
    (TABLE/"package_versions.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nmatplotlib={mpl.__version__}\nseaborn={sns.__version__}\nnetworkx={nx.__version__}\ncode={Path(__file__).resolve()}\nseed={SEED}\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()

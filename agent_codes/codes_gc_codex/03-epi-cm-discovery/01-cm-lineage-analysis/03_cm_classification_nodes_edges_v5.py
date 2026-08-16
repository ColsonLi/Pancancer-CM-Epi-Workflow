#!/usr/bin/env python3
"""Build canonical GC CM node and status-specific edge tables."""

from __future__ import annotations

import itertools
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
WF = CODE_DIR.parents[2]
BLOCK = "03-epi-cm-discovery"
NMF_TAG = "agent-led-current-selection-v5-nmf-r1"
TAG = "agent-led-current-selection-v5-cm-analysis-r1"
SEED = 42
TOP_N_SUBTYPES = 20
PLOT_TOP_N_SUBTYPES = 12
TOP_N_NODES = 10
EDGE_R_THRESHOLD = 0.25
NMF = WF / "tables" / BLOCK / "01-cm-lineage-analysis" / "02_balanced_joint_nmf" / NMF_TAG
OUT = WF / "tables" / BLOCK / "01-cm-lineage-analysis" / "03_cm_classification_nodes_and_edges" / TAG


def write_csv_once(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    frame.to_csv(partial, index=index)
    os.replace(partial, path)


def write_text_once(value: str, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(value)
    os.replace(partial, path)


def lineage(node: str) -> str:
    return str(node).split("_", 1)[0]


def main() -> None:
    if (OUT / "nodes_edges_completion.json").exists():
        raise FileExistsError(OUT / "nodes_edges_completion.json")
    classification = pd.read_csv(NMF / "joint_module_classification.csv")
    mapping = pd.read_csv(NMF / "raw_to_canonical_CM_mapping.csv")
    H = pd.read_csv(NMF / "H_df.csv").set_index("CM")
    V = pd.read_csv(NMF / "non_epi_subtype_frequency_column_minmax.csv").set_index("sample")
    activity = pd.read_csv(NMF / "activity_df_sample_by_CM.csv").set_index("sample")
    cm_order = classification.sort_values("global_order").CM.astype(str).tolist()
    if H.index.astype(str).tolist() != cm_order:
        raise RuntimeError("H canonical CM order differs from classification")
    if set(V.index) != set(activity.index):
        raise RuntimeError("V/activity sample IDs differ")
    status = activity.loc[V.index, "status"].astype(str)

    all_rows = []
    for cm in cm_order:
        ordered = H.loc[cm].sort_values(ascending=False, kind="mergesort")
        for rank, (node, loading) in enumerate(ordered.items(), 1):
            all_rows.append({
                "CM": cm, "cell_subtype": node, "loading": float(loading),
                "rank": rank, "cell_lineage": lineage(node),
            })
    all_nodes = pd.DataFrame(all_rows)
    top20 = all_nodes.groupby("CM", sort=False, group_keys=False).head(TOP_N_SUBTYPES).copy()
    top12 = all_nodes.groupby("CM", sort=False, group_keys=False).head(PLOT_TOP_N_SUBTYPES).copy()
    top10 = all_nodes.groupby("CM", sort=False, group_keys=False).head(TOP_N_NODES).copy()
    write_csv_once(all_nodes, OUT / "joint_cm_cell_subtype_nodes_all_from_H_df.csv")
    write_csv_once(top20, OUT / "joint_cm_cell_subtype_nodes_top20_from_H_df.csv")
    write_csv_once(top12, OUT / "joint_cm_cell_subtype_nodes_top12_from_H_df.csv")
    write_csv_once(top10, OUT / "joint_cm_cell_subtype_nodes_top10_from_H_df.csv")

    reference_rows = []
    diagnostic_rows = []
    edge_rows = []
    corr_long_rows = []
    for cm in cm_order:
        candidates = top10.loc[top10.CM.eq(cm)].sort_values("rank")
        nodes = candidates.cell_subtype.astype(str).tolist()
        corr_by_context = {}
        passing_nodes = set()
        for context in ("normal-like", "tumor"):
            matrix = V.loc[status.eq(context), nodes]
            corr = matrix.corr(method="pearson").fillna(0.0)
            np.fill_diagonal(corr.values, 1.0)
            corr_by_context[context] = corr
            for node_a, node_b in itertools.combinations(nodes, 2):
                value = float(corr.loc[node_a, node_b])
                corr_long_rows.append({
                    "context": context, "CM": cm, "node_a": node_a,
                    "node_b": node_b, "pearson_r": value,
                    "top10_diagnostic_no_edge_filter": True,
                })
                if value >= EDGE_R_THRESHOLD:
                    passing_nodes.update((node_a, node_b))
            write_csv_once(corr.reset_index(names="node"), OUT / "top10_node_correlation_matrices" / context / f"{cm}_top10_pearson_correlation.csv")
        retained = [node for node in nodes if node in passing_nodes]
        for ref_rank, node in enumerate(retained, 1):
            reference_rows.append({"CM": cm, "reference_node_rank": ref_rank, "node": node})
            original = candidates.loc[candidates.cell_subtype.eq(node)].iloc[0]
            diagnostic_rows.append({
                "CM": cm, "reference_node_rank": ref_rank, "node": node,
                "loading": float(original.loading), "loading_rank_in_H_top10": int(original["rank"]),
                "cell_lineage": lineage(node),
            })
        for context, corr in corr_by_context.items():
            for node_a, node_b in itertools.combinations(retained, 2):
                value = float(corr.loc[node_a, node_b])
                edge_rows.append({
                    "context": context, "CM": cm, "node_a": node_a, "node_b": node_b,
                    "pearson_r": value, "edge_pass_r_ge_0.25": value >= EDGE_R_THRESHOLD,
                })

    reference = pd.DataFrame(reference_rows, columns=["CM", "reference_node_rank", "node"])
    if reference.empty:
        raise RuntimeError("No CM nodes survived edge screening")
    edges = pd.DataFrame(edge_rows)
    write_csv_once(reference, OUT / "balanced_joint_cm_reference_node_sets_after_edge_threshold.csv")
    write_csv_once(pd.DataFrame(diagnostic_rows), OUT / "balanced_joint_cm_reference_node_loading_diagnostics.csv")
    write_csv_once(edges, OUT / "status_specific_nodeplot_edges.csv")
    write_csv_once(pd.DataFrame(corr_long_rows), OUT / "top10_node_correlations_no_edge_filter_long.csv")

    retained_detail = pd.DataFrame(diagnostic_rows)
    for context, filename in (("tumor", "tumor_network_nodes_from_H_df.csv"), ("normal-like", "normal_like_network_nodes_from_H_df.csv")):
        table = retained_detail.copy()
        table = table.rename(columns={"node": "cell_subtype", "reference_node_rank": "rank"})
        table["status_context"] = context
        table = table[["CM", "cell_subtype", "loading", "rank", "cell_lineage", "status_context"]]
        write_csv_once(table, OUT / filename)

    normal_pass = {
        (row.CM, tuple(sorted((row.node_a, row.node_b))))
        for row in edges.loc[edges.context.eq("normal-like") & edges["edge_pass_r_ge_0.25"]].itertuples()
    }
    origin_rows = []
    for row in edges.loc[edges.context.eq("tumor") & edges["edge_pass_r_ge_0.25"]].itertuples():
        key = (row.CM, tuple(sorted((row.node_a, row.node_b))))
        origin_rows.append({
            "CM": row.CM, "node_a": row.node_a, "node_b": row.node_b,
            "tumor_pearson_r": row.pearson_r,
            "edge_origin": "shared" if key in normal_pass else "tumor_only",
        })
    write_csv_once(pd.DataFrame(origin_rows), OUT / "tumor_centric_edge_origin.csv")
    write_csv_once(classification, OUT / "joint_module_classification.csv")
    write_csv_once(mapping, OUT / "raw_to_canonical_CM_mapping.csv")
    write_csv_once(pd.DataFrame([{
        "step": "cm_classification_nodes_and_edges", "n_cms": len(cm_order),
        "top_n_subtypes": TOP_N_SUBTYPES, "plot_top_n_subtypes": PLOT_TOP_N_SUBTYPES,
        "top_n_nodes": TOP_N_NODES, "edge_correlation_method": "pearson",
        "edge_r_threshold": EDGE_R_THRESHOLD, "contexts": "normal-like;tumor",
        "input_H": str(NMF / "H_df.csv"), "input_V": str(NMF / "non_epi_subtype_frequency_column_minmax.csv"),
        "random_seed": SEED, "code_file": str(Path(__file__).resolve()),
    }]), OUT / "node_edge_parameters.csv")
    result = {
        "n_cms": len(cm_order), "n_top10_candidate_rows": len(top10),
        "n_reference_node_rows": len(reference), "n_unique_reference_nodes": reference.node.nunique(),
        "n_edge_rows_all_contexts": len(edges),
        "n_passing_normal_like_edges": int(edges.loc[edges.context.eq("normal-like"), "edge_pass_r_ge_0.25"].sum()),
        "n_passing_tumor_edges": int(edges.loc[edges.context.eq("tumor"), "edge_pass_r_ge_0.25"].sum()),
    }
    write_text_once(json.dumps(result, indent=2) + "\n", OUT / "nodes_edges_completion.json")
    write_text_once(
        "Canonical GC CM node/edge tables. Reference nodes are H-top10 nodes with at least one Pearson r>=0.25 edge in tumor or normal-like.\n",
        OUT / "readme.txt",
    )
    write_text_once(
        f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\n"
        f"code={Path(__file__).resolve()}\nseed={SEED}\n",
        OUT / "package_versions.txt",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

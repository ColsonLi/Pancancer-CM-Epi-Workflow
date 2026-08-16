#!/usr/bin/env python3
"""Agent-led GC subtype annotation for one current selected lineage."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
sns.set_theme(style="ticks")

CODE_DIR = Path(__file__).resolve().parent
WF = CODE_DIR.parents[2]
MODULE = "02-cell_subtype_integration_clustering"
SELECT_TAG = "selected-user-override-b25-40-0p5-epi40-40-0p4-v5"
ANNOTATION_TAG = "agent-led-current-selection-v5"
SEED = 42
REVIEW_DEPTH = 300
MANIFEST = (
    WF / "tables" / MODULE / "04-subtype-selected-clustering" / SELECT_TAG
    / "final_selected_lineage_manifest.csv"
)

# Reuse the reviewed GC marker dictionaries and evidence helpers from the
# existing GC-specific annotation implementation, without using its old v2 data.
LEGACY_PATH = CODE_DIR.parent / "05-subtype-deg-annotation-v2" / "05_annotate_lineage_v2.py"
spec = importlib.util.spec_from_file_location("gc_annotation_markers", LEGACY_PATH)
markers = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(markers)
PREFIX = markers.PREFIX
ABBREV = markers.ABBREV
IDENTITY_MARKERS = markers.IDENTITY_MARKERS
STATE_MARKERS = markers.STATE_MARKERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage", required=True, choices=sorted({
        "b_cells", "endothelial_cells", "epithelial_cells", "myeloid_cells",
        "plasma_cells", "stromal_cells", "t_cells",
    }))
    return parser.parse_args()


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def write_csv_once(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def write_text_once(value: str, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(value)
    os.replace(partial, path)


def write_h5ad_once(adata: ad.AnnData, path: Path) -> None:
    partial = path.with_name(f".{path.stem}.partial{path.suffix}")
    if path.exists() or partial.exists():
        raise FileExistsError(f"Refusing overwrite: {path} / {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(partial)
    os.replace(partial, path)


def natural_key(value: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(value))]


def positive(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[np.isfinite(frame.logfoldchanges) & (frame.logfoldchanges > 0)].copy()
    return out.sort_values(
        ["pvals_adj", "pvals", "score", "logfoldchanges", "gene"],
        ascending=[True, True, False, False, True], kind="mergesort",
    ).reset_index(drop=True)


def rank_and_export(
    adata: ad.AnnData,
    groupby: str,
    out_dir: Path,
    key: str,
    token: str,
) -> dict[str, pd.DataFrame]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    sc.tl.rank_genes_groups(
        adata, groupby=groupby, method="t-test", use_raw=True,
        key_added=key, pts=True,
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    result = {}
    groups = sorted(adata.obs[groupby].astype(str).unique(), key=natural_key)
    for group in groups:
        frame = sc.get.rank_genes_groups_df(adata, group=group, key=key)
        frame = frame.rename(columns={"names": "gene", "scores": "score"})
        required = ["gene", "score", "logfoldchanges", "pvals", "pvals_adj"]
        frame = frame[required + [c for c in frame.columns if c not in required]]
        frame.insert(len(required), "group", str(group))
        filename = f"{safe(group)}_degs_{safe(groupby)}_{token}.csv"
        frame.to_csv(out_dir / filename, index=False)
        result[str(group)] = frame
    return result


def evidence_tuple(row: pd.Series):
    return (
        float(row.pvals_adj) if np.isfinite(row.pvals_adj) else np.inf,
        float(row.pvals) if np.isfinite(row.pvals) else np.inf,
        -float(row.score) if np.isfinite(row.score) else np.inf,
        -abs(float(row.logfoldchanges)) if np.isfinite(row.logfoldchanges) else np.inf,
    )


def assign_unique_genes(positive_degs: dict[str, pd.DataFrame]):
    """Resolve duplicate top genes in favor of the cluster with stronger evidence."""
    pointers = {cluster: 0 for cluster in positive_degs}
    owners: dict[str, tuple[str, pd.Series]] = {}
    assigned: dict[str, pd.Series] = {}
    conflicts: dict[str, list[str]] = {cluster: [] for cluster in positive_degs}
    pending = list(sorted(positive_degs, key=natural_key))
    while pending:
        cluster = pending.pop(0)
        frame = positive_degs[cluster]
        if pointers[cluster] >= len(frame):
            raise RuntimeError(f"No unique positive DEG remains for cluster {cluster}")
        row = frame.iloc[pointers[cluster]]
        gene = str(row.gene)
        if gene not in owners:
            owners[gene] = (cluster, row)
            assigned[cluster] = row
            continue
        old_cluster, old_row = owners[gene]
        if evidence_tuple(row) < evidence_tuple(old_row):
            conflicts[old_cluster].append(
                f"{gene} retained by cluster {cluster} with stronger adjusted-P/raw-P/score evidence"
            )
            assigned.pop(old_cluster, None)
            pointers[old_cluster] += 1
            pending.append(old_cluster)
            owners[gene] = (cluster, row)
            assigned[cluster] = row
        else:
            conflicts[cluster].append(
                f"{gene} retained by cluster {old_cluster} with stronger adjusted-P/raw-P/score evidence"
            )
            pointers[cluster] += 1
            pending.append(cluster)
    return assigned, conflicts


def mean_vector(matrix) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).ravel()


def identity_evidence(
    adata: ad.AnnData, cluster_key: str, cluster: str,
    top_genes: list[str], lineage: str,
):
    available_genes = set(adata.raw.var_names.astype(str))
    mask = adata.obs[cluster_key].astype(str).to_numpy() == cluster
    rows = []
    for identity, marker_set in IDENTITY_MARKERS[lineage].items():
        available = sorted(marker_set & available_genes)
        if not available:
            continue
        matrix = adata.raw[:, available].X
        delta = mean_vector(matrix[mask]) - mean_vector(matrix[~mask])
        overlap = sorted(set(top_genes[:REVIEW_DEPTH]) & set(available))
        rows.append((len(overlap), float(delta.mean()), identity, overlap[:8], available))
    if not rows:
        raise RuntimeError(f"No GC marker genes available for {lineage}")
    overlap_n, delta, identity, overlap, available = sorted(
        rows, key=lambda x: (-x[0], -x[1], x[2])
    )[0]
    evidence_genes = overlap or available[:5]
    evidence = (
        f"markers={';'.join(evidence_genes)};top{REVIEW_DEPTH}_overlap={overlap_n};"
        f"mean_logexpr_delta={delta:.6g}"
    )
    confidence = "high" if overlap_n >= 2 and delta > 0 else (
        "moderate" if overlap_n >= 1 or delta > 0 else "low"
    )
    return identity, evidence, overlap_n, delta, confidence


def state_evidence(top_genes: list[str]):
    top_set = set(top_genes[:100])
    hits = sorted(
        ((len(top_set & genes), state, sorted(top_set & genes)) for state, genes in STATE_MARKERS.items()),
        key=lambda x: (-x[0], x[1]),
    )
    count, state, genes = hits[0]
    return (state, ";".join(genes)) if count else ("baseline_or_unresolved", "")


def gene_identifier_audit(adata: ad.AnnData, input_h5ad: Path, table_dir: Path) -> None:
    pattern = re.compile(r"^ENS[A-Z]*G[0-9]+(?:\.[0-9]+)?$")
    rows = []
    examples = []
    for layer, index in (("var_names", adata.var_names), ("raw.var_names", adata.raw.var_names)):
        values = pd.Index(index).astype(str)
        matches = np.asarray([bool(pattern.match(x)) for x in values])
        fraction = float(matches.mean()) if len(values) else 0.0
        kind = "gene_id" if fraction >= 0.9 else "gene_name" if fraction <= 0.1 else "mixed_or_unknown"
        rows.append({
            "input_h5ad": str(input_h5ad), "object_layer": layer,
            "var_names_type": kind, "n_features": len(values),
            "n_unique": int(values.nunique()), "n_duplicates": int(values.duplicated().sum()),
            "ensembl_like_n": int(matches.sum()), "ensembl_like_fraction": fraction,
            "candidate_gene_id_columns": "", "candidate_gene_name_columns": "",
            "selected_gene_id_source": "", "selected_gene_name_source": layer,
            "action_taken": "none_gene_symbols_retained" if kind == "gene_name" else "none_audit_only",
        })
        examples.extend({"object_layer": layer, "example": x} for x in values[:20])
    write_csv_once(pd.DataFrame(rows), table_dir / "gene_identifier_audit.csv")
    write_csv_once(pd.DataFrame(examples), table_dir / "gene_identifier_examples.csv")


def main() -> None:
    lineage = parse_args().lineage
    manifest = pd.read_csv(MANIFEST)
    selected = manifest[manifest.lineage.eq(lineage)]
    if len(selected) != 1:
        raise RuntimeError(f"Manifest row missing or duplicated for {lineage}")
    selected = selected.iloc[0]
    input_h5ad = Path(selected.selected_h5ad)
    cluster_key = str(selected.cluster_key)
    token = (
        f"pcs{int(selected.n_pcs)}_nn{int(selected.n_neighbors)}_"
        f"res{str(float(selected.resolution)).replace('.', 'p')}"
    )
    table_dir = WF / "tables" / MODULE / "05-subtype-deg-annotation" / ANNOTATION_TAG / lineage
    figure_dir = WF / "figures" / MODULE / "05-subtype-deg-annotation" / ANNOTATION_TAG / lineage
    output_h5ad = (
        WF / "h5ad" / MODULE / "05-subtype-deg-annotation" / ANNOTATION_TAG / lineage
        / f"adata_{ABBREV[lineage]}.h5ad"
    )
    candidate_path = table_dir / "candidate_subtype_annotation_agent_led.csv"
    mapping_path = table_dir / "final_subtype_mapping.csv"
    completion_path = table_dir / "annotation_completion.json"
    if any(path.exists() for path in (output_h5ad, candidate_path, mapping_path, completion_path)):
        raise FileExistsError(f"Annotation output already exists for {lineage}")

    adata = ad.read_h5ad(input_h5ad)
    if adata.raw is None:
        raise RuntimeError("adata.raw normalized/log expression is required")
    if cluster_key not in adata.obs:
        raise RuntimeError(f"Missing selected cluster key: {cluster_key}")
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
    gene_identifier_audit(adata, input_h5ad, table_dir)

    raw_dir = table_dir / f"degs_{cluster_key}_{token}"
    raw_degs = rank_and_export(adata, cluster_key, raw_dir, "rank_raw_current", token)
    positive_degs = {cluster: positive(frame) for cluster, frame in raw_degs.items()}
    if any(frame.empty for frame in positive_degs.values()):
        raise RuntimeError("At least one raw cluster has no positive DEG")
    assigned, conflicts = assign_unique_genes(positive_degs)

    rows = []
    for cluster in sorted(raw_degs, key=natural_key):
        chosen = assigned[cluster]
        top_genes = positive_degs[cluster].gene.astype(str).tolist()
        identity, evidence, overlap, delta, confidence = identity_evidence(
            adata, cluster_key, cluster, top_genes, lineage,
        )
        state, state_genes = state_evidence(top_genes)
        conflict_note = "; ".join(conflicts[cluster])
        rationale = "most significant positive t-test DEG"
        if conflict_note:
            rationale += f"; duplicate resolution: {conflict_note}; next-ranked eligible positive DEG selected"
        subtype = f"{PREFIX[lineage]}_{chosen.gene}"
        rows.append({
            "cluster": cluster,
            "proposed_cell_subtype": subtype,
            "cell_subtype": subtype,
            "functional_state": state,
            "selected_prefix": PREFIX[lineage],
            "most_significant_deg_gene": str(chosen.gene),
            "gene_selection_rationale": rationale,
            "top_deg_score": float(chosen.score),
            "top_deg_logfoldchange": float(chosen.logfoldchanges),
            "top_deg_pvals": float(chosen.pvals),
            "top_deg_pvals_adj": float(chosen.pvals_adj),
            "top_supporting_markers": evidence,
            "marker_supported_lineage": identity,
            "canonical_identity": identity,
            "canonical_top300_overlap": overlap,
            "canonical_mean_logexpr_delta": delta,
            "functional_state_marker_evidence": state_genes,
            "prefix_rationale": "project GC broad-lineage abbreviation",
            "agent_led_authorization": "user replied continue and instructed the agent to write labels",
            "annotation_confidence": confidence,
            "needs_user_confirmation": False,
            "review_depth": REVIEW_DEPTH,
        })
    mapping = pd.DataFrame(rows)
    if len(mapping) != adata.obs[cluster_key].nunique() or mapping.cell_subtype.duplicated().any():
        raise AssertionError("Final mapping is not one-to-one and unique")
    write_csv_once(mapping, candidate_path)
    write_csv_once(mapping.drop(columns="proposed_cell_subtype"), mapping_path)

    lookup = mapping.set_index("cluster")
    clusters = adata.obs[cluster_key].astype(str)
    for column in ("cell_subtype", "functional_state", "canonical_identity", "annotation_confidence"):
        values = clusters.map(lookup[column])
        if values.isna().any():
            raise RuntimeError(f"Unmapped cells in {column}")
        adata.obs[column] = pd.Categorical(values)
    adata.obs["annotation_source"] = "GC_Block02_agent_led_current_selection_v5"

    post_dir = table_dir / f"degs_cell_subtype_{token}"
    post_degs = rank_and_export(adata, "cell_subtype", post_dir, "rank_post_current", token)
    write_csv_once(
        pd.DataFrame({
            "cell_subtype": sorted(post_degs), "deg_method": "t-test",
            "expression_source": "adata.raw", "use_raw": True,
        }),
        table_dir / "post_annotation_deg_contract.csv",
    )
    subtype_counts = (
        adata.obs.cell_subtype.astype(str).value_counts().rename_axis("cell_subtype")
        .reset_index(name="n_cells")
    )
    write_csv_once(subtype_counts, table_dir / "subtype_counts.csv")
    write_csv_once(pd.DataFrame([{
        "step": "raw_and_post_annotation_deg", "input_h5ad": str(input_h5ad),
        "output_h5ad_or_object": str(output_h5ad), "n_obs_before": adata.n_obs,
        "n_obs_after": adata.n_obs, "n_vars_before": adata.n_vars,
        "n_vars_after": adata.n_vars, "method": "scanpy_t_test",
        "expression_source": "adata.raw", "use_raw": True,
        "backend": "scanpy_cpu_no_equivalent_rapids_t_test_used",
        "review_depth": REVIEW_DEPTH, "random_seed": SEED,
        "code_file": str(Path(__file__).resolve()), "notes": "no subtype cell filtering",
    }]), table_dir / "deg_annotation_parameters.csv")

    figure_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.autoshow = False
    sc.settings.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
    sc.settings.figdir = figure_dir
    subtype_order = mapping.sort_values("cluster", key=lambda x: x.map(natural_key)).cell_subtype.tolist()
    adata.obs["cell_subtype"] = adata.obs.cell_subtype.cat.reorder_categories(subtype_order)
    sc.pl.umap(adata, color="cell_subtype", show=False, save="_cell_subtype.pdf")
    sc.pl.umap(adata, color="cell_subtype", show=False, save="_cell_subtype.svg")
    sc.pl.umap(
        adata, color=["canonical_identity", "functional_state"], ncols=1, wspace=0.4,
        show=False, save="_annotation_evidence.pdf",
    )
    sc.pl.umap(
        adata, color=["canonical_identity", "functional_state"], ncols=1, wspace=0.4,
        show=False, save="_annotation_evidence.svg",
    )
    marker_genes = [g for g in mapping.most_significant_deg_gene.astype(str) if g in adata.raw.var_names]
    sc.pl.dotplot(
        adata, var_names=marker_genes, groupby="cell_subtype", use_raw=True,
        standard_scale="var", show=False, save="_subtype_markers.pdf",
    )
    sc.pl.dotplot(
        adata, var_names=marker_genes, groupby="cell_subtype", use_raw=True,
        standard_scale="var", show=False, save="_subtype_markers.svg",
    )

    completion = {
        "lineage": lineage,
        "n_cells": int(adata.n_obs),
        "n_subtypes": int(mapping.shape[0]),
        "cluster_key": cluster_key,
        "n_low_confidence": int((mapping.annotation_confidence == "low").sum()),
        "all_clusters_mapped_once": True,
        "all_subtype_names_unique_within_lineage": True,
        "raw_deg_csv_count": len(raw_degs),
        "post_annotation_deg_csv_count": len(post_degs),
        "output_h5ad": str(output_h5ad),
        "agent_led_authorized": True,
    }
    write_text_once(json.dumps(completion, indent=2) + "\n", completion_path)
    write_text_once(
        "GC agent-led subtype annotation from the current selected clustering.\n"
        "No subtype-level cells were filtered. Raw and post-annotation DEGs use Scanpy t-test with adata.raw/use_raw=True.\n"
        "All clusters use the same top-300 marker-review depth; every final subtype maps one-to-one to a raw cluster.\n",
        table_dir / "readme.txt",
    )
    write_text_once(
        f"python={platform.python_version()}\nanndata={ad.__version__}\nscanpy={sc.__version__}\n"
        f"numpy={np.__version__}\npandas={pd.__version__}\nseaborn={sns.__version__}\n"
        f"code={Path(__file__).resolve()}\nmarker_reference_code={LEGACY_PATH}\nseed={SEED}\n",
        table_dir / "package_versions.txt",
    )
    write_h5ad_once(adata, output_h5ad)
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()

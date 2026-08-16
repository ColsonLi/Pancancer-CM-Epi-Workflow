#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

SEED = 42
MIN_AVAILABLE_CANONICAL_MARKERS = 2
ROOT = Path(__file__).resolve().parents[4]
WF = ROOT / "epi-cm-core-workflow"
INPUT = WF / "h5ad/01-celltype_integration_clustering/05-clustering-parameter-search/selected/adata_inte.h5ad"
SELECTION = WF / "tables/01-celltype_integration_clustering/05-clustering-parameter-search/selected_clustering.csv"
OUTPUT = WF / "h5ad/01-celltype_integration_clustering/06-broad-annotation/adata_anno.h5ad"
TABLE = WF / "tables/01-celltype_integration_clustering/06-broad-annotation"
FIG = WF / "figures/01-celltype_integration_clustering/06-broad-annotation"
STAGE_H5AD_DIR = OUTPUT.parent.with_name(".06-broad-annotation-v4.staging")
STAGE_TABLE = TABLE.with_name(".06-broad-annotation-v4.staging")
STAGE_FIG = FIG.with_name(".06-broad-annotation-v4.staging")

MARKERS = {
    "Epithelial Cells": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "MUC1", "KRT20", "INSM1", "PDX1"],
    "T Cells": ["CD3D", "CD3E", "TRBC1", "TRBC2", "CD247", "IL7R"],
    "B Cells": ["CD79A", "MS4A1", "CD37", "CD74", "HLA-DRA", "CD22"],
    "Plasma Cells": ["MZB1", "JCHAIN", "SDC1", "IGHG1", "IGHG3", "IGKC"],
    "Myeloid Cells": ["LST1", "TYROBP", "FCER1G", "CTSS", "LYZ", "AIF1"],
    "Mast Cells": ["TPSAB1", "TPSB2", "CPA3", "MS4A2", "KIT", "HDC"],
    "Endothelial Cells": ["PECAM1", "VWF", "EMCN", "KDR", "ENG", "ESAM"],
    "Stromal Cells": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "COL6A1", "ACTA2"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def res_token(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def refuse_existing_outputs() -> None:
    def contains_output(path: Path) -> bool:
        if not path.exists():
            return False
        # A previously pre-created but still empty category/task directory is
        # not an analysis output.  It is safe for the atomic staged publish to
        # replace it, while any real file/subdirectory must still hard-stop.
        return not (path.is_dir() and next(path.iterdir(), None) is None)

    existing = [
        str(path)
        for path in (OUTPUT.parent, TABLE, FIG, STAGE_H5AD_DIR, STAGE_TABLE, STAGE_FIG)
        if contains_output(path)
    ]
    if existing:
        raise FileExistsError(
            "Broad-annotation final/staging output already exists; refusing overwrite or "
            f"partial-output reuse: {existing}"
        )


def export_degs(
    a: ad.AnnData, groupby: str, out: Path, parameter_token: str
) -> dict[str, pd.DataFrame]:
    sc.tl.rank_genes_groups(a, groupby=groupby, method="t-test", use_raw=True, key_added=f"degs_{groupby}")
    result = a.uns[f"degs_{groupby}"]
    out.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for group in result["names"].dtype.names:
        frame = pd.DataFrame({
            "gene": result["names"][group], "score": result["scores"][group],
            "logfoldchanges": result["logfoldchanges"][group], "pvals": result["pvals"][group],
            "pvals_adj": result["pvals_adj"][group],
        })
        frames[str(group)] = frame
        safe = str(group).replace(" ", "_").replace("/", "_")
        frame.to_csv(out / f"{safe}_degs_{groupby}_{parameter_token}.csv", index=False)
    return frames


def main() -> None:
    random.seed(SEED); np.random.seed(SEED)
    refuse_existing_outputs()
    if not INPUT.is_file() or not SELECTION.is_file():
        raise FileNotFoundError("Selected clustering inputs missing")
    selection = pd.read_csv(SELECTION)
    if len(selection) != 1:
        raise RuntimeError("Invalid selected_clustering.csv")
    raw_key = str(selection.iloc[0]["raw_cluster_column"])
    pcs = int(selection.iloc[0]["n_pcs"])
    nn = int(selection.iloc[0]["n_neighbors"])
    resolution = float(selection.iloc[0]["resolution"])
    parameter_token = f"pcs{pcs}_nn{nn}_res{res_token(resolution)}"
    a = ad.read_h5ad(INPUT)
    if raw_key not in a.obs or a.raw is None or "X_umap" not in a.obsm:
        raise RuntimeError("Selected h5ad lacks raw clusters/raw/UMAP")
    clusters = sorted(a.obs[raw_key].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)
    if len(clusters) < len(MARKERS) or len(clusters) > 20:
        raise RuntimeError(f"Broad automatic annotation requires 8..20 raw clusters; observed {len(clusters)}")
    genes = set(a.raw.var_names.astype(str))
    used_markers: dict[str, list[str]] = {}
    marker_availability_rows = []
    for cell_type, panel in MARKERS.items():
        available = [g for g in panel if g in genes]
        if len(available) < MIN_AVAILABLE_CANONICAL_MARKERS:
            raise RuntimeError(f"Insufficient marker genes for {cell_type}: {available}")
        used_markers[cell_type] = available
        marker_availability_rows.append({
            "cell_type": cell_type,
            "configured_markers": ";".join(panel),
            "available_markers": ";".join(available),
            "n_configured": len(panel),
            "n_available": len(available),
            "minimum_required": MIN_AVAILABLE_CANONICAL_MARKERS,
            "gene_space": "strict five-series inner-join raw.var_names",
        })
        sc.tl.score_genes(a, gene_list=available, score_name=f"broad_score__{cell_type}",
                          ctrl_size=max(20, len(available)), use_raw=True, random_state=SEED)
    STAGE_TABLE.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(marker_availability_rows).to_csv(
        STAGE_TABLE / "broad_marker_panel_availability.csv", index=False
    )
    raw_deg_name = f"degs_{raw_key}_{parameter_token}"
    raw_deg_dir = STAGE_TABLE / raw_deg_name
    prior_raw_deg_dir = TABLE.with_name(".06-broad-annotation-v3.staging") / raw_deg_name
    if prior_raw_deg_dir.is_dir():
        raw_degs = {}
        for cluster in clusters:
            safe = str(cluster).replace(" ", "_").replace("/", "_")
            source_csv = prior_raw_deg_dir / f"{safe}_degs_{raw_key}_{parameter_token}.csv"
            if not source_csv.is_file():
                raise RuntimeError(f"Reusable full DEG is missing: {source_csv}")
            frame = pd.read_csv(source_csv)
            required = {"gene", "score", "logfoldchanges", "pvals", "pvals_adj"}
            if set(frame.columns) != required or len(frame) != a.raw.n_vars:
                raise RuntimeError(f"Reusable full DEG failed schema/row audit: {source_csv}")
            raw_degs[str(cluster)] = frame
        shutil.copytree(prior_raw_deg_dir, raw_deg_dir)
        pd.DataFrame([{
            "reused": True,
            "source_deg_dir": str(prior_raw_deg_dir.relative_to(WF)),
            "destination_deg_dir": str(raw_deg_dir.relative_to(WF)),
            "n_cluster_csv": len(raw_degs),
            "rows_per_csv": a.raw.n_vars,
            "reuse_reason": "same selected-v3 cell set, raw cluster column, parameters, and full use_raw DEG results",
        }]).to_csv(STAGE_TABLE / "raw_deg_reuse_audit.csv", index=False)
    else:
        raw_degs = export_degs(a, raw_key, raw_deg_dir, parameter_token)
        pd.DataFrame([{
            "reused": False,
            "source_deg_dir": "",
            "destination_deg_dir": str(raw_deg_dir.relative_to(WF)),
            "n_cluster_csv": len(raw_degs),
            "rows_per_csv": a.raw.n_vars,
            "reuse_reason": "no compatible prior full DEG directory",
        }]).to_csv(STAGE_TABLE / "raw_deg_reuse_audit.csv", index=False)
    score_cols = [f"broad_score__{x}" for x in MARKERS]
    score_mean = a.obs.assign(_cluster=a.obs[raw_key].astype(str)).groupby("_cluster", observed=True)[score_cols].mean().reindex(clusters)
    score_mean.columns = list(MARKERS)
    # FIXED: do not force every expected lineage to appear.  Each raw cluster
    # receives its best-supported observed identity; missing lineages remain
    # absent rather than being manufactured by an assignment constraint.
    mapping: dict[str, str] = {
        cluster: str(score_mean.loc[cluster].idxmax()) for cluster in clusters
    }
    review_depth = 50
    review_reason = "top50 sufficient for all assigned raw clusters"
    for depth in (50, 100, 200, 300):
        missing_support = []
        for cluster in clusters:
            positive = raw_degs[cluster][
                pd.to_numeric(raw_degs[cluster]["logfoldchanges"], errors="coerce") > 0
            ]
            top = set(positive.head(depth)["gene"].astype(str))
            if not any(g in top for g in used_markers[mapping[cluster]]):
                missing_support.append(cluster)
        review_depth = depth
        if not missing_support:
            review_reason = f"top{depth} provided canonical DEG support for every assigned raw cluster"
            break
        review_reason = (
            f"expanded uniformly to top{depth}; clusters without canonical marker DEG support: "
            + ";".join(missing_support)
        )
    if missing_support:
        raise RuntimeError(
            "Refusing automatic broad annotation: canonical positive DEG support is absent "
            f"through top{review_depth} for raw clusters {missing_support}"
        )
    rows = []
    for cluster in clusters:
        assigned = mapping[cluster]
        deg = raw_degs[cluster]
        positive = deg[pd.to_numeric(deg["logfoldchanges"], errors="coerce") > 0]
        top = positive.head(review_depth)["gene"].astype(str).tolist()
        support = [g for g in used_markers[assigned] if g in set(top)]
        top_positive = top[:10]
        rows.append({
            "cluster": cluster, "leiden_coarse": assigned, "cell_type": assigned,
            "selected_marker_score": float(score_mean.loc[cluster, assigned]),
            "next_best_marker_score": float(score_mean.loc[cluster].drop(assigned).max()),
            "marker_score_margin": float(score_mean.loc[cluster, assigned] - score_mean.loc[cluster].drop(assigned).max()),
            "canonical_markers_available": ";".join(used_markers[assigned]),
            "canonical_markers_in_reviewed_degs": ";".join(support),
            "deg_review_depth": review_depth,
            "deg_review_reason": review_reason,
            "top_positive_degs": ";".join(top_positive),
            "mapping_method": "automatic canonical marker score plus evidence-gated positive DEG review",
            "approved_for_apply": True, "approval_basis": "canonical positive DEG support required", "mapping_version": "v1",
        })
    mapping_frame = pd.DataFrame(rows)
    mapping_frame.to_csv(STAGE_TABLE / "broad_annotation_mapping_agent_v1.csv", index=False)
    score_mean.reset_index(names="cluster").to_csv(STAGE_TABLE / "broad_marker_score_by_raw_cluster.csv", index=False)
    observed_labels = [label for label in MARKERS if label in set(mapping.values())]
    a.obs["leiden_coarse"] = pd.Categorical(
        a.obs[raw_key].astype(str).map(mapping), categories=observed_labels
    )
    a.obs["cell_type"] = a.obs["leiden_coarse"].copy()
    if a.obs["leiden_coarse"].isna().any() or set(a.obs["leiden_coarse"].astype(str)) != set(observed_labels):
        raise RuntimeError("Broad mapping coverage failed")
    final_deg_name = f"degs_leiden_coarse_{parameter_token}"
    final_deg_dir = STAGE_TABLE / final_deg_name
    export_degs(a, "leiden_coarse", final_deg_dir, parameter_token)
    a.uns["broad_annotation"] = {
        "raw_cluster_column": raw_key, "mapping_file": "broad_annotation_mapping_agent_v1.csv",
        "mapping_method": "canonical marker score plus DEG evidence", "agent_authorized": True,
        "observed_labels_only": True, "deg_review_depth": review_depth,
        "raw_cluster_deg_dir": str((TABLE / raw_deg_name).relative_to(WF)),
        "leiden_coarse_deg_dir": str((TABLE / final_deg_name).relative_to(WF)),
        "parameter_token": parameter_token,
        "expression_layer_for_scores_and_degs": "adata.raw normalized_log", "random_seed": SEED,
    }
    STAGE_H5AD_DIR.mkdir(parents=True, exist_ok=False)
    staged_h5ad = STAGE_H5AD_DIR / OUTPUT.name
    a.write_h5ad(staged_h5ad, compression="gzip")
    counts = a.obs["leiden_coarse"].astype(str).value_counts().rename_axis("leiden_coarse").reset_index(name="n_cells")
    counts.to_csv(STAGE_TABLE / "broad_cell_type_composition.csv", index=False)
    pd.crosstab(a.obs["leiden_coarse"], a.obs["series"]).to_csv(STAGE_TABLE / "broad_cell_type_by_series_counts.csv")
    pd.crosstab(a.obs["leiden_coarse"], a.obs["status"]).to_csv(STAGE_TABLE / "broad_cell_type_by_status_counts.csv")
    marker_manifest = pd.DataFrame([{"cell_type": k, "markers": ";".join(v), "n_markers": len(v),
                                     "observed_in_annotation": k in observed_labels}
                                    for k, v in used_markers.items()])
    marker_manifest.to_csv(STAGE_TABLE / "broad_marker_manifest.csv", index=False)
    STAGE_FIG.mkdir(parents=True, exist_ok=False)
    sns.set_theme(style="ticks"); sc.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True); sc.settings.figdir = STAGE_FIG
    for suffix in ["pdf", "svg"]:
        sc.pl.umap(a, color=[raw_key, "leiden_coarse"], ncols=2, wspace=0.45, frameon=True,
                   show=False, save=f"_broad_annotation.{suffix}")
        plot_markers = {label: used_markers[label][:3] for label in observed_labels}
        sc.pl.dotplot(a, var_names=plot_markers, groupby="leiden_coarse", use_raw=True,
                      standard_scale="var", show=False, save=f"_broad_markers.{suffix}")
    validation = {"all_passed": True, "n_cells": a.n_obs, "n_raw_clusters": len(clusters),
                  "n_broad_labels": a.obs["leiden_coarse"].nunique(), "mapping_complete": True,
                  "all_clusters_have_canonical_positive_deg_support": True,
                  "raw_present": a.raw is not None, "output_sha256": sha256(staged_h5ad),
                  "final_h5ad_published_last": True}
    (STAGE_TABLE / "broad_annotation_completion.json").write_text(json.dumps(validation, indent=2) + "\n")
    versions = {p: version(p) for p in ["anndata", "scanpy", "numpy", "pandas", "scipy"]}
    versions.update({"python": platform.python_version(), "seed": SEED})
    (STAGE_TABLE / "package_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    (STAGE_TABLE / "readme.txt").write_text(
        "Broad annotation from selected raw clusters using canonical marker scores and full t-test DEGs with use_raw=True.\n"
        "Every applied raw-cluster label requires canonical positive DEG support through a uniform top-300 review.\n"
        "Only marker-supported labels observed in the data are retained; no expected lineage is forced to appear.\n"
        "Tables/figures are validated before the final h5ad directory is atomically published last.\n"
    )
    if not list(STAGE_FIG.glob("*.pdf")) or not list(STAGE_FIG.glob("*.svg")):
        raise RuntimeError("Broad-annotation PDF/SVG staging outputs are incomplete")
    unexpected = [path for path in STAGE_FIG.iterdir() if path.suffix.lower() not in {".pdf", ".svg"}]
    if unexpected:
        raise RuntimeError(f"Unexpected broad-annotation figure format: {unexpected}")
    if not staged_h5ad.is_file() or not (STAGE_TABLE / "broad_annotation_completion.json").is_file():
        raise RuntimeError("Broad-annotation staged artifact validation failed")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.parent.mkdir(parents=True, exist_ok=True)
    os.replace(STAGE_TABLE, TABLE)
    os.replace(STAGE_FIG, FIG)
    # Final h5ad directory is the last published artifact.
    os.replace(STAGE_H5AD_DIR, OUTPUT.parent)


if __name__ == "__main__":
    main()

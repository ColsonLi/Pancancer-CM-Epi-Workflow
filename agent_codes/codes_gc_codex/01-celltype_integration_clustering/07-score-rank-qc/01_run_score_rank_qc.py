#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

SEED = 42
ROOT = Path(__file__).resolve().parents[4]
WF = ROOT / "epi-cm-core-workflow"
RUN_VERSION = os.environ.get("GC_RUN_VERSION", "").strip()
RUN_PART = Path(RUN_VERSION) if RUN_VERSION else Path()
INPUT = WF / "h5ad/01-celltype_integration_clustering/06-broad-annotation" / RUN_PART / "adata_anno.h5ad"
OUT_ALL = WF / "h5ad/01-celltype_integration_clustering/07-score-rank-qc" / RUN_PART / "adata_anno_score_genes_rank.h5ad"
OUT_KEEP = WF / "h5ad/01-celltype_integration_clustering/07-score-rank-qc" / RUN_PART / "adata_anno_score_genes_rank_consistent.h5ad"
TABLE = WF / "tables/01-celltype_integration_clustering/07-score-rank-qc" / RUN_PART
FIG = WF / "figures/01-celltype_integration_clustering/07-score-rank-qc" / RUN_PART
MIN_SCORE_GENES = 10
MAX_SCORE_GENES = 100
STAGE_H5AD_DIR = OUT_ALL.parent.with_name(f".{OUT_ALL.parent.name}.staging")
STAGE_TABLE = TABLE.with_name(f".{TABLE.name}.staging")
STAGE_FIG = FIG.with_name(f".{FIG.name}.staging")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_h5ad_new(a: ad.AnnData, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.h5ad"); a.write_h5ad(tmp, compression="gzip"); os.replace(tmp, path)


def safe_label(label: str) -> str:
    return label.replace(" ", "_").replace("/", "_")


def refuse_existing_outputs() -> None:
    existing = [
        str(path)
        for path in (OUT_ALL.parent, TABLE, FIG, STAGE_H5AD_DIR, STAGE_TABLE, STAGE_FIG)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Score-rank final/staging output already exists; refusing overwrite or "
            f"partial-output reuse: {existing}"
        )


def main() -> None:
    random.seed(SEED); np.random.seed(SEED)
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    refuse_existing_outputs()
    a = ad.read_h5ad(INPUT)
    if "leiden_coarse" not in a.obs or a.raw is None:
        raise RuntimeError("Broad annotated raw-preserved source required")
    labels = list(a.obs["leiden_coarse"].cat.categories if hasattr(a.obs["leiden_coarse"], "cat") else sorted(a.obs["leiden_coarse"].astype(str).unique()))
    if len(labels) < 2:
        raise RuntimeError("Score-rank QC requires at least two observed broad labels")
    annotation_meta = a.uns.get("broad_annotation", {})
    deg_rel = annotation_meta.get("leiden_coarse_deg_dir")
    if not deg_rel:
        raise RuntimeError("Broad annotation does not record its full leiden_coarse DEG directory")
    deg_dir = WF / str(deg_rel)
    if not deg_dir.is_dir():
        raise FileNotFoundError(deg_dir)
    raw_genes = set(a.raw.var_names.astype(str))
    import cupy as cp
    import rapids_singlecell as rsc
    score_map: dict[str, str] = {}
    score_gene_rows: list[dict[str, object]] = []
    for label in labels:
        safe = safe_label(str(label))
        matches = sorted(deg_dir.glob(f"{safe}_degs_leiden_coarse_*.csv"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one full DEG CSV for {label}; found={matches}")
        deg = pd.read_csv(matches[0])
        required = {"gene", "logfoldchanges", "pvals_adj"}
        missing = sorted(required - set(deg.columns))
        if missing:
            raise KeyError(f"DEG table lacks required columns {missing}: {matches[0]}")
        logfc = pd.to_numeric(deg["logfoldchanges"], errors="coerce")
        fdr = pd.to_numeric(deg["pvals_adj"], errors="coerce")
        eligible = deg[(logfc > 0) & (fdr < 0.05)].copy()
        genes: list[str] = []
        seen: set[str] = set()
        for gene in eligible["gene"].astype(str):
            if gene in raw_genes and gene not in seen:
                genes.append(gene)
                seen.add(gene)
            if len(genes) == MAX_SCORE_GENES:
                break
        if len(genes) < MIN_SCORE_GENES:
            raise RuntimeError(
                f"Too few positive significant usable DEG score genes for {label}: "
                f"{len(genes)} < required {MIN_SCORE_GENES}; no unfiltered/negative fallback is allowed"
            )
        score_col = f"deg_score__{safe}"
        rsc.tl.score_genes(
            a,
            gene_list=genes,
            score_name=score_col,
            ctrl_size=max(20, len(genes)),
            use_raw=True,
            random_state=SEED,
        )
        score_map[str(label)] = score_col
        score_gene_rows.append({
            "broad_label": str(label),
            "score_column": score_col,
            "source_deg_csv": str(matches[0].relative_to(WF)),
            "n_genes": len(genes),
            "n_positive_fdr_lt_0p05_rows": len(eligible),
            "genes": ";".join(genes),
            "selection_rule": (
                f"first up to {MAX_SCORE_GENES} unique raw-available genes with "
                f"logfoldchanges>0 and pvals_adj<0.05; require at least {MIN_SCORE_GENES}"
            ),
            "insufficient_gene_rule": "stop before publishing outputs; no fallback",
            "expression_layer": "adata.raw normalized_log",
            "backend": "rapids-singlecell GPU",
        })
    cp.get_default_memory_pool().free_all_blocks()
    score_cols = [score_map[x] for x in labels]
    values = a.obs[score_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite marker scores")
    score_frame = pd.DataFrame(values, index=a.obs_names, columns=labels)
    rank_frame = score_frame.rank(axis=0, ascending=False, method="average")
    rank_pct_frame = score_frame.rank(axis=0, ascending=False, pct=True, method="average")
    for j, label in enumerate(labels):
        safe = safe_label(str(label))
        a.obs[f"{safe}_score"] = values[:, j]
        a.obs[f"{safe}_score_rank"] = rank_frame.iloc[:, j].to_numpy(dtype=float)
        a.obs[f"{safe}_score_rank_pct"] = rank_pct_frame.iloc[:, j].to_numpy(dtype=float)
    highest_idx = np.argmax(values, axis=1)
    rank_order = np.argsort(rank_pct_frame.to_numpy(dtype=float), axis=1, kind="stable")
    best_idx = rank_order[:, 0]
    second_idx = rank_order[:, 1]
    a.obs["highest_score_type"] = pd.Categorical([labels[i] for i in highest_idx], categories=labels)
    a.obs["best_rank_type_global"] = pd.Categorical([labels[i] for i in best_idx], categories=labels)
    a.obs["best_rank_score"] = values[np.arange(len(values)), best_idx]
    a.obs["second_best_rank_score"] = values[np.arange(len(values)), second_idx]
    rank_pct_values = rank_pct_frame.to_numpy(dtype=float)
    a.obs["best_rank_pct"] = rank_pct_values[np.arange(len(values)), best_idx]
    a.obs["second_best_rank_pct"] = rank_pct_values[np.arange(len(values)), second_idx]
    a.obs["best_rank_score_margin"] = a.obs["second_best_rank_pct"] - a.obs["best_rank_pct"]
    a.obs["n_best_rank_ties"] = (
        rank_pct_values == rank_pct_values.min(axis=1, keepdims=True)
    ).sum(axis=1)
    a.obs["score_rank_consistent"] = a.obs["leiden_coarse"].astype(str).to_numpy() == a.obs["best_rank_type_global"].astype(str).to_numpy()
    a.uns["score_rank_qc"] = {
        "assigned_label": "leiden_coarse", "best_rank_label": "best_rank_type_global",
        "keep_rule": "leiden_coarse == best_rank_type_global",
        "score_source": (
            f"first up to {MAX_SCORE_GENES} unique raw-available genes per broad label with "
            f"logfoldchanges>0 and pvals_adj<0.05; minimum {MIN_SCORE_GENES}; score_genes use_raw=True"
        ),
        "rank_rule": "within each score column across cells, descending percentile; smallest percentile wins per cell",
        "unfiltered_retained_for_audit": True, "random_seed": SEED,
        "score_backend": "rapids-singlecell GPU", "run_version": RUN_VERSION,
    }
    keep = a.obs["score_rank_consistent"].to_numpy(dtype=bool)
    if keep.sum() == 0:
        raise RuntimeError("Score-rank filter removed all cells")
    b = a[keep].copy()
    STAGE_H5AD_DIR.mkdir(parents=True, exist_ok=False)
    staged_all = STAGE_H5AD_DIR / OUT_ALL.name
    staged_keep = STAGE_H5AD_DIR / OUT_KEEP.name
    write_h5ad_new(a, staged_all)
    write_h5ad_new(b, staged_keep)
    STAGE_TABLE.mkdir(parents=True, exist_ok=False)
    per_cell = pd.DataFrame({
        "cell_id": a.obs_names.astype(str), "leiden_coarse": a.obs["leiden_coarse"].astype(str).to_numpy(),
        "cell_type": a.obs["cell_type"].astype(str).to_numpy(), "best_rank_type_global": a.obs["best_rank_type_global"].astype(str).to_numpy(),
        "best_rank_score": a.obs["best_rank_score"].to_numpy(), "best_rank_score_margin": a.obs["best_rank_score_margin"].to_numpy(),
        "n_best_rank_ties": a.obs["n_best_rank_ties"].to_numpy(), "score_rank_consistent": keep,
    })
    per_cell.to_csv(STAGE_TABLE / "score_rank_qc_per_cell.csv", index=False)
    per_cell[keep].to_csv(STAGE_TABLE / "score_rank_qc_consistent_cells.csv", index=False)
    per_cell[~keep].to_csv(STAGE_TABLE / "score_rank_qc_inconsistent_cells.csv", index=False)
    summary = (per_cell.groupby("leiden_coarse", observed=True)["score_rank_consistent"]
               .agg(n_cells="size", n_consistent="sum").reset_index())
    summary["n_inconsistent"] = summary["n_cells"] - summary["n_consistent"]
    summary["consistent_fraction"] = summary["n_consistent"] / summary["n_cells"]
    summary.to_csv(STAGE_TABLE / "score_rank_qc_summary_by_leiden_coarse.csv", index=False)
    pd.DataFrame(score_gene_rows).to_csv(STAGE_TABLE / "score_gene_sets_from_leiden_coarse_degs.csv", index=False)
    pd.DataFrame([{"score_column": score_map[x], "broad_label": x} for x in labels]).to_csv(STAGE_TABLE / "score_column_label_mapping.csv", index=False)
    STAGE_FIG.mkdir(parents=True, exist_ok=False)
    sns.set_theme(style="ticks"); sc.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True); sc.settings.figdir = STAGE_FIG
    for suffix in ["pdf", "svg"]:
        sc.pl.umap(a, color=["leiden_coarse", "best_rank_type_global", "score_rank_consistent"],
                   ncols=3, wspace=0.45, frameon=True, show=False, save=f"_score_rank_qc.{suffix}")
        sc.pl.umap(b, color="leiden_coarse", frameon=True, show=False, save=f"_consistent_cells.{suffix}")
    validation = {"all_passed": True, "n_cells_unfiltered": a.n_obs, "n_cells_consistent": b.n_obs,
                  "n_cells_inconsistent": int((~keep).sum()), "consistent_fraction": float(keep.mean()),
                  "all_consistent_handoff": bool(b.obs["score_rank_consistent"].all()),
                  "marker_rule": "logfoldchanges>0 and pvals_adj<0.05",
                  "minimum_marker_genes_per_label": MIN_SCORE_GENES,
                  "unfiltered_sha256": sha256(staged_all), "consistent_sha256": sha256(staged_keep),
                  "final_h5ad_directory_published_last": True}
    (STAGE_TABLE / "score_rank_qc_completion.json").write_text(json.dumps(validation, indent=2) + "\n")
    versions = {p: version(p) for p in ["anndata", "scanpy", "numpy", "pandas", "rapids-singlecell", "cupy-cuda12x"]}
    versions.update({"python": platform.python_version(), "seed": SEED})
    (STAGE_TABLE / "package_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    (STAGE_TABLE / "readme.txt").write_text(
        "Broad score gene sets use only saved leiden_coarse DEGs with logfoldchanges>0 and pvals_adj<0.05.\n"
        f"Up to {MAX_SCORE_GENES} genes are used per label; fewer than {MIN_SCORE_GENES} stops without fallback or publication.\n"
        "score_genes uses adata.raw through rapids-singlecell GPU; descending within-score percentiles are compared per cell and the smallest percentile wins.\n"
        "The unfiltered object is retained; the default downstream handoff keeps only leiden_coarse == best_rank_type_global.\n"
        "Tables/figures are validated before the complete h5ad directory is atomically published last.\n"
    )
    if not list(STAGE_FIG.glob("*.pdf")) or not list(STAGE_FIG.glob("*.svg")):
        raise RuntimeError("Score-rank PDF/SVG staging outputs are incomplete")
    unexpected = [path for path in STAGE_FIG.iterdir() if path.suffix.lower() not in {".pdf", ".svg"}]
    if unexpected:
        raise RuntimeError(f"Unexpected score-rank figure format: {unexpected}")
    if not staged_all.is_file() or not staged_keep.is_file() or not (STAGE_TABLE / "score_rank_qc_completion.json").is_file():
        raise RuntimeError("Score-rank staged artifact validation failed")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    OUT_ALL.parent.parent.mkdir(parents=True, exist_ok=True)
    os.replace(STAGE_TABLE, TABLE)
    os.replace(STAGE_FIG, FIG)
    # Atomically publish both h5ads together as the final artifact directory.
    os.replace(STAGE_H5AD_DIR, OUT_ALL.parent)


if __name__ == "__main__":
    main()

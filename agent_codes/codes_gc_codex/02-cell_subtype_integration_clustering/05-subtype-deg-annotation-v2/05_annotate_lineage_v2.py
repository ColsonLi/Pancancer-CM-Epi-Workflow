#!/usr/bin/env python3
"""GC-specific DEG-assisted annotation of one v2 lineage clustering."""

from __future__ import annotations

import argparse
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

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

CODE_DIR = Path(__file__).resolve().parent
WF = CODE_DIR.parents[2]
MODULE = "02-cell_subtype_integration_clustering"
SEED = 42
REVIEW_DEPTH = 300

PREFIX = {
    "b_cells": "B", "endothelial_cells": "Endo", "epithelial_cells": "Epi",
    "mast_cells": "Mast", "myeloid_cells": "Mye", "plasma_cells": "Plasma",
    "stromal_cells": "Stromal", "t_cells": "T",
}
ABBREV = {
    "b_cells": "b", "endothelial_cells": "endo", "epithelial_cells": "epi",
    "mast_cells": "mast", "myeloid_cells": "mye", "plasma_cells": "plasma",
    "stromal_cells": "stromal", "t_cells": "t",
}

# Gastric-cancer atlas marker panel. Generic lineage markers are retained, but
# gastric epithelial identities and tumor-associated immune/stromal states are
# explicitly represented rather than using a kidney/example panel.
IDENTITY_MARKERS = {
    "epithelial_cells": {
        "pit_mucous": {"MUC5AC", "GKN1", "GKN2", "TFF1", "KLF4"},
        "mucous_neck_SPEM": {"MUC6", "TFF2", "AQP5", "WFDC2", "LYZ"},
        "chief": {"PGA3", "PGA4", "PGA5", "PGC", "LIPF", "GIF"},
        "parietal": {"ATP4A", "ATP4B", "KCNQ1", "KCNE2", "CLIC6"},
        "intestinal_enterocyte": {"FABP1", "FABP2", "APOA1", "APOA4", "SI", "ALDOB", "KRT20", "CDX2"},
        "goblet": {"MUC2", "TFF3", "CLCA1", "SPINK4", "ZG16"},
        "enteroendocrine": {"CHGA", "CHGB", "SCG5", "NEUROD1", "PAX6", "PCSK1N"},
        "tuft": {"POU2F3", "TRPM5", "AVIL", "IL17RB", "SOX9"},
        "basal_tumor_like": {"KRT17", "KRT5", "KRT14", "TP63", "SFN"},
        "malignant_like": {"CEACAM5", "CEACAM6", "EPCAM", "KRT19", "KRT8", "KRT18", "TACSTD2"},
    },
    "t_cells": {
        "CD4_naive": {"CCR7", "LTB", "MAL", "TCF7", "LEF1", "SELL"},
        "CD4_memory": {"IL7R", "LTB", "MALAT1", "CXCR4", "LTST1"},
        "Treg": {"FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2"},
        "CD8_cytotoxic": {"CD8A", "CD8B", "CCL5", "NKG7", "GZMK", "GZMH"},
        "exhausted_T": {"PDCD1", "HAVCR2", "LAG3", "TOX", "CXCL13", "TIGIT"},
        "NK": {"NKG7", "GNLY", "PRF1", "KLRD1", "FCGR3A", "XCL1", "XCL2"},
    },
    "b_cells": {
        "naive_B": {"MS4A1", "CD79A", "TCL1A", "IGHM", "IGHD", "HVCN1"},
        "memory_B": {"CD79A", "CD37", "CD27", "AIM2", "BANK1", "CD44"},
        "germinal_center_B": {"BCL6", "AICDA", "RGS13", "MEF2B", "CD83"},
        "plasmablast_like": {"MZB1", "JCHAIN", "SDC1", "DERL3", "XBP1", "IGHG1"},
    },
    "plasma_cells": {
        "plasma_IgA": {"MZB1", "JCHAIN", "SDC1", "DERL3", "XBP1", "IGHA1", "IGHA2"},
        "plasma_IgG": {"MZB1", "JCHAIN", "SDC1", "XBP1", "IGHG1", "IGHG3"},
        "plasmablast": {"MZB1", "JCHAIN", "DERL3", "XBP1", "MKI67", "STMN1"},
        "B_like": {"MS4A1", "CD79A", "CD74", "HLA-DRA", "CD37"},
    },
    "myeloid_cells": {
        "classical_monocyte": {"S100A8", "S100A9", "LYZ", "CTSS", "FCN1", "VCAN"},
        "nonclassical_monocyte": {"LST1", "FCGR3A", "MS4A7", "IFITM3", "LILRB1"},
        "C1Q_macrophage": {"C1QA", "C1QB", "C1QC", "APOE", "CD68", "CTSB"},
        "SPP1_macrophage": {"SPP1", "APOC1", "TREM2", "GPNMB", "CTSL"},
        "CD1C_DC": {"FCER1A", "CD1C", "CLEC10A", "CST3", "HLA-DQA1"},
        "LAMP3_DC": {"LAMP3", "CCR7", "FSCN1", "CCL19", "CCL22"},
        "pDC": {"GZMB", "GZMB", "JCHAIN", "GZMB", "TCF4", "IRF7"},
        "neutrophil": {"CSF3R", "FCGR3B", "CXCR2", "FPR1", "MNDA"},
    },
    "mast_cells": {
        "MC_T": {"TPSAB1", "TPSB2", "KIT", "MS4A2", "HDC"},
        "MC_TC": {"TPSAB1", "TPSB2", "CPA3", "CMA1", "CTSG"},
        "activated_mast": {"IL1RL1", "HDC", "LTC4S", "AREG", "JUN", "FOS"},
    },
    "endothelial_cells": {
        "capillary": {"CA4", "RGCC", "VWF", "PECAM1", "EMCN", "KDR"},
        "arterial": {"EFNB2", "GJA5", "SOX17", "FBLN5", "HEY1"},
        "venous_HEV": {"ACKR1", "SELP", "VWF", "NR2F2", "VCAM1"},
        "lymphatic": {"PROX1", "PDPN", "LYVE1", "FLT4", "CCL21"},
        "tip_angiogenic": {"ESM1", "KDR", "ANGPT2", "APLNR", "PGF"},
    },
    "stromal_cells": {
        "fibroblast": {"COL1A1", "COL1A2", "DCN", "COL3A1", "LUM", "PDGFRA"},
        "myofibroblast": {"ACTA2", "TAGLN", "MYL9", "TPM2", "COL1A1"},
        "pericyte": {"RGS5", "CSPG4", "MCAM", "PDGFRB", "ABCC9", "KCNJ8"},
        "smooth_muscle": {"MYH11", "ACTG2", "CNN1", "DES", "TAGLN"},
        "telocyte": {"PDGFRA", "COL15A1", "FOXL1", "WNT2B", "BMP4"},
        "iCAF": {"CXCL12", "IL6", "CFD", "COL14A1", "APOD"},
        "myCAF": {"ACTA2", "TAGLN", "COL1A1", "COL3A1", "POSTN", "FAP"},
    },
}

STATE_MARKERS = {
    "cycling": {"MKI67", "TOP2A", "UBE2C", "CENPF", "STMN1", "PCNA"},
    "interferon_response": {"ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "OAS1", "IFI6"},
    "stress_response": {"FOS", "JUN", "JUNB", "DUSP1", "HSPA1A", "HSPA1B"},
    "hypoxia": {"CA9", "VEGFA", "NDRG1", "ENO1", "LDHA", "BNIP3"},
    "antigen_presentation": {"HLA-DRA", "HLA-DRB1", "HLA-DPA1", "CD74", "B2M"},
    "cytotoxic": {"NKG7", "GNLY", "PRF1", "GZMB", "GZMH", "CCL5"},
    "EMT_like": {"VIM", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "FN1"},
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lineage", required=True, choices=sorted(PREFIX))
    return p.parse_args()


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def write_csv_once(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_h5ad_once(adata: ad.AnnData, path: Path) -> None:
    tmp = path.with_name(f".{path.stem}.partial{path.suffix}")
    if path.exists() or tmp.exists():
        raise FileExistsError(f"Refusing overwrite: {path} / {tmp}")
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def rank_and_export(adata: ad.AnnData, groupby: str, out_dir: Path, key: str) -> dict[str, pd.DataFrame]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="t-test", use_raw=True, key_added=key, pts=True)
    out_dir.mkdir(parents=True, exist_ok=False)
    result = {}
    for group in sorted(adata.obs[groupby].astype(str).unique(), key=lambda x: (len(x), x)):
        x = sc.get.rank_genes_groups_df(adata, group=group, key=key)
        x = x.rename(columns={"names": "gene", "scores": "score"})
        required = ["gene", "score", "logfoldchanges", "pvals", "pvals_adj"]
        x = x[required + [c for c in x.columns if c not in required]]
        x.insert(0, "group", group)
        x.to_csv(out_dir / f"{safe(group)}.csv", index=False)
        result[group] = x
    return result


def positive(df: pd.DataFrame) -> pd.DataFrame:
    x = df[np.isfinite(df.logfoldchanges) & (df.logfoldchanges > 0)].copy()
    return x.sort_values(["pvals_adj", "pvals", "score", "logfoldchanges", "gene"],
                         ascending=[True, True, False, False, True], kind="mergesort")


def mean_vec(x) -> np.ndarray:
    return np.asarray(x.mean(axis=0)).ravel()


def identity_evidence(adata: ad.AnnData, cluster_key: str, cluster: str, top: list[str], lineage: str):
    genes = set(adata.raw.var_names.astype(str))
    mask = adata.obs[cluster_key].astype(str).to_numpy() == cluster
    rows = []
    for identity, marker_set in IDENTITY_MARKERS[lineage].items():
        available = sorted(marker_set & genes)
        if not available:
            continue
        matrix = adata.raw[:, available].X
        delta = mean_vec(matrix[mask]) - mean_vec(matrix[~mask])
        overlap = sorted(set(top[:REVIEW_DEPTH]) & set(available))
        rows.append((len(overlap), float(delta.mean()), identity, overlap[:8], available))
    if not rows:
        raise RuntimeError(f"No GC marker genes available for {lineage}")
    overlap_n, delta, identity, overlap, available = sorted(rows, key=lambda x: (-x[0], -x[1], x[2]))[0]
    evidence_genes = overlap or available[:5]
    evidence = f"markers={';'.join(evidence_genes)};top{REVIEW_DEPTH}_overlap={overlap_n};mean_logexpr_delta={delta:.6g}"
    confidence = "high" if overlap_n >= 2 and delta > 0 else "moderate" if overlap_n >= 1 or delta > 0 else "low"
    return identity, evidence, overlap_n, delta, confidence


def state_evidence(top: list[str]):
    top_set = set(top[:100])
    hits = sorted(((len(top_set & genes), state, sorted(top_set & genes)) for state, genes in STATE_MARKERS.items()),
                  key=lambda x: (-x[0], x[1]))
    count, state, genes = hits[0]
    return (state, ";".join(genes)) if count else ("baseline_or_unresolved", "")


def main() -> None:
    a = args()
    lineage = a.lineage
    params_path = WF / "tables" / MODULE / "04-subtype-selected-clustering-v2" / lineage / "selected_clustering_parameters_v2.json"
    params = json.loads(params_path.read_text())
    cluster_key = params["cluster_key"]
    inp = WF / "h5ad" / MODULE / "04-subtype-selected-clustering-v2" / lineage / f"adata_{lineage}_selected_clustered_v2.h5ad"
    table_dir = WF / "tables" / MODULE / "05-subtype-deg-annotation-v2" / lineage
    fig_dir = WF / "figures" / MODULE / "05-subtype-deg-annotation-v2" / lineage
    out = WF / "h5ad" / MODULE / "05-subtype-deg-annotation-v2" / lineage / f"adata_{ABBREV[lineage]}.h5ad"
    mapping_path = table_dir / "final_subtype_mapping.csv"
    candidate_path = table_dir / "candidate_subtype_annotation_agent_led.csv"
    if any(p.exists() for p in (out, mapping_path, candidate_path)):
        raise FileExistsError(f"v2 annotation output exists for {lineage}")

    adata = ad.read_h5ad(inp)
    if adata.raw is None:
        raise RuntimeError("adata.raw is required")
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
    token = f"pcs{params['n_pcs']}_nn{params['n_neighbors']}_res{str(params['resolution']).replace('.', 'p')}"
    raw_degs = rank_and_export(adata, cluster_key, table_dir / f"degs_{cluster_key}_{token}", "rank_raw_v2")

    used = set()
    rows = []
    for cluster, df in raw_degs.items():
        pos = positive(df)
        if pos.empty:
            raise RuntimeError(f"No positive DEG for cluster {cluster}")
        chosen = None
        for _, deg in pos.iterrows():
            label = f"{PREFIX[lineage]}_{deg.gene}"
            if label not in used:
                chosen = deg
                used.add(label)
                break
        if chosen is None:
            raise RuntimeError(f"No unique positive DEG label for cluster {cluster}")
        top = pos.gene.astype(str).tolist()
        identity, evidence, overlap, delta, confidence = identity_evidence(adata, cluster_key, cluster, top, lineage)
        state, state_genes = state_evidence(top)
        rows.append({
            "cluster": cluster,
            "proposed_cell_subtype": f"{PREFIX[lineage]}_{chosen.gene}",
            "cell_subtype": f"{PREFIX[lineage]}_{chosen.gene}",
            "functional_state": state,
            "selected_prefix": PREFIX[lineage],
            "most_significant_deg_gene": str(chosen.gene),
            "top_deg_score": float(chosen.score),
            "top_deg_logfoldchange": float(chosen.logfoldchanges),
            "top_deg_pvals_adj": float(chosen.pvals_adj),
            "gene_selection_rationale": "most significant positive t-test DEG; next eligible DEG used only if needed for unique label",
            "top_supporting_markers": evidence,
            "marker_supported_lineage": identity,
            "canonical_identity": identity,
            "canonical_top300_overlap": overlap,
            "canonical_mean_logexpr_delta": delta,
            "functional_state_marker_evidence": state_genes,
            "prefix_rationale": "project-recorded GC broad-lineage abbreviation",
            "agent_led_authorization": "user previously authorized agent-led selection and continuation",
            "annotation_confidence": confidence,
            "needs_user_confirmation": confidence == "low",
            "review_depth": REVIEW_DEPTH,
        })
    mapping = pd.DataFrame(rows)
    if mapping.cell_subtype.duplicated().any() or len(mapping) != adata.obs[cluster_key].nunique():
        raise AssertionError("Invalid one-to-one subtype mapping")
    write_csv_once(mapping, candidate_path)
    write_csv_once(mapping.drop(columns=["proposed_cell_subtype"]), mapping_path)

    lookup = mapping.set_index("cluster")
    clusters = adata.obs[cluster_key].astype(str)
    for col in ("cell_subtype", "functional_state", "canonical_identity", "annotation_confidence"):
        adata.obs[col] = clusters.map(lookup[col]).astype("category")
    adata.obs["annotation_source"] = "GC_Block02_v2_DEG_marker_agent_led"
    post = rank_and_export(adata, "cell_subtype", table_dir / f"degs_cell_subtype_{token}", "rank_post_v2")
    write_csv_once(pd.DataFrame({"cell_subtype": sorted(post), "deg_method": "t-test", "use_raw": True}),
                   table_dir / "post_annotation_deg_contract.csv")

    fig_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.autoshow = False
    sc.settings.figdir = str(fig_dir)
    sc.pl.umap(adata, color=["cell_subtype", "canonical_identity", "functional_state"], ncols=1,
               show=False, save="_annotated_v2.pdf")
    sc.pl.umap(adata, color=["cell_subtype", "canonical_identity", "functional_state"], ncols=1,
               show=False, save="_annotated_v2.svg")
    marker_genes = [g for g in mapping.most_significant_deg_gene.astype(str) if g in adata.raw.var_names]
    sc.pl.dotplot(adata, var_names=marker_genes, groupby="cell_subtype", use_raw=True,
                  standard_scale="var", show=False, save="_subtype_markers_v2.pdf")
    sc.pl.dotplot(adata, var_names=marker_genes, groupby="cell_subtype", use_raw=True,
                  standard_scale="var", show=False, save="_subtype_markers_v2.svg")

    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "readme.txt").write_text(
        "GC v2 subtype annotation; no subtype-level cell filtering was performed.\n"
        "Raw and post-annotation DEGs use Scanpy t-test with adata.raw/use_raw=True.\n"
        "All clusters used a common top-300 review depth because the multi-study GC atlas has weak/heterogeneous marker separation.\n"
        "Agent-led final labeling was authorized by the user's earlier instruction to choose and continue.\n"
    )
    (table_dir / "package_versions.txt").write_text(
        f"python={platform.python_version()}\nanndata={ad.__version__}\nscanpy={sc.__version__}\n"
        f"numpy={np.__version__}\npandas={pd.__version__}\ncode={Path(__file__).resolve()}\nseed={SEED}\n"
    )
    write_h5ad_once(adata, out)
    print(json.dumps({"lineage": lineage, "n_cells": adata.n_obs, "n_subtypes": len(mapping),
                      "n_low_confidence": int((mapping.annotation_confidence == 'low').sum()),
                      "output": str(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

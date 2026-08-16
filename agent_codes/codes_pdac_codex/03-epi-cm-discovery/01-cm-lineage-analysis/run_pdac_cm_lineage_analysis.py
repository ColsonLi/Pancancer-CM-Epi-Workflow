#!/usr/bin/env python3
"""Canonical PDAC Block 03 table analysis.

Build sample-by-subtype frequencies from the projected atlas, fit the fixed
status-balanced joint NMF route, assign canonical CM names, derive node/edge
tables, and compute exhaustive Spearman and Pearson Epi-CM associations.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import random
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, nnls
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.multitest import multipletests


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
INPUT_H5AD = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata/adata_anno_cellsubtype.h5ad"
)
TABLE_ROOT = PROJECT_ROOT / "tables/03-epi-cm-discovery/01-cm-lineage-analysis"
PREP_DIR = TABLE_ROOT / "01-prepare-inputs-and-frequency-tables"
NMF_DIR = TABLE_ROOT / "02-balanced-joint-nmf"
NODE_DIR = TABLE_ROOT / "03-cm-classification-nodes-and-edges"
SPEARMAN_DIR = TABLE_ROOT / "04-epi-cm-association-spearman"
PEARSON_DIR = TABLE_ROOT / "05-epi-cm-association-pearson"


@dataclass(frozen=True)
class Config:
    sample_col: str = "sample"
    series_col: str = "series"
    status_col: str = "status"
    celltype_col: str = "cell_type"
    subtype_col: str = "cell_subtype"
    epithelial_cell_type: str = "Epithelial Cells"
    normal_status: str = "normal-like"
    tumor_status: str = "tumor"
    min_non_epi_cells_per_sample: int = 50
    min_epi_cells_per_sample: int = 1
    forced_cm_k: int | None = None
    cm_k_min: int = 2
    cm_k_max: int = 20
    rank_selection_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    nmf_max_iter: int = 3000
    nmf_tol: float = 1e-5
    nmf_alpha_w: float = 0.0
    nmf_alpha_h: float = 1e-3
    nmf_l1_ratio: float = 0.1
    normal_group_total_weight: float = 0.5
    tumor_group_total_weight: float = 0.5
    normal_specific_max_ratio: float = 0.5
    tumor_specific_min_ratio: float = 2.0
    min_active_fraction_for_specific: float = 0.05
    top_n_subtypes: int = 20
    plot_top_n_subtypes: int = 12
    top_n_nodes: int = 10
    edge_r_threshold: float = 0.25


CFG = Config()


def write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def bh_fdr(values) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.any():
        q[valid] = multipletests(p[valid], method="fdr_bh")[1]
    return q


def prefix_cell_type(subtypes: pd.Series) -> pd.Series:
    mapping = {
        "Epi": "Epithelial Cells",
        "T": "T/NK Cells",
        "NK": "T/NK Cells",
        "Mye": "Myeloid Cells",
        "B": "B Cells",
        "Endo": "Endothelial Cells",
        "Str": "Stromal Cells",
        "Mast": "Mast Cells",
    }
    prefix = subtypes.astype(str).str.split("_", n=1).str[0]
    result = prefix.map(mapping)
    if result.isna().any():
        raise ValueError(
            "No broad-cell mapping for projected subtype prefixes: "
            + ",".join(sorted(prefix.loc[result.isna()].unique()))
        )
    return result.astype(str)


def load_obs() -> pd.DataFrame:
    if not INPUT_H5AD.is_file():
        raise FileNotFoundError(INPUT_H5AD)
    a = ad.read_h5ad(INPUT_H5AD, backed="r")
    required = [
        CFG.sample_col,
        CFG.series_col,
        CFG.status_col,
        CFG.celltype_col,
        CFG.subtype_col,
    ]
    missing = [column for column in required if column not in a.obs]
    if missing:
        a.file.close()
        raise KeyError(f"Projected atlas lacks columns: {missing}")
    obs = a.obs[required].copy()
    a.file.close()
    if not obs.index.is_unique or obs[required].isna().any().any():
        raise ValueError("Projected atlas requires unique cell IDs and complete CM metadata")
    for column in required:
        obs[column] = obs[column].astype(str)

    raw_status = obs[CFG.status_col].copy()
    obs[CFG.status_col] = obs[CFG.status_col].replace({"normal": CFG.normal_status})
    status_mapping = pd.DataFrame(
        {"raw_status": raw_status, "canonical_status": obs[CFG.status_col]}
    ).drop_duplicates()
    status_mapping.to_csv(PREP_DIR / "raw_to_canonical_status_mapping.csv", index=False)
    observed_status = set(obs[CFG.status_col].unique())
    expected_status = {CFG.normal_status, CFG.tumor_status}
    if observed_status != expected_status:
        raise ValueError(f"Expected {sorted(expected_status)}, observed {sorted(observed_status)}")
    for column in [CFG.status_col, CFG.series_col]:
        per_sample = obs[[CFG.sample_col, column]].drop_duplicates()
        if per_sample[CFG.sample_col].duplicated().any():
            raise ValueError(f"Samples have conflicting {column} values")

    obs["cell_type_from_subtype_prefix"] = prefix_cell_type(obs[CFG.subtype_col])
    conflicts = obs.loc[
        obs[CFG.celltype_col].ne(obs["cell_type_from_subtype_prefix"]),
        [CFG.sample_col, CFG.celltype_col, CFG.subtype_col, "cell_type_from_subtype_prefix"],
    ]
    conflicts.to_csv(
        PREP_DIR / "cell_subtype_prefix_cell_type_conflicts.csv",
        index=True,
        index_label="cell_id",
    )
    # Prefix-derived identity is the explicit CM matrix compartment contract.
    obs[CFG.celltype_col] = obs["cell_type_from_subtype_prefix"]
    return obs


def prepare_frequency_tables(obs: pd.DataFrame):
    metadata = obs[[CFG.sample_col, CFG.status_col, CFG.series_col]].drop_duplicates()
    sample_status_all = metadata.set_index(CFG.sample_col).sort_index()
    sample_status_all.index.name = "sample"
    epi_mask = obs[CFG.celltype_col].eq(CFG.epithelial_cell_type)
    epi = obs.loc[epi_mask]
    non_epi = obs.loc[~epi_mask]
    if epi.empty or non_epi.empty:
        raise ValueError(f"Empty analysis compartment: epithelial={len(epi)}, non_epi={len(non_epi)}")

    non_epi_counts_all = pd.crosstab(
        non_epi[CFG.sample_col], non_epi[CFG.subtype_col]
    ).sort_index().astype(int)
    epi_counts_all = pd.crosstab(
        epi[CFG.sample_col], epi[CFG.subtype_col]
    ).sort_index().astype(int)
    compartment = pd.DataFrame(index=sample_status_all.index)
    compartment["non_epi_cells"] = (
        non_epi_counts_all.sum(axis=1).reindex(compartment.index).fillna(0).astype(int)
    )
    compartment["epi_cells"] = (
        epi_counts_all.sum(axis=1).reindex(compartment.index).fillna(0).astype(int)
    )
    inclusion = compartment.copy()
    inclusion["has_status"] = True
    inclusion["has_non_epi"] = inclusion["non_epi_cells"].ge(CFG.min_non_epi_cells_per_sample)
    inclusion["has_epi_for_association"] = inclusion["epi_cells"].ge(CFG.min_epi_cells_per_sample)
    inclusion["keep_for_cm"] = inclusion["has_status"] & inclusion["has_non_epi"]
    inclusion["keep_for_epi_cm"] = inclusion["keep_for_cm"] & inclusion[
        "has_epi_for_association"
    ]

    def exclusion_reason(row: pd.Series) -> str:
        reasons = []
        if not row["has_status"]:
            reasons.append("missing_status")
        if not row["has_non_epi"]:
            reasons.append("non_epi_cells_below_50")
        if not row["has_epi_for_association"]:
            reasons.append("epi_cells_below_1_for_association")
        return ";".join(reasons) if reasons else "kept"

    inclusion["exclusion_reason"] = inclusion.apply(exclusion_reason, axis=1)
    keep_cm = inclusion.index[inclusion["keep_for_cm"]]
    keep_epi_cm = inclusion.index[inclusion["keep_for_epi_cm"]]
    if len(keep_cm) < 2:
        raise ValueError("Fewer than two samples pass non-epithelial CM eligibility")
    non_epi_counts = non_epi_counts_all.reindex(keep_cm).fillna(0).astype(int)
    non_epi_frequency = non_epi_counts.div(non_epi_counts.sum(axis=1), axis=0).fillna(0.0)
    epi_counts = epi_counts_all.reindex(keep_epi_cm).fillna(0).astype(int)
    epi_frequency = epi_counts.div(epi_counts.sum(axis=1), axis=0).fillna(0.0)
    sample_status = sample_status_all.loc[keep_cm].copy()
    sample_status["non_epi_cells"] = compartment.loc[keep_cm, "non_epi_cells"]
    sample_status["epi_cells"] = compartment.loc[keep_cm, "epi_cells"]

    non_epi_counts.to_csv(PREP_DIR / "non_epi_subtype_counts.csv")
    non_epi_frequency.to_csv(PREP_DIR / "non_epi_subtype_frequency.csv")
    epi_counts.to_csv(PREP_DIR / "epi_subtype_counts.csv")
    epi_frequency.to_csv(PREP_DIR / "epi_subtype_frequency.csv")
    sample_status.to_csv(PREP_DIR / "sample_status.csv")
    compartment.to_csv(PREP_DIR / "sample_compartment_cell_counts.csv")
    inclusion.to_csv(PREP_DIR / "sample_inclusion_exclusion.csv")

    status_counts = sample_status.groupby(CFG.status_col).size()
    if set(status_counts.index) != {CFG.normal_status, CFG.tumor_status}:
        raise ValueError("Balanced joint NMF requires both normal-like and tumor samples")
    pd.DataFrame(
        {
            "canonical_status": status_counts.index,
            "n_samples": status_counts.values,
            "detected_mode": "tumor_normal",
            "n_total_samples": len(sample_status),
            "status_balance_applied": True,
        }
    ).to_csv(PREP_DIR / "cm_status_mode_detection.csv", index=False)
    write_json(
        {
            "detected_mode": "tumor_normal",
            "n_total_samples": int(len(sample_status)),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "status_balance_applied": True,
            "single_status_routes": "disabled",
        },
        PREP_DIR / "cm_status_mode.json",
    )
    return non_epi_counts, non_epi_frequency, epi_counts, epi_frequency, sample_status, inclusion


def column_minmax_normalize(frequency: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    column_min = frequency.min(axis=0)
    column_range = (frequency.max(axis=0) - column_min).replace(0, np.nan)
    V_column_minmax = ((frequency - column_min) / column_range).fillna(0.0).clip(lower=0.0)
    params = pd.DataFrame({"min": column_min, "range": column_range.fillna(0.0)})
    params.index.name = "cell_subtype"
    if int(V_column_minmax.var(axis=0).gt(0).sum()) == 0:
        raise ValueError("Column-minmax matrix has no non-zero-variance subtype columns")
    return V_column_minmax.astype(float), params


def make_status_balanced_weights(sample_status: pd.DataFrame) -> pd.Series:
    status = sample_status[CFG.status_col].astype(str)
    normal_mask = status.eq(CFG.normal_status)
    tumor_mask = status.eq(CFG.tumor_status)
    if normal_mask.sum() == 0 or tumor_mask.sum() == 0:
        raise ValueError("Both status groups are required for balanced weights")
    weights = pd.Series(0.0, index=sample_status.index, name="joint_nmf_row_weight")
    weights.loc[normal_mask] = CFG.normal_group_total_weight / normal_mask.sum()
    weights.loc[tumor_mask] = CFG.tumor_group_total_weight / tumor_mask.sum()
    return weights / weights.mean()


def make_nmf(k: int, seed: int) -> NMF:
    return NMF(
        n_components=k,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        random_state=seed,
        max_iter=CFG.nmf_max_iter,
        tol=CFG.nmf_tol,
        alpha_W=CFG.nmf_alpha_w,
        alpha_H=CFG.nmf_alpha_h,
        l1_ratio=CFG.nmf_l1_ratio,
    )


def row_l1(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix, dtype=float),
        where=denominator != 0,
    )


def matched_cosine(H_a: np.ndarray, H_b: np.ndarray) -> float:
    similarities = cosine_similarity(row_l1(H_a), row_l1(H_b))
    row_index, column_index = linear_sum_assignment(-similarities)
    return float(similarities[row_index, column_index].mean())


def evaluate_rank(V_weighted: np.ndarray) -> tuple[pd.DataFrame, int]:
    min_rank = max(2, CFG.cm_k_min)
    max_rank = min(CFG.cm_k_max, V_weighted.shape[0], V_weighted.shape[1])
    if max_rank < min_rank:
        raise ValueError(f"Invalid candidate K range {min_rank}..{max_rank}")
    denominator = float(np.square(V_weighted).sum())
    if denominator == 0:
        raise ValueError("Weighted NMF input is all zero")
    rows = []
    for k in range(min_rank, max_rank + 1):
        H_by_seed, errors, explained = [], [], []
        for seed in CFG.rank_selection_seeds:
            model = make_nmf(k, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                W = model.fit_transform(V_weighted)
            H = model.components_
            error = float(np.square(V_weighted - W @ H).sum())
            H_by_seed.append(H)
            errors.append(error)
            explained.append(1.0 - error / denominator)
        stability = [
            matched_cosine(H_by_seed[i], H_by_seed[j])
            for i in range(len(H_by_seed))
            for j in range(i + 1, len(H_by_seed))
        ]
        best_index = int(np.argmin(errors))
        rows.append(
            {
                "k": k,
                "n_seeds": len(CFG.rank_selection_seeds),
                "mean_balanced_explained_fraction": float(np.mean(explained)),
                "best_balanced_explained_fraction": float(np.max(explained)),
                "mean_reconstruction_error": float(np.mean(errors)),
                "best_reconstruction_error": float(np.min(errors)),
                "stability_matched_cosine": float(np.mean(stability)),
                "best_seed": int(CFG.rank_selection_seeds[best_index]),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["selection_score"] = (
        metrics["best_balanced_explained_fraction"]
        + 0.05 * metrics["stability_matched_cosine"]
        - 0.01 * metrics["k"]
    )
    if CFG.forced_cm_k is None:
        selected_k = int(
            metrics.sort_values(
                ["selection_score", "best_balanced_explained_fraction"], ascending=False
            ).iloc[0]["k"]
        )
    else:
        selected_k = int(CFG.forced_cm_k)
        if selected_k not in metrics["k"].tolist():
            raise ValueError(f"forced_cm_k={selected_k} was not evaluated")
    metrics["selected"] = metrics["k"].eq(selected_k)
    return metrics, selected_k


def refit_activity_by_nnls(H: np.ndarray, V_unweighted: np.ndarray) -> np.ndarray:
    design = H.T
    W = np.zeros((V_unweighted.shape[0], H.shape[0]), dtype=float)
    for index in range(V_unweighted.shape[0]):
        W[index, :] = nnls(
            design, V_unweighted[index, :], maxiter=design.shape[1] * 10
        )[0]
    return W


def classify_cms(W_raw: pd.DataFrame, sample_status: pd.DataFrame) -> pd.DataFrame:
    status = sample_status.loc[W_raw.index, CFG.status_col].astype(str)
    normal = W_raw.loc[status.eq(CFG.normal_status)]
    tumor = W_raw.loc[status.eq(CFG.tumor_status)]
    rows = []
    for raw_component in W_raw.columns:
        normal_values = normal[raw_component].astype(float)
        tumor_values = tumor[raw_component].astype(float)
        normal_mean = float(normal_values.mean())
        tumor_mean = float(tumor_values.mean())
        ratio = tumor_mean / normal_mean if normal_mean > 0 else np.inf
        normal_active = float(normal_values.gt(1e-8).mean())
        tumor_active = float(tumor_values.gt(1e-8).mean())
        if ratio >= CFG.tumor_specific_min_ratio and tumor_active >= CFG.min_active_fraction_for_specific:
            cm_class = "tCM"
        elif ratio <= CFG.normal_specific_max_ratio and normal_active >= CFG.min_active_fraction_for_specific:
            cm_class = "normalCM"
        else:
            cm_class = "sharedCM"
        _, p_value = mannwhitneyu(tumor_values, normal_values, alternative="two-sided")
        rows.append(
            {
                "raw_component": raw_component,
                "raw_component_order": int(re.sub(r"\D", "", raw_component)),
                "class": cm_class,
                "normal_like_mean": normal_mean,
                "tumor_mean": tumor_mean,
                "delta": tumor_mean - normal_mean,
                "tumor_to_normal_like_mean_ratio": ratio,
                "normal_like_active_fraction": normal_active,
                "tumor_active_fraction": tumor_active,
                "p": float(p_value),
            }
        )
    classification = pd.DataFrame(rows).sort_values("raw_component_order", kind="stable")
    classification["q"] = bh_fdr(classification["p"])
    classification["global_order"] = np.arange(1, len(classification) + 1)
    classification["class_prefix"] = classification["class"].map(
        {"sharedCM": "s", "normalCM": "n", "tCM": "t"}
    )
    classification["CM"] = (
        classification["class_prefix"] + "_CM" + classification["global_order"].astype(str)
    )
    return classification


def transform_columns(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    data = frame.astype(float).copy()
    if method == "raw":
        return data
    if method == "standard_scale_col":
        minimum = data.min(axis=0)
        value_range = (data.max(axis=0) - minimum).replace(0, np.nan)
        return data.sub(minimum, axis=1).div(value_range, axis=1).fillna(0.0).clip(0, 1)
    if method == "zscore":
        return data.sub(data.mean(axis=0), axis=1).div(
            data.std(axis=0, ddof=0).replace(0, np.nan), axis=1
        ).fillna(0.0)
    if method == "robust":
        median = data.median(axis=0)
        iqr = (data.quantile(0.75) - data.quantile(0.25)).replace(0, np.nan)
        return data.sub(median, axis=1).div(iqr, axis=1).fillna(0.0)
    raise ValueError(method)


def fit_balanced_joint_nmf(non_epi_frequency: pd.DataFrame, sample_status: pd.DataFrame):
    V_column_minmax, minmax_params = column_minmax_normalize(non_epi_frequency)
    sample_weights = make_status_balanced_weights(sample_status.loc[V_column_minmax.index])
    V = V_column_minmax.to_numpy(dtype=float)
    V_weighted = V * np.sqrt(sample_weights.to_numpy(dtype=float))[:, None]
    metrics, selected_k = evaluate_rank(V_weighted)
    best_seed = int(metrics.loc[metrics["selected"], "best_seed"].iloc[0])
    model = make_nmf(selected_k, best_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_transform(V_weighted)
    H_raw = model.components_
    W_raw = refit_activity_by_nnls(H_raw, V)

    raw_components = [f"CM{index + 1}" for index in range(selected_k)]
    W_raw_df = pd.DataFrame(W_raw, index=V_column_minmax.index, columns=raw_components)
    H_raw_df = pd.DataFrame(H_raw, index=raw_components, columns=V_column_minmax.columns)
    classification = classify_cms(W_raw_df, sample_status)
    raw_to_cm = dict(zip(classification["raw_component"], classification["CM"]))
    cm_order = classification["CM"].tolist()
    W_df = W_raw_df.rename(columns=raw_to_cm).loc[:, cm_order]
    H_df = H_raw_df.rename(index=raw_to_cm).loc[cm_order]
    loading = H_df.T.copy()
    loading_fraction = loading.div(
        loading.sum(axis=0).replace(0, np.nan), axis=1
    ).fillna(0.0)

    V_column_minmax.to_csv(NMF_DIR / "non_epi_subtype_frequency_column_minmax.csv")
    minmax_params.to_csv(NMF_DIR / "column_minmax_params.csv")
    sample_weights.to_frame().join(
        sample_status[[CFG.status_col, CFG.series_col, "non_epi_cells"]]
    ).to_csv(NMF_DIR / "group_balanced_sample_weights.csv")
    metrics.to_csv(NMF_DIR / "joint_nmf_k_selection_metrics.csv", index=False)
    write_json(
        {
            "selected_module_k": selected_k,
            "best_seed": best_seed,
            "candidate_Ks": metrics["k"].astype(int).tolist(),
            "rank_selection_seeds": list(CFG.rank_selection_seeds),
            "selection_score": "best_explained + 0.05*stability - 0.01*K",
            "forced": CFG.forced_cm_k is not None,
        },
        NMF_DIR / "selected_module_k.json",
    )
    classification.to_csv(NODE_DIR / "joint_module_classification.csv", index=False)
    classification[
        ["raw_component", "raw_component_order", "CM", "class", "class_prefix", "global_order"]
    ].to_csv(NODE_DIR / "raw_to_canonical_CM_mapping.csv", index=False)
    W_df.to_csv(NMF_DIR / "W_df.csv")
    H_df.to_csv(NMF_DIR / "H_df.csv")
    W_df.T.to_csv(NMF_DIR / "activity_df_CM_by_sample.csv")
    loading.to_csv(NMF_DIR / "loading_df_cell_subtype_by_CM.csv")
    loading_fraction.to_csv(NMF_DIR / "loading_df_cell_subtype_by_CM_fraction.csv")
    loading.to_csv(NMF_DIR / "balanced_joint_cm_subtype_loadings_raw_from_H_df.csv")
    loading_fraction.to_csv(NMF_DIR / "balanced_joint_cm_subtype_loadings_fraction_from_H_df.csv")
    activity = W_df.join(
        sample_status[[CFG.status_col, CFG.series_col, "non_epi_cells", "epi_cells"]]
    )
    activity.to_csv(NMF_DIR / "activity_df_sample_by_CM.csv")
    for method in ["raw", "zscore", "robust", "standard_scale_col"]:
        transform_columns(W_df, method).to_csv(
            NMF_DIR / f"w_df_activity_sample_by_CM_{method}.csv"
        )
        transform_columns(loading, method).T.to_csv(
            NMF_DIR / f"h_df_loading_cell_subtype_by_CM_{method}.csv"
        )
    transform_columns(W_df, "standard_scale_col").to_csv(
        NMF_DIR / "w_df_activity_sample_by_CM_standard_scale_col_with_series_plot_matrix.csv"
    )
    sample_status.loc[W_df.index, [CFG.series_col, CFG.status_col]].rename(
        columns={CFG.series_col: "Series", CFG.status_col: "Status"}
    ).to_csv(
        NMF_DIR / "w_df_activity_sample_by_CM_standard_scale_col_with_series_row_annotations.csv"
    )
    pd.DataFrame(
        [
            ("W_df.csv", "sample x canonical CM"),
            ("H_df.csv", "canonical CM x non-epithelial node"),
            ("activity_df_sample_by_CM.csv", "sample x canonical CM plus metadata"),
            ("loading_df_cell_subtype_by_CM.csv", "non-epithelial node x canonical CM"),
            ("h_df_loading_cell_subtype_by_CM_*.csv", "canonical CM x non-epithelial node"),
        ],
        columns=["table", "orientation"],
    ).to_csv(NMF_DIR / "matrix_orientation.csv", index=False)

    summary_rows = []
    status = sample_status.loc[W_df.index, CFG.status_col].astype(str)
    for cm in cm_order:
        normal_values = W_df.loc[status.eq(CFG.normal_status), cm]
        tumor_values = W_df.loc[status.eq(CFG.tumor_status), cm]
        _, p_value = mannwhitneyu(tumor_values, normal_values, alternative="two-sided")
        summary_rows.append(
            {
                "CM": cm,
                "normal_like_mean": float(normal_values.mean()),
                "normal_like_sd": float(normal_values.std(ddof=1)),
                "tumor_mean": float(tumor_values.mean()),
                "tumor_sd": float(tumor_values.std(ddof=1)),
                "delta_tumor_minus_normal_like": float(tumor_values.mean() - normal_values.mean()),
                "mwu_p": float(p_value),
            }
        )
    status_summary = pd.DataFrame(summary_rows)
    status_summary["mwu_q"] = bh_fdr(status_summary["mwu_p"])
    status_summary.to_csv(
        NODE_DIR / "joint_CM_activity_tumor_vs_normal_mean_sd_summary.csv", index=False
    )
    return V_column_minmax, W_df, H_df, classification, metrics


def lineage_from_subtype(subtype: str) -> str:
    return str(subtype).split("_", 1)[0]


def correlation_and_q(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = list(frame.columns)
    corr = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=float)
    p_matrix = pd.DataFrame(np.zeros((len(columns), len(columns))), index=columns, columns=columns)
    pairs, p_values = [], []
    for node_a, node_b in combinations(columns, 2):
        x, y = frame[node_a].astype(float), frame[node_b].astype(float)
        valid = x.notna() & y.notna()
        if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
            r_value, p_value = np.nan, np.nan
        else:
            r_value, p_value = pearsonr(x[valid], y[valid])
        corr.loc[node_a, node_b] = corr.loc[node_b, node_a] = r_value
        p_matrix.loc[node_a, node_b] = p_matrix.loc[node_b, node_a] = p_value
        pairs.append((node_a, node_b))
        p_values.append(p_value)
    q_values = bh_fdr(p_values)
    q_matrix = pd.DataFrame(np.zeros((len(columns), len(columns))), index=columns, columns=columns)
    for (node_a, node_b), q_value in zip(pairs, q_values):
        q_matrix.loc[node_a, node_b] = q_matrix.loc[node_b, node_a] = q_value
    return corr, p_matrix, q_matrix


def build_nodes_and_edges(V: pd.DataFrame, H: pd.DataFrame, sample_status: pd.DataFrame) -> dict:
    node_rows = []
    for cm in H.index:
        ranked = H.loc[cm].sort_values(ascending=False, kind="stable")
        for rank, (subtype, loading) in enumerate(ranked.items(), start=1):
            node_rows.append(
                {
                    "CM": cm,
                    "cell_subtype": subtype,
                    "loading": float(loading),
                    "rank": rank,
                    "cell_lineage": lineage_from_subtype(subtype),
                }
            )
    all_nodes = pd.DataFrame(node_rows)
    top20 = all_nodes.loc[all_nodes["rank"].le(CFG.top_n_subtypes)].copy()
    top10 = all_nodes.loc[all_nodes["rank"].le(CFG.top_n_nodes)].copy()
    all_nodes.to_csv(NODE_DIR / "joint_cm_cell_subtype_nodes_all_from_H_df.csv", index=False)
    top20.to_csv(NODE_DIR / "joint_cm_cell_subtype_nodes_top20_from_H_df.csv", index=False)
    top10.to_csv(NODE_DIR / "joint_cm_cell_subtype_nodes_top10_from_H_df.csv", index=False)

    status = sample_status.loc[V.index, CFG.status_col].astype(str)
    retained_rows, edge_rows, node_context_rows = [], [], []
    cm_without_retained_nodes = []
    for cm in H.index:
        candidates = (
            top10.loc[top10["CM"].eq(cm)].sort_values("rank")["cell_subtype"].astype(str).tolist()
        )
        context_corr = {
            CFG.normal_status: V.loc[status.eq(CFG.normal_status), candidates].corr(method="pearson"),
            CFG.tumor_status: V.loc[status.eq(CFG.tumor_status), candidates].corr(method="pearson"),
        }
        retained = set()
        for node_a, node_b in combinations(candidates, 2):
            for matrix in context_corr.values():
                r_value = matrix.loc[node_a, node_b]
                if np.isfinite(r_value) and r_value >= CFG.edge_r_threshold:
                    retained.update([node_a, node_b])
        ordered_retained = [node for node in candidates if node in retained]
        if not ordered_retained:
            cm_without_retained_nodes.append(cm)
        for rank, node in enumerate(ordered_retained, start=1):
            retained_rows.append({"CM": cm, "reference_node_rank": rank, "node": node})
            loading_rank = int(
                top10.loc[
                    top10["CM"].eq(cm) & top10["cell_subtype"].eq(node), "rank"
                ].iloc[0]
            )
            for context in [CFG.normal_status, CFG.tumor_status]:
                node_context_rows.append(
                    {
                        "CM": cm,
                        "cell_subtype": node,
                        "loading": float(H.loc[cm, node]),
                        "rank": loading_rank,
                        "cell_lineage": lineage_from_subtype(node),
                        "status_context": context,
                    }
                )
        for context, matrix in context_corr.items():
            for node_a, node_b in combinations(ordered_retained, 2):
                r_value = matrix.loc[node_a, node_b]
                edge_rows.append(
                    {
                        "context": context,
                        "CM": cm,
                        "node_a": node_a,
                        "node_b": node_b,
                        "pearson_r": float(r_value) if np.isfinite(r_value) else np.nan,
                        "edge_pass_r_ge_0.25": bool(
                            np.isfinite(r_value) and r_value >= CFG.edge_r_threshold
                        ),
                    }
                )

        # Top-node correlation matrices are diagnostics and never edge-filtered.
        for context, matrix in context_corr.items():
            _, p_matrix, q_matrix = correlation_and_q(
                V.loc[status.eq(context), candidates]
            )
            matrix.to_csv(
                NODE_DIR / f"top10_node_correlation_{context}_{cm}_no_edge_filter.csv"
            )
            p_matrix.to_csv(
                NODE_DIR / f"top10_node_p_{context}_{cm}_no_edge_filter.csv"
            )
            q_matrix.to_csv(
                NODE_DIR / f"top10_node_q_{context}_{cm}_no_edge_filter.csv"
            )

    membership = pd.DataFrame(
        retained_rows, columns=["CM", "reference_node_rank", "node"]
    )
    membership.to_csv(
        NODE_DIR / "balanced_joint_cm_reference_node_sets_after_edge_threshold.csv", index=False
    )
    edges = pd.DataFrame(
        edge_rows,
        columns=[
            "context",
            "CM",
            "node_a",
            "node_b",
            "pearson_r",
            "edge_pass_r_ge_0.25",
        ],
    )
    edges.to_csv(NODE_DIR / "status_specific_nodeplot_edges.csv", index=False)
    context_nodes = pd.DataFrame(
        node_context_rows,
        columns=[
            "CM",
            "cell_subtype",
            "loading",
            "rank",
            "cell_lineage",
            "status_context",
        ],
    )
    for context, filename in [
        (CFG.normal_status, "normal_like_network_nodes_from_H_df.csv"),
        (CFG.tumor_status, "tumor_network_nodes_from_H_df.csv"),
    ]:
        context_nodes.loc[context_nodes["status_context"].eq(context)].to_csv(
            NODE_DIR / filename, index=False
        )

    tumor_edges = edges.loc[
        edges["context"].eq(CFG.tumor_status) & edges["edge_pass_r_ge_0.25"]
    ].copy()
    normal_pass = {
        (row.CM, *sorted([str(row.node_a), str(row.node_b)]))
        for row in edges.loc[
            edges["context"].eq(CFG.normal_status) & edges["edge_pass_r_ge_0.25"]
        ].itertuples(index=False)
    }
    tumor_edges["edge_origin"] = [
        "shared"
        if (row.CM, *sorted([str(row.node_a), str(row.node_b)])) in normal_pass
        else "tumor_only"
        for row in tumor_edges.itertuples(index=False)
    ]
    tumor_edges.to_csv(NODE_DIR / "tumor_centric_edges_with_origin.csv", index=False)
    write_json(
        {
            "top_n_subtypes": CFG.top_n_subtypes,
            "top_n_nodes": CFG.top_n_nodes,
            "edge_r_threshold": CFG.edge_r_threshold,
            "n_membership_rows": int(len(membership)),
            "n_edge_rows": int(len(edges)),
            "cms_without_edge_retained_nodes": cm_without_retained_nodes,
        },
        NODE_DIR / "cm_nodes_edges_summary.json",
    )
    return {
        "membership_rows": len(membership),
        "edge_rows": len(edges),
        "cms_without_retained_nodes": cm_without_retained_nodes,
    }


def safe_correlation(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float, int]:
    valid = x.notna() & y.notna()
    n_samples = int(valid.sum())
    if n_samples < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan, np.nan, n_samples
    if method == "spearman":
        statistic, p_value = spearmanr(x[valid], y[valid])
    elif method == "pearson":
        statistic, p_value = pearsonr(x[valid], y[valid])
    else:
        raise ValueError(method)
    return float(statistic), float(p_value), n_samples


def run_association(
    epi_frequency: pd.DataFrame,
    W: pd.DataFrame,
    sample_status: pd.DataFrame,
    inclusion: pd.DataFrame,
    method: str,
    output_dir: Path,
) -> dict:
    eligible = inclusion.index[inclusion["keep_for_epi_cm"]]
    common = (
        epi_frequency.index.intersection(W.index)
        .intersection(sample_status.index)
        .intersection(eligible)
    )
    E = epi_frequency.loc[common].astype(float)
    C = W.loc[common].astype(float)
    status = sample_status.loc[common, CFG.status_col].astype(str)
    statistic_column = "rho" if method == "spearman" else "r"
    all_rows = []
    context_samples = {
        "overall": common,
        CFG.normal_status: status.index[status.eq(CFG.normal_status)],
        CFG.tumor_status: status.index[status.eq(CFG.tumor_status)],
    }
    context_counts = {}
    for context, samples in context_samples.items():
        rows = []
        for epithelial_subtype in E.columns:
            for cm in C.columns:
                statistic, p_value, n_samples = safe_correlation(
                    E.loc[samples, epithelial_subtype], C.loc[samples, cm], method
                )
                rows.append(
                    {
                        "method": method,
                        "status_context": context,
                        "CM": cm,
                        "epi_subtype": epithelial_subtype,
                        statistic_column: statistic,
                        "p_value": p_value,
                        "n_samples": n_samples,
                    }
                )
        context_table = pd.DataFrame(rows)
        context_table["q_value"] = bh_fdr(context_table["p_value"])
        all_rows.append(context_table)
        context_counts[context] = len(context_table)
        value_matrix = context_table.pivot(
            index="epi_subtype", columns="CM", values=statistic_column
        ).reindex(index=E.columns, columns=C.columns)
        p_matrix = context_table.pivot(
            index="epi_subtype", columns="CM", values="p_value"
        ).reindex(index=E.columns, columns=C.columns)
        q_matrix = context_table.pivot(
            index="epi_subtype", columns="CM", values="q_value"
        ).reindex(index=E.columns, columns=C.columns)
        if method == "spearman":
            stem = f"balanced_joint_cm_epi_cm_association_{context}"
            value_matrix.to_csv(output_dir / f"{stem}_rho_matrix.csv")
        else:
            stem = f"balanced_joint_cm_epi_cm_association_{context}_pearson"
            value_matrix.to_csv(output_dir / f"{stem}_r_matrix.csv")
        p_matrix.to_csv(output_dir / f"{stem}_p_matrix.csv")
        q_matrix.to_csv(output_dir / f"{stem}_q_matrix.csv")
        context_table.to_csv(output_dir / f"{stem}_{method}_long.csv", index=False)
        context_table.loc[context_table["q_value"].lt(0.05)].to_csv(
            output_dir / f"{stem}_{method}_significant_q0.05.csv", index=False
        )
    combined = pd.concat(all_rows, ignore_index=True)
    expected_pairs = len(E.columns) * len(C.columns)
    if any(count != expected_pairs for count in context_counts.values()):
        raise RuntimeError(f"{method}: incomplete epithelial x CM Cartesian product")
    combined.to_csv(output_dir / f"epi_cm_{method}_all_contexts_all_pairs_long.csv", index=False)
    summary = {
        "method": method,
        "statistic_column": statistic_column,
        "contexts": list(context_samples),
        "n_epi_subtypes": int(E.shape[1]),
        "n_cms": int(C.shape[1]),
        "pairs_per_context": int(expected_pairs),
        "rows_total": int(len(combined)),
        "bh_fdr_scope": "separately across the full epithelial_subtype x CM Cartesian product within each status context",
    }
    write_json(summary, output_dir / f"association_{method}_summary.json")
    return summary


def write_provenance(
    obs: pd.DataFrame,
    non_epi_frequency: pd.DataFrame,
    epi_frequency: pd.DataFrame,
    W: pd.DataFrame,
    H: pd.DataFrame,
    metrics: pd.DataFrame,
    node_summary: dict,
    spearman_summary: dict,
    pearson_summary: dict,
) -> None:
    parameters = {
        "input_h5ad": str(INPUT_H5AD),
        "code_file": str(Path(__file__).resolve()),
        "random_seed": SEED,
        "config": asdict(CFG),
        "n_cells": int(len(obs)),
        "n_samples_cm": int(W.shape[0]),
        "n_non_epi_nodes": int(H.shape[1]),
        "n_epi_subtypes": int(epi_frequency.shape[1]),
        "selected_k": int(W.shape[1]),
        "status_values": sorted(obs[CFG.status_col].unique().tolist()),
        "non_epi_resolution": "all finalized non-epithelial direct-clustering subtypes",
        "gpu_note": "Canonical sklearn NMF, scipy NNLS and correlation tests are CPU algorithms",
        "formal_route": "status-balanced joint NMF only; single-status routes disabled",
    }
    write_json(parameters, TABLE_ROOT / "run_parameters.json")
    versions = [
        "environment_category=rapids_environment_cpu_algorithms",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"numpy={importlib.metadata.version('numpy')}",
        f"pandas={importlib.metadata.version('pandas')}",
        f"scipy={importlib.metadata.version('scipy')}",
        f"scikit-learn={importlib.metadata.version('scikit-learn')}",
        f"statsmodels={importlib.metadata.version('statsmodels')}",
        f"anndata={importlib.metadata.version('anndata')}",
        f"random_seed={SEED}",
        f"code_file={Path(__file__).resolve()}",
    ]
    (TABLE_ROOT / "package_versions.txt").write_text("\n".join(versions) + "\n", encoding="utf-8")
    (TABLE_ROOT / "readme.txt").write_text(
        "Canonical PDAC Block 03. All Epi_*, T_*, NK_*, Mye_*, B_*, Endo_*, Str_*, "
        "and Mast_* labels are finalized Block 02 direct-clustering subtypes. Balanced joint NMF uses only "
        "keep_for_cm samples; epithelial eligibility is applied later only to Epi-CM "
        "associations. Spearman and Pearson branches contain separate exhaustive Cartesian "
        "products and separate BH-FDR calculations.\n",
        encoding="utf-8",
    )
    validation = {
        "input_h5ad_exists": INPUT_H5AD.is_file(),
        "n_cells": int(len(obs)),
        "n_samples_cm": int(W.shape[0]),
        "n_non_epi_nodes": int(H.shape[1]),
        "candidate_k_rows": int(len(metrics)),
        "selected_k_rows": int(metrics["selected"].sum()),
        "W_orientation": [int(W.shape[0]), int(W.shape[1])],
        "H_orientation": [int(H.shape[0]), int(H.shape[1])],
        "all_public_cm_names_canonical": bool(
            all(re.fullmatch(r"[snt]_CM\d+", name) for name in W.columns)
        ),
        "node_summary": node_summary,
        "spearman_summary": spearman_summary,
        "pearson_summary": pearson_summary,
        "single_status_routes_disabled": True,
        "validation_passed": True,
    }
    write_json(validation, TABLE_ROOT / "block03_analysis_validation.json")


def main() -> None:
    if TABLE_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite Block 03 analysis outputs: {TABLE_ROOT}")
    for directory in [PREP_DIR, NMF_DIR, NODE_DIR, SPEARMAN_DIR, PEARSON_DIR]:
        directory.mkdir(parents=True, exist_ok=False)
    obs = load_obs()
    (
        _,
        non_epi_frequency,
        _,
        epi_frequency,
        sample_status,
        inclusion,
    ) = prepare_frequency_tables(obs)
    V, W, H, _, metrics = fit_balanced_joint_nmf(non_epi_frequency, sample_status)
    node_summary = build_nodes_and_edges(V, H, sample_status)
    spearman_summary = run_association(
        epi_frequency, W, sample_status, inclusion, "spearman", SPEARMAN_DIR
    )
    pearson_summary = run_association(
        epi_frequency, W, sample_status, inclusion, "pearson", PEARSON_DIR
    )
    write_provenance(
        obs,
        non_epi_frequency,
        epi_frequency,
        W,
        H,
        metrics,
        node_summary,
        spearman_summary,
        pearson_summary,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "n_cells": len(obs),
                "n_samples_cm": len(W),
                "n_epi_subtypes": epi_frequency.shape[1],
                "n_non_epi_nodes": H.shape[1],
                "selected_k": W.shape[1],
                "canonical_CMs": W.columns.tolist(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

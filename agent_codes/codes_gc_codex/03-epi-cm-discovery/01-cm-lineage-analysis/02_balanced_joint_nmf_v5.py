#!/usr/bin/env python3
"""Canonical status-balanced joint NMF for GC non-epithelial subtypes."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, nnls
from scipy.stats import mannwhitneyu
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.multitest import multipletests


CODE_DIR = Path(__file__).resolve().parent
WF = CODE_DIR.parents[2]
BLOCK = "03-epi-cm-discovery"
PREFLIGHT_TAG = "agent-led-current-selection-v5-preflight-r2"
TAG = "agent-led-current-selection-v5-nmf-r1"
SEED = 42
RANK_SELECTION_SEEDS = (0, 1, 2, 3, 4)
K_RANGE = (2, 20)
NMF_MAX_ITER = 3000
NMF_TOL = 1e-5
NMF_ALPHA_W = 0.0
NMF_ALPHA_H = 1e-3
NMF_L1_RATIO = 0.1
NORMAL_TOTAL_WEIGHT = 0.5
TUMOR_TOTAL_WEIGHT = 0.5
NORMAL_SPECIFIC_MAX_RATIO = 0.5
TUMOR_SPECIFIC_MIN_RATIO = 2.0
MIN_ACTIVE_FRACTION = 0.05

INPUT = (
    WF / "tables" / BLOCK / "01-cm-lineage-analysis"
    / "01_prepare_inputs_and_frequency_tables" / PREFLIGHT_TAG
)
OUT = (
    WF / "tables" / BLOCK / "01-cm-lineage-analysis"
    / "02_balanced_joint_nmf" / TAG
)


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


def column_minmax_normalize(frame: pd.DataFrame):
    column_min = frame.min(axis=0)
    column_range = (frame.max(axis=0) - column_min).replace(0, np.nan)
    normalized = ((frame - column_min) / column_range).fillna(0.0).clip(lower=0.0)
    params = pd.DataFrame({"min": column_min, "range": column_range.fillna(0.0)})
    params.index.name = "cell_subtype"
    return normalized.astype(float), params


def make_nmf(k: int, seed: int) -> NMF:
    return NMF(
        n_components=k, init="nndsvda", solver="cd", beta_loss="frobenius",
        random_state=seed, max_iter=NMF_MAX_ITER, tol=NMF_TOL,
        alpha_W=NMF_ALPHA_W, alpha_H=NMF_ALPHA_H, l1_ratio=NMF_L1_RATIO,
    )


def row_l1(matrix: np.ndarray) -> np.ndarray:
    denom = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denom, out=np.zeros_like(matrix, dtype=float), where=denom != 0)


def matched_cosine(left: np.ndarray, right: np.ndarray) -> float:
    similarity = cosine_similarity(row_l1(left), row_l1(right))
    rows, columns = linear_sum_assignment(-similarity)
    return float(similarity[rows, columns].mean())


def stability(h_by_seed: list[np.ndarray]) -> float:
    values = [
        matched_cosine(h_by_seed[i], h_by_seed[j])
        for i in range(len(h_by_seed)) for j in range(i + 1, len(h_by_seed))
    ]
    return float(np.mean(values))


def robust_rows(frame: pd.DataFrame) -> pd.DataFrame:
    median = frame.median(axis=1)
    mad = frame.sub(median, axis=0).abs().median(axis=1).replace(0, np.nan)
    return frame.sub(median, axis=0).div(mad, axis=0).fillna(0.0)


def zscore_rows(frame: pd.DataFrame) -> pd.DataFrame:
    std = frame.std(axis=1, ddof=0).replace(0, np.nan)
    return frame.sub(frame.mean(axis=1), axis=0).div(std, axis=0).fillna(0.0)


def standard_scale_columns(frame: pd.DataFrame) -> pd.DataFrame:
    minimum = frame.min(axis=0)
    span = (frame.max(axis=0) - minimum).replace(0, np.nan)
    return frame.sub(minimum, axis=1).div(span, axis=1).fillna(0.0)


def main() -> None:
    if (OUT / "selected_module_k.json").exists():
        raise FileExistsError(OUT / "selected_module_k.json")
    random.seed(SEED)
    np.random.seed(SEED)
    frequency = pd.read_csv(INPUT / "non_epi_subtype_frequency.csv").set_index("sample")
    status = pd.read_csv(INPUT / "sample_status.csv").set_index("sample")
    inclusion = pd.read_csv(INPUT / "sample_inclusion_exclusion.csv").set_index("sample")
    mode = json.loads((INPUT / "cm_status_mode.json").read_text())["detected_mode"]
    if mode != "tumor_normal":
        raise RuntimeError(f"This canonical balanced route requires tumor_normal, got {mode}")
    status = status.loc[frequency.index]
    if set(status.status) != {"tumor", "normal-like"}:
        raise RuntimeError("Both tumor and normal-like samples are required")
    non_epi_cells = inclusion.loc[frequency.index, "non_epi_cells"].astype(int)
    if frequency.shape != (93, 50):
        raise RuntimeError(f"Unexpected NMF input shape: {frequency.shape}")

    V_column_minmax, minmax_params = column_minmax_normalize(frequency)
    valid_columns = V_column_minmax.columns[V_column_minmax.var(axis=0) > 0]
    if len(valid_columns) == 0:
        raise RuntimeError("Column-minmax matrix has no non-zero-variance subtype columns")
    status_values = status.status.astype(str)
    normal_mask = status_values.eq("normal-like")
    tumor_mask = status_values.eq("tumor")
    weights = pd.Series(0.0, index=frequency.index, name="joint_nmf_row_weight")
    weights.loc[normal_mask] = NORMAL_TOTAL_WEIGHT / int(normal_mask.sum())
    weights.loc[tumor_mask] = TUMOR_TOTAL_WEIGHT / int(tumor_mask.sum())
    weights = weights / weights.mean()
    V = V_column_minmax.to_numpy(dtype=float)
    V_weighted = np.sqrt(weights.to_numpy())[:, None] * V
    denominator = float(np.square(V_weighted).sum())
    if denominator <= 0:
        raise RuntimeError("Weighted NMF input is all zero")
    max_k = min(K_RANGE[1], V_weighted.shape[0], len(valid_columns))
    candidate_ks = list(range(max(2, K_RANGE[0]), max_k + 1))

    summary_rows = []
    seed_rows = []
    best_h_by_k = {}
    for k in candidate_ks:
        h_by_seed = []
        errors = []
        explained = []
        iterations = []
        for seed in RANK_SELECTION_SEEDS:
            model = make_nmf(k, seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                W = model.fit_transform(V_weighted)
            H = model.components_
            error = float(np.square(V_weighted - W @ H).sum())
            fraction = 1.0 - error / denominator
            h_by_seed.append(H)
            errors.append(error)
            explained.append(fraction)
            iterations.append(int(model.n_iter_))
            seed_rows.append({
                "k": k, "seed": seed, "reconstruction_error": error,
                "balanced_explained_fraction": fraction, "n_iter": int(model.n_iter_),
                "convergence_warning": any("maximum number of iterations" in str(x.message).lower() for x in caught),
            })
        best_index = int(np.argmin(errors))
        best_h_by_k[k] = h_by_seed[best_index]
        summary_rows.append({
            "k": k, "n_seeds": len(RANK_SELECTION_SEEDS),
            "mean_balanced_explained_fraction": float(np.mean(explained)),
            "best_balanced_explained_fraction": float(np.max(explained)),
            "mean_reconstruction_error": float(np.mean(errors)),
            "best_reconstruction_error": float(np.min(errors)),
            "stability_matched_cosine": stability(h_by_seed),
            "best_seed": int(RANK_SELECTION_SEEDS[best_index]),
            "max_n_iter": max(iterations),
        })
    metrics = pd.DataFrame(summary_rows)
    metrics["selection_score"] = (
        metrics.best_balanced_explained_fraction
        + 0.05 * metrics.stability_matched_cosine
        - 0.01 * metrics.k
    )
    selected = metrics.sort_values(
        ["selection_score", "best_balanced_explained_fraction"],
        ascending=False, kind="mergesort",
    ).iloc[0]
    selected_k = int(selected.k)
    best_seed = int(selected.best_seed)
    metrics["selected"] = metrics.k.eq(selected_k)

    final_model = make_nmf(selected_k, best_seed)
    with warnings.catch_warnings(record=True) as final_warnings:
        warnings.simplefilter("always")
        final_model.fit_transform(V_weighted)
    H_raw = final_model.components_
    design = H_raw.T
    W_raw = np.zeros((V.shape[0], selected_k), dtype=float)
    for index in range(V.shape[0]):
        W_raw[index, :] = nnls(design, V[index, :], maxiter=design.shape[1] * 10)[0]
    raw_components = [f"CM{i}" for i in range(1, selected_k + 1)]
    W_raw_df = pd.DataFrame(W_raw, index=frequency.index, columns=raw_components)
    H_raw_df = pd.DataFrame(H_raw, index=raw_components, columns=frequency.columns)

    class_rows = []
    p_values = []
    for order, component in enumerate(raw_components, 1):
        normal = W_raw_df.loc[normal_mask, component]
        tumor = W_raw_df.loc[tumor_mask, component]
        normal_mean = float(normal.mean())
        tumor_mean = float(tumor.mean())
        ratio = tumor_mean / normal_mean if normal_mean > 0 else np.inf
        normal_active = float((normal > 1e-8).mean())
        tumor_active = float((tumor > 1e-8).mean())
        if ratio >= TUMOR_SPECIFIC_MIN_RATIO and tumor_active >= MIN_ACTIVE_FRACTION:
            cm_class = "tCM"
        elif ratio <= NORMAL_SPECIFIC_MAX_RATIO and normal_active >= MIN_ACTIVE_FRACTION:
            cm_class = "normalCM"
        else:
            cm_class = "sharedCM"
        p_value = float(mannwhitneyu(tumor, normal, alternative="two-sided").pvalue)
        p_values.append(p_value)
        class_rows.append({
            "raw_component": component, "raw_component_order": order,
            "class": cm_class, "tumor_mean": tumor_mean,
            "normal_like_mean": normal_mean, "delta": tumor_mean - normal_mean,
            "p": p_value, "tumor_to_normal_mean_ratio": ratio,
            "normal_active_fraction": normal_active, "tumor_active_fraction": tumor_active,
            "classification_basis": "activity_ratio_and_active_fraction_thresholds",
        })
    q_values = multipletests(p_values, method="fdr_bh")[1]
    classification = pd.DataFrame(class_rows)
    classification["q"] = q_values
    classification["global_order"] = np.arange(1, selected_k + 1)
    classification["class_prefix"] = classification["class"].map({"sharedCM": "s", "normalCM": "n", "tCM": "t"})
    classification["CM"] = classification.class_prefix + "_CM" + classification.global_order.astype(str)
    classification = classification[[
        "raw_component", "raw_component_order", "CM", "class", "class_prefix",
        "global_order", "tumor_mean", "normal_like_mean", "delta", "p", "q",
        "tumor_to_normal_mean_ratio", "normal_active_fraction", "tumor_active_fraction",
        "classification_basis",
    ]]
    raw_to_canonical = dict(zip(classification.raw_component, classification.CM))
    cm_order = classification.CM.tolist()
    W_df = W_raw_df.rename(columns=raw_to_canonical).loc[:, cm_order]
    H_df = H_raw_df.rename(index=raw_to_canonical).loc[cm_order]
    loading = H_df.T
    loading_fraction = loading.div(loading.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    activity = W_df.join(status[["status"]]).join(non_epi_cells.rename("non_epi_cells"))
    activity_summary = (
        activity.reset_index().melt(id_vars=["sample", "status", "non_epi_cells"], value_vars=cm_order, var_name="CM", value_name="activity")
        .groupby(["CM", "status"], observed=True).activity.agg(["mean", "std", "count"]).reset_index()
    )

    write_csv_once(V_column_minmax.reset_index(), OUT / "non_epi_subtype_frequency_column_minmax.csv")
    write_csv_once(minmax_params.reset_index(), OUT / "column_minmax_params.csv")
    weight_audit = weights.to_frame().join(status[["status"]]).join(non_epi_cells.rename("non_epi_cells")).reset_index()
    write_csv_once(weight_audit, OUT / "group_balanced_sample_weights.csv")
    write_csv_once(pd.DataFrame(seed_rows), OUT / "joint_nmf_k_selection_per_seed.csv")
    write_csv_once(metrics, OUT / "joint_nmf_k_selection_metrics.csv")
    write_csv_once(W_raw_df.reset_index(), OUT / "W_df_raw_components_nnls.csv")
    write_csv_once(H_raw_df.reset_index(names="raw_component"), OUT / "H_df_raw_components_weighted_fit.csv")
    write_csv_once(classification, OUT / "joint_module_classification.csv")
    write_csv_once(classification[["raw_component", "raw_component_order", "CM", "class", "class_prefix", "global_order"]], OUT / "raw_to_canonical_CM_mapping.csv")
    write_csv_once(W_df.reset_index(), OUT / "W_df.csv")
    write_csv_once(H_df.reset_index(names="CM"), OUT / "H_df.csv")
    write_csv_once(W_df.T.reset_index(names="CM"), OUT / "activity_df_CM_by_sample.csv")
    write_csv_once(activity.reset_index(), OUT / "activity_df_sample_by_CM.csv")
    write_csv_once(loading.reset_index(names="cell_subtype"), OUT / "loading_df_cell_subtype_by_CM.csv")
    write_csv_once(loading_fraction.reset_index(names="cell_subtype"), OUT / "loading_df_cell_subtype_by_CM_fraction.csv")
    write_csv_once(H_df.reset_index(names="CM"), OUT / "h_df_loading_cell_subtype_by_CM_raw.csv")
    write_csv_once(zscore_rows(H_df).reset_index(names="CM"), OUT / "h_df_loading_cell_subtype_by_CM_zscore.csv")
    write_csv_once(robust_rows(H_df).reset_index(names="CM"), OUT / "h_df_loading_cell_subtype_by_CM_robust.csv")
    write_csv_once(standard_scale_columns(H_df).reset_index(names="CM"), OUT / "h_df_loading_cell_subtype_by_CM_standard_scale_col.csv")
    write_csv_once(activity_summary, OUT / "joint_CM_activity_tumor_vs_normal_mean_sd_summary.csv")
    write_csv_once(pd.DataFrame([
        {"table": "W_df.csv", "orientation": "sample x canonical_CM", "values": "NNLS activity on unweighted column-minmax V"},
        {"table": "H_df.csv", "orientation": "canonical_CM x cell_subtype", "values": "weighted-fit loading"},
        {"table": "loading_df_cell_subtype_by_CM.csv", "orientation": "cell_subtype x canonical_CM", "values": "H transpose"},
        {"table": "activity_df_sample_by_CM.csv", "orientation": "sample x canonical_CM plus metadata", "values": "NNLS activity"},
    ]), OUT / "matrix_orientation_contract.csv")

    selected_record = {
        "selected_module_k": selected_k, "selection_mode": "canonical_data_driven_score",
        "best_seed": best_seed, "candidate_Ks": candidate_ks,
        "rank_selection_seeds": list(RANK_SELECTION_SEEDS),
        "selection_score_formula": "best_balanced_explained_fraction + 0.05*stability_matched_cosine - 0.01*K",
        "n_samples": V.shape[0], "n_subtypes_total": V.shape[1],
        "n_subtypes_valid": len(valid_columns), "actual_max_k": max_k,
        "final_nmf_n_iter": int(final_model.n_iter_),
        "final_convergence_warning": any("maximum number of iterations" in str(x.message).lower() for x in final_warnings),
        "status_counts": status.status.value_counts().to_dict(),
    }
    write_text_once(json.dumps(selected_record, indent=2) + "\n", OUT / "selected_module_k.json")
    write_csv_once(pd.DataFrame([{
        "step": "balanced_joint_nmf", "input_frequency": str(INPUT / "non_epi_subtype_frequency.csv"),
        "n_samples": V.shape[0], "n_subtypes_total": V.shape[1], "n_subtypes_valid": len(valid_columns),
        "candidate_k_min": min(candidate_ks), "candidate_k_max": max(candidate_ks),
        "rank_selection_seeds": ";".join(map(str, RANK_SELECTION_SEEDS)),
        "nmf_init": "nndsvda", "solver": "cd", "beta_loss": "frobenius",
        "max_iter": NMF_MAX_ITER, "tol": NMF_TOL, "alpha_W": NMF_ALPHA_W,
        "alpha_H": NMF_ALPHA_H, "l1_ratio": NMF_L1_RATIO,
        "normal_group_total_weight": NORMAL_TOTAL_WEIGHT, "tumor_group_total_weight": TUMOR_TOTAL_WEIGHT,
        "selected_k": selected_k, "best_seed": best_seed, "final_activity_refit": "NNLS_unweighted_column_minmax_V",
        "backend": "scikit_learn_cpu", "script_seed": SEED, "code_file": str(Path(__file__).resolve()),
    }]), OUT / "run_parameters.csv")
    write_text_once(
        "Canonical GC tumor/normal status-balanced joint NMF. K=2..20 and seeds 0..4 were evaluated.\n"
        "Public W/H tables use canonical CM IDs; W/activity was refit by NNLS on unweighted column-minmax V.\n",
        OUT / "readme.txt",
    )
    write_text_once(
        f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\n"
        f"scikit-learn NMF backend=CPU\ncode={Path(__file__).resolve()}\nscript_seed={SEED}\n",
        OUT / "package_versions.txt",
    )
    print(json.dumps(selected_record, indent=2), flush=True)


if __name__ == "__main__":
    main()

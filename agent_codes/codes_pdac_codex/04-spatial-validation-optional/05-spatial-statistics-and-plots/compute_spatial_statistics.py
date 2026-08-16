#!/usr/bin/env python3
"""Compute CSV-driven PDAC Tangram Epi-CM spatial statistics and heatmaps."""

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
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests


SEED = 42
PERCENTILE_THRESHOLD = 0.5
random.seed(SEED)
np.random.seed(SEED)
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
sns.set_theme(style="ticks")

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
MAPPING_DIR = PROJECT_ROOT / "tables/04-spatial-validation-optional/04-tangram-mapping"
SCORE_DIR = MAPPING_DIR / "spot-scores"
REFERENCE_DIR = PROJECT_ROOT / "tables/04-spatial-validation-optional/03-tangram-reference"
TABLE_DIR = PROJECT_ROOT / "tables/04-spatial-validation-optional/05-spatial-statistics-and-plots"
FIGURE_DIR = PROJECT_ROOT / "figures/04-spatial-validation-optional/05-spatial-statistics-and-plots"


def bh(values) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.any():
        q[valid] = multipletests(p[valid], method="fdr_bh")[1]
    return q


def signed_z_from_p(p_value: float, direction: float) -> float:
    if not np.isfinite(p_value) or not np.isfinite(direction) or direction == 0:
        return np.nan
    clipped = min(max(float(p_value), np.finfo(float).tiny), 1.0)
    return float(np.sign(direction) * stats.norm.isf(clipped / 2.0))


def q_star(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


def cm_sort_key(value: str) -> int:
    match = re.search(r"CM(\d+)", str(value))
    return int(match.group(1)) if match else 999


def per_sample_statistics(
    score_path: Path,
    expected_cms: list[str],
    expected_epi: list[str],
) -> tuple[list[dict], list[dict]]:
    score = pd.read_csv(score_path)
    sample_values = score["sample"].astype(str).unique()
    if len(sample_values) != 1:
        raise ValueError(f"{score_path}: expected one sample")
    sample = sample_values[0]
    status = str(score["status"].iloc[0])
    cm_columns = [f"CMact__{name}" for name in expected_cms]
    epi_columns = [f"EPIfrac__{name}" for name in expected_epi]
    missing = sorted(set(cm_columns + epi_columns) - set(score.columns))
    if missing:
        raise KeyError(f"{sample}: missing spot-score features: {missing}")
    features = score[cm_columns + epi_columns].apply(pd.to_numeric, errors="coerce")
    percentiles = features.rank(method="average", pct=True)
    spearman_rows, fisher_rows = [], []
    for cm in expected_cms:
        cm_column = f"CMact__{cm}"
        cm_high = percentiles[cm_column].ge(PERCENTILE_THRESHOLD)
        for epithelial_subtype in expected_epi:
            epi_column = f"EPIfrac__{epithelial_subtype}"
            epi_high = percentiles[epi_column].ge(PERCENTILE_THRESHOLD)
            x = features[cm_column]
            y = features[epi_column]
            valid = x.notna() & y.notna()
            if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
                rho, spearman_p = np.nan, np.nan
            else:
                rho, spearman_p = stats.spearmanr(x[valid], y[valid])
            spearman_rows.append(
                {
                    "sample": sample,
                    "status": status,
                    "CM": cm,
                    "epi_subtype": epithelial_subtype,
                    "n_spots": int(valid.sum()),
                    "spearman_rho": float(rho),
                    "spearman_p": float(spearman_p),
                }
            )
            both_high = int((cm_high & epi_high).sum())
            cm_high_only = int((cm_high & ~epi_high).sum())
            epi_high_only = int((~cm_high & epi_high).sum())
            both_low = int((~cm_high & ~epi_high).sum())
            odds_ratio, fisher_p = stats.fisher_exact(
                [[both_high, cm_high_only], [epi_high_only, both_low]], alternative="two-sided"
            )
            corrected_odds = ((both_high + 0.5) * (both_low + 0.5)) / (
                (cm_high_only + 0.5) * (epi_high_only + 0.5)
            )
            log2_odds = float(np.log2(corrected_odds))
            fisher_rows.append(
                {
                    "sample": sample,
                    "status": status,
                    "CM": cm,
                    "epi_subtype": epithelial_subtype,
                    "percentile_threshold": PERCENTILE_THRESHOLD,
                    "both_high": both_high,
                    "cm_high_only": cm_high_only,
                    "epi_high_only": epi_high_only,
                    "both_low": both_low,
                    "n_spots": int(both_high + cm_high_only + epi_high_only + both_low),
                    "fisher_odds_ratio": float(odds_ratio),
                    "haldane_log2_odds_ratio": log2_odds,
                    "fisher_p": float(fisher_p),
                    "signed_z": signed_z_from_p(float(fisher_p), log2_odds),
                }
            )
    return spearman_rows, fisher_rows


def combine_signed_stouffer(fisher: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cm, epithelial_subtype), group in fisher.groupby(["CM", "epi_subtype"], sort=False):
        z_values = pd.to_numeric(group["signed_z"], errors="coerce").dropna().to_numpy()
        if len(z_values):
            combined_z = float(z_values.sum() / math.sqrt(len(z_values)))
            p_value = float(2.0 * stats.norm.sf(abs(combined_z)))
        else:
            combined_z, p_value = np.nan, np.nan
        rows.append(
            {
                "scope": "all_14_eligible_spatial_samples",
                "CM": cm,
                "epi_subtype": epithelial_subtype,
                "n_samples": len(z_values),
                "combined_signed_z": combined_z,
                "p_value": p_value,
                "direction": "positive" if combined_z > 0 else "negative" if combined_z < 0 else "neutral",
            }
        )
    output = pd.DataFrame(rows)
    output["q_value_bh"] = bh(output["p_value"])
    return output


def pooled_fisher(fisher: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cm, epithelial_subtype), group in fisher.groupby(["CM", "epi_subtype"], sort=False):
        both_high = int(group["both_high"].sum())
        cm_high_only = int(group["cm_high_only"].sum())
        epi_high_only = int(group["epi_high_only"].sum())
        both_low = int(group["both_low"].sum())
        odds_ratio, p_value = stats.fisher_exact(
            [[both_high, cm_high_only], [epi_high_only, both_low]], alternative="two-sided"
        )
        corrected_odds = ((both_high + 0.5) * (both_low + 0.5)) / (
            (cm_high_only + 0.5) * (epi_high_only + 0.5)
        )
        log2_odds = float(np.log2(corrected_odds))
        rows.append(
            {
                "scope": "pooled_spots_across_14_eligible_samples",
                "CM": cm,
                "epi_subtype": epithelial_subtype,
                "both_high": both_high,
                "cm_high_only": cm_high_only,
                "epi_high_only": epi_high_only,
                "both_low": both_low,
                "n_spots": both_high + cm_high_only + epi_high_only + both_low,
                "fisher_odds_ratio": float(odds_ratio),
                "haldane_log2_odds_ratio": log2_odds,
                "p_value": float(p_value),
                "signed_z": signed_z_from_p(float(p_value), log2_odds),
            }
        )
    output = pd.DataFrame(rows)
    output["q_value_bh"] = bh(output["p_value"])
    return output


def combine_spearman(spearman: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cm, epithelial_subtype), group in spearman.groupby(["CM", "epi_subtype"], sort=False):
        rho = pd.to_numeric(group["spearman_rho"], errors="coerce").clip(-0.999999, 0.999999)
        n = pd.to_numeric(group["n_spots"], errors="coerce")
        valid = rho.notna() & n.gt(3)
        if valid.any():
            z = np.arctanh(rho[valid].to_numpy())
            weights = np.sqrt(n[valid].to_numpy() - 3.0)
            meta_z = float(np.sum(weights * z) / np.sqrt(np.sum(np.square(weights))))
            p_value = float(2.0 * stats.norm.sf(abs(meta_z)))
            weighted_mean_rho = float(
                np.sum(weights * rho[valid].to_numpy()) / np.sum(weights)
            )
        else:
            meta_z, p_value, weighted_mean_rho = np.nan, np.nan, np.nan
        rows.append(
            {
                "scope": "all_14_eligible_spatial_samples",
                "CM": cm,
                "epi_subtype": epithelial_subtype,
                "n_samples": int(valid.sum()),
                "weighted_mean_spearman_rho": weighted_mean_rho,
                "fisher_z_meta_statistic": meta_z,
                "p_value": p_value,
            }
        )
    output = pd.DataFrame(rows)
    output["q_value_bh"] = bh(output["p_value"])
    return output


def plot_meta_heatmap(
    table: pd.DataFrame,
    value_column: str,
    label: str,
    stem: Path,
    title: str,
) -> None:
    values = table.pivot(index="epi_subtype", columns="CM", values=value_column)
    q_values = table.pivot(index="epi_subtype", columns="CM", values="q_value_bh")
    columns = sorted(values.columns, key=cm_sort_key)
    rows = sorted(values.index)
    values = values.reindex(index=rows, columns=columns)
    q_values = q_values.reindex(index=rows, columns=columns)
    annotation = q_values.applymap(q_star)
    finite = values.to_numpy()[np.isfinite(values.to_numpy())]
    maximum = float(np.max(np.abs(finite))) if len(finite) else 1.0
    maximum = maximum if maximum > 0 else 1.0
    fig, axis = plt.subplots(
        figsize=(max(6.0, 0.65 * len(columns) + 2), max(4.5, 0.45 * len(rows) + 1.5))
    )
    sns.heatmap(
        values,
        cmap="RdBu_r",
        center=0,
        vmin=-maximum,
        vmax=maximum,
        annot=annotation,
        fmt="",
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": label},
        ax=axis,
    )
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=90)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    for extension in [".pdf", ".svg"]:
        fig.savefig(stem.with_suffix(extension), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if TABLE_DIR.exists() or FIGURE_DIR.exists():
        raise FileExistsError("Refusing to overwrite spatial statistics outputs")
    mapping_validation = MAPPING_DIR / "tangram_mapping_validation.json"
    if not mapping_validation.is_file():
        raise FileNotFoundError(mapping_validation)
    validation = json.loads(mapping_validation.read_text(encoding="utf-8"))
    if not validation.get("validation_passed") or validation.get("samples_completed") != 14:
        raise ValueError("Tangram mapping is not complete 14/14")
    loadings = pd.read_csv(
        REFERENCE_DIR / "cm_loading_fraction_non_epi_node_by_CM.csv", index_col=0
    )
    expected_cms = loadings.columns.astype(str).tolist()
    expected_epi = pd.read_csv(REFERENCE_DIR / "epithelial_subtypes.csv")[
        "cell_subtype"
    ].astype(str).tolist()
    score_paths = sorted(SCORE_DIR.glob("*_tangram_pseudobulk_epi_cm_spot_scores.csv"))
    if len(score_paths) != 14:
        raise ValueError(f"Expected 14 spot-score CSVs, found {len(score_paths)}")
    TABLE_DIR.mkdir(parents=True, exist_ok=False)
    FIGURE_DIR.mkdir(parents=True, exist_ok=False)
    all_spearman, all_fisher = [], []
    pair_manifest_rows = []
    for index, path in enumerate(score_paths, start=1):
        spearman_rows, fisher_rows = per_sample_statistics(path, expected_cms, expected_epi)
        all_spearman.extend(spearman_rows)
        all_fisher.extend(fisher_rows)
        sample = spearman_rows[0]["sample"]
        for cm in expected_cms:
            for epithelial_subtype in expected_epi:
                pair_manifest_rows.append(
                    {
                        "sample": sample,
                        "CM": cm,
                        "epi_subtype": epithelial_subtype,
                        "spot_score_csv": str(path),
                    }
                )
        print(f"DONE statistics {index}/14 {path.name}", flush=True)
    spearman = pd.DataFrame(all_spearman)
    fisher = pd.DataFrame(all_fisher)
    spearman["spearman_q_within_sample"] = spearman.groupby("sample")["spearman_p"].transform(
        lambda values: bh(values)
    )
    fisher["fisher_q_within_sample"] = fisher.groupby("sample")["fisher_p"].transform(
        lambda values: bh(values)
    )
    stouffer = combine_signed_stouffer(fisher)
    pooled = pooled_fisher(fisher)
    spearman_meta = combine_spearman(spearman)
    spearman.to_csv(TABLE_DIR / "per_sample_spearman.csv", index=False)
    fisher.to_csv(TABLE_DIR / "percentile_quadrant_fisher_per_sample.csv", index=False)
    stouffer.to_csv(TABLE_DIR / "percentile_quadrant_fisher_sample_stouffer_14_samples.csv", index=False)
    pooled.to_csv(TABLE_DIR / "percentile_quadrant_fisher_pooled_spots_14_samples.csv", index=False)
    spearman_meta.to_csv(TABLE_DIR / "spearman_fisher_z_meta_14_samples.csv", index=False)
    pair_manifest = pd.DataFrame(pair_manifest_rows)
    pair_manifest.to_csv(TABLE_DIR / "all_sample_cm_epi_pair_manifest.csv", index=False)
    plot_meta_heatmap(
        stouffer,
        "combined_signed_z",
        "Combined signed Stouffer Z",
        FIGURE_DIR / "all_14_samples_fisher_stouffer_signed_z_qstars",
        "14 eligible PDAC Visium samples: percentile-quadrant Fisher/Stouffer",
    )
    plot_meta_heatmap(
        spearman_meta,
        "weighted_mean_spearman_rho",
        "Weighted mean Spearman rho",
        FIGURE_DIR / "all_14_samples_spearman_meta_qstars",
        "14 eligible PDAC Visium samples: spot-level Spearman meta-analysis",
    )
    expected_pairs = len(expected_cms) * len(expected_epi)
    summary = {
        "n_samples": int(spearman["sample"].nunique()),
        "all_samples_are_tumor": bool(spearman["status"].eq("tumor").all()),
        "tumor_only_scope_equals_all_samples": True,
        "duplicate_tumor_only_outputs_suppressed": True,
        "n_cms": len(expected_cms),
        "n_epithelial_subtypes": len(expected_epi),
        "pairs_per_sample": expected_pairs,
        "pair_sample_rows": int(len(spearman)),
        "combined_pair_rows": int(len(stouffer)),
        "stouffer_weighting": "unweighted sample-level signed Fisher Z; spatial samples are replicates",
        "spearman_meta_weighting": "sqrt(n_spots-3) weighted Fisher-z meta-analysis",
        "downstream_source": "reloaded canonical per-sample Tangram spot-score CSV files only",
        "validation_passed": bool(
            spearman["sample"].nunique() == 14
            and len(spearman) == 14 * expected_pairs
            and len(fisher) == 14 * expected_pairs
            and len(stouffer) == expected_pairs
        ),
    }
    (TABLE_DIR / "spatial_statistics_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    versions = [
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"numpy={importlib.metadata.version('numpy')}",
        f"pandas={importlib.metadata.version('pandas')}",
        f"scipy={importlib.metadata.version('scipy')}",
        f"statsmodels={importlib.metadata.version('statsmodels')}",
        f"matplotlib={importlib.metadata.version('matplotlib')}",
        f"seaborn={importlib.metadata.version('seaborn')}",
        f"random_seed={SEED}",
        f"code_file={Path(__file__).resolve()}",
    ]
    (TABLE_DIR / "package_versions.txt").write_text("\n".join(versions) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

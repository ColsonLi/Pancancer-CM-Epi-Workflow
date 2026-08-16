#!/usr/bin/env python3
"""Compute complete CSV-driven spatial Spearman/Fisher/Stouffer statistics."""

from __future__ import annotations

import json
import math
import platform
import random
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

WF = Path(__file__).resolve().parents[3]
TASK = "01-tangram-pseudobulk-v3-final-v5"
BASE = WF / f"tables/04-spatial-validation-optional/{TASK}"
MAPPING = BASE / "02-tangram-mapping"
SCORE_DIR = MAPPING / "spot_scores"
TABLE_DIR = BASE / "03-all-pair-statistics-and-plots"
FIG_DIR = WF / f"figures/04-spatial-validation-optional/{TASK}/03-all-pair-statistics-and-plots/stouffer_heatmaps"
SCOPE = BASE / "01-reference-and-manifest/spatial_sample_scope_manifest.csv"

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
    }
)


def bh(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    q = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if valid.any():
        q[valid] = multipletests(p[valid], method="fdr_bh")[1]
    return pd.Series(q, index=values.index)


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    ok = x.notna() & y.notna()
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan, np.nan
    result = stats.spearmanr(x[ok].to_numpy(float), y[ok].to_numpy(float))
    return float(result.statistic), float(result.pvalue)


def signed_z_from_p(p: float, direction: float) -> float:
    if not np.isfinite(p) or not np.isfinite(direction) or direction == 0:
        return np.nan
    bounded = max(min(float(p), 1.0), np.finfo(float).tiny)
    return float(np.sign(direction) * stats.norm.isf(bounded / 2.0))


def sample_statistics(path: Path, cms: list[str], epis: list[str], metadata: pd.DataFrame):
    score = pd.read_csv(path)
    samples = score["sample"].astype(str).unique()
    if len(samples) != 1:
        raise ValueError(f"{path}: expected exactly one sample")
    sample = samples[0]
    meta = metadata.loc[sample]
    features = [f"CMact__{x}" for x in cms] + [f"EPIfrac__{x}" for x in epis]
    numeric = score[features].apply(pd.to_numeric, errors="coerce")
    percentile = numeric.rank(method="average", pct=True)
    spearman_rows, fisher_rows = [], []
    common = {
        "sample": sample,
        "status": str(meta["status"]),
        "patient_id": str(meta["patient_id"]),
        "biological_region": str(meta["biological_region"]),
        "section_role": str(meta["section_role"]),
        "n_spatial_spots": len(score),
    }
    for cm in cms:
        cm_col = f"CMact__{cm}"
        cm_high = percentile[cm_col].ge(0.5)
        for epi in epis:
            epi_col = f"EPIfrac__{epi}"
            epi_high = percentile[epi_col].ge(0.5)
            rho, p_s = safe_spearman(numeric[cm_col], numeric[epi_col])
            spearman_rows.append(
                {**common, "CM": cm, "epi_subtype": epi, "spearman_rho": rho, "spearman_p": p_s}
            )
            both = int((cm_high & epi_high).sum())
            cm_only = int((cm_high & ~epi_high).sum())
            epi_only = int((~cm_high & epi_high).sum())
            both_low = int((~cm_high & ~epi_high).sum())
            odds, p_f = stats.fisher_exact([[both, cm_only], [epi_only, both_low]])
            corrected_or = ((both + 0.5) * (both_low + 0.5)) / ((cm_only + 0.5) * (epi_only + 0.5))
            log2_or = float(np.log2(corrected_or))
            fisher_rows.append(
                {
                    **common,
                    "CM": cm,
                    "epi_subtype": epi,
                    "percentile_cutoff": 0.5,
                    "both_high": both,
                    "cm_high_only": cm_only,
                    "epi_high_only": epi_only,
                    "both_low": both_low,
                    "fisher_odds_ratio": float(odds),
                    "haldane_log2_odds_ratio": log2_or,
                    "fisher_p": float(p_f),
                    "signed_z": signed_z_from_p(float(p_f), log2_or),
                }
            )
    return spearman_rows, fisher_rows


def combine_stouffer(frame: pd.DataFrame, scope: str, unit: str = "section") -> pd.DataFrame:
    source = frame.copy()
    if unit == "biological_region":
        source = (
            source.groupby(["biological_region", "CM", "epi_subtype"], as_index=False)
            .agg(
                signed_z_sum=("signed_z", "sum"),
                n_sections_in_region=("signed_z", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
            )
        )
        source["signed_z"] = source["signed_z_sum"] / np.sqrt(source["n_sections_in_region"].replace(0, np.nan))
    rows = []
    for (cm, epi), group in source.groupby(["CM", "epi_subtype"], sort=False):
        z = pd.to_numeric(group["signed_z"], errors="coerce").dropna().to_numpy(float)
        combined = float(z.sum() / math.sqrt(len(z))) if len(z) else np.nan
        p = float(2 * stats.norm.sf(abs(combined))) if len(z) else np.nan
        rows.append(
            {
                "scope": scope,
                "aggregation_unit": unit,
                "CM": cm,
                "epi_subtype": epi,
                "n_units": len(z),
                "combined_signed_z": combined,
                "p_value": p,
            }
        )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = bh(result["p_value"])
    return result


def cm_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"CM(\d+)", str(value))
    return (int(match.group(1)) if match else 999, str(value))


def q_label(q: float) -> str:
    if not np.isfinite(q):
        return "ns"
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "ns"


def plot_heatmap(table: pd.DataFrame, stem: Path, title: str) -> None:
    z = table.pivot(index="epi_subtype", columns="CM", values="combined_signed_z")
    q = table.pivot(index="epi_subtype", columns="CM", values="q_value_bh")
    z = z.reindex(index=sorted(z.index), columns=sorted(z.columns, key=cm_sort_key))
    q = q.reindex(index=z.index, columns=z.columns)
    finite = z.to_numpy(float)[np.isfinite(z.to_numpy(float))]
    vmax = float(np.max(np.abs(finite))) if len(finite) else 1.0
    vmax = vmax if vmax > 0 else 1.0
    fig, ax = plt.subplots(figsize=(max(8, 0.52 * z.shape[1] + 2.6), max(5, 0.42 * z.shape[0] + 1.9)))
    sns.heatmap(
        z,
        cmap="coolwarm",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.25,
        linecolor="white",
        annot=q.map(q_label),
        fmt="",
        annot_kws={"fontsize": 6.5, "color": "black"},
        square=True,
        cbar_kws={"label": "Combined signed Stouffer Z"},
        ax=ax,
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("CM")
    ax.set_ylabel("Epithelial subtype")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = time.time()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(SCOPE).set_index("sample")
    pairs = pd.read_csv(MAPPING / "all_sample_cm_epi_pair_manifest.csv")
    cms = pairs["CM"].drop_duplicates().astype(str).tolist()
    epis = pairs["epi_subtype"].drop_duplicates().astype(str).tolist()
    paths = sorted(SCORE_DIR.glob("*_tangram_pseudobulk_epi_cm_spot_scores.csv"))
    if len(paths) != 72:
        raise ValueError(f"Expected 72 score CSVs, found {len(paths)}")
    all_spearman, all_fisher = [], []
    for i, path in enumerate(paths, 1):
        spearman, fisher = sample_statistics(path, cms, epis, metadata)
        all_spearman.extend(spearman)
        all_fisher.extend(fisher)
        print(f"DONE statistics {i}/72 {path.name}", flush=True)
    spearman = pd.DataFrame(all_spearman)
    fisher = pd.DataFrame(all_fisher)
    spearman["spearman_q_within_section"] = spearman.groupby("sample")["spearman_p"].transform(bh)
    fisher["fisher_q_within_section"] = fisher.groupby("sample")["fisher_p"].transform(bh)
    spearman.to_csv(TABLE_DIR / "per_section_spearman.csv", index=False)
    fisher.to_csv(TABLE_DIR / "percentile_quadrant_fisher_per_section.csv", index=False)

    all_sections = combine_stouffer(fisher, "all-sections", "section")
    tumor_sections = combine_stouffer(fisher.loc[fisher["status"].eq("tumor")], "tumor-only", "section")
    regions = combine_stouffer(fisher, "all-biological-regions", "biological_region")
    outputs = {
        "percentile_quadrant_fisher_stouffer_all_72_sections": all_sections,
        "percentile_quadrant_fisher_stouffer_tumor_72_sections": tumor_sections,
        "percentile_quadrant_fisher_stouffer_replicate_aware_37_regions": regions,
    }
    for stem, table in outputs.items():
        table.to_csv(TABLE_DIR / f"{stem}.csv", index=False)
        title = stem.replace("_", " ").title()
        plot_heatmap(table, FIG_DIR / f"{stem}_signedZ_qstars", title)
    if not all_sections[["CM", "epi_subtype", "combined_signed_z", "q_value_bh"]].equals(
        tumor_sections[["CM", "epi_subtype", "combined_signed_z", "q_value_bh"]]
    ):
        raise ValueError("All-section and tumor-only results must be identical because all 72 sections are tumor")
    summary = {
        "n_sections": fisher["sample"].nunique(),
        "n_tumor_sections": fisher.loc[fisher["status"].eq("tumor"), "sample"].nunique(),
        "n_biological_regions": fisher["biological_region"].nunique(),
        "n_donors": fisher["patient_id"].nunique(),
        "n_cm": len(cms),
        "n_epi_subtypes": len(epis),
        "n_pair_section_rows": len(fisher),
        "n_pairs_per_scope": len(all_sections),
        "section_stouffer": "unweighted signed Z across 72 sections",
        "replicate_aware_sensitivity": "within-region Stouffer followed by equal-weight Stouffer across 37 biological regions",
        "elapsed_seconds": round(time.time() - started, 3),
        "python": platform.python_version(),
        "scipy": scipy.__version__,
        "seed": SEED,
    }
    (TABLE_DIR / "statistics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()


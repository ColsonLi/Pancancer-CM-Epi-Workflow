#!/usr/bin/env python3
"""Render every canonical CM-Epi pair for one PDAC sample from its score CSV."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
SCORE_DIR = PROJECT_ROOT / "tables/04-spatial-validation-optional/04-tangram-mapping/spot-scores"
STAT_DIR = PROJECT_ROOT / "tables/04-spatial-validation-optional/05-spatial-statistics-and-plots"
PAIR_MANIFEST = STAT_DIR / "all_sample_cm_epi_pair_manifest.csv"
MANIFEST_DIR = STAT_DIR / "plot-manifests-by-sample"
FIGURE_ROOT = PROJECT_ROOT / "figures/04-spatial-validation-optional/05-spatial-statistics-and-plots"

CM_CMAP = LinearSegmentedColormap.from_list(
    "cm_percentile", ["#F2F2F2", "#B7D7EA", "#4C78A8", "#2B6CB0"]
)
EPI_CMAP = LinearSegmentedColormap.from_list(
    "epi_percentile", ["#F2F2F2", "#FDE6B3", "#FBBF24", "#F59E0B"]
)
CATEGORY_ORDER = ["both_low", "epi_high_only", "cm_high_only", "both_high"]
CATEGORY_LABELS = {
    "both_low": "CM<0.5, Epi<0.5",
    "epi_high_only": "Epi≥0.5 only",
    "cm_high_only": "CM≥0.5 only",
    "both_high": "Both≥0.5",
}
CATEGORY_COLORS = {
    "both_low": "#D9D9D9",
    "epi_high_only": "#F59E0B",
    "cm_high_only": "#2B6CB0",
    "both_high": "#B91C1C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    return parser.parse_args()


def robust_limits(values: pd.Series) -> tuple[float, float]:
    array = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy()
    )
    if not len(array):
        return 0.0, 1.0
    lower, upper = np.nanquantile(array, [0.01, 0.99])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower, upper = float(np.nanmin(array)), float(np.nanmax(array))
    if lower == upper:
        upper = lower + 1e-9
    return float(lower), float(upper)


def set_spatial_axes(axis) -> None:
    axis.invert_yaxis()
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("full-resolution pixel x")
    axis.set_ylabel("full-resolution pixel y")


def save_pdf_svg(fig, stem: Path) -> None:
    for extension in [".pdf", ".svg"]:
        path = stem.with_suffix(extension)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def raw_plot(
    score: pd.DataFrame,
    sample: str,
    cm: str,
    epithelial_subtype: str,
    output_dir: Path,
    point_size: float,
) -> None:
    cm_raw = pd.to_numeric(score[f"CMact__{cm}"], errors="coerce")
    epi_raw = pd.to_numeric(score[f"EPIfrac__{epithelial_subtype}"], errors="coerce")
    x = score["spatial_x"].to_numpy(dtype=float)
    y = score["spatial_y"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), constrained_layout=True)
    cm_image = axes[0].scatter(
        x,
        y,
        c=cm_raw,
        s=point_size,
        cmap="magma",
        norm=Normalize(*robust_limits(cm_raw)),
        linewidths=0,
        rasterized=True,
    )
    set_spatial_axes(axes[0])
    axes[0].set_title(f"CMact: {cm}")
    fig.colorbar(cm_image, ax=axes[0], fraction=0.046, pad=0.02).set_label("raw CMact")
    epi_image = axes[1].scatter(
        x,
        y,
        c=epi_raw,
        s=point_size,
        cmap="viridis",
        norm=Normalize(*robust_limits(epi_raw)),
        linewidths=0,
        rasterized=True,
    )
    set_spatial_axes(axes[1])
    axes[1].set_title(f"EPIfrac: {epithelial_subtype}")
    fig.colorbar(epi_image, ax=axes[1], fraction=0.046, pad=0.02).set_label(
        "within-epithelial EPIfrac"
    )
    fig.suptitle(f"{sample}: {cm} | {epithelial_subtype}", y=1.03, fontsize=9)
    save_pdf_svg(
        fig, output_dir / f"{sample}__{cm}__{epithelial_subtype}__raw_cmact_epifrac"
    )


def percentile_plot(
    score: pd.DataFrame,
    percentiles: pd.DataFrame,
    sample: str,
    cm: str,
    epithelial_subtype: str,
    output_dir: Path,
    point_size: float,
) -> None:
    cm_percentile = percentiles[f"CMact__{cm}"]
    epi_percentile = percentiles[f"EPIfrac__{epithelial_subtype}"]
    cm_high = cm_percentile.ge(0.5)
    epi_high = epi_percentile.ge(0.5)
    category = pd.Series("both_low", index=score.index)
    category.loc[epi_high & ~cm_high] = "epi_high_only"
    category.loc[cm_high & ~epi_high] = "cm_high_only"
    category.loc[cm_high & epi_high] = "both_high"
    x = score["spatial_x"].to_numpy(dtype=float)
    y = score["spatial_y"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), constrained_layout=True)
    cm_image = axes[0].scatter(
        x,
        y,
        c=cm_percentile,
        cmap=CM_CMAP,
        vmin=0,
        vmax=1,
        s=point_size,
        linewidths=0,
        rasterized=True,
    )
    set_spatial_axes(axes[0])
    axes[0].set_title(f"CM percentile: {cm}")
    fig.colorbar(cm_image, ax=axes[0], fraction=0.046, pad=0.02).set_label("CMact percentile")
    epi_image = axes[1].scatter(
        x,
        y,
        c=epi_percentile,
        cmap=EPI_CMAP,
        vmin=0,
        vmax=1,
        s=point_size,
        linewidths=0,
        rasterized=True,
    )
    set_spatial_axes(axes[1])
    axes[1].set_title(f"Epi percentile: {epithelial_subtype}")
    fig.colorbar(epi_image, ax=axes[1], fraction=0.046, pad=0.02).set_label(
        "EPIfrac percentile"
    )
    for label in CATEGORY_ORDER:
        mask = category.eq(label).to_numpy()
        if mask.any():
            axes[2].scatter(
                x[mask],
                y[mask],
                c=CATEGORY_COLORS[label],
                s=point_size,
                linewidths=0,
                rasterized=True,
            )
    set_spatial_axes(axes[2])
    axes[2].set_title("Within-sample percentile quadrant")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label=CATEGORY_LABELS[label],
            markerfacecolor=CATEGORY_COLORS[label],
            markersize=6,
        )
        for label in CATEGORY_ORDER
    ]
    axes[2].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6)
    fig.suptitle(f"{sample}: {cm} | {epithelial_subtype}", y=1.03, fontsize=9)
    save_pdf_svg(
        fig,
        output_dir / f"{sample}__{cm}__{epithelial_subtype}__percentile_quadrant_0p5",
    )


def main() -> None:
    sample = parse_args().sample
    started = time.time()
    score_path = SCORE_DIR / f"{sample}_tangram_pseudobulk_epi_cm_spot_scores.csv"
    if not score_path.is_file() or not PAIR_MANIFEST.is_file():
        raise FileNotFoundError("Canonical score CSV or all-pair manifest is missing")
    score = pd.read_csv(score_path)
    pairs = pd.read_csv(PAIR_MANIFEST)
    pairs = pairs.loc[pairs["sample"].astype(str).eq(sample)].copy()
    if pairs.empty:
        raise ValueError(f"{sample}: no canonical CM-Epi pairs")
    expected_pairs = pairs[["CM", "epi_subtype"]].drop_duplicates()
    if len(pairs) != len(expected_pairs):
        raise ValueError(f"{sample}: duplicate pair rows")
    feature_columns = [
        column
        for column in score.columns
        if column.startswith("CMact__") or column.startswith("EPIfrac__")
    ]
    percentiles = score[feature_columns].rank(method="average", pct=True)
    point_size = max(0.2, min(5.0, 25000.0 / len(score)))
    raw_dir = FIGURE_ROOT / "raw-cmact-epifrac-by-sample" / sample
    percentile_dir = FIGURE_ROOT / "percentile-quadrant-by-sample" / sample
    raw_dir.mkdir(parents=True, exist_ok=True)
    percentile_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair_index, pair in enumerate(pairs.itertuples(index=False), start=1):
        cm = str(pair.CM)
        epithelial_subtype = str(pair.epi_subtype)
        raw_stem = raw_dir / f"{sample}__{cm}__{epithelial_subtype}__raw_cmact_epifrac"
        percentile_stem = (
            percentile_dir
            / f"{sample}__{cm}__{epithelial_subtype}__percentile_quadrant_0p5"
        )
        raw_complete = raw_stem.with_suffix(".pdf").is_file() and raw_stem.with_suffix(".svg").is_file()
        percentile_complete = (
            percentile_stem.with_suffix(".pdf").is_file()
            and percentile_stem.with_suffix(".svg").is_file()
        )
        if not raw_complete:
            if raw_stem.with_suffix(".pdf").exists() or raw_stem.with_suffix(".svg").exists():
                raise FileExistsError(f"Partial raw pair figure requires review: {raw_stem}")
            raw_plot(score, sample, cm, epithelial_subtype, raw_dir, point_size)
        if not percentile_complete:
            if percentile_stem.with_suffix(".pdf").exists() or percentile_stem.with_suffix(".svg").exists():
                raise FileExistsError(f"Partial percentile pair figure requires review: {percentile_stem}")
            percentile_plot(
                score,
                percentiles,
                sample,
                cm,
                epithelial_subtype,
                percentile_dir,
                point_size,
            )
        rows.append(
            {
                "sample": sample,
                "CM": cm,
                "epi_subtype": epithelial_subtype,
                "spot_score_csv": str(score_path),
                "raw_pdf": str(raw_stem.with_suffix(".pdf")),
                "raw_svg": str(raw_stem.with_suffix(".svg")),
                "percentile_pdf": str(percentile_stem.with_suffix(".pdf")),
                "percentile_svg": str(percentile_stem.with_suffix(".svg")),
                "completed": True,
            }
        )
        if pair_index % 10 == 0 or pair_index == len(pairs):
            print(f"{sample}: {pair_index}/{len(pairs)} pairs", flush=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{sample}_plot_manifest.csv"
    summary_path = MANIFEST_DIR / f"{sample}_plot_summary.json"
    if manifest_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite plot completion record for {sample}")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    summary = {
        "sample": sample,
        "n_pairs": len(rows),
        "n_spots": len(score),
        "point_size": point_size,
        "real_coordinate_columns": ["spatial_x", "spatial_y"],
        "raw_pair_figure_files": 2 * len(rows),
        "percentile_pair_figure_files": 2 * len(rows),
        "downstream_source": str(score_path),
        "elapsed_seconds": round(time.time() - started, 3),
        "validation_passed": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

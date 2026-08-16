#!/usr/bin/env python3
"""Enrich completed grid metrics and prepare a non-binding manual-review shortlist."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
TABLE_ROOT = PROJECT_ROOT / "tables/01-celltype_integration_clustering/05-clustering-parameter-search"
FIGURE_ROOT = PROJECT_ROOT / "figures/01-celltype_integration_clustering/05-clustering-parameter-search"
REVIEW_PATH = TABLE_ROOT / "candidate_review_manifest.csv"
ENRICHED_PATH = TABLE_ROOT / "candidate_review_enriched.csv"
SHORTLIST_PATH = TABLE_ROOT / "candidate_manual_review_shortlist.csv"
SUMMARY_PATH = TABLE_ROOT / "clustering_grid_summary_by_pcs.csv"
README_PATH = TABLE_ROOT / "candidate_manual_review_readme.txt"
MANUAL_FIGURE_DIR = FIGURE_ROOT / "candidate-manual-review"
RENDER_DIR = MANUAL_FIGURE_DIR / "renders"
MONTAGE_PNG = MANUAL_FIGURE_DIR / "candidate_manual_review_shortlist.png"
MONTAGE_PDF = MANUAL_FIGURE_DIR / "candidate_manual_review_shortlist.pdf"


def require_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite manual-review artifact: {path}")


def main() -> None:
    for path in [ENRICHED_PATH, SHORTLIST_PATH, SUMMARY_PATH, README_PATH, MANUAL_FIGURE_DIR]:
        require_absent(path)
    review = pd.read_csv(REVIEW_PATH)
    if review.shape[0] != 422 or review["candidate_label"].nunique() != 422:
        raise ValueError("Completed-candidate review manifest is not the audited 422-row grid")

    size_rows = []
    for row in review.itertuples(index=False):
        counts_path = TABLE_ROOT / row.candidate_label / "cluster_counts.csv"
        counts = pd.read_csv(counts_path)["n_cells"].astype(int)
        size_rows.append(
            {
                "candidate_label": row.candidate_label,
                "min_cluster_n_cells": int(counts.min()),
                "median_cluster_n_cells": float(counts.median()),
                "max_cluster_n_cells": int(counts.max()),
                "small_clusters_lt100": int((counts < 100).sum()),
                "small_clusters_lt500": int((counts < 500).sum()),
                "small_clusters_lt1000": int((counts < 1000).sum()),
                "smallest_cluster_fraction": float(counts.min() / counts.sum()),
                "cluster_size_cv": float(counts.std(ddof=0) / counts.mean()),
            }
        )
    enriched = review.merge(pd.DataFrame(size_rows), on="candidate_label", validate="one_to_one")
    enriched["distance_from_16_clusters"] = (enriched["n_clusters"] - 16).abs()
    enriched.to_csv(ENRICHED_PATH, index=False)

    # This shortlist is deliberately non-binding. It gives one representative
    # per PC setting and keeps broad-lineage-scale solutions (14-18 clusters).
    # Within each PC, it first avoids tiny clusters, then favors sample mixing,
    # normalized sample entropy, proximity to 16 clusters, and cluster size.
    eligible = enriched.loc[enriched["n_clusters"].between(14, 18)].copy()
    eligible = eligible.sort_values(
        [
            "n_pcs",
            "small_clusters_lt500",
            "worst_cluster_max_sample_fraction",
            "median_sample_entropy_normalized",
            "distance_from_16_clusters",
            "min_cluster_n_cells",
            "median_cluster_n_samples",
        ],
        ascending=[True, True, True, False, True, False, False],
    )
    shortlist = eligible.groupby("n_pcs", as_index=False).head(1).copy()
    if shortlist.shape[0] != 9 or shortlist["n_pcs"].nunique() != 9:
        raise ValueError("Could not derive one representative candidate per PC setting")
    shortlist = shortlist.sort_values(
        [
            "small_clusters_lt500",
            "worst_cluster_max_sample_fraction",
            "median_sample_entropy_normalized",
            "distance_from_16_clusters",
            "min_cluster_n_cells",
        ],
        ascending=[True, True, False, True, False],
    ).reset_index(drop=True)
    shortlist.insert(0, "shortlist_review_order", np.arange(1, len(shortlist) + 1))
    shortlist["recommended_for_first_manual_inspection"] = False
    shortlist.loc[0, "recommended_for_first_manual_inspection"] = True
    shortlist["final_selection_made"] = False
    shortlist["selection_contract"] = "non_binding_shortlist_user_must_manually_select_completed_candidate"
    shortlist.to_csv(SHORTLIST_PATH, index=False)

    manifest = pd.read_csv(TABLE_ROOT / "clustering_grid_manifest.csv")
    summary = (
        manifest.groupby(["n_pcs", "status"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ["completed", "skipped_user_approved", "failed", "planned", "running"]:
        if column not in summary:
            summary[column] = 0
    summary["manifest_rows"] = summary[
        ["completed", "skipped_user_approved", "failed", "planned", "running"]
    ].sum(axis=1)
    summary.to_csv(SUMMARY_PATH, index=False)

    RENDER_DIR.mkdir(parents=True, exist_ok=False)
    rendered = []
    for row in shortlist.itertuples(index=False):
        source_pdf = Path(row.figure_path)
        output_prefix = RENDER_DIR / row.candidate_label
        subprocess.run(
            ["pdftoppm", "-png", "-singlefile", "-r", "120", str(source_pdf), str(output_prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        output_png = output_prefix.with_suffix(".png")
        if not output_png.is_file():
            raise FileNotFoundError(output_png)
        rendered.append((row, Image.open(output_png).convert("RGB")))

    tile_width = max(image.width for _, image in rendered)
    image_height = max(image.height for _, image in rendered)
    header_height = 70
    tile_height = header_height + image_height
    canvas = Image.new("RGB", (tile_width * 3, tile_height * 3), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (row, image) in enumerate(rendered):
        x = (index % 3) * tile_width
        y = (index // 3) * tile_height
        canvas.paste(image, (x + (tile_width - image.width) // 2, y + header_height))
        title = f"#{row.shortlist_review_order} {row.candidate_label} | clusters={row.n_clusters}"
        detail = (
            f"min cells={row.min_cluster_n_cells}; worst sample frac={row.worst_cluster_max_sample_fraction:.3f}; "
            f"entropy={row.median_sample_entropy_normalized:.3f}"
        )
        draw.text((x + 8, y + 10), title, fill="black")
        draw.text((x + 8, y + 36), detail, fill="black")
    canvas.save(MONTAGE_PNG, optimize=True)
    canvas.save(MONTAGE_PDF, "PDF", resolution=120.0)

    lead = shortlist.iloc[0]
    README_PATH.write_text(
        "PDAC Block 1 broad-clustering manual review\n\n"
        "The full user grid is complete: 810 manifest rows, 422 executed candidates, "
        "388 user-rule skips, and 0 failures. No candidate H5AD or final selection record "
        "was created.\n\n"
        "The shortlist is non-binding and contains one representative per PC value. "
        "It restricts display candidates to 14-18 clusters, avoids clusters below 500 cells "
        "where possible, and then prioritizes lower worst-cluster single-sample fraction, "
        "higher median normalized sample entropy, proximity to 16 clusters, and larger "
        "minimum cluster size. The user must inspect the UMAPs and manually select any "
        "completed candidate; the shortlist does not constitute a final selection.\n\n"
        f"First candidate suggested for visual inspection only: {lead['candidate_label']}.\n"
        f"Shortlist table: {SHORTLIST_PATH}\n"
        f"Shortlist montage: {MONTAGE_PDF}\n"
        f"All completed metrics: {ENRICHED_PATH}\n",
        encoding="utf-8",
    )
    print(shortlist[["shortlist_review_order", "candidate_label", "n_clusters"]].to_string(index=False))


if __name__ == "__main__":
    main()

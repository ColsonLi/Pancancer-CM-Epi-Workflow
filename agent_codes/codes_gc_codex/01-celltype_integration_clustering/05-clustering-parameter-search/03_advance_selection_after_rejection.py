#!/usr/bin/env python3
"""Advance automatic clustering selection after a clean-rerun stability rejection.

The rejected selection is preserved as immutable evidence before the canonical
selection state is advanced to the next row of the completed automatic ranking.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
WF = ROOT / "epi-cm-core-workflow"
TABLE = WF / "tables/01-celltype_integration_clustering/05-clustering-parameter-search"
RANKING = TABLE / "automatic_candidate_ranking.csv"
SELECTION = TABLE / "selected_clustering.csv"
REJECTIONS = TABLE / "selection-rejections"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rejected-rank", type=int, required=True)
    parser.add_argument("--next-rank", type=int, required=True)
    parser.add_argument("--observed-ari", type=float, required=True)
    parser.add_argument("--required-ari", type=float, required=True)
    parser.add_argument("--grid-clusters", type=int, required=True)
    parser.add_argument("--rerun-clusters", type=int, required=True)
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    if args.next_rank != args.rejected_rank + 1:
        raise ValueError("Fallback must advance exactly one automatic rank")
    selection = pd.read_csv(SELECTION)
    ranking = pd.read_csv(RANKING)
    if len(selection) != 1 or int(selection.iloc[0]["automatic_rank"]) != args.rejected_rank:
        raise RuntimeError("Canonical selection does not match the rejected rank")
    candidate = ranking.loc[ranking["automatic_rank"].eq(args.next_rank)]
    if len(candidate) != 1 or str(candidate.iloc[0]["status"]) != "completed":
        raise RuntimeError("Next ranked candidate is not a unique completed grid result")

    rejection_csv = REJECTIONS / f"rank-{args.rejected_rank:03d}_clean-rerun-rejection.csv"
    rejection_txt = REJECTIONS / f"rank-{args.rejected_rank:03d}_clean-rerun-rejection.txt"
    if rejection_csv.exists() or rejection_txt.exists():
        raise FileExistsError("Rejection evidence already exists; refusing duplicate advance")
    REJECTIONS.mkdir(parents=True, exist_ok=True)
    rejected = selection.copy()
    rejected["clean_rerun_checkpoint_ari"] = args.observed_ari
    rejected["required_checkpoint_ari"] = args.required_ari
    rejected["grid_n_clusters"] = args.grid_clusters
    rejected["rerun_n_clusters"] = args.rerun_clusters
    rejected["rejection_reason"] = "clean_rerun_checkpoint_ari_below_required_threshold"
    atomic_csv(rejected, rejection_csv)
    rejection_txt.write_text(
        f"Automatic rank {args.rejected_rank} rejected after clean full-cell GPU rerun.\n"
        f"Observed checkpoint ARI: {args.observed_ari:.9f}\n"
        f"Required checkpoint ARI: {args.required_ari:.9f}\n"
        f"Grid clusters: {args.grid_clusters}; rerun clusters: {args.rerun_clusters}\n"
        f"Action: advance without grid reduction to automatic rank {args.next_rank}.\n"
    )

    row = candidate.iloc[0]
    parent = rejection_csv.relative_to(WF)
    final = pd.DataFrame(
        [
            {
                "n_pcs": int(row.n_pcs),
                "n_neighbors": int(row.n_neighbors),
                "resolution": float(row.resolution),
                "n_clusters": int(row.n_clusters),
                "raw_cluster_column": str(row.raw_cluster_column),
                "integration_source": "Harmony",
                "source_h5ad": str(row.source_h5ad),
                "source_sha256": str(row.source_sha256),
                "selection_mode": "automatic_fallback_after_clean_rerun_rejection",
                "automatic_rank": int(row.automatic_rank),
                "composite_score": float(row.composite_score),
                "selection_note": (
                    f"Automatic rank {args.rejected_rank} rejected at ARI "
                    f"{args.observed_ari:.9f} < {args.required_ari:.9f}; testing next rank"
                ),
                "parent_rejection_record": str(parent),
                "random_seed": int(row.seed),
                "downsampled": False,
            }
        ]
    )
    atomic_csv(final, SELECTION)
    print(final.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

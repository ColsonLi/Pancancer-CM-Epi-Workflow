#!/usr/bin/env python3
"""Build the seven-lineage manifest after the user's B/Epi parameter overrides."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
MODULE = "02-cell_subtype_integration_clustering"
TASK = "04-subtype-selected-clustering"
OLD = "selected-manual-pcs25-nn50-res0p3-v4"
NEW = "selected-user-override-b25-40-0p5-epi40-40-0p4-v5"
OVERRIDDEN = {"b_cells", "epithelial_cells"}
LINEAGES = (
    "b_cells",
    "endothelial_cells",
    "epithelial_cells",
    "myeloid_cells",
    "plasma_cells",
    "stromal_cells",
    "t_cells",
)


def main() -> None:
    table_root = ROOT / "tables" / MODULE / TASK
    output = table_root / NEW / "final_selected_lineage_manifest.csv"
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for lineage in LINEAGES:
        version = NEW if lineage in OVERRIDDEN else OLD
        source = table_root / version / lineage / "selected_clustering_parameters.csv"
        row = pd.read_csv(source).iloc[0].to_dict()
        selected_h5ad = Path(row["selected_h5ad"])
        obj = ad.read_h5ad(selected_h5ad, backed="r")
        cluster_key = str(row["cluster_key"])
        observed_k = int(obj.obs[cluster_key].astype(str).nunique())
        audited = {
            "lineage": lineage,
            "n_cells": int(obj.n_obs),
            "n_pcs": int(row["n_pcs"]),
            "n_neighbors": int(row["n_neighbors"]),
            "resolution": float(row["resolution"]),
            "n_clusters": observed_k,
            "cluster_key": cluster_key,
            "clean_rerun_checkpoint_ari": float(row["clean_rerun_checkpoint_ari"]),
            "selected_h5ad": str(selected_h5ad),
            "selected_parameters_csv": str(source),
            "source_version": version,
            "user_overridden_in_v5": lineage in OVERRIDDEN,
            "reused_validated_v4_result": lineage not in OVERRIDDEN,
            "h5ad_readable": True,
            "cluster_key_present": cluster_key in obj.obs,
            "cluster_count_matches_record": observed_k == int(row["n_clusters"]),
            "umap_present": "X_umap" in obj.obsm,
            "raw_present": obj.raw is not None,
            "backend": row["backend"],
            "random_seed": int(row["random_seed"]),
            "downsampled": bool(row["downsampled"]),
        }
        obj.file.close()
        rows.append(audited)
    frame = pd.DataFrame(rows)
    if not frame[[
        "h5ad_readable", "cluster_key_present", "cluster_count_matches_record",
        "umap_present", "raw_present",
    ]].all().all():
        raise RuntimeError("Final selected-lineage audit failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    (output.parent / "final_selection_readme.txt").write_text(
        "Current seven-lineage GC subtype clustering selection.\n"
        "B and epithelial were clean-rerun from their v4 Harmony objects using the user's exact v5 parameters.\n"
        "Endothelial, myeloid, plasma, stromal, and T reuse their already validated v4 selected objects.\n"
        "No lineage was downsampled; clustering used rapids-singlecell GPU and seed 42.\n"
    )
    print(frame.to_csv(index=False))


if __name__ == "__main__":
    main()

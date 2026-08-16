#!/usr/bin/env python3
"""Project all GC v2 subtype annotations by exact cell IDs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
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

LINEAGES = {
    "b_cells": ("b", "B Cells"),
    "endothelial_cells": ("endo", "Endothelial Cells"),
    "epithelial_cells": ("epi", "Epithelial Cells"),
    "mast_cells": ("mast", "Mast Cells"),
    "myeloid_cells": ("mye", "Myeloid Cells"),
    "plasma_cells": ("plasma", "Plasma Cells"),
    "stromal_cells": ("stromal", "Stromal Cells"),
    "t_cells": ("t", "T Cells"),
}


def digest(ids) -> str:
    h = hashlib.sha256()
    for value in ids:
        h.update(str(value).encode())
        h.update(b"\0")
    return h.hexdigest()


def write_csv_once(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    target_path = WF / "h5ad" / "01-celltype_integration_clustering" / "07-score-rank-qc" / "adata_anno_score_genes_rank_consistent.h5ad"
    out_dir = WF / "h5ad" / MODULE / "06-project-subtypes-to-full-adata-v2"
    table_dir = WF / "tables" / MODULE / "06-project-subtypes-to-full-adata-v2"
    fig_dir = WF / "figures" / MODULE / "06-project-subtypes-to-full-adata-v2"
    output_path = out_dir / "adata_anno_cellsubtype.h5ad"
    audit_path = table_dir / "exact_projection_audit.json"
    if output_path.exists() or output_path.with_name(f".{output_path.stem}.partial{output_path.suffix}").exists() or audit_path.exists():
        raise FileExistsError("v2 projection output already/residually exists")

    full = ad.read_h5ad(target_path)
    if not full.obs_names.is_unique:
        raise RuntimeError("Projection target cell IDs are not unique")
    target_ids = full.obs_names
    transfer_columns = ["cell_subtype", "functional_state", "canonical_identity", "annotation_confidence", "annotation_source"]
    projected = pd.DataFrame(index=target_ids, columns=transfer_columns + ["cell_type", "subtype_lineage", "subtype_cluster"])
    rows = []
    seen = set()

    for lineage, (abbrev, broad_label) in LINEAGES.items():
        source_path = WF / "h5ad" / MODULE / "05-subtype-deg-annotation-v2" / lineage / f"adata_{abbrev}.h5ad"
        source = ad.read_h5ad(source_path, backed="r")
        if not source.obs_names.is_unique:
            raise RuntimeError(f"Duplicate source IDs: {lineage}")
        ids = source.obs_names
        duplicate_across = sum(i in seen for i in ids)
        missing = ids.difference(target_ids)
        if duplicate_across or len(missing):
            raise RuntimeError(f"Projection mismatch for {lineage}: duplicate_across={duplicate_across}, missing={len(missing)}")
        seen.update(ids)
        cluster_cols = [c for c in source.obs.columns if c.startswith("leiden_selected_v2_")]
        if len(cluster_cols) != 1:
            raise RuntimeError(f"Expected one v2 cluster column for {lineage}, found {cluster_cols}")
        frame = source.obs[transfer_columns].copy()
        frame["cell_type"] = broad_label
        frame["subtype_lineage"] = lineage
        frame["subtype_cluster"] = lineage + ":" + source.obs[cluster_cols[0]].astype(str)
        projected.loc[ids, frame.columns] = frame.to_numpy()
        rows.append({
            "lineage": lineage, "broad_label": broad_label, "source_h5ad": str(source_path),
            "n_source_cells": source.n_obs, "n_matched_exact": len(ids), "n_missing": len(missing),
            "n_duplicate_across_lineages": duplicate_across, "source_cell_id_sha256": digest(ids),
            "n_subtypes": source.obs.cell_subtype.nunique(),
        })
        source.file.close()

    missing_target = projected.cell_subtype.isna()
    if missing_target.any() or len(seen) != full.n_obs:
        raise RuntimeError(f"Incomplete exact projection: missing_target={int(missing_target.sum())}, seen={len(seen)}, target={full.n_obs}")
    if projected.cell_subtype.nunique() != 67:
        raise RuntimeError(f"Expected 67 globally unique v2 subtypes, found {projected.cell_subtype.nunique()}")

    for column in ("cell_type", "cell_subtype", "functional_state", "canonical_identity", "annotation_confidence", "annotation_source", "subtype_lineage", "subtype_cluster"):
        if column in full.obs:
            backup = f"{column}_pre_v2"
            if backup in full.obs:
                raise RuntimeError(f"Backup column already exists: {backup}")
            full.obs[backup] = full.obs[column].copy()
        full.obs[column] = pd.Categorical(projected[column].astype(str).to_numpy())

    report = pd.DataFrame(rows)
    write_csv_once(report, table_dir / "exact_projection_by_lineage.csv")
    counts = full.obs.cell_subtype.astype(str).value_counts().rename_axis("cell_subtype").reset_index(name="n_cells")
    write_csv_once(counts, table_dir / "projected_subtype_counts.csv")
    uniqueness = (
        full.obs.assign(cell_subtype=full.obs.cell_subtype.astype(str), subtype_lineage=full.obs.subtype_lineage.astype(str))
        .groupby("cell_subtype", observed=True).subtype_lineage.nunique().rename("n_lineages").reset_index()
    )
    if not uniqueness.n_lineages.eq(1).all():
        raise RuntimeError("Subtype names are not globally lineage-unique")
    write_csv_once(uniqueness, table_dir / "cell_subtype_global_uniqueness_audit.csv")

    broad_order = list(dict.fromkeys(v[1] for v in LINEAGES.values()))
    subtype_order = sorted(full.obs.cell_subtype.astype(str).unique())
    full.obs["cell_type"] = full.obs.cell_type.cat.reorder_categories([x for x in broad_order if x in full.obs.cell_type.cat.categories])
    full.obs["cell_subtype"] = full.obs.cell_subtype.cat.reorder_categories(subtype_order)
    broad_palette = sc.pl.palettes.default_20[: len(broad_order)]
    subtype_palette = sc.pl.palettes.default_102[: len(subtype_order)]
    broad_color = dict(zip(broad_order, broad_palette))
    subtype_color = dict(zip(subtype_order, subtype_palette))
    write_csv_once(pd.DataFrame({"cell_type": broad_order, "color": broad_palette}), table_dir / "broad_palette.csv")
    write_csv_once(pd.DataFrame({"cell_subtype": subtype_order, "color": subtype_palette}), table_dir / "subtype_palette.csv")

    fig_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.autoshow = False
    sc.settings.figdir = str(fig_dir)
    coarse_order = [str(x) for x in full.obs.leiden_coarse.cat.categories] if hasattr(full.obs.leiden_coarse, "cat") else sorted(full.obs.leiden_coarse.astype(str).unique())
    coarse_palette = [broad_color.get(x, "#808080") for x in coarse_order]
    sc.pl.umap(full, color=["leiden_coarse", "cell_type"], palette=coarse_palette, ncols=2,
               show=False, save="_leiden_coarse_vs_projected_cell_type.pdf")
    sc.pl.umap(full, color=["leiden_coarse", "cell_type"], palette=coarse_palette, ncols=2,
               show=False, save="_leiden_coarse_vs_projected_cell_type.svg")
    sc.pl.umap(full, color="cell_subtype", palette=subtype_palette, show=False, save="_projected_cell_subtype.pdf")
    sc.pl.umap(full, color="cell_subtype", palette=subtype_palette, show=False, save="_projected_cell_subtype.svg")

    for lineage, (abbrev, _) in LINEAGES.items():
        source_path = WF / "h5ad" / MODULE / "05-subtype-deg-annotation-v2" / lineage / f"adata_{abbrev}.h5ad"
        source = ad.read_h5ad(source_path)
        cats = sorted(source.obs.cell_subtype.astype(str).unique())
        source.obs["cell_subtype"] = pd.Categorical(source.obs.cell_subtype.astype(str), categories=cats)
        palette = [subtype_color[x] for x in cats]
        lineage_fig = fig_dir / lineage
        lineage_fig.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = str(lineage_fig)
        sc.pl.umap(source, color="cell_subtype", palette=palette, show=False, save="_projected_cell_subtype.pdf")
        sc.pl.umap(source, color="cell_subtype", palette=palette, show=False, save="_projected_cell_subtype.svg")

    audit = {
        "project": "gastric_cancer", "version": "v2", "source_full_h5ad": str(target_path),
        "output_h5ad": str(output_path), "n_target_cells": full.n_obs, "n_projected_exact": len(seen),
        "n_unmatched": 0, "n_duplicate_source_ids": 0, "n_lineages": len(LINEAGES),
        "n_global_subtypes": int(full.obs.cell_subtype.nunique()),
        "n_low_confidence_cells": int((full.obs.annotation_confidence.astype(str) == "low").sum()),
        "target_cell_id_sha256": digest(target_ids), "output_cell_id_sha256": digest(full.obs_names),
        "no_subtype_cell_filtering": True, "random_seed": SEED,
    }
    table_dir.mkdir(parents=True, exist_ok=True)
    tmp_audit = audit_path.with_name(f".{audit_path.name}.partial")
    tmp_audit.write_text(json.dumps(audit, indent=2) + "\n")
    os.replace(tmp_audit, audit_path)
    (table_dir / "readme.txt").write_text(
        "Exact-ID projection of all eight GC v2 lineage annotations to the Module 01 score/rank-consistent atlas.\n"
        "No cells were removed at the subtype stage; low-confidence marker annotations are retained and flagged.\n"
    )
    (table_dir / "package_versions.txt").write_text(
        f"python={platform.python_version()}\nanndata={ad.__version__}\nscanpy={sc.__version__}\n"
        f"numpy={np.__version__}\npandas={pd.__version__}\ncode={Path(__file__).resolve()}\nseed={SEED}\n"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_h5ad = output_path.with_name(f".{output_path.stem}.partial{output_path.suffix}")
    full.write_h5ad(tmp_h5ad)
    os.replace(tmp_h5ad, output_path)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

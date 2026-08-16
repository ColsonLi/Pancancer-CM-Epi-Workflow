#!/usr/bin/env python3
"""Project all current GC subtype annotations to the exact consistent atlas."""

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
import seaborn as sns

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
sns.set_theme(style="ticks")

CODE_DIR = Path(__file__).resolve().parent
WF = CODE_DIR.parents[2]
MODULE = "02-cell_subtype_integration_clustering"
TAG = "agent-led-current-selection-v5"
SELECT_TAG = "selected-user-override-b25-40-0p5-epi40-40-0p4-v5"
SEED = 42
LINEAGES = {
    "b_cells": ("b", "B Cells"),
    "endothelial_cells": ("endo", "Endothelial Cells"),
    "epithelial_cells": ("epi", "Epithelial Cells"),
    "myeloid_cells": ("mye", "Myeloid Cells"),
    "plasma_cells": ("plasma", "Plasma Cells"),
    "stromal_cells": ("stromal", "Stromal Cells"),
    "t_cells": ("t", "T Cells"),
}


def digest(ids) -> str:
    value = hashlib.sha256()
    for item in ids:
        value.update(str(item).encode())
        value.update(b"\0")
    return value.hexdigest()


def write_csv_once(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def write_text_once(value: str, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(value)
    os.replace(partial, path)


def write_h5ad_once(adata: ad.AnnData, path: Path) -> None:
    partial = path.with_name(f".{path.stem}.partial{path.suffix}")
    if path.exists() or partial.exists():
        raise FileExistsError(f"Refusing overwrite: {path} / {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(partial)
    os.replace(partial, path)


def main() -> None:
    target_path = (
        WF / "h5ad" / "01-celltype_integration_clustering" / "07-score-rank-qc"
        / "selected-manual-pcs25-nn50-res0p3-v4"
        / "adata_anno_score_genes_rank_consistent.h5ad"
    )
    selected_manifest_path = (
        WF / "tables" / MODULE / "04-subtype-selected-clustering" / SELECT_TAG
        / "final_selected_lineage_manifest.csv"
    )
    selected_manifest = pd.read_csv(selected_manifest_path).set_index("lineage")
    output_path = WF / "h5ad" / MODULE / "06-project-subtypes-to-full-adata" / TAG / "adata_anno_cellsubtype.h5ad"
    table_dir = WF / "tables" / MODULE / "06-project-subtypes-to-full-adata" / TAG
    figure_dir = WF / "figures" / MODULE / "06-project-subtypes-to-full-adata" / TAG
    audit_path = table_dir / "exact_projection_audit.json"
    if output_path.exists() or audit_path.exists():
        raise FileExistsError("Current exact-projection output already exists")

    full = ad.read_h5ad(target_path)
    if not full.obs_names.is_unique:
        raise RuntimeError("Projection target cell IDs are not unique")
    target_ids = full.obs_names
    transfer_columns = [
        "cell_subtype", "functional_state", "canonical_identity",
        "annotation_confidence", "annotation_source",
    ]
    projected = pd.DataFrame(
        index=target_ids,
        columns=transfer_columns + ["cell_type", "subtype_lineage", "subtype_cluster"],
    )
    seen = set()
    rows = []
    source_paths = {}
    for lineage, (abbrev, broad_label) in LINEAGES.items():
        source_path = (
            WF / "h5ad" / MODULE / "05-subtype-deg-annotation" / TAG / lineage
            / f"adata_{abbrev}.h5ad"
        )
        source_paths[lineage] = source_path
        source = ad.read_h5ad(source_path, backed="r")
        if not source.obs_names.is_unique:
            raise RuntimeError(f"Duplicate source IDs: {lineage}")
        ids = source.obs_names
        missing = ids.difference(target_ids)
        duplicate_across = sum(item in seen for item in ids)
        if len(missing) or duplicate_across:
            raise RuntimeError(
                f"Projection mismatch for {lineage}: missing={len(missing)}, duplicate={duplicate_across}"
            )
        cluster_key = str(selected_manifest.loc[lineage, "cluster_key"])
        if cluster_key not in source.obs:
            raise RuntimeError(f"Missing source cluster key for {lineage}: {cluster_key}")
        frame = source.obs[transfer_columns].copy()
        frame["cell_type"] = broad_label
        frame["subtype_lineage"] = lineage
        frame["subtype_cluster"] = lineage + ":" + source.obs[cluster_key].astype(str)
        projected.loc[ids, frame.columns] = frame.to_numpy()
        seen.update(ids)
        row = {
            "lineage": lineage,
            "broad_label": broad_label,
            "source_h5ad": str(source_path),
            "cluster_key": cluster_key,
            "n_source_cells": int(source.n_obs),
            "n_matched_exact": len(ids),
            "n_missing": len(missing),
            "n_duplicate_across_lineages": duplicate_across,
            "source_cell_id_sha256": digest(ids),
            "n_subtypes": int(source.obs.cell_subtype.nunique()),
        }
        rows.append(row)
        write_csv_once(pd.DataFrame([row]), table_dir / lineage / "projection_match_report.csv")
        source.file.close()

    missing_target = projected.cell_subtype.isna()
    if missing_target.any() or len(seen) != full.n_obs:
        raise RuntimeError(
            f"Incomplete exact projection: missing_target={int(missing_target.sum())}, "
            f"seen={len(seen)}, target={full.n_obs}"
        )
    if projected.cell_subtype.nunique() != 62:
        raise RuntimeError(f"Expected 62 current subtypes, found {projected.cell_subtype.nunique()}")

    for column in (
        "cell_type", "cell_subtype", "functional_state", "canonical_identity",
        "annotation_confidence", "annotation_source", "subtype_lineage", "subtype_cluster",
    ):
        if column in full.obs:
            backup = f"{column}_pre_subtype_v5"
            if backup in full.obs:
                raise RuntimeError(f"Backup column already exists: {backup}")
            full.obs[backup] = full.obs[column].copy()
        full.obs[column] = pd.Categorical(projected[column].astype(str).to_numpy())

    report = pd.DataFrame(rows)
    write_csv_once(report, table_dir / "exact_projection_by_lineage.csv")
    counts = (
        full.obs.cell_subtype.astype(str).value_counts().rename_axis("cell_subtype")
        .reset_index(name="n_cells")
    )
    write_csv_once(counts, table_dir / "projected_subtype_counts.csv")
    uniqueness = (
        full.obs.assign(
            cell_subtype=full.obs.cell_subtype.astype(str),
            subtype_lineage=full.obs.subtype_lineage.astype(str),
        )
        .groupby("cell_subtype", observed=True).subtype_lineage.nunique()
        .rename("n_lineages").reset_index()
    )
    if not uniqueness.n_lineages.eq(1).all():
        raise RuntimeError("Subtype names are not globally lineage-unique")
    write_csv_once(uniqueness, table_dir / "cell_subtype_global_uniqueness_audit.csv")

    broad_order = [value[1] for value in LINEAGES.values()]
    subtype_order = sorted(full.obs.cell_subtype.astype(str).unique())
    full.obs["leiden_coarse"] = pd.Categorical(
        full.obs.leiden_coarse.astype(str), categories=broad_order,
    )
    full.obs["cell_type"] = pd.Categorical(
        full.obs.cell_type.astype(str), categories=broad_order,
    )
    full.obs["cell_subtype"] = pd.Categorical(
        full.obs.cell_subtype.astype(str), categories=subtype_order,
    )
    broad_palette = sc.pl.palettes.default_20[:len(broad_order)]
    subtype_palette = sc.pl.palettes.default_102[:len(subtype_order)]
    broad_color = dict(zip(broad_order, broad_palette))
    subtype_color = dict(zip(subtype_order, subtype_palette))
    write_csv_once(pd.DataFrame({"cell_type": broad_order, "color": broad_palette}), table_dir / "broad_palette.csv")
    write_csv_once(pd.DataFrame({"cell_subtype": subtype_order, "color": subtype_palette}), table_dir / "subtype_palette.csv")

    figure_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.autoshow = False
    sc.settings.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
    sc.settings.figdir = figure_dir
    sc.pl.umap(
        full, color=["leiden_coarse", "cell_type"], palette=broad_palette,
        ncols=2, wspace=0.4, show=False,
        save="_leiden_coarse_vs_projected_cell_type.pdf",
    )
    sc.pl.umap(
        full, color=["leiden_coarse", "cell_type"], palette=broad_palette,
        ncols=2, wspace=0.4, show=False,
        save="_leiden_coarse_vs_projected_cell_type.svg",
    )
    sc.pl.umap(full, color="cell_subtype", palette=subtype_palette, show=False, save="_projected_cell_subtype.pdf")
    sc.pl.umap(full, color="cell_subtype", palette=subtype_palette, show=False, save="_projected_cell_subtype.svg")

    for lineage, source_path in source_paths.items():
        source = ad.read_h5ad(source_path)
        categories = sorted(source.obs.cell_subtype.astype(str).unique())
        source.obs["cell_subtype"] = pd.Categorical(
            source.obs.cell_subtype.astype(str), categories=categories,
        )
        palette = [subtype_color[value] for value in categories]
        lineage_dir = figure_dir / lineage
        lineage_dir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = lineage_dir
        sc.pl.umap(
            source, color="cell_subtype", palette=palette, show=False,
            save="_projected_cell_subtype.pdf",
        )
        sc.pl.umap(
            source, color="cell_subtype", palette=palette, show=False,
            save="_projected_cell_subtype.svg",
        )

    audit = {
        "project": "gastric_cancer",
        "annotation_tag": TAG,
        "source_full_h5ad": str(target_path),
        "output_h5ad": str(output_path),
        "n_target_cells": int(full.n_obs),
        "n_projected_exact": len(seen),
        "n_unmatched": 0,
        "n_duplicate_source_ids": 0,
        "n_lineages": len(LINEAGES),
        "n_global_subtypes": int(full.obs.cell_subtype.nunique()),
        "n_low_confidence_cells": int((full.obs.annotation_confidence.astype(str) == "low").sum()),
        "target_cell_id_sha256": digest(target_ids),
        "output_cell_id_sha256": digest(full.obs_names),
        "no_subtype_cell_filtering": True,
        "random_seed": SEED,
    }
    write_text_once(json.dumps(audit, indent=2) + "\n", audit_path)
    write_text_once(
        "Exact-ID projection of all seven current GC lineage annotations to the Module 01 score/rank-consistent atlas.\n"
        "No subtype-stage cells were removed; all 284341 target cells were matched exactly once.\n",
        table_dir / "readme.txt",
    )
    write_text_once(
        f"python={platform.python_version()}\nanndata={ad.__version__}\nscanpy={sc.__version__}\n"
        f"numpy={np.__version__}\npandas={pd.__version__}\nseaborn={sns.__version__}\n"
        f"code={Path(__file__).resolve()}\nseed={SEED}\n",
        table_dir / "package_versions.txt",
    )
    write_h5ad_once(full, output_path)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

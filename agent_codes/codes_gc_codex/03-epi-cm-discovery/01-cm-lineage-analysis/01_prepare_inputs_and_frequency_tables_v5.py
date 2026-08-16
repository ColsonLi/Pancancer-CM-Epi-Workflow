#!/usr/bin/env python3
"""Prepare GC sample/subtype frequency tables and Block 03 status routing."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
WF = CODE_DIR.parents[2]
BLOCK = "03-epi-cm-discovery"
INPUT_TAG = "agent-led-current-selection-v5"
TAG = "agent-led-current-selection-v5-preflight-r2"
MIN_NON_EPI_CELLS = 50
MIN_EPI_CELLS = 1
SEED = 42
INPUT_H5AD = (
    WF / "h5ad" / "02-cell_subtype_integration_clustering"
    / "06-project-subtypes-to-full-adata" / INPUT_TAG / "adata_anno_cellsubtype.h5ad"
)
OUT = (
    WF / "tables" / BLOCK / "01-cm-lineage-analysis"
    / "01_prepare_inputs_and_frequency_tables" / TAG
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


def main() -> None:
    completion = OUT / "preflight_completion.json"
    if completion.exists():
        raise FileExistsError(completion)
    adata = ad.read_h5ad(INPUT_H5AD, backed="r")
    required = {"sample", "status", "cell_type", "cell_subtype"}
    missing = required - set(adata.obs.columns)
    if missing:
        raise RuntimeError(f"Missing required obs columns: {sorted(missing)}")
    obs = adata.obs[list(required)].copy()
    obs["sample"] = obs["sample"].astype(str)
    obs["raw_status"] = obs["status"].astype(str)
    status_map = {"tumor": "tumor", "normal": "normal-like", "normal-like": "normal-like"}
    obs["status"] = obs.raw_status.map(status_map)
    if obs.status.isna().any():
        raise RuntimeError(f"Unresolved statuses: {sorted(obs.loc[obs.status.isna(), 'raw_status'].unique())}")
    obs["cell_type"] = obs["cell_type"].astype(str)
    obs["cell_subtype"] = obs["cell_subtype"].astype(str)

    sample_status_all = obs[["sample", "status", "raw_status"]].drop_duplicates()
    conflicts = sample_status_all.groupby("sample").status.nunique()
    if (conflicts > 1).any():
        raise RuntimeError("Conflicting canonical statuses within biological samples")
    sample_status_all = sample_status_all.drop_duplicates("sample").sort_values("sample")
    prefix = obs.cell_subtype.str.split("_", n=1).str[0]
    prefix_map = {
        "Epi": "Epithelial Cells", "T": "T Cells", "Mye": "Myeloid Cells",
        "B": "B Cells", "Plasma": "Plasma Cells", "Endo": "Endothelial Cells",
        "Stromal": "Stromal Cells",
    }
    obs["cell_type_from_subtype_prefix"] = prefix.map(prefix_map)
    unresolved_prefix = sorted(prefix[obs.cell_type_from_subtype_prefix.isna()].unique())
    if unresolved_prefix:
        raise RuntimeError(f"Unresolved subtype prefixes: {unresolved_prefix}")
    conflict_mask = obs.cell_type.ne(obs.cell_type_from_subtype_prefix)
    conflict_table = (
        obs.loc[conflict_mask, ["sample", "cell_type", "cell_subtype", "cell_type_from_subtype_prefix"]]
        .value_counts().rename("n_cells").reset_index()
    )
    write_csv_once(conflict_table, OUT / "cell_type_subtype_prefix_conflicts.csv")
    if conflict_mask.any():
        obs.loc[:, "cell_type"] = obs.cell_type_from_subtype_prefix

    epi_mask = obs.cell_type.eq("Epithelial Cells")
    non_epi_counts_all = pd.crosstab(obs.loc[~epi_mask, "sample"], obs.loc[~epi_mask, "cell_subtype"]).sort_index()
    epi_counts_all = pd.crosstab(obs.loc[epi_mask, "sample"], obs.loc[epi_mask, "cell_subtype"]).sort_index()
    all_samples = pd.Index(sample_status_all["sample"], name="sample")
    compartment = pd.DataFrame(index=all_samples)
    compartment["non_epi_cells"] = non_epi_counts_all.sum(axis=1).reindex(all_samples).fillna(0).astype(int)
    compartment["epi_cells"] = epi_counts_all.sum(axis=1).reindex(all_samples).fillna(0).astype(int)
    compartment = compartment.join(sample_status_all.set_index("sample")[["status", "raw_status"]])
    compartment["has_status"] = compartment.status.notna()
    compartment["has_non_epi"] = compartment.non_epi_cells.ge(MIN_NON_EPI_CELLS)
    compartment["has_epi_for_association"] = compartment.epi_cells.ge(MIN_EPI_CELLS)
    compartment["keep_for_cm"] = compartment.has_status & compartment.has_non_epi
    compartment["keep_for_epi_cm"] = compartment.keep_for_cm & compartment.has_epi_for_association

    def exclusion_reason(row):
        reasons = []
        if not row.has_status:
            reasons.append("missing_status")
        if not row.has_non_epi:
            reasons.append("non_epi_cells_below_threshold")
        if not row.has_epi_for_association:
            reasons.append("epi_cells_below_association_threshold")
        return ";".join(reasons) if reasons else "kept"

    compartment["exclusion_reason"] = compartment.apply(exclusion_reason, axis=1)
    cm_samples = compartment.index[compartment.keep_for_cm].sort_values()
    epi_cm_samples = compartment.index[compartment.keep_for_epi_cm].sort_values()
    if len(cm_samples) == 0:
        raise RuntimeError("No samples pass CM eligibility")

    non_epi_counts = non_epi_counts_all.reindex(index=cm_samples, fill_value=0).sort_index(axis=1)
    non_epi_freq = non_epi_counts.div(non_epi_counts.sum(axis=1), axis=0).fillna(0)
    epi_counts = epi_counts_all.reindex(index=epi_cm_samples, fill_value=0).sort_index(axis=1)
    epi_freq = epi_counts.div(epi_counts.sum(axis=1), axis=0).fillna(0)
    sample_status = compartment.loc[cm_samples, ["status", "raw_status"]].copy()

    if non_epi_counts.shape[1] != 50 or epi_counts.shape[1] != 12:
        raise RuntimeError(f"Expected 50 non-epi and 12 epi subtypes, got {non_epi_counts.shape[1]} and {epi_counts.shape[1]}")
    if not np.allclose(non_epi_freq.sum(axis=1), 1.0):
        raise RuntimeError("Non-epithelial frequency rows do not sum to one")
    if not np.allclose(epi_freq.sum(axis=1), 1.0):
        raise RuntimeError("Epithelial frequency rows do not sum to one")

    write_csv_once(sample_status_all, OUT / "sample_status_all.csv")
    write_csv_once(sample_status.reset_index(), OUT / "sample_status.csv")
    write_csv_once(compartment.reset_index(), OUT / "sample_inclusion_exclusion.csv")
    write_csv_once(compartment.reset_index()[["sample", "non_epi_cells", "epi_cells"]], OUT / "sample_compartment_cell_counts.csv")
    write_csv_once(non_epi_counts.reset_index(), OUT / "non_epi_subtype_counts.csv")
    write_csv_once(non_epi_freq.reset_index(), OUT / "non_epi_subtype_frequency.csv")
    write_csv_once(epi_counts.reset_index(), OUT / "epi_subtype_counts.csv")
    write_csv_once(epi_freq.reset_index(), OUT / "epi_subtype_frequency.csv")
    subtype_map = obs[["cell_subtype", "cell_type", "cell_type_from_subtype_prefix"]].drop_duplicates().sort_values("cell_subtype")
    write_csv_once(subtype_map, OUT / "cell_subtype_to_broad_cell_type.csv")

    eligible_status = sample_status.reset_index()
    observed = set(eligible_status.status)
    if observed == {"tumor", "normal-like"}:
        mode = "tumor_normal"
    elif observed == {"tumor"}:
        mode = "tumor_only"
    elif observed == {"normal-like"}:
        mode = "normal_only"
    else:
        raise RuntimeError(f"Unsupported eligible status set: {sorted(observed)}")
    mode_rows = (
        eligible_status.groupby("status", observed=True)["sample"].nunique()
        .rename("n_samples").reset_index().rename(columns={"status": "canonical_status"})
    )
    mode_rows.insert(0, "detected_mode", mode)
    mode_rows["n_total_samples"] = len(cm_samples)
    mode_rows["status_source"] = "annotated_h5ad_obs_status"
    mode_rows["status_balance_applied"] = mode == "tumor_normal"
    mode_rows["cm_classification_available"] = mode == "tumor_normal"
    write_csv_once(mode_rows, OUT / "cm_status_mode_detection.csv")
    write_text_once(json.dumps({
        "detected_mode": mode,
        "n_total_samples": len(cm_samples),
        "status_source": "annotated_h5ad_obs_status",
        "status_balance_applied": mode == "tumor_normal",
        "cm_classification_available": mode == "tumor_normal",
        "raw_to_canonical_status": {"tumor": "tumor", "normal": "normal-like"},
        "included_samples": cm_samples.astype(str).tolist(),
    }, indent=2) + "\n", OUT / "cm_status_mode.json")

    parameters = pd.DataFrame([{
        "step": "prepare_inputs_and_frequency_tables", "input_h5ad": str(INPUT_H5AD),
        "output_h5ad_or_object": "sample_by_subtype_frequency_tables",
        "n_obs_before": int(adata.n_obs), "n_obs_after": int(adata.n_obs),
        "n_vars_before": int(adata.n_vars), "n_vars_after": int(adata.n_vars),
        "sample_column": "sample", "status_column": "status",
        "cell_type_column": "cell_type", "cell_subtype_column": "cell_subtype",
        "epithelial_type": "Epithelial Cells",
        "min_non_epi_cells_per_sample": MIN_NON_EPI_CELLS,
        "min_epi_cells_per_sample": MIN_EPI_CELLS,
        "n_all_samples": len(all_samples), "n_keep_for_cm": len(cm_samples),
        "n_keep_for_epi_cm": len(epi_cm_samples), "n_non_epi_subtypes": non_epi_counts.shape[1],
        "n_epi_subtypes": epi_counts.shape[1], "random_seed": SEED,
        "code_file": str(Path(__file__).resolve()),
    }])
    write_csv_once(parameters, OUT / "prepare_inputs_parameters.csv")
    status_counts = compartment.groupby(["status", "keep_for_cm", "keep_for_epi_cm"], observed=True).size().rename("n_samples").reset_index()
    write_csv_once(status_counts, OUT / "sample_eligibility_summary_by_status.csv")
    result = {
        "input_h5ad": str(INPUT_H5AD), "n_cells": int(adata.n_obs),
        "n_all_samples": len(all_samples), "n_keep_for_cm": len(cm_samples),
        "n_keep_for_epi_cm": len(epi_cm_samples), "n_non_epi_subtypes": 50,
        "n_epi_subtypes": 12, "detected_mode": mode,
        "eligible_status_counts": sample_status.status.value_counts().to_dict(),
    }
    write_text_once(json.dumps(result, indent=2) + "\n", completion)
    write_text_once(
        "GC Block 03 frequency-table and status-routing preflight.\n"
        "CM/NMF eligibility uses non-epithelial cells only; epithelial eligibility is applied only to Epi-CM association.\n",
        OUT / "readme.txt",
    )
    write_text_once(
        f"python={platform.python_version()}\nanndata={ad.__version__}\n"
        f"numpy={np.__version__}\npandas={pd.__version__}\n"
        f"code={Path(__file__).resolve()}\nseed={SEED}\n",
        OUT / "package_versions.txt",
    )
    adata.file.close()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare canonical display-only W/H matrices and row annotations."""

from __future__ import annotations

import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
WF = CODE_DIR.parents[2]
BLOCK = "03-epi-cm-discovery"
TAG = "agent-led-current-selection-v5-cm-analysis-display-r2"
NMF_TAG = "agent-led-current-selection-v5-nmf-r1"
NMF = WF / "tables" / BLOCK / "01-cm-lineage-analysis" / "02_balanced_joint_nmf" / NMF_TAG
OUT = WF / "tables" / BLOCK / "01-cm-lineage-analysis" / "05_prepare_display_matrices" / TAG
ATLAS = WF / "h5ad" / "02-cell_subtype_integration_clustering" / "06-project-subtypes-to-full-adata" / "agent-led-current-selection-v5" / "adata_anno_cellsubtype.h5ad"


def write(frame: pd.DataFrame, name: str, index_name: str) -> None:
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index(names=index_name).to_csv(path, index=False)


def scales(frame: pd.DataFrame):
    z = frame.sub(frame.mean(axis=0), axis=1).div(frame.std(axis=0, ddof=0).replace(0, np.nan), axis=1).fillna(0)
    q1, q3 = frame.quantile(0.25), frame.quantile(0.75)
    robust = frame.sub(frame.median(axis=0), axis=1).div((q3 - q1).replace(0, np.nan), axis=1).fillna(0)
    minimum = frame.min(axis=0)
    minmax = frame.sub(minimum, axis=1).div((frame.max(axis=0) - minimum).replace(0, np.nan), axis=1).fillna(0)
    return z, robust, minmax


def main() -> None:
    if (OUT / "display_matrix_completion.json").exists():
        raise FileExistsError(OUT / "display_matrix_completion.json")
    activity = pd.read_csv(NMF / "W_df.csv").set_index("sample")
    loading = pd.read_csv(NMF / "loading_df_cell_subtype_by_CM.csv").set_index("cell_subtype")
    loading_fraction = pd.read_csv(NMF / "loading_df_cell_subtype_by_CM_fraction.csv").set_index("cell_subtype")
    wz, wr, wm = scales(activity)
    hz, hr, hm = scales(loading)
    write(activity, "w_df_activity_sample_by_CM_raw.csv", "sample")
    write(wz, "w_df_activity_sample_by_CM_zscore.csv", "sample")
    write(wr, "w_df_activity_sample_by_CM_robust.csv", "sample")
    write(wm, "w_df_activity_sample_by_CM_standard_scale_col.csv", "sample")
    write(loading, "h_df_loading_cell_subtype_by_CM_raw.csv", "cell_subtype")
    write(hz, "h_df_loading_cell_subtype_by_CM_zscore.csv", "cell_subtype")
    write(hr, "h_df_loading_cell_subtype_by_CM_robust.csv", "cell_subtype")
    write(hm, "h_df_loading_cell_subtype_by_CM_standard_scale_col.csv", "cell_subtype")
    write(loading, "balanced_joint_cm_subtype_loadings_raw_from_H_df.csv", "cell_subtype")
    write(loading_fraction, "balanced_joint_cm_subtype_loadings_fraction_from_H_df.csv", "cell_subtype")

    atlas = ad.read_h5ad(ATLAS, backed="r")
    metadata = atlas.obs[["sample", "series", "status"]].astype(str).drop_duplicates()
    conflicts = metadata.groupby("sample").agg({"series": "nunique", "status": "nunique"})
    if (conflicts > 1).any().any():
        raise RuntimeError("Sample-level series/status conflict")
    metadata = metadata.drop_duplicates("sample").set_index("sample").loc[activity.index]
    metadata["Status"] = metadata.status.map({"normal": "normal-like", "tumor": "tumor"})
    metadata["Series"] = metadata.series
    annotations = metadata[["Series", "Status"]]
    order = annotations.reset_index().sort_values(["Series", "Status", "sample"])["sample"]
    plot_matrix = wm.loc[order]
    write(plot_matrix, "w_df_activity_sample_by_CM_standard_scale_col_with_series_plot_matrix.csv", "sample")
    write(annotations.loc[order], "w_df_activity_sample_by_CM_standard_scale_col_with_series_row_annotations.csv", "sample")
    atlas.file.close()
    result = {"n_samples": activity.shape[0], "n_cms": activity.shape[1], "n_subtypes": loading.shape[0], "display_scales": ["raw", "zscore_ddof0", "robust_median_IQR", "standard_scale_col_minmax"]}
    (OUT / "display_matrix_completion.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "readme.txt").write_text("Display-only activity/loading matrices; none are used for NMF, classification, edges, or associations.\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

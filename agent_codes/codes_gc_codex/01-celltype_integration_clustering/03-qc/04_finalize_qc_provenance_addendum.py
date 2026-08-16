#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parents[4]
TABLE = ROOT / "epi-cm-core-workflow/tables/01-celltype_integration_clustering/03-qc"


def write_new(name: str, frame: pd.DataFrame) -> None:
    path = TABLE / name
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False)


def main() -> None:
    steps = pd.read_csv(TABLE / "qc_step_summary.csv").set_index("step")
    failed = pd.read_csv(TABLE / "02_scrublet_failed_sample_exclusions.csv")
    points = pd.DataFrame([
        {"execution_point": "merged_input_before_initial_filter", "n_obs": int(steps.loc["merged_input", "n_cells"]), "n_vars": int(steps.loc["merged_input", "n_genes"])},
        {"execution_point": "after_initial_min_genes_before_scrublet", "n_obs": int(steps.loc["after_initial_min_genes", "n_cells"]), "n_vars": int(steps.loc["after_initial_min_genes", "n_genes"])},
        {"execution_point": "after_gene_filter_before_final_cell_filter", "n_obs": int(steps.loc["after_gene_filter", "n_cells"]), "n_vars": int(steps.loc["after_gene_filter", "n_genes"])},
        {"execution_point": "after_final_cell_filter_canonical_output", "n_obs": int(steps.loc["after_final_cell_filter", "n_cells"]), "n_vars": int(steps.loc["after_final_cell_filter", "n_genes"])},
    ])
    points["parameters"] = "MT prefix=MT-; RIBO prefix=RPS; percent_top=None; log1p=False"
    points["backend_package"] = "scanpy CPU"
    points["code_file"] = str(Path(__file__).with_name("02_run_qc.py"))
    points["random_seed"] = SEED
    write_new("04_mt_ribo_qc_metric_execution_points.csv", points)

    write_new("02_scrublet_failed_sample_filter_parameters.csv", pd.DataFrame([{
        "step": "exclude_entire_samples_with_invalid_scrublet",
        "n_obs_before": int(steps.loc["after_initial_min_genes", "n_cells"]),
        "n_obs_after": int(steps.loc["after_failed_scrublet_sample_exclusions", "n_cells"]),
        "n_failed_samples": int(len(failed)),
        "n_cells_excluded": int(pd.to_numeric(failed.get("n_cells_excluded", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "parameters": "exclude whole sample after permitted same-method attempt/fallback returns invalid Scrublet calls",
        "backend_package": "rapids-singlecell GPU Scrublet; AnnData boolean filter on CPU",
        "code_file": str(Path(__file__).with_name("02_run_qc.py")),
        "random_seed": SEED,
    }]))
    print("QC execution-point provenance addendum written")


if __name__ == "__main__":
    main()

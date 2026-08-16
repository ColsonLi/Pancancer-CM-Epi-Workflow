#!/usr/bin/env python3
"""Record the evidence-backed QC threshold choice used for the canonical run."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
PROFILE_DIR = PROJECT_ROOT / "tables/01-celltype_integration_clustering/03-qc/01-qc-distribution-profile"
OUTPUT = PROJECT_ROOT / "tables/01-celltype_integration_clustering/03-qc/threshold_selection_rationale.csv"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    global_q = pd.read_csv(PROFILE_DIR / "qc_distribution_quantiles_global.csv")
    series_q = pd.read_csv(PROFILE_DIR / "qc_distribution_quantiles_by_series.csv")
    rows = [
        {
            "parameter": "min_genes_initial",
            "selected_value": 200,
            "distribution_evidence": "global q0.05=95; GSE278688 minimum=201; GSE229413 q0.05=77",
            "rationale": "remove extreme low-complexity cells before Scrublet while retaining every eligible sample",
            "prospective_cells_retained_before_scrublet": "not_final_filter",
            "prospective_samples_retained": 42,
        },
        {
            "parameter": "min_cells_gene",
            "selected_value": 3,
            "distribution_evidence": "2.5% of shared genes occur in <=2 cells",
            "rationale": "remove genes without reproducible detection across at least three cells",
            "prospective_cells_retained_before_scrublet": "not_applicable",
            "prospective_samples_retained": "not_applicable",
        },
        {
            "parameter": "min_counts_gene",
            "selected_value": 3,
            "distribution_evidence": "2.5% gene total-count quantile=2",
            "rationale": "remove genes with fewer than three total molecules after min-cell filtering",
            "prospective_cells_retained_before_scrublet": "not_applicable",
            "prospective_samples_retained": "not_applicable",
        },
        {
            "parameter": "min_genes_final",
            "selected_value": 500,
            "distribution_evidence": "global q0.05=95 and q0.25=647; GSE278688 q0.025=538",
            "rationale": "exclude low-complexity/dead-cell tail while preserving well-profiled cells in both series",
            "prospective_cells_retained_before_scrublet": 158680,
            "prospective_samples_retained": 42,
        },
        {
            "parameter": "max_genes_final",
            "selected_value": 9000,
            "distribution_evidence": "global q0.995=9155; GSE229413 q0.99=8881",
            "rationale": "remove extreme high-complexity residual doublet tail after Scrublet",
            "prospective_cells_retained_before_scrublet": 158680,
            "prospective_samples_retained": 42,
        },
        {
            "parameter": "max_pct_mt",
            "selected_value": 20,
            "distribution_evidence": "GSE278688 q0.99=18.25 and q0.995=21.80; GSE229413 has a distinct high-MT dead-cell tail",
            "rationale": "retain the high-quality body of both series while excluding strong mitochondrial-stress cells",
            "prospective_cells_retained_before_scrublet": 158680,
            "prospective_samples_retained": 42,
        },
        {
            "parameter": "RIBO_metric_availability",
            "selected_value": "unavailable_after_inner_join",
            "distribution_evidence": "0 RPS-prefixed genes; all pct_counts_RIBO values are 0",
            "rationale": "GSE278688 source features contain no RPS genes, so the required inner join structurally removes them; no ribosomal threshold is fabricated",
            "prospective_cells_retained_before_scrublet": "not_applicable",
            "prospective_samples_retained": "not_applicable",
        },
    ]
    evidence = {
        "global_profile_rows": global_q.shape[0],
        "series_profile_rows": series_q.shape[0],
        "profile_directory": str(PROFILE_DIR),
        "code_file": str(Path(__file__).resolve()),
        "random_seed": SEED,
    }
    frame = pd.DataFrame(rows)
    for key, value in evidence.items():
        frame[key] = value
    frame.to_csv(OUTPUT, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

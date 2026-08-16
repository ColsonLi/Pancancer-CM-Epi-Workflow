#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[4]
TABLE_DIR = ROOT / "epi-cm-core-workflow/tables/01-celltype_integration_clustering/03-qc"
INPUT = ROOT / "epi-cm-core-workflow/h5ad/01-celltype_integration_clustering/02-merge-metadata/adata_merge.h5ad"
OUTPUT = ROOT / "epi-cm-core-workflow/h5ad/01-celltype_integration_clustering/03-qc/adata_qc.h5ad"


def write_new(name: str, frame: pd.DataFrame) -> None:
    path = TABLE_DIR / name
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    frame.to_csv(path, index=False)


def row(step: str, before_cells: int, after_cells: int, before_genes: int,
        after_genes: int, parameters: str, backend: str, notes: str = "") -> pd.DataFrame:
    return pd.DataFrame([{
        "step": step,
        "input_h5ad": str(INPUT),
        "output_h5ad_or_object": str(OUTPUT),
        "n_obs_before": before_cells,
        "n_obs_after": after_cells,
        "n_vars_before": before_genes,
        "n_vars_after": after_genes,
        "parameters": parameters,
        "backend_package": backend,
        "code_file": str(Path(__file__).with_name("02_run_qc.py")),
        "random_seed": SEED,
        "notes_fallbacks": notes,
    }])


def main() -> None:
    steps = pd.read_csv(TABLE_DIR / "qc_step_summary.csv").set_index("step")
    thresholds = pd.read_csv(TABLE_DIR / "01_selected_qc_thresholds.csv").set_index("parameter")["value"]
    n0, g0 = map(int, steps.loc["merged_input", ["n_cells", "n_genes"]])
    n1 = int(steps.loc["after_initial_min_genes", "n_cells"])
    n2 = int(steps.loc["after_failed_scrublet_sample_exclusions", "n_cells"])
    n3 = int(steps.loc["after_predicted_doublet_filter", "n_cells"])
    n4, g4 = map(int, steps.loc["after_gene_filter", ["n_cells", "n_genes"]])
    n5, g5 = map(int, steps.loc["after_final_cell_filter", ["n_cells", "n_genes"]])
    failed = pd.read_csv(TABLE_DIR / "02_scrublet_failed_sample_exclusions.csv")
    write_new("01_initial_cell_filter_parameters.csv", row(
        "initial_cell_filter", n0, n1, g0, g0,
        f"min_genes={int(thresholds['min_genes_initial'])}", "scanpy CPU"))
    write_new("03_doublet_filter_parameters.csv", row(
        "predicted_doublet_filter", n2, n3, g0, g0,
        "remove predicted_doublet=True", "rapids-singlecell GPU calls; AnnData boolean filter on CPU",
        f"failed_samples_excluded={len(failed)}"))
    write_new("04_mt_ribo_qc_metric_parameters.csv", row(
        "mt_ribo_qc_metrics", n3, n3, g0, g0,
        "MT prefix=MT-; RIBO prefix=RPS; percent_top=None; log1p=False",
        "scanpy CPU"))
    write_new("05_gene_filter_parameters.csv", row(
        "gene_filter", n3, n4, g0, g4,
        f"min_cells={int(thresholds['min_cells_gene'])};min_counts={int(thresholds['min_counts_gene'])}",
        "scanpy CPU"))
    write_new("06_final_cell_filter_parameters.csv", row(
        "final_cell_filter", n4, n5, g4, g5,
        f"min_genes={int(thresholds['min_genes_final'])};max_genes={int(thresholds['max_genes_final'])};max_pct_mt={float(thresholds['max_pct_mt'])}",
        "AnnData boolean filter on CPU"))

    versions_json = TABLE_DIR / "package_versions_02_run_qc.json"
    versions = json.loads(versions_json.read_text())
    lines = [
        "Block 01 / 03-qc package versions",
        f"environment={ROOT / 'uv_envs/rapids/.venv'}",
        f"code_file={Path(__file__).with_name('02_run_qc.py')}",
        f"contract_export_code={Path(__file__)}",
    ] + [f"{key}={value}" for key, value in sorted(versions.items())]
    package_path = TABLE_DIR / "package_versions.txt"
    if package_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {package_path}")
    package_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    a = ad.read_h5ad(OUTPUT, backed="r")
    if a.shape != (n5, g5) or not a.obs_names.is_unique or not a.var_names.is_unique:
        raise RuntimeError("QC object does not match the completed QC step summary")
    a.file.close()
    print(f"QC skill-contract tables completed for {n5} cells and {g5} genes")


if __name__ == "__main__":
    main()

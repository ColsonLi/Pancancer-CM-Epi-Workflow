#!/usr/bin/env python3
"""Run the required sample-wise Scrublet and explicit-threshold PDAC QC."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import sys
import traceback
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
from scipy import sparse


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
cp.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
INPUT_H5AD = PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/02-merge-metadata/adata_merge.h5ad"
H5AD_DIR = PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/03-qc"
TABLE_DIR = PROJECT_ROOT / "tables/01-celltype_integration_clustering/03-qc"
FIGURE_DIR = PROJECT_ROOT / "figures/01-celltype_integration_clustering/03-qc"
SCRUBLET_FILTERED_H5AD = H5AD_DIR / "adata_scrublet_filtered.h5ad"
OUTPUT_H5AD = H5AD_DIR / "adata_qc.h5ad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-genes-initial", type=int, required=True)
    parser.add_argument("--min-cells-gene", type=int, required=True)
    parser.add_argument("--min-counts-gene", type=int, required=True)
    parser.add_argument("--min-genes-final", type=int, required=True)
    parser.add_argument("--max-genes-final", type=int, required=True)
    parser.add_argument("--max-pct-mt", type=float, required=True)
    return parser.parse_args()


def require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def common_row(
    step: str,
    input_path: str,
    output_path: str,
    n_obs_before: int,
    n_obs_after: int,
    n_vars_before: int,
    n_vars_after: int,
    backend: str,
    notes: str = "",
) -> dict[str, object]:
    return {
        "step": step,
        "input_h5ad": input_path,
        "output_h5ad_or_object": output_path,
        "n_obs_before": n_obs_before,
        "n_obs_after": n_obs_after,
        "n_vars_before": n_vars_before,
        "n_vars_after": n_vars_after,
        "backend_package": backend,
        "code_file": str(Path(__file__).resolve()),
        "random_seed": SEED,
        "notes_fallbacks": notes,
    }


def release_gpu() -> None:
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def is_oom(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in message or "outofmemory" in message or "cuda_error_out_of_memory" in message


def validate_scrublet_result(sample: ad.AnnData, input_names: pd.Index) -> None:
    required = {"doublet_score", "predicted_doublet"}
    if not required.issubset(sample.obs.columns):
        raise ValueError("missing required Scrublet output columns")
    if not sample.obs_names.equals(input_names):
        raise ValueError("Scrublet output cell IDs do not match input cell IDs")
    if sample.obs["doublet_score"].isna().any():
        raise ValueError("doublet_score contains missing values")
    if sample.obs["predicted_doublet"].isna().any():
        raise ValueError("predicted_doublet contains missing values")


def run_rsc_scrublet(sample: ad.AnnData) -> ad.AnnData:
    input_names = sample.obs_names.copy()
    if "counts" in sample.layers:
        del sample.layers["counts"]
    rsc.get.anndata_to_GPU(sample)
    rsc.pp.scrublet(sample, random_state=SEED, verbose=False)
    rsc.get.anndata_to_CPU(sample)
    validate_scrublet_result(sample, input_names)
    return sample


def run_cpu_scrublet(sample: ad.AnnData) -> ad.AnnData:
    input_names = sample.obs_names.copy()
    if "counts" in sample.layers:
        del sample.layers["counts"]
    sc.pp.scrublet(sample, random_state=SEED, verbose=False)
    validate_scrublet_result(sample, input_names)
    return sample


def fresh_sample(adata: ad.AnnData, sample_id: str) -> ad.AnnData:
    mask = adata.obs["sample"].astype(str).eq(sample_id).to_numpy()
    return adata[mask].copy()


def sample_counts(adata: ad.AnnData, stage: str) -> pd.DataFrame:
    counts = adata.obs.groupby(["series", "sample", "status"], observed=True).size()
    return counts.rename("n_cells").reset_index().assign(stage=stage)


def main() -> None:
    args = parse_args()
    if args.min_genes_initial <= 0 or args.min_cells_gene <= 0 or args.min_counts_gene <= 0:
        raise ValueError("Minimum QC thresholds must be positive")
    if args.min_genes_final < args.min_genes_initial:
        raise ValueError("min_genes_final must be >= min_genes_initial")
    if args.max_genes_final <= args.min_genes_final:
        raise ValueError("max_genes_final must exceed min_genes_final")
    if not 0 < args.max_pct_mt <= 100:
        raise ValueError("max_pct_mt must be in (0, 100]")
    if not INPUT_H5AD.is_file():
        raise FileNotFoundError(INPUT_H5AD)

    outputs = [
        SCRUBLET_FILTERED_H5AD,
        OUTPUT_H5AD,
        TABLE_DIR / "01_initial_cell_filter_parameters.csv",
        TABLE_DIR / "02_scrublet_parameters.csv",
        TABLE_DIR / "02_scrublet_failed_sample_exclusions.csv",
        TABLE_DIR / "03_doublet_filter_parameters.csv",
        TABLE_DIR / "04_mt_ribo_qc_metric_parameters.csv",
        TABLE_DIR / "05_gene_filter_parameters.csv",
        TABLE_DIR / "06_final_cell_filter_parameters.csv",
        TABLE_DIR / "qc_report.csv",
        TABLE_DIR / "gpu_backend_capability_summary.csv",
        TABLE_DIR / "package_versions.txt",
        TABLE_DIR / "qc_validation.json",
    ]
    for path in outputs:
        require_new(path)
    H5AD_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(INPUT_H5AD)
    required_obs = {"sample", "series", "status", "original_barcode"}
    if not required_obs.issubset(adata.obs.columns):
        raise ValueError(f"Input lacks required obs fields: {required_obs - set(adata.obs.columns)}")
    if "counts" not in adata.layers:
        raise ValueError("Input lacks layers['counts']")
    if not sparse.issparse(adata.X) or np.count_nonzero(adata.X.data != np.floor(adata.X.data)):
        raise ValueError("Input X must be sparse integer counts")
    if not sparse.issparse(adata.layers["counts"]):
        raise ValueError("layers['counts'] must be sparse")

    stage_reports = [sample_counts(adata, "input_merged_counts")]
    n_obs_input, n_vars_input = adata.shape
    before_initial = adata.shape
    sc.pp.filter_cells(adata, min_genes=args.min_genes_initial)
    stage_reports.append(sample_counts(adata, "after_initial_min_genes"))
    initial_row = common_row(
        "initial_cell_filter",
        str(INPUT_H5AD),
        "in_memory_before_scrublet",
        before_initial[0], adata.n_obs, before_initial[1], adata.n_vars,
        "scanpy",
    )
    initial_row["min_genes_initial"] = args.min_genes_initial
    pd.DataFrame([initial_row]).to_csv(TABLE_DIR / "01_initial_cell_filter_parameters.csv", index=False)

    adata.obs["doublet_score"] = np.nan
    adata.obs["predicted_doublet"] = pd.NA
    scrublet_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    backend_rows: list[dict[str, object]] = []
    failed_samples: list[str] = []

    for order, sample_id in enumerate(sorted(adata.obs["sample"].astype(str).unique()), start=1):
        sample = fresh_sample(adata, sample_id)
        n_cells = sample.n_obs
        backend = "rapids-singlecell_GPU"
        attempts = 1
        errors: list[str] = []
        result: ad.AnnData | None = None
        try:
            result = run_rsc_scrublet(sample)
        except Exception as exc:
            errors.append(f"attempt1:{type(exc).__name__}:{exc}")
            release_gpu()
            if is_oom(exc):
                backend = "scanpy_CPU_after_GPU_OOM"
                attempts += 1
                try:
                    result = run_cpu_scrublet(fresh_sample(adata, sample_id))
                except Exception as cpu_exc:
                    errors.append(f"cpu_fallback:{type(cpu_exc).__name__}:{cpu_exc}")
            else:
                backend = "rapids-singlecell_GPU_repair_retry"
                attempts += 1
                try:
                    result = run_rsc_scrublet(fresh_sample(adata, sample_id))
                except Exception as retry_exc:
                    errors.append(f"repair_retry:{type(retry_exc).__name__}:{retry_exc}")
        finally:
            release_gpu()

        if result is None:
            failed_samples.append(sample_id)
            reason = " | ".join(errors)[:4000]
            failure_rows.append(
                {
                    "sample": sample_id,
                    "n_cells_excluded": n_cells,
                    "failure_reason": reason,
                    "action": "exclude_entire_sample",
                }
            )
            scrublet_rows.append(
                {
                    "sample_order": order,
                    "sample": sample_id,
                    "n_cells_input": n_cells,
                    "n_cells_output": 0,
                    "backend": backend,
                    "attempts": attempts,
                    "doublets_predicted": "",
                    "status": "failed_sample_excluded",
                    "error_summary": reason,
                    "random_state": SEED,
                    "sim_doublet_ratio": 2.0,
                    "expected_doublet_rate": 0.05,
                    "stdev_doublet_rate": 0.02,
                    "synthetic_doublet_umi_subsampling": 1.0,
                    "n_prin_comps": 30,
                    "threshold": "automatic",
                }
            )
            backend_rows.append(
                {
                    "step": f"scrublet:{sample_id}",
                    "planned_backend": "rapids-singlecell_GPU",
                    "attempted_backend": backend,
                    "status": "failed_sample_excluded",
                    "error_summary": reason,
                    "fallback_backend": "scanpy_CPU" if "CPU" in backend else "",
                    "clean_input_reloaded": attempts > 1,
                    "final_backend_for_rerun": backend,
                }
            )
            continue

        adata.obs.loc[result.obs_names, "doublet_score"] = result.obs["doublet_score"].astype(float)
        adata.obs.loc[result.obs_names, "predicted_doublet"] = result.obs["predicted_doublet"].astype(bool)
        n_doublets = int(result.obs["predicted_doublet"].astype(bool).sum())
        scrublet_rows.append(
            {
                "sample_order": order,
                "sample": sample_id,
                "n_cells_input": n_cells,
                "n_cells_output": result.n_obs,
                "backend": backend,
                "attempts": attempts,
                "doublets_predicted": n_doublets,
                "status": "completed",
                "error_summary": " | ".join(errors)[:4000],
                "random_state": SEED,
                "sim_doublet_ratio": 2.0,
                "expected_doublet_rate": 0.05,
                "stdev_doublet_rate": 0.02,
                "synthetic_doublet_umi_subsampling": 1.0,
                "n_prin_comps": 30,
                "threshold": "automatic",
            }
        )
        backend_rows.append(
            {
                "step": f"scrublet:{sample_id}",
                "planned_backend": "rapids-singlecell_GPU",
                "attempted_backend": backend,
                "status": "completed",
                "error_summary": " | ".join(errors)[:1000],
                "fallback_backend": "scanpy_CPU" if "CPU" in backend else "",
                "clean_input_reloaded": attempts > 1,
                "final_backend_for_rerun": backend,
            }
        )

    pd.DataFrame(scrublet_rows).to_csv(TABLE_DIR / "02_scrublet_parameters.csv", index=False)
    failure_columns = ["sample", "n_cells_excluded", "failure_reason", "action"]
    pd.DataFrame(failure_rows, columns=failure_columns).to_csv(
        TABLE_DIR / "02_scrublet_failed_sample_exclusions.csv", index=False
    )
    if failed_samples:
        adata = adata[~adata.obs["sample"].astype(str).isin(failed_samples)].copy()
    if adata.n_obs == 0:
        raise RuntimeError("No samples remain after excluding failed Scrublet samples")
    if adata.obs[["doublet_score", "predicted_doublet"]].isna().any().any():
        raise RuntimeError("Retained samples do not all have complete Scrublet outputs")
    adata.obs["doublet_score"] = adata.obs["doublet_score"].astype(float)
    adata.obs["predicted_doublet"] = adata.obs["predicted_doublet"].astype(bool)
    stage_reports.append(sample_counts(adata, "after_failed_scrublet_sample_exclusion"))

    before_doublet = adata.shape
    adata = adata[~adata.obs["predicted_doublet"]].copy()
    stage_reports.append(sample_counts(adata, "after_predicted_doublet_filter"))
    doublet_row = common_row(
        "predicted_doublet_filter",
        str(INPUT_H5AD),
        str(SCRUBLET_FILTERED_H5AD),
        before_doublet[0], adata.n_obs, before_doublet[1], adata.n_vars,
        "rapids-singlecell_or_recorded_sample_fallback",
        f"failed_samples_excluded={len(failed_samples)}",
    )
    doublet_row["filter_rule"] = "predicted_doublet == False"
    pd.DataFrame([doublet_row]).to_csv(TABLE_DIR / "03_doublet_filter_parameters.csv", index=False)
    adata.write_h5ad(SCRUBLET_FILTERED_H5AD, compression="gzip", compression_opts=1)

    adata.var["MT"] = adata.var_names.astype(str).str.startswith("MT-")
    adata.var["RIBO"] = adata.var_names.astype(str).str.startswith("RPS")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["MT", "RIBO"], percent_top=None, log1p=False, inplace=True
    )
    metric_row = common_row(
        "mt_ribo_qc_metrics",
        str(SCRUBLET_FILTERED_H5AD),
        "in_memory_before_gene_filter",
        adata.n_obs, adata.n_obs, adata.n_vars, adata.n_vars,
        "scanpy",
    )
    metric_row.update(
        {
            "mt_prefix": "MT-",
            "ribo_prefix": "RPS",
            "mt_gene_count": int(adata.var["MT"].sum()),
            "ribo_gene_count": int(adata.var["RIBO"].sum()),
            "percent_top": "None",
            "log1p": False,
        }
    )
    pd.DataFrame([metric_row]).to_csv(TABLE_DIR / "04_mt_ribo_qc_metric_parameters.csv", index=False)

    before_gene = adata.shape
    sc.pp.filter_genes(adata, min_cells=args.min_cells_gene)
    after_min_cells = adata.shape
    sc.pp.filter_genes(adata, min_counts=args.min_counts_gene)
    after_min_counts = adata.shape
    gene_rows = []
    row = common_row(
        "gene_filter_min_cells", str(SCRUBLET_FILTERED_H5AD), "in_memory",
        before_gene[0], after_min_cells[0], before_gene[1], after_min_cells[1], "scanpy",
    )
    row["min_cells_gene"] = args.min_cells_gene
    gene_rows.append(row)
    row = common_row(
        "gene_filter_min_counts", "in_memory_after_min_cells", "in_memory",
        after_min_cells[0], after_min_counts[0], after_min_cells[1], after_min_counts[1], "scanpy",
    )
    row["min_counts_gene"] = args.min_counts_gene
    gene_rows.append(row)
    pd.DataFrame(gene_rows).to_csv(TABLE_DIR / "05_gene_filter_parameters.csv", index=False)

    adata.var["MT"] = adata.var_names.astype(str).str.startswith("MT-")
    adata.var["RIBO"] = adata.var_names.astype(str).str.startswith("RPS")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["MT", "RIBO"], percent_top=None, log1p=False, inplace=True
    )
    before_final = adata.shape
    keep = (
        adata.obs["n_genes_by_counts"].ge(args.min_genes_final)
        & adata.obs["n_genes_by_counts"].le(args.max_genes_final)
        & adata.obs["pct_counts_MT"].le(args.max_pct_mt)
    )
    adata = adata[keep].copy()
    stage_reports.append(sample_counts(adata, "after_final_cell_filter"))
    final_row = common_row(
        "final_cell_filter",
        str(SCRUBLET_FILTERED_H5AD), str(OUTPUT_H5AD),
        before_final[0], adata.n_obs, before_final[1], adata.n_vars, "anndata_boolean_subset",
    )
    final_row.update(
        {
            "min_genes_final": args.min_genes_final,
            "max_genes_final": args.max_genes_final,
            "max_pct_mt": args.max_pct_mt,
            "filter_rule": "n_genes_by_counts between inclusive bounds and pct_counts_MT <= max_pct_mt",
        }
    )
    pd.DataFrame([final_row]).to_csv(TABLE_DIR / "06_final_cell_filter_parameters.csv", index=False)

    if adata.n_obs == 0:
        raise RuntimeError("No cells remain after final QC thresholds")
    if np.count_nonzero(adata.X.data != np.floor(adata.X.data)):
        raise ValueError("QC X is no longer integer counts")
    if np.count_nonzero(adata.layers["counts"].data != np.floor(adata.layers["counts"].data)):
        raise ValueError("QC counts layer is no longer integer counts")
    adata.uns["qc_thresholds"] = {
        "min_genes_initial": args.min_genes_initial,
        "min_cells_gene": args.min_cells_gene,
        "min_counts_gene": args.min_counts_gene,
        "min_genes_final": args.min_genes_final,
        "max_genes_final": args.max_genes_final,
        "max_pct_mt": args.max_pct_mt,
        "random_seed": SEED,
    }
    adata.write_h5ad(OUTPUT_H5AD, compression="gzip", compression_opts=1)

    all_samples = pd.concat(stage_reports, ignore_index=True)
    all_samples.to_csv(TABLE_DIR / "qc_report.csv", index=False)
    pd.DataFrame(backend_rows).to_csv(TABLE_DIR / "gpu_backend_capability_summary.csv", index=False)

    sc.set_figure_params(figsize=(3, 3), dpi=150)
    sc.settings.figdir = FIGURE_DIR
    metrics = ["n_genes_by_counts", "total_counts", "pct_counts_MT", "pct_counts_RIBO", "doublet_score"]
    sc.pl.violin(
        adata, keys=metrics, groupby="series", jitter=0, multi_panel=True,
        show=False, save="_post_qc_by_series.pdf",
    )
    sc.pl.violin(
        adata, keys=metrics, groupby="status", jitter=0, multi_panel=True,
        show=False, save="_post_qc_by_status.pdf",
    )

    package_names = [
        "rapids-singlecell-cu13", "cudf-cu13", "cuml-cu13", "cugraph-cu13",
        "cuvs-cu13", "cupy-cuda13x", "scanpy", "anndata", "numpy", "pandas",
        "scipy", "h5py",
    ]
    lines = [
        "environment_category=rapids",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        "install_manager=uv",
        "index_url=https://pypi.tuna.tsinghua.edu.cn/simple",
        f"random_seed={SEED}",
        f"code_file={Path(__file__).resolve()}",
    ]
    for package in package_names:
        lines.append(f"{package}={importlib.metadata.version(package)}")
    (TABLE_DIR / "package_versions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    validation = {
        "input_h5ad": str(INPUT_H5AD),
        "scrublet_filtered_h5ad": str(SCRUBLET_FILTERED_H5AD),
        "output_h5ad": str(OUTPUT_H5AD),
        "n_obs_input": n_obs_input,
        "n_obs_output": adata.n_obs,
        "n_vars_input": n_vars_input,
        "n_vars_output": adata.n_vars,
        "n_samples_input": int(stage_reports[0]["sample"].nunique()),
        "n_samples_output": int(adata.obs["sample"].nunique()),
        "failed_scrublet_samples": failed_samples,
        "predicted_doublets_removed": int(before_doublet[0] - stage_reports[-2]["n_cells"].sum()) if len(stage_reports) >= 2 else None,
        "required_qc_columns_present": all(
            column in adata.obs.columns
            for column in [
                "n_genes_by_counts", "total_counts", "pct_counts_MT", "total_counts_MT",
                "pct_counts_RIBO", "total_counts_RIBO", "doublet_score", "predicted_doublet",
            ]
        ),
        "counts_layer_present": "counts" in adata.layers,
        "counts_nonnegative_integer": bool(
            adata.layers["counts"].data.size == 0
            or (
                float(adata.layers["counts"].data.min()) >= 0
                and np.count_nonzero(
                    adata.layers["counts"].data != np.floor(adata.layers["counts"].data)
                ) == 0
            )
        ),
        "obs_names_unique": bool(adata.obs_names.is_unique),
        "thresholds": vars(args),
        "random_seed": SEED,
    }
    (TABLE_DIR / "qc_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

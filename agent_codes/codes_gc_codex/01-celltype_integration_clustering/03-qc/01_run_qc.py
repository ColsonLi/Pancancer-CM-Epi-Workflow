#!/usr/bin/env python3
"""Full-cell basic QC and per-sample Scrublet for the GC scRNA merge.

The count matrix is never downsampled.  Scrublet runs independently per sample
on raw counts after basic cell/mitochondrial filters.  RAPIDS-SingleCell is
attempted first in an isolated process; any CPU fallback is explicit in the
audit tables. If all allowed same-method routes fail, that complete sample is
excluded and the remaining samples continue; the run stops only if none remain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


SEED = 42
ROOT = Path(__file__).resolve().parents[4]
WF = ROOT / "epi-cm-core-workflow"
DEFAULT_INPUT = WF / "h5ad/01-celltype_integration_clustering/02-merge-metadata/adata_merge.h5ad"
DEFAULT_OUTPUT_DIR = WF / "h5ad/01-celltype_integration_clustering/03-qc"
DEFAULT_TABLE_DIR = WF / "tables/01-celltype_integration_clustering/03-qc"
DEFAULT_FIGURE_DIR = WF / "figures/01-celltype_integration_clustering/03-qc"
DEFAULT_MANIFEST = (
    WF
    / "tables/01-celltype_integration_clustering/01-sample-manifest/"
    "all_datasets_all_samples_inclusion_v4.csv"
)
DEFAULT_MAIN_PYTHON = ROOT / "uv_envs/main/.venv/bin/python"
DEFAULT_RAPIDS_PYTHON = ROOT / "uv_envs/rapids/.venv/bin/python"
WORKER = Path(__file__).with_name("02_scrublet_worker.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(names: pd.Index) -> str:
    return hashlib.sha256("\n".join(names.astype(str)).encode()).hexdigest()


def atomic_h5ad(adata: ad.AnnData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.h5ad")
    if tmp.exists():
        tmp.unlink()
    adata.write_h5ad(tmp, compression="gzip")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "".join(path.suffixes)
    tmp = path.with_name(path.name[: -len(suffix)] + ".part" + suffix) if suffix else path.with_name(path.name + ".part")
    frame.to_csv(tmp, index=False, **kwargs)
    os.replace(tmp, path)


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in (
        "anndata",
        "scanpy",
        "scrublet",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "h5py",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def enforce_raw_integer_counts(adata: ad.AnnData, context: str) -> None:
    if adata.raw is not None or len(adata.layers) != 0:
        raise ValueError(f"{context}: expected counts only in X, without raw/layers")
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    else:
        adata.X = sparse.csr_matrix(adata.X)
    adata.X.sum_duplicates()
    adata.X.eliminate_zeros()
    values = adata.X.data
    if values.size and (
        not np.isfinite(values).all()
        or float(values.min()) < 0
        or not np.equal(values, np.rint(values)).all()
    ):
        raise ValueError(f"{context}: X is not non-negative integer raw counts")
    adata.X = adata.X.astype(np.int32, copy=False)


def scrublet_key(series: str, sample: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{series}__{sample}")


def result_is_current(csv_path: Path, json_path: Path, expected: dict[str, object]) -> bool:
    if not csv_path.is_file() or not json_path.is_file():
        return False
    try:
        metadata = json.loads(json_path.read_text())
        for key, value in expected.items():
            if metadata.get(key) != value:
                return False
        frame = pd.read_csv(csv_path)
        return (
            len(frame) == int(expected["n_cells"])
            and frame["obs_name"].is_unique
            and {"doublet_score", "predicted_doublet"}.issubset(frame.columns)
        )
    except Exception:
        return False


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)
    parsed = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if parsed.isna().any():
        raise ValueError("Boolean column contains values other than true/false")
    return parsed.astype(bool)


def worker_command(
    python: Path,
    input_path: Path,
    csv_path: Path,
    json_path: Path,
    backend: str,
    args: argparse.Namespace,
) -> list[str]:
    return [
        str(python),
        str(WORKER),
        "--input",
        str(input_path),
        "--output-csv",
        str(csv_path),
        "--output-json",
        str(json_path),
        "--backend",
        backend,
        "--expected-doublet-rate",
        str(args.expected_doublet_rate),
        "--sim-doublet-ratio",
        str(args.sim_doublet_ratio),
        "--n-prin-comps",
        str(args.scrublet_pcs),
        "--seed",
        str(SEED),
    ]


def run_worker(
    python: Path,
    input_path: Path,
    csv_path: Path,
    json_path: Path,
    backend: str,
    log_path: Path,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if not python.is_file():
        return False, f"Python executable missing: {python}"
    command = worker_command(python, input_path, csv_path, json_path, backend, args)
    started = time.time()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "command=" + " ".join(command) + "\n"
        + f"returncode={completed.returncode}\n"
        + f"elapsed_seconds={time.time() - started:.6f}\n"
        + "stdout:\n"
        + completed.stdout
        + "\nstderr:\n"
        + completed.stderr
    )
    detail = f"returncode={completed.returncode}; log={log_path}"
    return completed.returncode == 0, detail


def make_qc_figures(cell_metrics: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes[0, 0].hist(cell_metrics["total_counts"], bins=100, color="#4477AA")
    axes[0, 0].set_xlabel("Total counts")
    axes[0, 0].set_ylabel("Cells")
    axes[0, 1].hist(cell_metrics["n_genes_by_counts"], bins=100, color="#66AA55")
    axes[0, 1].set_xlabel("Detected genes")
    axes[1, 0].hist(cell_metrics["pct_counts_MT"], bins=100, color="#CC6677")
    axes[1, 0].set_xlabel("Mitochondrial counts (%)")
    axes[1, 1].scatter(
        cell_metrics["total_counts"],
        cell_metrics["pct_counts_MT"],
        s=1,
        alpha=0.15,
        rasterized=True,
        color="#555555",
    )
    axes[1, 1].set_xlabel("Total counts")
    axes[1, 1].set_ylabel("Mitochondrial counts (%)")
    fig.suptitle("GC scRNA-seq full-cell QC distributions")
    fig.tight_layout()
    fig.savefig(figure_dir / "qc_distributions.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "qc_distributions.svg", bbox_inches="tight")
    plt.close(fig)

    finite = cell_metrics["doublet_score"].dropna()
    if not finite.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(finite, bins=100, color="#AA4499")
        ax.set_xlabel("Scrublet doublet score")
        ax.set_ylabel("Cells")
        ax.set_title("Per-sample Scrublet scores (all samples pooled for display)")
        fig.tight_layout()
        fig.savefig(figure_dir / "scrublet_score_distribution.pdf", bbox_inches="tight")
        fig.savefig(figure_dir / "scrublet_score_distribution.svg", bbox_inches="tight")
        plt.close(fig)


def gene_identifier_row(stage: str, names: pd.Index) -> dict[str, object]:
    values = pd.Index(names.astype(str))
    upper = values.str.upper()
    return {
        "stage": stage,
        "n_genes": len(values),
        "n_unique_identifiers": int(values.nunique()),
        "n_duplicate_identifiers": int(values.duplicated().sum()),
        "n_blank_identifiers": int((values.str.strip() == "").sum()),
        "n_ensembl_like": int(values.str.match(r"^ENSG\d+(?:\.\d+)?$", case=False).sum()),
        "n_mt_prefix": int(upper.str.startswith("MT-").sum()),
        "n_ribo_prefix": int(upper.str.match(r"^RP[SL]").sum()),
        "identifier_axis": "gene symbol in var_names",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--main-python", type=Path, default=DEFAULT_MAIN_PYTHON)
    parser.add_argument("--rapids-python", type=Path, default=DEFAULT_RAPIDS_PYTHON)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--min-cells-per-gene", type=int, default=3)
    parser.add_argument("--max-pct-mt", type=float, default=20.0)
    parser.add_argument("--expected-doublet-rate", type=float, default=0.05)
    parser.add_argument("--sim-doublet-ratio", type=float, default=2.0)
    parser.add_argument("--scrublet-pcs", type=int, default=30)
    parser.add_argument(
        "--allow-cpu-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After an audited GPU failure, retry that sample with Scanpy CPU.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse Scrublet results only when cell IDs and parameters match.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    started = time.time()
    output = args.output_dir / "adata_qc.h5ad"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite completed QC output: {output}")
    if not args.input.is_file() or not args.manifest.is_file():
        raise FileNotFoundError("Merge h5ad or canonical v4 manifest is missing")
    if args.min_genes < 1 or args.min_cells_per_gene < 1 or args.max_pct_mt <= 0:
        raise ValueError("QC thresholds must be positive")
    adata = ad.read_h5ad(args.input)
    enforce_raw_integer_counts(adata, "merge input")
    required_obs = {
        "sample",
        "series",
        "status",
        "patient",
        "tissue_class",
        "original_barcode",
    }
    missing = sorted(required_obs - set(adata.obs.columns))
    if missing or not adata.obs_names.is_unique:
        raise ValueError(f"Invalid merge metadata; missing={missing}")
    original_shape = adata.shape
    original_samples = set(
        zip(adata.obs["series"].astype(str), adata.obs["sample"].astype(str))
    )
    input_gene_names = adata.var_names.astype(str)
    adata.var["MT"] = input_gene_names.str.upper().str.startswith("MT-")
    adata.var["RIBO"] = input_gene_names.str.upper().str.match(r"^RP[SL]")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["MT", "RIBO"], percent_top=None, log1p=False, inplace=True
    )
    initial_metrics = adata.obs[
        [
            "series",
            "sample",
            "status",
            "total_counts",
            "n_genes_by_counts",
            "total_counts_MT",
            "pct_counts_MT",
            "total_counts_RIBO",
            "pct_counts_RIBO",
        ]
    ].copy()
    initial_metrics["pass_min_genes"] = (
        initial_metrics["n_genes_by_counts"] >= args.min_genes
    )
    initial_metrics["pass_mt"] = initial_metrics["pct_counts_MT"] <= args.max_pct_mt
    initial_metrics["pass_basic_cell_qc"] = (
        initial_metrics["pass_min_genes"] & initial_metrics["pass_mt"]
    )
    adata = adata[initial_metrics["pass_basic_cell_qc"].to_numpy()].copy()
    genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=args.min_cells_per_gene)
    genes_after_first_filter = adata.n_vars
    genes_after_first_names = adata.var_names.copy()
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise RuntimeError("Basic QC removed every cell or gene")
    basic_qc_samples = set(
        zip(adata.obs["series"].astype(str), adata.obs["sample"].astype(str))
    )
    exclusions: list[dict[str, object]] = []
    input_sample_counts = initial_metrics.groupby(["series", "sample"], observed=True).size()
    for series, sample in sorted(original_samples - basic_qc_samples):
        exclusions.append(
            {
                "series": series,
                "sample": sample,
                "exclusion_reason": "no_cells_after_basic_qc",
                "n_cells_input": int(input_sample_counts.loc[(series, sample)]),
                "n_cells_after_basic_qc": 0,
                "gpu_detail": "not_attempted",
                "cpu_detail": "not_attempted",
            }
        )

    checkpoint_dir = args.output_dir / "scrublet_checkpoints"
    input_dir = checkpoint_dir / "inputs"
    result_dir = checkpoint_dir / "results"
    log_dir = args.table_dir / "scrublet_logs"
    for directory in (input_dir, result_dir, log_dir, args.table_dir, args.figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    scrublet_audit: list[dict[str, object]] = []
    result_frames: list[pd.DataFrame] = []
    groups = adata.obs.groupby(["series", "sample"], observed=True, sort=True).indices
    for (series, sample), positions in groups.items():
        key = scrublet_key(str(series), str(sample))
        sample_adata = adata[np.asarray(positions)].copy()
        input_path = input_dir / f"{key}.h5ad"
        csv_path = result_dir / f"{key}.csv"
        json_path = result_dir / f"{key}.json"
        expected = {
            "n_cells": int(sample_adata.n_obs),
            "n_genes": int(sample_adata.n_vars),
            "input_obs_names_sha256": names_sha256(sample_adata.obs_names),
            "expected_doublet_rate": args.expected_doublet_rate,
            "sim_doublet_ratio": args.sim_doublet_ratio,
            "random_seed": SEED,
            "downsampled": False,
        }
        if args.resume and result_is_current(csv_path, json_path, expected):
            metadata = json.loads(json_path.read_text())
            frame = pd.read_csv(csv_path)
            frame["series"] = str(series)
            frame["sample"] = str(sample)
            result_frames.append(frame)
            scrublet_audit.append(
                {
                    "series": series,
                    "sample": sample,
                    "state": "reused_checkpoint",
                    "backend": metadata["backend"],
                    "n_cells": sample_adata.n_obs,
                    "n_predicted_doublets": int(parse_bool_series(frame["predicted_doublet"]).sum()),
                    "gpu_attempted": metadata["backend"] == "gpu",
                    "cpu_fallback": metadata["backend"] == "cpu",
                    "detail": str(json_path.relative_to(WF)),
                    "threshold": metadata.get("threshold"),
                }
            )
            continue
        for path in (csv_path, json_path):
            if path.exists():
                path.unlink()
        atomic_h5ad(sample_adata, input_path)
        gpu_ok, gpu_detail = run_worker(
            args.rapids_python,
            input_path,
            csv_path,
            json_path,
            "gpu",
            log_dir / f"{key}.gpu.log",
            args,
        )
        backend = "gpu"
        cpu_detail = "not_attempted"
        if not gpu_ok and args.allow_cpu_fallback:
            for path in (csv_path, json_path):
                if path.exists():
                    path.unlink()
            cpu_ok, cpu_detail = run_worker(
                args.main_python,
                input_path,
                csv_path,
                json_path,
                "cpu",
                log_dir / f"{key}.cpu.log",
                args,
            )
            backend = "cpu"
        else:
            cpu_ok = False
        success = gpu_ok or cpu_ok
        if not success or not result_is_current(csv_path, json_path, expected):
            exclusions.append(
                {
                    "series": series,
                    "sample": sample,
                    "exclusion_reason": "scrublet_failed_after_gpu_and_allowed_cpu_routes",
                    "n_cells_input": int(input_sample_counts.loc[(series, sample)]),
                    "n_cells_after_basic_qc": sample_adata.n_obs,
                    "gpu_detail": gpu_detail,
                    "cpu_detail": cpu_detail,
                }
            )
            scrublet_audit.append(
                {
                    "series": series,
                    "sample": sample,
                    "state": "failed",
                    "backend": "none",
                    "n_cells": sample_adata.n_obs,
                    "n_predicted_doublets": np.nan,
                    "gpu_attempted": True,
                    "cpu_fallback": bool(args.allow_cpu_fallback),
                    "detail": f"gpu={gpu_detail}; cpu={cpu_detail}",
                }
            )
            continue
        metadata = json.loads(json_path.read_text())
        frame = pd.read_csv(csv_path)
        frame["series"] = str(series)
        frame["sample"] = str(sample)
        result_frames.append(frame)
        scrublet_audit.append(
            {
                "series": series,
                "sample": sample,
                "state": "completed",
                "backend": backend,
                "n_cells": sample_adata.n_obs,
                "n_predicted_doublets": int(parse_bool_series(frame["predicted_doublet"]).sum()),
                "gpu_attempted": True,
                "cpu_fallback": backend == "cpu",
                "detail": f"gpu={gpu_detail}; cpu={cpu_detail}",
                "threshold": metadata.get("threshold"),
            }
        )

    exclusion_frame = pd.DataFrame(
        exclusions,
        columns=[
            "series",
            "sample",
            "exclusion_reason",
            "n_cells_input",
            "n_cells_after_basic_qc",
            "gpu_detail",
            "cpu_detail",
        ],
    )
    excluded_keys = set(zip(exclusion_frame["series"], exclusion_frame["sample"]))
    keep_successful_sample = np.array(
        [
            (str(series), str(sample)) not in excluded_keys
            for series, sample in zip(adata.obs["series"], adata.obs["sample"])
        ]
    )
    adata = adata[keep_successful_sample].copy()
    n_after_sample_exclusions = adata.n_obs
    if adata.n_obs == 0 or not result_frames:
        atomic_csv(exclusion_frame, args.table_dir / "02_scrublet_failed_sample_exclusions.csv")
        raise RuntimeError("No sample remains after basic-QC/Scrublet sample exclusions")

    results = pd.concat(result_frames, ignore_index=True)
    if results["obs_name"].duplicated().any() or set(results["obs_name"]) != set(adata.obs_names):
        raise RuntimeError("Scrublet results do not map one-to-one to post-basic-QC cells")
    results = results.set_index("obs_name").loc[adata.obs_names]
    adata.obs["doublet_score"] = pd.to_numeric(results["doublet_score"]).to_numpy()
    predicted = parse_bool_series(results["predicted_doublet"])
    adata.obs["predicted_doublet"] = predicted.to_numpy()
    adata.obs["pass_scrublet"] = ~adata.obs["predicted_doublet"]
    post_scrublet_metrics = adata.obs[
        [
            "series",
            "sample",
            "status",
            "total_counts",
            "n_genes_by_counts",
            "total_counts_MT",
            "pct_counts_MT",
            "total_counts_RIBO",
            "pct_counts_RIBO",
            "doublet_score",
            "predicted_doublet",
            "pass_scrublet",
        ]
    ].copy()
    adata = adata[adata.obs["pass_scrublet"].to_numpy()].copy()
    sc.pp.filter_genes(adata, min_cells=args.min_cells_per_gene)
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise RuntimeError("No cell or gene remains after doublet removal")
    enforce_raw_integer_counts(adata, "final QC")
    final_samples = set(
        zip(adata.obs["series"].astype(str), adata.obs["sample"].astype(str))
    )
    expected_final_samples = original_samples - excluded_keys
    if final_samples != expected_final_samples:
        raise RuntimeError(
            f"Final sample set disagrees with audited exclusions: "
            f"missing={sorted(expected_final_samples-final_samples)}, "
            f"extra={sorted(final_samples-expected_final_samples)}"
        )
    if not np.isfinite(adata.obs["pct_counts_MT"]).all() or (
        adata.obs["pct_counts_MT"] > args.max_pct_mt
    ).any():
        raise RuntimeError("Final QC contains cells over the mitochondrial threshold")
    if (adata.obs["n_genes_by_counts"] < args.min_genes).any() or adata.obs[
        "predicted_doublet"
    ].any():
        raise RuntimeError("Final QC contains a failed basic-QC cell or predicted doublet")

    initial_metrics["doublet_score"] = np.nan
    initial_metrics["predicted_doublet"] = pd.NA
    initial_metrics.loc[post_scrublet_metrics.index, "doublet_score"] = post_scrublet_metrics[
        "doublet_score"
    ]
    initial_metrics.loc[post_scrublet_metrics.index, "predicted_doublet"] = (
        post_scrublet_metrics["predicted_doublet"].astype(bool)
    )
    initial_metrics["retained_final"] = initial_metrics.index.isin(adata.obs_names)
    initial_metrics.index.name = "obs_name"
    cell_metrics = initial_metrics.reset_index()
    atomic_csv(
        cell_metrics,
        args.table_dir / "cell_qc_metrics.csv.gz",
        compression="gzip",
    )
    make_qc_figures(cell_metrics, args.figure_dir)

    sample_initial = (
        cell_metrics.groupby(["series", "sample", "status"], observed=True)
        .agg(
            n_cells_input=("obs_name", "size"),
            n_pass_basic=("pass_basic_cell_qc", "sum"),
            n_predicted_doublet=("predicted_doublet", lambda x: int((x == True).sum())),
            n_cells_final=("retained_final", "sum"),
            median_total_counts=("total_counts", "median"),
            median_n_genes=("n_genes_by_counts", "median"),
            median_pct_mt=("pct_counts_MT", "median"),
            median_pct_ribo=("pct_counts_RIBO", "median"),
        )
        .reset_index()
    )
    backend_table = pd.DataFrame(scrublet_audit)[
        ["series", "sample", "backend", "state", "threshold", "cpu_fallback"]
    ]
    sample_summary = sample_initial.merge(
        backend_table, on=["series", "sample"], how="left", validate="one_to_one"
    )
    parameters = {
        "input_h5ad": str(args.input),
        "input_sha256": sha256(args.input),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "output_h5ad": str(output),
        "min_genes_per_cell": args.min_genes,
        "max_pct_mitochondrial_counts": args.max_pct_mt,
        "mitochondrial_gene_rule": "case-insensitive prefix MT-",
        "ribosomal_gene_rule": "case-insensitive prefix RPS or RPL",
        "min_cells_per_gene": args.min_cells_per_gene,
        "gene_filter_applied_before_and_after_doublet_removal": True,
        "scrublet_unit": "independent per series+sample",
        "scrublet_backend_priority": "rapids-singlecell GPU then audited Scanpy CPU fallback",
        "cpu_fallback_allowed": args.allow_cpu_fallback,
        "expected_doublet_rate": args.expected_doublet_rate,
        "sim_doublet_ratio": args.sim_doublet_ratio,
        "scrublet_pcs_requested": args.scrublet_pcs,
        "scrublet_threshold": "automatic per sample",
        "seed": SEED,
        "downsampled": False,
        "normalization": False,
        "input_shape": list(original_shape),
        "genes_before_first_filter": genes_before,
        "genes_after_first_filter": genes_after_first_filter,
        "output_shape": list(adata.shape),
        "failed_complete_samples_excluded": int(len(excluded_keys)),
        "failed_sample_rule": "exclude complete sample and continue; stop only if none remain",
    }
    required_qc_columns = [
        "total_counts",
        "n_genes_by_counts",
        "total_counts_MT",
        "pct_counts_MT",
        "total_counts_RIBO",
        "pct_counts_RIBO",
        "doublet_score",
        "predicted_doublet",
    ]
    missing_qc_columns = [column for column in required_qc_columns if column not in adata.obs]
    if missing_qc_columns:
        raise RuntimeError(f"Final h5ad lacks required QC columns: {missing_qc_columns}")

    scrublet_audit_frame = pd.DataFrame(scrublet_audit)
    input_summary = pd.DataFrame(
        [
            {
                "input_h5ad": str(args.input),
                "n_cells": original_shape[0],
                "n_genes": original_shape[1],
                "n_samples": len(original_samples),
                "n_series": initial_metrics["series"].astype(str).nunique(),
                "n_tumor_cells": int((initial_metrics["status"] == "tumor").sum()),
                "n_normal_cells": int((initial_metrics["status"] == "normal").sum()),
                "raw_integer_counts": True,
                "downsampled": False,
            }
        ]
    )
    cell_filter_audit = pd.DataFrame(
        [
            {"stage": "input", "n_cells_remaining": original_shape[0], "n_removed_at_stage": 0},
            {
                "stage": "basic_cell_qc",
                "n_cells_remaining": int(initial_metrics["pass_basic_cell_qc"].sum()),
                "n_removed_at_stage": int((~initial_metrics["pass_basic_cell_qc"]).sum()),
            },
            {
                "stage": "complete_failed_sample_exclusion",
                "n_cells_remaining": n_after_sample_exclusions,
                "n_removed_at_stage": int(initial_metrics["pass_basic_cell_qc"].sum())
                - n_after_sample_exclusions,
            },
            {
                "stage": "predicted_doublet_removal",
                "n_cells_remaining": adata.n_obs,
                "n_removed_at_stage": n_after_sample_exclusions - adata.n_obs,
            },
        ]
    )
    gene_filter_audit = pd.DataFrame(
        [
            {
                "stage": "merge_input",
                "n_genes_remaining": original_shape[1],
                "n_removed_at_stage": 0,
                "rule": "hierarchical inner-join output",
            },
            {
                "stage": "after_basic_cell_qc",
                "n_genes_remaining": genes_after_first_filter,
                "n_removed_at_stage": genes_before - genes_after_first_filter,
                "rule": f"detected in >= {args.min_cells_per_gene} cells",
            },
            {
                "stage": "final_after_doublet_removal",
                "n_genes_remaining": adata.n_vars,
                "n_removed_at_stage": genes_after_first_filter - adata.n_vars,
                "rule": f"detected in >= {args.min_cells_per_gene} retained cells",
            },
        ]
    )
    parameter_table = pd.DataFrame(
        [
            {
                "parameter": key,
                "value": json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list, tuple, bool))
                else value,
            }
            for key, value in parameters.items()
        ]
    )
    gene_identifier_audit = pd.DataFrame(
        [
            gene_identifier_row("merge_input", input_gene_names),
            gene_identifier_row("after_first_gene_filter", genes_after_first_names),
            gene_identifier_row("final_qc", adata.var_names),
        ]
    )

    # Seven numbered, compact audit/parameter tables required by this block.
    atomic_csv(input_summary, args.table_dir / "01_qc_input_summary.csv")
    atomic_csv(exclusion_frame, args.table_dir / "02_scrublet_failed_sample_exclusions.csv")
    atomic_csv(scrublet_audit_frame, args.table_dir / "03_scrublet_sample_audit.csv")
    atomic_csv(sample_summary, args.table_dir / "04_sample_qc_summary.csv")
    atomic_csv(cell_filter_audit, args.table_dir / "05_cell_filter_audit.csv")
    atomic_csv(gene_filter_audit, args.table_dir / "06_gene_filter_audit.csv")
    atomic_csv(parameter_table, args.table_dir / "07_qc_parameters.csv")
    atomic_csv(gene_identifier_audit, args.table_dir / "gene_identifier_audit.csv")

    adata.uns["qc_parameters"] = parameters
    atomic_h5ad(adata, output)
    validation = {
        "all_passed": True,
        "input_shape": list(original_shape),
        "output_shape": list(adata.shape),
        "n_samples_input": len(original_samples),
        "n_samples_output": len(final_samples),
        "n_complete_samples_excluded": len(excluded_keys),
        "sample_set_matches_audited_exclusions": final_samples == expected_final_samples,
        "raw_integer_counts_preserved": True,
        "no_predicted_doublets_retained": not bool(adata.obs["predicted_doublet"].any()),
        "max_observed_pct_mt": float(adata.obs["pct_counts_MT"].max()),
        "min_observed_n_genes": int(adata.obs["n_genes_by_counts"].min()),
        "required_qc_columns": required_qc_columns,
        "all_required_qc_columns_present": not missing_qc_columns,
        "downsampled": False,
        "elapsed_seconds": time.time() - started,
        "output_sha256": sha256(output),
    }
    atomic_json(parameters, args.table_dir / "qc_parameters.json")
    atomic_json(validation, args.table_dir / "qc_validation.json")
    atomic_json(package_versions(), args.table_dir / "package_versions.json")
    (args.table_dir / "qc_report.txt").write_text(
        f"QC completed with seed {SEED}.\n"
        f"Input: {original_shape[0]} cells, {original_shape[1]} genes, {len(original_samples)} samples.\n"
        f"Final: {adata.n_obs} cells, {adata.n_vars} genes, {len(final_samples)} samples.\n"
        f"Complete samples excluded after allowed routes failed: {len(excluded_keys)}.\n"
        f"Predicted doublets removed from successful samples: {int(post_scrublet_metrics['predicted_doublet'].sum())}.\n"
        "Counts remain raw integers; no normalization or downsampling was performed.\n"
        "Required QC columns: " + ", ".join(required_qc_columns) + ".\n"
        "Figures: PDF/SVG only.\n"
    )
    (args.table_dir / "readme.txt").write_text(
        "adata_qc.h5ad retains raw integer counts in X and contains no normalized layer.\n"
        f"Cells require >= {args.min_genes} detected genes and <= {args.max_pct_mt}% MT counts.\n"
        f"Genes require detection in >= {args.min_cells_per_gene} retained cells.\n"
        "Scrublet is run independently on every full sample with seed 42; no cell "
        "downsampling is performed. GPU is attempted first and every fallback/failure "
        "is recorded. If all allowed routes fail, that complete sample is excluded and "
        "the remaining samples continue; the run stops only if no sample remains.\n"
        "Eight required cell-QC columns include total/pct MT, total/pct RIBO, total "
        "counts, detected genes, doublet_score, and predicted_doublet.\n"
        "Figures are emitted only as PDF and SVG.\n"
    )


if __name__ == "__main__":
    main()

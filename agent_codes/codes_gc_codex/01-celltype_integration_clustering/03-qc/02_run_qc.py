#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
from scipy import sparse


ROOT = Path(__file__).resolve().parents[4]
INPUT = ROOT / "epi-cm-core-workflow/h5ad/01-celltype_integration_clustering/02-merge-metadata/adata_merge.h5ad"
OUTPUT = ROOT / "epi-cm-core-workflow/h5ad/01-celltype_integration_clustering/03-qc/adata_qc.h5ad"
TABLE_DIR = ROOT / "epi-cm-core-workflow/tables/01-celltype_integration_clustering/03-qc"
FIGURE_DIR = ROOT / "epi-cm-core-workflow/figures/01-celltype_integration_clustering/03-qc"
CHECKPOINT_DIR = TABLE_DIR / "02-scrublet-per-sample"

SEED = 42
MIN_GENES_INITIAL = 200
EXPECTED_DOUBLET_RATE = 0.05
SCRUBLET_N_PCS = 30
MIN_CELLS_GENE = 3
MIN_COUNTS_GENE = 3
MIN_GENES_FINAL = 200
MAX_GENES_FINAL = 6000
MAX_PCT_MT = 20.0


def write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_qc(adata: ad.AnnData) -> None:
    adata.X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    mt_mask = adata.var_names.astype(str).str.startswith("MT-")
    ribo_mask = adata.var_names.astype(str).str.startswith("RPS")
    adata.var["MT"] = mt_mask
    adata.var["RIBO"] = ribo_mask
    total = np.asarray(adata.X.sum(axis=1)).ravel()
    n_genes = np.diff(adata.X.indptr)
    mt = np.asarray(adata.X[:, mt_mask].sum(axis=1)).ravel()
    ribo = np.asarray(adata.X[:, ribo_mask].sum(axis=1)).ravel()
    adata.obs["n_genes_by_counts"] = n_genes
    adata.obs["total_counts"] = total
    adata.obs["total_counts_MT"] = mt
    adata.obs["pct_counts_MT"] = np.divide(
        mt, total, out=np.zeros_like(mt, dtype=float), where=total > 0
    ) * 100
    adata.obs["total_counts_RIBO"] = ribo
    adata.obs["pct_counts_RIBO"] = np.divide(
        ribo, total, out=np.zeros_like(ribo, dtype=float), where=total > 0
    ) * 100


def release_gpu() -> None:
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, cp.cuda.memory.OutOfMemoryError) or "out of memory" in str(exc).lower()


def validate_calls(calls: pd.DataFrame, expected_names: pd.Index) -> None:
    required = {"cell_id", "doublet_score", "predicted_doublet"}
    if not required.issubset(calls.columns):
        raise ValueError(f"missing Scrublet columns: {sorted(required - set(calls.columns))}")
    if not pd.Index(calls["cell_id"]).equals(expected_names):
        raise ValueError("Scrublet output cell IDs or order do not match input")
    if calls["doublet_score"].isna().any() or calls["predicted_doublet"].isna().any():
        raise ValueError("Scrublet calls contain missing values")


def run_rsc_scrublet(sample_adata: ad.AnnData) -> tuple[pd.DataFrame, float | None]:
    input_names = sample_adata.obs_names.copy()
    actual_pcs = min(SCRUBLET_N_PCS, sample_adata.n_obs - 2, sample_adata.n_vars - 1)
    if actual_pcs < 2:
        raise ValueError(f"Scrublet input has fewer than two usable PCs: {sample_adata.shape}")
    # CuPy sparse matrices do not accept integer dtypes.  This is an isolated
    # Scrublet working copy; the caller's raw-count AnnData remains int32.
    sample_adata.X = sample_adata.X.astype(np.float32, copy=False)
    rsc.get.anndata_to_GPU(sample_adata)
    rsc.pp.scrublet(
        sample_adata,
        expected_doublet_rate=EXPECTED_DOUBLET_RATE,
        n_prin_comps=actual_pcs,
        random_state=SEED,
        verbose=False,
    )
    threshold = sample_adata.uns.get("scrublet", {}).get("threshold")
    rsc.get.anndata_to_CPU(sample_adata)
    calls = pd.DataFrame(
        {
            "cell_id": sample_adata.obs_names,
            "doublet_score": sample_adata.obs["doublet_score"].astype(float).to_numpy(),
            "predicted_doublet": sample_adata.obs["predicted_doublet"].astype(bool).to_numpy(),
        }
    )
    validate_calls(calls, input_names)
    return calls, float(threshold) if threshold is not None else None


def run_cpu_scrublet(sample_adata: ad.AnnData) -> tuple[pd.DataFrame, float | None]:
    input_names = sample_adata.obs_names.copy()
    actual_pcs = min(SCRUBLET_N_PCS, sample_adata.n_obs - 2, sample_adata.n_vars - 1)
    if actual_pcs < 2:
        raise ValueError(f"Scrublet input has fewer than two usable PCs: {sample_adata.shape}")
    sc.pp.scrublet(
        sample_adata,
        expected_doublet_rate=EXPECTED_DOUBLET_RATE,
        n_prin_comps=actual_pcs,
        random_state=SEED,
        verbose=False,
    )
    threshold = sample_adata.uns.get("scrublet", {}).get("threshold")
    calls = pd.DataFrame(
        {
            "cell_id": sample_adata.obs_names,
            "doublet_score": sample_adata.obs["doublet_score"].astype(float).to_numpy(),
            "predicted_doublet": sample_adata.obs["predicted_doublet"].astype(bool).to_numpy(),
        }
    )
    validate_calls(calls, input_names)
    return calls, float(threshold) if threshold is not None else None


def save_qc_figure(adata: ad.AnnData, stem: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    specs = [
        (np.log10(adata.obs["n_genes_by_counts"].to_numpy() + 1), "log10(n_genes + 1)"),
        (np.log10(adata.obs["total_counts"].to_numpy() + 1), "log10(total_counts + 1)"),
        (adata.obs["pct_counts_MT"].to_numpy(), "pct_counts_MT"),
        (adata.obs["doublet_score"].to_numpy(), "doublet_score"),
    ]
    for ax, (values, label) in zip(axes.ravel(), specs):
        ax.hist(values, bins=120, color="#4C78A8", edgecolor="none")
        ax.set_xlabel(label)
        ax.set_ylabel("Cells")
    fig.suptitle(title)
    fig.tight_layout()
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".svg"]:
        path = stem.with_suffix(suffix)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def group_counts(adata: ad.AnnData) -> dict[str, int]:
    return adata.obs["sample"].astype(str).value_counts(sort=False).astype(int).to_dict()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {OUTPUT}")
    np.random.seed(SEED)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"reading {INPUT}", flush=True)
    adata = ad.read_h5ad(INPUT)
    compute_qc(adata)
    input_counts = group_counts(adata)
    n_obs_input, n_vars_input = adata.shape

    initial_keep = adata.obs["n_genes_by_counts"].to_numpy() >= MIN_GENES_INITIAL
    adata = adata[initial_keep].copy()
    compute_qc(adata)
    after_initial_counts = group_counts(adata)
    if adata.obs["sample"].nunique() != len(input_counts):
        raise RuntimeError("Initial cell filter removed an entire sample")
    print(f"after initial min_genes={MIN_GENES_INITIAL}: {adata.n_obs} cells", flush=True)

    scores = pd.Series(np.nan, index=adata.obs_names, dtype=float)
    predictions = pd.Series(pd.NA, index=adata.obs_names, dtype="boolean")
    scrublet_rows = []
    failure_rows = []
    sample_order = list(dict.fromkeys(adata.obs["sample"].astype(str)))

    for order, sample in enumerate(sample_order, start=1):
        sample_mask = adata.obs["sample"].astype(str).eq(sample).to_numpy()
        sample_adata = adata[sample_mask].copy()
        actual_pcs = min(SCRUBLET_N_PCS, sample_adata.n_obs - 2, sample_adata.n_vars - 1)
        checkpoint = CHECKPOINT_DIR / f"{sample}_scrublet_calls.csv.gz"
        start = time.monotonic()
        backend = "rapids-singlecell_gpu"
        gpu_attempts = 0
        cpu_fallback = False
        fallback_reason = ""
        threshold = None
        valid = False
        failure_reason = ""
        print(f"[{order:02d}/{len(sample_order)}] Scrublet {sample}: {sample_adata.n_obs} cells", flush=True)

        if checkpoint.exists():
            calls = pd.read_csv(checkpoint, dtype={"cell_id": str, "predicted_doublet": bool})
            validate_calls(calls, sample_adata.obs_names)
            backend = "validated_checkpoint_reuse"
            valid = True
        else:
            calls = pd.DataFrame()
            for attempt in [1, 2]:
                gpu_attempts = attempt
                try:
                    calls, threshold = run_rsc_scrublet(sample_adata.copy())
                    valid = True
                    break
                except Exception as exc:
                    release_gpu()
                    if is_oom(exc):
                        fallback_reason = f"GPU OOM on attempt {attempt}: {exc}"
                        cpu_fallback = True
                        backend = "scanpy_cpu_after_gpu_oom"
                        try:
                            calls, threshold = run_cpu_scrublet(sample_adata.copy())
                            valid = True
                        except Exception as cpu_exc:
                            failure_reason = f"GPU OOM then CPU Scrublet failed: {cpu_exc}"
                        break
                    failure_reason = f"non-OOM GPU attempt {attempt} failed: {exc}"
                    if attempt == 1:
                        print(f"  non-OOM failure; releasing GPU and retrying same rsc method: {exc}", flush=True)
                        continue
            release_gpu()
            if valid:
                if checkpoint.exists():
                    raise FileExistsError(checkpoint)
                calls.to_csv(checkpoint, index=False, compression="gzip")

        elapsed = time.monotonic() - start
        if valid:
            scores.loc[calls["cell_id"]] = calls["doublet_score"].to_numpy(dtype=float)
            predictions.loc[calls["cell_id"]] = calls["predicted_doublet"].to_numpy(dtype=bool)
            predicted_n = int(calls["predicted_doublet"].astype(bool).sum())
            print(f"  pass via {backend}: {predicted_n} predicted doublets", flush=True)
        else:
            predicted_n = 0
            failure_rows.append(
                {
                    "sample": sample,
                    "n_cells_excluded": sample_adata.n_obs,
                    "failure_reason": failure_reason,
                    "action": "exclude_entire_sample",
                }
            )
            print(f"  failed; entire sample will be excluded: {failure_reason}", flush=True)
        scrublet_rows.append(
            {
                "sample_order": order,
                "sample": sample,
                "n_cells_input": sample_adata.n_obs,
                "n_cells_output": len(calls) if valid else 0,
                "backend": backend,
                "gpu_attempts": gpu_attempts,
                "cpu_fallback": cpu_fallback,
                "fallback_reason": fallback_reason,
                "valid_complete_calls": valid,
                "n_predicted_doublets": predicted_n,
                "doublet_fraction": predicted_n / sample_adata.n_obs if valid else np.nan,
                "threshold": threshold,
                "expected_doublet_rate": EXPECTED_DOUBLET_RATE,
                "n_prin_comps_requested": SCRUBLET_N_PCS,
                "n_prin_comps_used": actual_pcs,
                "random_seed": SEED,
                "elapsed_seconds": elapsed,
                "checkpoint": str(checkpoint) if valid else "",
            }
        )
        del sample_adata, calls
        gc.collect()

    failed_samples = {row["sample"] for row in failure_rows}
    if failed_samples:
        adata = adata[~adata.obs["sample"].astype(str).isin(failed_samples)].copy()
        scores = scores.loc[adata.obs_names]
        predictions = predictions.loc[adata.obs_names]
    if adata.n_obs == 0:
        raise RuntimeError("No samples remain after failed Scrublet exclusions")
    if scores.isna().any() or predictions.isna().any():
        raise RuntimeError("Retained samples do not all have complete Scrublet calls")
    adata.obs["doublet_score"] = scores.loc[adata.obs_names].to_numpy(dtype=float)
    adata.obs["predicted_doublet"] = predictions.loc[adata.obs_names].to_numpy(dtype=bool)

    save_qc_figure(adata, FIGURE_DIR / "02_qc_before_filter_histograms", "Before doublet and final QC filters")
    call_table = adata.obs[["sample", "series", "status", "doublet_score", "predicted_doublet"]].copy()
    call_table.insert(0, "cell_id", call_table.index)
    call_path = TABLE_DIR / "02_scrublet_cell_calls.csv.gz"
    if call_path.exists():
        raise FileExistsError(call_path)
    call_table.to_csv(call_path, index=False, compression="gzip")

    before_doublet_counts = group_counts(adata)
    adata = adata[~adata.obs["predicted_doublet"].to_numpy(dtype=bool)].copy()
    after_doublet_counts = group_counts(adata)
    n_obs_after_doublet = adata.n_obs

    x_csc = adata.X.tocsc()
    gene_n_cells = np.diff(x_csc.indptr)
    gene_counts = np.asarray(x_csc.sum(axis=0)).ravel()
    gene_keep = (gene_n_cells >= MIN_CELLS_GENE) & (gene_counts >= MIN_COUNTS_GENE)
    gene_filter = pd.DataFrame(
        {
            "gene_symbol": adata.var_names,
            "n_cells_by_counts_before_gene_filter": gene_n_cells,
            "total_counts_before_gene_filter": gene_counts,
            "kept": gene_keep,
        }
    )
    write_csv_new(TABLE_DIR / "03_gene_filter_audit.csv", gene_filter)
    adata = adata[:, gene_keep].copy()
    compute_qc(adata)
    n_vars_after_gene = adata.n_vars
    before_final_counts = group_counts(adata)

    final_keep = (
        (adata.obs["n_genes_by_counts"].to_numpy() >= MIN_GENES_FINAL)
        & (adata.obs["n_genes_by_counts"].to_numpy() <= MAX_GENES_FINAL)
        & (adata.obs["pct_counts_MT"].to_numpy() <= MAX_PCT_MT)
    )
    adata = adata[final_keep].copy()
    compute_qc(adata)
    final_counts = group_counts(adata)
    adata.obs["qc_pass"] = True
    adata.uns["matrix_semantics"] = "raw unnormalized integer counts in X after QC"
    adata.uns["qc_parameters"] = {
        "input_h5ad": str(INPUT),
        "min_genes_initial": MIN_GENES_INITIAL,
        "expected_doublet_rate": EXPECTED_DOUBLET_RATE,
        "scrublet_n_prin_comps": SCRUBLET_N_PCS,
        "scrublet_random_seed": SEED,
        "min_cells_gene": MIN_CELLS_GENE,
        "min_counts_gene": MIN_COUNTS_GENE,
        "min_genes_final": MIN_GENES_FINAL,
        "max_genes_final": MAX_GENES_FINAL,
        "max_pct_mt": MAX_PCT_MT,
        "failed_scrublet_sample_action": "exclude_entire_sample",
    }
    if adata.X.data.size and not np.allclose(adata.X.data, np.round(adata.X.data)):
        raise RuntimeError("Post-QC X is no longer integer-valued raw counts")

    save_qc_figure(adata, FIGURE_DIR / "03_qc_after_filter_histograms", "After QC")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp.h5ad")
    if temporary.exists():
        temporary.unlink()
    adata.write_h5ad(temporary, compression="gzip")
    os.replace(temporary, OUTPUT)

    selected = pd.DataFrame(
        [
            {"parameter": "min_genes_initial", "value": MIN_GENES_INITIAL, "rationale": "Current 532520-cell audit: removes 30617 cells (5.75%) before Scrublet."},
            {"parameter": "expected_doublet_rate", "value": EXPECTED_DOUBLET_RATE, "rationale": "rapids-singlecell Scrublet default project-style rate."},
            {"parameter": "min_cells_gene", "value": MIN_CELLS_GENE, "rationale": "Current audit retained all 8290 cross-series shared genes at min_cells=3."},
            {"parameter": "min_counts_gene", "value": MIN_COUNTS_GENE, "rationale": "Current audit retained all 8290 cross-series shared genes at min_counts=3."},
            {"parameter": "min_genes_final", "value": MIN_GENES_FINAL, "rationale": "Same conservative lower cutoff as the initial filter; preserves rare low-RNA cell types."},
            {"parameter": "max_genes_final", "value": MAX_GENES_FINAL, "rationale": "Current audit: only 82/532520 cells exceed 6000 detected shared genes; Scrublet runs first."},
            {"parameter": "max_pct_mt", "value": MAX_PCT_MT, "rationale": "Current audit median is 4.48%; 20% removes the distinct high-MT tail (54036/532520 before joint filters)."},
        ]
    )
    write_csv_new(TABLE_DIR / "01_selected_qc_thresholds.csv", selected)
    write_csv_new(TABLE_DIR / "02_scrublet_parameters.csv", pd.DataFrame(scrublet_rows))
    write_csv_new(
        TABLE_DIR / "02_scrublet_failed_sample_exclusions.csv",
        pd.DataFrame(
            failure_rows,
            columns=["sample", "n_cells_excluded", "failure_reason", "action"],
        ),
    )

    all_samples = list(input_counts)
    retention = []
    for sample in all_samples:
        retention.append(
            {
                "sample": sample,
                "series": "",
                "n_cells_input": input_counts.get(sample, 0),
                "n_after_initial_min_genes": after_initial_counts.get(sample, 0),
                "n_excluded_whole_sample_scrublet_failure": after_initial_counts.get(sample, 0) if sample in failed_samples else 0,
                "n_before_doublet_filter": before_doublet_counts.get(sample, 0),
                "n_after_doublet_filter": after_doublet_counts.get(sample, 0),
                "n_before_final_cell_filter": before_final_counts.get(sample, 0),
                "n_after_qc": final_counts.get(sample, 0),
                "retained_fraction_from_merge": final_counts.get(sample, 0) / input_counts[sample],
            }
        )
    retention_frame = pd.DataFrame(retention)
    sample_to_series = (
        pd.read_csv(
            ROOT / "epi-cm-core-workflow/tables/01-celltype_integration_clustering/01-sample-manifest/all_datasets_curated_xlsx_inclusion_v5.csv"
        )
        .set_index("sample")["series"]
        .to_dict()
    )
    retention_frame["series"] = retention_frame["sample"].map(sample_to_series)
    write_csv_new(TABLE_DIR / "04_cell_retention_by_sample.csv", retention_frame)

    qc_report = (
        adata.obs.groupby(["series", "sample", "status", "patient"], observed=True)
        .agg(
            n_cells_after_qc=("sample", "size"),
            median_n_genes=("n_genes_by_counts", "median"),
            median_total_counts=("total_counts", "median"),
            median_pct_mt=("pct_counts_MT", "median"),
            median_pct_ribo=("pct_counts_RIBO", "median"),
            median_doublet_score=("doublet_score", "median"),
        )
        .reset_index()
    )
    write_csv_new(TABLE_DIR / "qc_report.csv", qc_report)

    summary = pd.DataFrame(
        [
            {"step": "merged_input", "n_cells": n_obs_input, "n_genes": n_vars_input},
            {"step": "after_initial_min_genes", "n_cells": sum(after_initial_counts.values()), "n_genes": n_vars_input},
            {"step": "after_failed_scrublet_sample_exclusions", "n_cells": sum(before_doublet_counts.values()), "n_genes": n_vars_input},
            {"step": "after_predicted_doublet_filter", "n_cells": n_obs_after_doublet, "n_genes": n_vars_input},
            {"step": "after_gene_filter", "n_cells": n_obs_after_doublet, "n_genes": n_vars_after_gene},
            {"step": "after_final_cell_filter", "n_cells": adata.n_obs, "n_genes": adata.n_vars},
        ]
    )
    write_csv_new(TABLE_DIR / "qc_step_summary.csv", summary)

    validation = pd.DataFrame(
        [
            {"check": "all_retained_samples_have_complete_scrublet_calls", "passed": not scores.isna().any() and not predictions.isna().any(), "value": adata.obs["sample"].nunique()},
            {"check": "all_predicted_doublets_removed", "passed": not adata.obs["predicted_doublet"].any(), "value": int(adata.obs["predicted_doublet"].sum())},
            {"check": "required_qc_columns_present", "passed": {"n_genes_by_counts", "total_counts", "pct_counts_MT", "total_counts_MT", "pct_counts_RIBO", "total_counts_RIBO", "doublet_score", "predicted_doublet"}.issubset(adata.obs), "value": len(adata.obs.columns)},
            {"check": "final_cell_thresholds_satisfied", "passed": bool(((adata.obs["n_genes_by_counts"] >= MIN_GENES_FINAL) & (adata.obs["n_genes_by_counts"] <= MAX_GENES_FINAL) & (adata.obs["pct_counts_MT"] <= MAX_PCT_MT)).all()), "value": adata.n_obs},
            {"check": "post_qc_counts_integer_valued", "passed": bool(np.allclose(adata.X.data, np.round(adata.X.data))), "value": adata.X.nnz},
            {"check": "obs_and_var_names_unique", "passed": adata.obs_names.is_unique and adata.var_names.is_unique, "value": adata.n_vars},
            {"check": "adata_qc_written", "passed": OUTPUT.exists(), "value": OUTPUT.stat().st_size},
        ]
    )
    write_csv_new(TABLE_DIR / "qc_validation.csv", validation)
    artifact = pd.DataFrame(
        [
            {
                "file": str(OUTPUT),
                "size_bytes": OUTPUT.stat().st_size,
                "sha256": sha256(OUTPUT),
                "n_cells": adata.n_obs,
                "n_genes": adata.n_vars,
                "nnz": adata.X.nnz,
                "n_samples": adata.obs["sample"].nunique(),
                "n_failed_scrublet_samples": len(failed_samples),
            }
        ]
    )
    write_csv_new(TABLE_DIR / "adata_qc_artifact.csv", artifact)

    versions = {
        package: version(package)
        for package in [
            "rapids-singlecell",
            "cudf-cu12",
            "cuml-cu12",
            "cupy-cuda12x",
            "scanpy",
            "anndata",
            "numpy",
            "scipy",
        ]
    }
    versions.update(
        {
            "python": platform.python_version(),
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "random_seed": SEED,
        }
    )
    write_text_new(TABLE_DIR / "package_versions_02_run_qc.json", json.dumps(versions, indent=2))
    write_csv_new(
        TABLE_DIR / "gpu_compatibility_repair.csv",
        pd.DataFrame(
            [
                {
                    "attempt": 1,
                    "status": "failed_non_oom",
                    "error": "CuPy rejected raw int32 sparse working matrix",
                    "action": "stop profiling run; reload clean adata_merge.h5ad",
                    "caller_raw_counts_mutated": False,
                },
                {
                    "attempt": 2,
                    "status": "completed",
                    "error": "",
                    "action": "cast only each isolated Scrublet working copy to float32 before GPU transfer",
                    "caller_raw_counts_mutated": False,
                },
            ]
        ),
    )
    write_text_new(
        TABLE_DIR / "readme.txt",
        "\n".join(
            [
                "Block 01 / 03-qc",
                f"Input: {INPUT}",
                "Code order: 01_audit_qc_distributions.py, then 02_run_qc.py.",
                f"Output: {OUTPUT}",
                "Scrublet: one sample at a time on GPU; GPU OOM permits same-method Scanpy CPU fallback; repeated non-OOM GPU failure excludes the whole sample.",
                "Raw unnormalized integer counts remain in X. Thresholds and per-step/per-sample retention are recorded in CSV tables.",
                "Figures are PDF/SVG only.",
            ]
        )
        + "\n",
    )

    if not validation["passed"].all():
        raise RuntimeError("One or more post-QC validation checks failed")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "shape": adata.shape,
                "failed_scrublet_samples": sorted(failed_samples),
                "predicted_doublets_removed": int(sum(before_doublet_counts.values()) - n_obs_after_doublet),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

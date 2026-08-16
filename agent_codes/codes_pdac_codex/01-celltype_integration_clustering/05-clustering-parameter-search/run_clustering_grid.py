#!/usr/bin/env python3
"""Run the user-specified full PDAC broad-clustering grid with early stopping."""

from __future__ import annotations

import importlib.metadata
import json
import math
import random
import sys
import time
from decimal import Decimal
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc


SEED = 42
PCS_VALUES = list(range(10, 51, 5))
NN_VALUES = list(range(10, 51, 5))
RESOLUTIONS = [Decimal(i) / Decimal(10) for i in range(1, 11)]
EARLY_STOP_CLUSTER_COUNT = 20
INTEGRATED_BASIS = "X_pca_inte"

random.seed(SEED)
np.random.seed(SEED)
cp.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
SOURCE_H5AD = PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_harmony.h5ad"
TABLE_ROOT = PROJECT_ROOT / "tables/01-celltype_integration_clustering/05-clustering-parameter-search"
FIGURE_ROOT = PROJECT_ROOT / "figures/01-celltype_integration_clustering/05-clustering-parameter-search"
MANIFEST_PATH = TABLE_ROOT / "clustering_grid_manifest.csv"
REVIEW_PATH = TABLE_ROOT / "candidate_review_manifest.csv"
COMPLETION_PATH = TABLE_ROOT / "clustering_grid_completion_check.csv"


def res_text(resolution: Decimal) -> str:
    return format(resolution, ".1f")


def res_token(resolution: Decimal) -> str:
    return res_text(resolution).replace(".", "p")


def candidate_label(n_pcs: int, n_neighbors: int, resolution: Decimal) -> str:
    return f"pcs-{n_pcs}_nn-{n_neighbors}_res-{res_token(resolution)}"


def atomic_manifest_write(frame: pd.DataFrame) -> None:
    temporary = MANIFEST_PATH.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(MANIFEST_PATH)


def normalized_entropy(values: pd.Series) -> float:
    counts = values.value_counts().to_numpy(dtype=float)
    if counts.size <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(counts.size))


def mixing_table(adata, cluster_key: str) -> pd.DataFrame:
    frame = adata.obs[[cluster_key, "sample", "series", "status"]].copy()
    rows = []
    for cluster, subset in frame.groupby(cluster_key, observed=True):
        sample_counts = subset["sample"].astype(str).value_counts()
        series_counts = subset["series"].astype(str).value_counts()
        status_counts = subset["status"].astype(str).value_counts()
        rows.append(
            {
                "cluster": str(cluster),
                "n_cells": subset.shape[0],
                "n_samples": sample_counts.size,
                "n_series": series_counts.size,
                "max_sample_fraction": float(sample_counts.iloc[0] / subset.shape[0]),
                "sample_entropy_normalized": normalized_entropy(subset["sample"].astype(str)),
                "max_series_fraction": float(series_counts.iloc[0] / subset.shape[0]),
                "tumor_fraction": float(status_counts.get("tumor", 0) / subset.shape[0]),
                "normal_like_fraction": float(status_counts.get("normal-like", 0) / subset.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not SOURCE_H5AD.is_file():
        raise FileNotFoundError(SOURCE_H5AD)
    root_outputs = [
        MANIFEST_PATH,
        REVIEW_PATH,
        COMPLETION_PATH,
        TABLE_ROOT / "grid_parameters.csv",
        TABLE_ROOT / "gpu_backend_capability_summary.csv",
        TABLE_ROOT / "package_versions.txt",
        TABLE_ROOT / "readme.txt",
    ]
    for path in root_outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for n_pcs in PCS_VALUES:
        for n_neighbors in NN_VALUES:
            for resolution in RESOLUTIONS:
                label = candidate_label(n_pcs, n_neighbors, resolution)
                manifest_rows.append(
                    {
                        "n_pcs": n_pcs,
                        "n_neighbors": n_neighbors,
                        "resolution": float(resolution),
                        "candidate_label": label,
                        "expected_cluster_count_table": str(TABLE_ROOT / label / "cluster_counts.csv"),
                        "expected_parameter_table": str(TABLE_ROOT / label / "candidate_parameters.csv"),
                        "expected_figure_dir": str(FIGURE_ROOT / label),
                        "status": "planned",
                        "completed": False,
                        "reason_if_skipped": "",
                    }
                )
    manifest = pd.DataFrame(manifest_rows)
    if manifest.shape[0] != 810:
        raise AssertionError(f"Expected 810 manifest rows, observed {manifest.shape[0]}")
    atomic_manifest_write(manifest)
    pd.DataFrame(
        [
            {
                "source_h5ad": str(SOURCE_H5AD),
                "n_pcs_values": ";".join(map(str, PCS_VALUES)),
                "n_neighbors_values": ";".join(map(str, NN_VALUES)),
                "resolution_values": ";".join(map(res_text, RESOLUTIONS)),
                "expected_manifest_rows": 810,
                "graph_candidates": len(PCS_VALUES) * len(NN_VALUES),
                "maximum_resolution_candidates": 810,
                "early_stop_rule": "after first resolution with >20 clusters, skip all higher resolutions for that PC/NN graph",
                "early_stop_user_specified": True,
                "candidate_h5ad_saved": False,
                "random_seed": SEED,
                "code_file": str(Path(__file__).resolve()),
            }
        ]
    ).to_csv(TABLE_ROOT / "grid_parameters.csv", index=False)

    # The candidate grid only consumes obs metadata and the Harmony basis.
    # Keep expression/raw matrices in the canonical source H5AD and construct
    # a graph-only transient template to avoid duplicating several GB of
    # unused scaled expression on the 11-GB GPU for every PC/NN graph.
    source_backed = sc.read_h5ad(SOURCE_H5AD, backed="r")
    if INTEGRATED_BASIS not in source_backed.obsm:
        raise ValueError(f"Source lacks obsm[{INTEGRATED_BASIS!r}]")
    if source_backed.obsm[INTEGRATED_BASIS].shape[1] < max(PCS_VALUES):
        raise ValueError("Harmony basis has fewer than 50 dimensions")
    if "neighbors" in source_backed.uns or "X_umap" in source_backed.obsm:
        raise ValueError("Harmony source is not a clean pre-clustering object")
    required_obs = ["sample", "series", "status"]
    missing_obs = [column for column in required_obs if column not in source_backed.obs]
    if missing_obs:
        raise ValueError(f"Source lacks required obs metadata: {missing_obs}")
    graph_obs = source_backed.obs[required_obs].copy()
    integrated_cpu = np.asarray(source_backed.obsm[INTEGRATED_BASIS])
    source_backed.file.close()
    del source_backed
    source = sc.AnnData(obs=graph_obs)
    source.obsm[INTEGRATED_BASIS] = cp.asarray(integrated_cpu)
    del integrated_cpu, graph_obs
    source_template_transferred = True
    review_rows = []
    sc.set_figure_params(figsize=(3, 3), dpi=150)

    for n_pcs in PCS_VALUES:
        for n_neighbors in NN_VALUES:
            graph_started = time.time()
            free_before, total_vram = cp.cuda.runtime.memGetInfo()
            adata_run = source.copy()
            rsc.pp.neighbors(
                adata_run,
                n_neighbors=n_neighbors,
                n_pcs=n_pcs,
                use_rep=INTEGRATED_BASIS,
                random_state=SEED,
                algorithm="brute",
            )
            rsc.tl.umap(adata_run, random_state=SEED)
            if isinstance(adata_run.obsm["X_umap"], cp.ndarray):
                adata_run.obsm["X_umap"] = cp.asnumpy(adata_run.obsm["X_umap"])

            stop_after_current = False
            for resolution_index, resolution in enumerate(RESOLUTIONS):
                label = candidate_label(n_pcs, n_neighbors, resolution)
                row_mask = manifest["candidate_label"].eq(label)
                if stop_after_current:
                    manifest.loc[row_mask, "status"] = "skipped_user_approved"
                    manifest.loc[row_mask, "completed"] = False
                    manifest.loc[row_mask, "reason_if_skipped"] = (
                        "higher_resolution_skipped_after_first_resolution_exceeded_20_clusters_per_user_rule"
                    )
                    atomic_manifest_write(manifest)
                    continue

                manifest.loc[row_mask, "status"] = "running"
                atomic_manifest_write(manifest)
                cluster_key = f"leiden_res{res_token(resolution)}"
                candidate_started = time.time()
                rsc.tl.leiden(
                    adata_run,
                    resolution=float(resolution),
                    key_added=cluster_key,
                    random_state=SEED,
                )
                cluster_counts = (
                    adata_run.obs[cluster_key].astype(str).value_counts().sort_index().rename("n_cells").reset_index()
                )
                cluster_counts.columns = ["cluster", "n_cells"]
                n_clusters = cluster_counts.shape[0]
                table_dir = TABLE_ROOT / label
                figure_dir = FIGURE_ROOT / label
                table_dir.mkdir(parents=True, exist_ok=False)
                figure_dir.mkdir(parents=True, exist_ok=False)
                cluster_counts.to_csv(table_dir / "cluster_counts.csv", index=False)
                mixing = mixing_table(adata_run, cluster_key)
                mixing.to_csv(table_dir / "cluster_batch_mixing.csv", index=False)

                sc.settings.figdir = figure_dir
                sc.pl.umap(
                    adata_run,
                    color=cluster_key,
                    show=False,
                    save=f"_{label}.pdf",
                )
                figure_path = figure_dir / f"umap_{label}.pdf"
                if not figure_path.is_file():
                    raise FileNotFoundError(f"Expected Scanpy UMAP output missing: {figure_path}")
                elapsed = time.time() - candidate_started
                parameters = {
                    "source_h5ad": str(SOURCE_H5AD),
                    "source_template_transferred_to_gpu_before_loop": source_template_transferred,
                    "source_template_content": "obs_sample_series_status_plus_X_pca_inte_only",
                    "source_expression_omitted_from_transient_grid_object": True,
                    "candidate_object_created_by_copy": True,
                    "n_pcs": n_pcs,
                    "n_neighbors": n_neighbors,
                    "resolution": float(resolution),
                    "neighbor_use_rep": INTEGRATED_BASIS,
                    "neighbor_algorithm": "brute",
                    "umap_random_state": SEED,
                    "clustering_algorithm": "leiden",
                    "raw_cluster_column": cluster_key,
                    "n_clusters": n_clusters,
                    "cluster_count_table": str(table_dir / "cluster_counts.csv"),
                    "cluster_batch_mixing_table": str(table_dir / "cluster_batch_mixing.csv"),
                    "output_figure": str(figure_path),
                    "code_file": str(Path(__file__).resolve()),
                    "random_seed": SEED,
                    "backend_package": "rapids-singlecell_GPU",
                    "candidate_h5ad_saved": False,
                    "candidate_object_deleted_after_outputs": True,
                    "gpu_memory_release_status": "released_after_graph_candidate_outputs",
                    "gpu_free_bytes_before_graph": int(free_before),
                    "gpu_total_bytes": int(total_vram),
                    "concurrency_limit": 1,
                    "gpu_assignment": "CUDA_VISIBLE_DEVICES default GPU 0",
                    "elapsed_seconds_candidate_resolution": elapsed,
                    "review_notes": (
                        "first_resolution_exceeding_20_clusters; higher_resolutions_skipped"
                        if n_clusters > EARLY_STOP_CLUSTER_COUNT
                        else "within_20_cluster_early_stop_limit"
                    ),
                }
                pd.DataFrame([parameters]).to_csv(table_dir / "candidate_parameters.csv", index=False)
                review_rows.append(
                    {
                        "candidate_label": label,
                        "n_pcs": n_pcs,
                        "n_neighbors": n_neighbors,
                        "resolution": float(resolution),
                        "n_clusters": n_clusters,
                        "median_cluster_n_samples": float(mixing["n_samples"].median()),
                        "worst_cluster_max_sample_fraction": float(mixing["max_sample_fraction"].max()),
                        "median_sample_entropy_normalized": float(mixing["sample_entropy_normalized"].median()),
                        "figure_path": str(figure_path),
                        "cluster_count_table": str(table_dir / "cluster_counts.csv"),
                        "mixing_table": str(table_dir / "cluster_batch_mixing.csv"),
                        "early_stop_triggered": n_clusters > EARLY_STOP_CLUSTER_COUNT,
                    }
                )
                manifest.loc[row_mask, "status"] = "completed"
                manifest.loc[row_mask, "completed"] = True
                manifest.loc[row_mask, "reason_if_skipped"] = ""
                atomic_manifest_write(manifest)
                if n_clusters > EARLY_STOP_CLUSTER_COUNT:
                    stop_after_current = True

            del adata_run
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            graph_elapsed = time.time() - graph_started
            if graph_elapsed <= 0:
                raise AssertionError("Invalid graph elapsed time")

    del source
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    review = pd.DataFrame(review_rows)
    review.to_csv(REVIEW_PATH, index=False)

    completed = int(manifest["status"].eq("completed").sum())
    skipped = int(manifest["status"].eq("skipped_user_approved").sum())
    failed = int(manifest["status"].eq("failed").sum())
    unresolved = int(manifest["status"].isin(["planned", "running"]).sum())
    output_checks = []
    for _, row in manifest.loc[manifest["completed"]].iterrows():
        output_checks.append(
            Path(row["expected_cluster_count_table"]).is_file()
            and Path(row["expected_parameter_table"]).is_file()
            and Path(row["expected_figure_dir"]).is_dir()
            and any(Path(row["expected_figure_dir"]).glob("*.pdf"))
        )
    all_outputs = bool(output_checks and all(output_checks))
    manual_ready = bool(failed == 0 and unresolved == 0 and completed + skipped == 810 and all_outputs)
    pd.DataFrame(
        [
            {
                "expected_candidates": 810,
                "completed_candidates": completed,
                "failed_candidates": failed,
                "skipped_candidates": skipped,
                "missing_candidates": unresolved,
                "all_expected_lightweight_outputs_exist": all_outputs,
                "manual_selection_ready": manual_ready,
                "user_early_stop_rule_applied": True,
                "graph_candidates": 81,
                "graphs_with_early_stop": int(review["early_stop_triggered"].sum()),
                "candidate_h5ad_files_saved": 0,
            }
        ]
    ).to_csv(COMPLETION_PATH, index=False)
    pd.DataFrame(
        [
            {
                "step": "full_clustering_grid",
                "planned_backend": "rapids-singlecell_GPU",
                "attempted_backend": "rapids-singlecell_GPU",
                "status": "completed" if manual_ready else "incomplete",
                "error_summary": "",
                "fallback_backend": "",
                "clean_input_reloaded": False,
                "final_backend_for_rerun": "rapids-singlecell_GPU",
            }
        ]
    ).to_csv(TABLE_ROOT / "gpu_backend_capability_summary.csv", index=False)
    packages = [
        "rapids-singlecell-cu13", "cudf-cu13", "cuml-cu13", "cugraph-cu13",
        "cuvs-cu13", "cupy-cuda13x", "scanpy", "anndata", "numpy", "pandas",
        "scipy", "matplotlib",
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
    for package in packages:
        lines.append(f"{package}={importlib.metadata.version(package)}")
    (TABLE_ROOT / "package_versions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (TABLE_ROOT / "readme.txt").write_text(
        "PDAC broad-clustering candidate grid\n\n"
        f"Source: {SOURCE_H5AD}\n"
        "Executed user grid: PCs 10-50 step 5; neighbors 10-50 step 5; "
        "Leiden resolution 0.1-1.0 step 0.1.  For each PC/NN graph, the first "
        "resolution producing more than 20 clusters was retained as an executed "
        "candidate and every higher resolution was recorded as skipped_user_approved.\n"
        "Candidate outputs are lightweight tables and PDF UMAPs only; no candidate "
        "H5AD was saved.  A final selected H5AD and selected_clustering.csv must not "
        "be created until the user manually chooses a completed candidate.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "expected_candidates": 810,
                "completed_candidates": completed,
                "skipped_candidates": skipped,
                "failed_candidates": failed,
                "manual_selection_ready": manual_ready,
                "review_manifest": str(REVIEW_PATH),
            },
            indent=2,
        )
    )
    if not manual_ready:
        raise RuntimeError("Clustering grid did not reach manual-selection readiness")


if __name__ == "__main__":
    main()

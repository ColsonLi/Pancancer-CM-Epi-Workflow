#!/usr/bin/env python3
"""Run the canonical 81-graph, 15-resolution epithelial subtype grid on GPU."""

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
from anndata import AnnData


SEED = 42
PCS_VALUES = list(range(10, 51, 5))
NN_VALUES = list(range(10, 51, 5))
RESOLUTIONS = [Decimal(i) / Decimal(10) for i in range(1, 16)]
INTEGRATED_BASIS = "X_pca_inte"
PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
SOURCE_H5AD = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/02-subtype-harmony/epithelial/adata_epithelial_harmony.h5ad"
)
TABLE_ROOT = (
    PROJECT_ROOT
    / "tables/02-cell_subtype_integration_clustering/03-subtype-clustering-grid/epithelial"
)
FIGURE_ROOT = (
    PROJECT_ROOT
    / "figures/02-cell_subtype_integration_clustering/03-subtype-clustering-grid/epithelial"
)
MANIFEST_PATH = TABLE_ROOT / "epithelial_clustering_grid_manifest.csv"


random.seed(SEED)
np.random.seed(SEED)
cp.random.seed(SEED)


def res_token(value: Decimal) -> str:
    return format(value, "0.1f").replace(".", "p")


def graph_label(n_pcs: int, n_neighbors: int) -> str:
    return f"pcs-{n_pcs}_nn-{n_neighbors}_res-0p1-1p5"


def candidate_label(n_pcs: int, n_neighbors: int, resolution: Decimal) -> str:
    return f"pcs-{n_pcs}_nn-{n_neighbors}_res-{res_token(resolution)}"


def atomic_manifest_write(frame: pd.DataFrame) -> None:
    temporary = MANIFEST_PATH.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(MANIFEST_PATH)


def initial_manifest() -> pd.DataFrame:
    rows = []
    for n_pcs in PCS_VALUES:
        for n_neighbors in NN_VALUES:
            graph = graph_label(n_pcs, n_neighbors)
            for resolution in RESOLUTIONS:
                rows.append(
                    {
                        "candidate_label": candidate_label(n_pcs, n_neighbors, resolution),
                        "graph_label": graph,
                        "n_pcs": n_pcs,
                        "n_neighbors": n_neighbors,
                        "resolution": float(resolution),
                        "status": "planned",
                        "completed": False,
                        "algorithm": "",
                        "raw_cluster_column": "",
                        "n_clusters": pd.NA,
                        "error_summary": "",
                        "candidate_h5ad_saved": False,
                        "aggregate_figure": str(FIGURE_ROOT / graph / f"umap_{graph}.pdf"),
                    }
                )
    manifest = pd.DataFrame(rows)
    if manifest.shape[0] != 1215 or manifest["graph_label"].nunique() != 81:
        raise AssertionError("Epithelial subtype grid manifest dimensions are incorrect")
    return manifest


def cluster_mixing(adata: AnnData, key: str, algorithm: str, resolution: float) -> pd.DataFrame:
    rows = []
    n_samples_total = max(int(adata.obs["sample"].nunique()), 2)
    for cluster in adata.obs[key].cat.categories:
        subset = adata.obs.loc[adata.obs[key] == cluster]
        sample_counts = subset["sample"].value_counts()
        sample_counts = sample_counts.loc[sample_counts > 0]
        probabilities = sample_counts.to_numpy(dtype=float) / subset.shape[0]
        entropy = float(-(probabilities * np.log(probabilities)).sum() / math.log(n_samples_total))
        status_counts = subset["status"].astype(str).value_counts()
        rows.append(
            {
                "resolution": resolution,
                "algorithm": algorithm,
                "raw_cluster_column": key,
                "cluster": str(cluster),
                "n_cells": int(subset.shape[0]),
                "n_samples": int(sample_counts.shape[0]),
                "max_sample_fraction": float(sample_counts.iloc[0] / subset.shape[0]),
                "sample_entropy_normalized": entropy,
                "tumor_fraction": float(status_counts.get("tumor", 0) / subset.shape[0]),
                "normal_like_fraction": float(status_counts.get("normal-like", 0) / subset.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def package_version_lines() -> list[str]:
    packages = [
        "rapids-singlecell-cu13",
        "cudf-cu13",
        "cuml-cu13",
        "cugraph-cu13",
        "cuvs-cu13",
        "cupy-cuda13x",
        "scanpy",
        "anndata",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
    ]
    lines = [
        "environment_category=rapids",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"random_seed={SEED}",
        f"code_files={Path(__file__).resolve()}",
    ]
    lines.extend(f"{package}={importlib.metadata.version(package)}" for package in packages)
    return lines


def main() -> None:
    if not SOURCE_H5AD.is_file():
        raise FileNotFoundError(SOURCE_H5AD)
    if TABLE_ROOT.exists() or FIGURE_ROOT.exists():
        raise FileExistsError("Fresh epithelial subtype grid requires absent table and figure roots")
    TABLE_ROOT.mkdir(parents=True, exist_ok=False)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=False)
    manifest = initial_manifest()
    atomic_manifest_write(manifest)
    pd.DataFrame(
        [
            {
                "source_h5ad": str(SOURCE_H5AD),
                "n_pcs_values": ";".join(map(str, PCS_VALUES)),
                "n_neighbors_values": ";".join(map(str, NN_VALUES)),
                "resolution_values": ";".join(format(value, "0.1f") for value in RESOLUTIONS),
                "n_graphs": 81,
                "n_resolution_candidates": 1215,
                "all_resolutions_required": True,
                "combined_review_figure_per_graph": True,
                "candidate_h5ad_saved": False,
                "abnormal_leiden_threshold_clusters": 100,
                "abnormal_leiden_action": "switch that resolution candidate to Louvain",
                "random_seed": SEED,
                "code_file": str(Path(__file__).resolve()),
            }
        ]
    ).to_csv(TABLE_ROOT / "grid_parameters.csv", index=False)

    source_backed = sc.read_h5ad(SOURCE_H5AD, backed="r")
    if INTEGRATED_BASIS not in source_backed.obsm:
        raise KeyError("Saved epithelial Harmony object lacks X_pca_inte")
    if source_backed.obsm[INTEGRATED_BASIS].shape[1] < max(PCS_VALUES):
        raise ValueError("Epithelial Harmony basis has fewer than 50 dimensions")
    if "neighbors" in source_backed.uns or "X_umap" in source_backed.obsm:
        raise ValueError("Epithelial Harmony source is not clean")
    forbidden = [
        column
        for column in source_backed.obs
        if column.startswith(("leiden_res", "louvain_res"))
    ]
    if forbidden:
        raise ValueError(f"Epithelial Harmony source already has cluster columns: {forbidden}")
    required_obs = ["sample", "series", "status"]
    missing_obs = [column for column in required_obs if column not in source_backed.obs]
    if missing_obs:
        raise KeyError(f"Epithelial Harmony source lacks obs columns: {missing_obs}")
    graph_obs = source_backed.obs[required_obs].copy()
    integrated_cpu = np.asarray(source_backed.obsm[INTEGRATED_BASIS])
    n_obs = int(source_backed.n_obs)
    source_backed.file.close()
    del source_backed
    source = AnnData(obs=graph_obs)
    source.obsm[INTEGRATED_BASIS] = cp.asarray(integrated_cpu)
    del graph_obs, integrated_cpu
    sc.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)

    review_rows = []
    backend_rows = []
    started = time.time()
    for n_pcs in PCS_VALUES:
        for n_neighbors in NN_VALUES:
            graph = graph_label(n_pcs, n_neighbors)
            table_dir = TABLE_ROOT / graph
            figure_dir = FIGURE_ROOT / graph
            table_dir.mkdir(parents=True, exist_ok=False)
            figure_dir.mkdir(parents=True, exist_ok=False)
            row_mask = manifest["graph_label"].eq(graph)
            manifest.loc[row_mask, "status"] = "running"
            atomic_manifest_write(manifest)
            graph_started = time.time()
            free_before, total_vram = cp.cuda.runtime.memGetInfo()
            adata_run = source.copy()
            try:
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
                plot_keys = []
                count_frames = []
                mixing_frames = []
                parameter_rows = []
                for resolution in RESOLUTIONS:
                    resolution_float = float(resolution)
                    token = res_token(resolution)
                    label = candidate_label(n_pcs, n_neighbors, resolution)
                    raw_key = f"leiden_res{token}"
                    algorithm = "leiden"
                    candidate_started = time.time()
                    rsc.tl.leiden(
                        adata_run,
                        resolution=resolution_float,
                        key_added=raw_key,
                        random_state=SEED,
                    )
                    n_clusters = int(adata_run.obs[raw_key].nunique())
                    if n_clusters > 100:
                        algorithm = "louvain"
                        abnormal_key = raw_key
                        raw_key = f"louvain_res{token}"
                        rsc.tl.louvain(
                            adata_run,
                            resolution=resolution_float,
                            key_added=raw_key,
                            random_state=SEED,
                        )
                        del adata_run.obs[abnormal_key]
                        n_clusters = int(adata_run.obs[raw_key].nunique())
                    counts = (
                        adata_run.obs[raw_key]
                        .astype(str)
                        .value_counts()
                        .sort_index()
                        .rename("n_cells")
                        .reset_index()
                    )
                    counts.columns = ["cluster", "n_cells"]
                    counts.insert(0, "raw_cluster_column", raw_key)
                    counts.insert(0, "algorithm", algorithm)
                    counts.insert(0, "resolution", resolution_float)
                    count_frames.append(counts)
                    mixing = cluster_mixing(adata_run, raw_key, algorithm, resolution_float)
                    mixing_frames.append(mixing)
                    elapsed_candidate = time.time() - candidate_started
                    review_rows.append(
                        {
                            "candidate_label": label,
                            "graph_label": graph,
                            "n_pcs": n_pcs,
                            "n_neighbors": n_neighbors,
                            "resolution": resolution_float,
                            "algorithm": algorithm,
                            "raw_cluster_column": raw_key,
                            "n_clusters": n_clusters,
                            "minimum_cluster_n_cells": int(counts["n_cells"].min()),
                            "clusters_below_100_cells": int((counts["n_cells"] < 100).sum()),
                            "clusters_below_500_cells": int((counts["n_cells"] < 500).sum()),
                            "worst_cluster_max_sample_fraction": float(mixing["max_sample_fraction"].max()),
                            "median_sample_entropy_normalized": float(mixing["sample_entropy_normalized"].median()),
                            "aggregate_figure": str(figure_dir / f"umap_{graph}.pdf"),
                        }
                    )
                    parameter_rows.append(
                        {
                            "candidate_label": label,
                            "source_h5ad": str(SOURCE_H5AD),
                            "n_pcs": n_pcs,
                            "n_neighbors": n_neighbors,
                            "resolution": resolution_float,
                            "neighbor_use_rep": INTEGRATED_BASIS,
                            "neighbor_algorithm": "brute",
                            "umap_random_state": SEED,
                            "clustering_algorithm": algorithm,
                            "raw_cluster_column": raw_key,
                            "n_clusters": n_clusters,
                            "elapsed_seconds_resolution": elapsed_candidate,
                            "candidate_h5ad_saved": False,
                        }
                    )
                    candidate_mask = manifest["candidate_label"].eq(label)
                    manifest.loc[candidate_mask, "algorithm"] = algorithm
                    manifest.loc[candidate_mask, "raw_cluster_column"] = raw_key
                    manifest.loc[candidate_mask, "n_clusters"] = n_clusters
                    plot_keys.append(raw_key)

                pd.concat(count_frames, ignore_index=True).to_csv(
                    table_dir / "cluster_counts_by_resolution.csv", index=False
                )
                pd.concat(mixing_frames, ignore_index=True).to_csv(
                    table_dir / "cluster_batch_mixing_by_resolution.csv", index=False
                )
                pd.DataFrame(parameter_rows).to_csv(
                    table_dir / "candidate_parameters_by_resolution.csv", index=False
                )
                sc.settings.figdir = figure_dir
                sc.pl.umap(
                    adata_run,
                    color=plot_keys,
                    ncols=5,
                    wspace=0.35,
                    show=False,
                    save=f"_{graph}.pdf",
                )
                plt.close("all")
                figure_path = figure_dir / f"umap_{graph}.pdf"
                if not figure_path.is_file() or figure_path.stat().st_size == 0:
                    raise FileNotFoundError(f"Combined graph review PDF missing: {figure_path}")
                manifest.loc[row_mask, "status"] = "completed"
                manifest.loc[row_mask, "completed"] = True
                manifest.loc[row_mask, "error_summary"] = ""
                atomic_manifest_write(manifest)
                backend_rows.append(
                    {
                        "graph_label": graph,
                        "step": "neighbors_umap_clustering_grid",
                        "planned_backend": "rapids-singlecell_GPU",
                        "attempted_backend": "rapids-singlecell_GPU",
                        "status": "completed",
                        "error_summary": "",
                        "fallback_backend": "",
                        "clean_input_reloaded": True,
                        "final_backend_for_rerun": "rapids-singlecell_GPU",
                        "gpu_free_bytes_before_graph": int(free_before),
                        "gpu_total_bytes": int(total_vram),
                        "elapsed_seconds_graph": time.time() - graph_started,
                    }
                )
            except Exception as exc:
                manifest.loc[row_mask, "status"] = "failed"
                manifest.loc[row_mask, "completed"] = False
                manifest.loc[row_mask, "error_summary"] = str(exc)
                atomic_manifest_write(manifest)
                raise
            finally:
                del adata_run
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

    del source
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    review = pd.DataFrame(review_rows)
    review.to_csv(TABLE_ROOT / "candidate_review_metrics.csv", index=False)
    pd.DataFrame(backend_rows).to_csv(TABLE_ROOT / "gpu_backend_capability_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "attempt": 1,
                "status": "interrupted_not_formal",
                "error_summary": "categorical sample value_counts retained zero-count categories and produced NaN entropy metrics",
                "biological_method_changed": False,
            },
            {
                "attempt": 2,
                "status": "completed",
                "error_summary": "",
                "biological_method_changed": False,
            },
        ]
    ).to_csv(TABLE_ROOT / "execution_attempts.csv", index=False)
    (TABLE_ROOT / "package_versions.txt").write_text(
        "\n".join(package_version_lines()) + "\n", encoding="utf-8"
    )
    completed = int(manifest["completed"].sum())
    failures = int(manifest["status"].eq("failed").sum())
    pdf_count = len(list(FIGURE_ROOT.glob("*/umap_*.pdf")))
    validation = {
        "source_h5ad": str(SOURCE_H5AD),
        "n_obs": n_obs,
        "n_graphs_expected": 81,
        "n_graphs_with_pdf": pdf_count,
        "n_candidates_expected": 1215,
        "n_candidates_completed": completed,
        "n_candidates_failed": failures,
        "all_resolutions_executed": completed == 1215,
        "candidate_h5ad_saved": False,
        "random_seed": SEED,
        "total_elapsed_seconds": time.time() - started,
        "manual_or_agent_selection_ready": completed == 1215 and failures == 0 and pdf_count == 81,
    }
    (TABLE_ROOT / "grid_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

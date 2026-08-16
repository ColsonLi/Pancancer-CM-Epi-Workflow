#!/usr/bin/env python3
"""Shared implementation for one-lineage-at-a-time Module 02 preprocessing."""

from __future__ import annotations

import importlib.metadata
import json
import random
import sys
import time
import warnings
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cpsparse
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
from anndata import AnnData


SEED = 42
N_TOP_GENES = 3_000
REGRESSION_KEYS = ["total_counts", "pct_counts_MT"]
REGRESSION_BATCHSIZE = 1_000
SCALE_MAX_VALUE = 10.0
PCA_N_COMPS = 50
HARMONY_MAX_ITER = 50
HARMONY_MAX_CLUSTERING_ITER = 200
BATCH_KEY = "sample"
PCA_BASIS = "X_pca"
INTEGRATED_BASIS = "X_pca_inte"

CLUSTER_N_PCS = 20
CLUSTER_N_NEIGHBORS = 20
CLUSTER_INITIAL_RESOLUTION = 0.4
CLUSTER_ADJUSTED_RESOLUTION = 0.4
CLUSTER_MAX_INITIAL_CLUSTERS = 20
CLUSTER_FIXED_BY_USER = True

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
ANNOTATION_H5AD = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/07-score-rank-qc/adata_anno_score_genes_rank_consistent.h5ad"
)
EXPRESSION_H5AD = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_normalized_log_all_genes.h5ad"
)


def package_version_lines(code_file: Path, category: str) -> list[str]:
    packages = [
        "rapids-singlecell-cu13", "cudf-cu13", "cuml-cu13", "cugraph-cu13",
        "cuvs-cu13", "cupy-cuda13x", "scanpy", "anndata", "numpy",
        "pandas", "scipy", "harmonypy", "h5py",
    ]
    lines = [
        f"environment_category={category}",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"random_seed={SEED}",
        f"code_files={code_file.resolve()};{Path(__file__).resolve()}",
    ]
    lines.extend(f"{package}={importlib.metadata.version(package)}" for package in packages)
    return lines


def release_gpu_arrays(adata: sc.AnnData | None = None) -> None:
    if adata is not None:
        rsc.get.anndata_to_CPU(adata, convert_all=True)
        for key in list(adata.obsm):
            value = adata.obsm[key]
            if isinstance(value, cp.ndarray):
                adata.obsm[key] = cp.asnumpy(value)
        for key in list(adata.obsp):
            value = adata.obsp[key]
            if cpsparse.issparse(value):
                adata.obsp[key] = value.get()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def is_gpu_oom(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in message
        for token in ["out of memory", "outofmemory", "cuda_error_out_of_memory"]
    )


def resolution_token(resolution: float) -> str:
    return f"{resolution:g}".replace(".", "p")


def validate_clean_lineage_harmony(adata: sc.AnnData, target_label: str) -> None:
    if adata.raw is None:
        raise ValueError(f"Clean {target_label} Harmony source lacks adata.raw")
    if INTEGRATED_BASIS not in adata.obsm:
        raise ValueError(f"Clean {target_label} Harmony source lacks {INTEGRATED_BASIS}")
    if adata.obsm[INTEGRATED_BASIS].shape[1] < CLUSTER_N_PCS:
        raise ValueError(
            f"Clean {target_label} Harmony source has fewer than {CLUSTER_N_PCS} dimensions"
        )
    graph_fields = [key for key in adata.obsp if key in {"connectivities", "distances"}]
    cluster_fields = [
        key for key in adata.obs if key.startswith(("leiden_res", "louvain_res"))
    ]
    if graph_fields or "neighbors" in adata.uns or "X_umap" in adata.obsm or cluster_fields:
        raise ValueError(
            f"Clean {target_label} Harmony source contains downstream state: "
            f"graph={graph_fields}, neighbors={'neighbors' in adata.uns}, "
            f"umap={'X_umap' in adata.obsm}, clusters={cluster_fields}"
        )
    if "cell_subtype" in adata.obs or "functional_state" in adata.obs:
        raise ValueError(f"Clean {target_label} Harmony source already contains subtype labels")


def run_lineage_gpu_graph(
    adata: sc.AnnData, resolution: float, raw_cluster_column: str
) -> None:
    graph_obs_columns = [
        column for column in ["sample", "series", "status"] if column in adata.obs
    ]
    graph_adata = AnnData(obs=adata.obs[graph_obs_columns].copy())
    try:
        graph_adata.obsm[INTEGRATED_BASIS] = cp.asarray(
            np.asarray(adata.obsm[INTEGRATED_BASIS])
        )
        rsc.pp.neighbors(
            graph_adata,
            n_neighbors=CLUSTER_N_NEIGHBORS,
            n_pcs=CLUSTER_N_PCS,
            use_rep=INTEGRATED_BASIS,
            random_state=SEED,
            algorithm="brute",
        )
        rsc.tl.umap(graph_adata, random_state=SEED)
        rsc.tl.leiden(
            graph_adata,
            resolution=resolution,
            key_added=raw_cluster_column,
            random_state=SEED,
        )
        release_gpu_arrays(graph_adata)
        adata.obs[raw_cluster_column] = graph_adata.obs[raw_cluster_column].copy()
        adata.obsm["X_umap"] = np.asarray(graph_adata.obsm["X_umap"])
        adata.obsp["distances"] = graph_adata.obsp["distances"].copy()
        adata.obsp["connectivities"] = graph_adata.obsp["connectivities"].copy()
        adata.uns["neighbors"] = graph_adata.uns["neighbors"].copy()
    except Exception:
        release_gpu_arrays(graph_adata)
        raise
    finally:
        del graph_adata
        release_gpu_arrays()


def run_lineage_cpu_graph(
    source_h5ad: Path, target_label: str, resolution: float, raw_cluster_column: str
) -> sc.AnnData:
    adata = sc.read_h5ad(source_h5ad)
    validate_clean_lineage_harmony(adata, target_label)
    sc.pp.neighbors(
        adata,
        n_neighbors=CLUSTER_N_NEIGHBORS,
        n_pcs=CLUSTER_N_PCS,
        use_rep=INTEGRATED_BASIS,
        random_state=SEED,
        method="umap",
    )
    sc.tl.umap(adata, random_state=SEED)
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=raw_cluster_column,
        random_state=SEED,
    )
    return adata


def execute_clean_lineage_clustering(
    source_h5ad: Path, target_label: str, resolution: float
) -> tuple[sc.AnnData, dict[str, object]]:
    raw_cluster_column = f"leiden_res{resolution_token(resolution)}"
    adata = sc.read_h5ad(source_h5ad)
    validate_clean_lineage_harmony(adata, target_label)
    started = time.time()
    backend = "rapids-singlecell_GPU"
    error_summary = ""
    fallback_backend = ""
    clean_input_reloaded = False
    free_before, total_vram = cp.cuda.runtime.memGetInfo()
    try:
        run_lineage_gpu_graph(adata, resolution, raw_cluster_column)
    except Exception as exc:
        release_gpu_arrays(adata)
        error_summary = f"GPU:{type(exc).__name__}:{exc}"
        if not is_gpu_oom(exc):
            raise
        del adata
        release_gpu_arrays()
        backend = "scanpy_CPU_after_GPU_OOM"
        fallback_backend = "scanpy_CPU"
        clean_input_reloaded = True
        adata = run_lineage_cpu_graph(
            source_h5ad, target_label, resolution, raw_cluster_column
        )
    free_after, _ = cp.cuda.runtime.memGetInfo()
    return adata, {
        "resolution": resolution,
        "raw_cluster_column": raw_cluster_column,
        "backend": backend,
        "error_summary": error_summary,
        "fallback_backend": fallback_backend,
        "clean_input_reloaded": clean_input_reloaded,
        "elapsed_seconds": time.time() - started,
        "gpu_total_mib": round(total_vram / 1024**2, 2),
        "gpu_free_before_mib": round(free_before / 1024**2, 2),
        "gpu_free_after_mib": round(free_after / 1024**2, 2),
    }


def run_direct_clustering(
    target_label: str,
    slug: str,
    cell_abbrev: str,
    code_file: Path,
    fixed_resolution: float = CLUSTER_INITIAL_RESOLUTION,
) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    cp.random.seed(SEED)
    source_h5ad = (
        PROJECT_ROOT
        / f"h5ad/02-cell_subtype_integration_clustering/02-subtype-harmony/{slug}/adata_{slug}_harmony.h5ad"
    )
    task = "03-subtype-clustering"
    h5ad_dir = PROJECT_ROOT / f"h5ad/02-cell_subtype_integration_clustering/{task}/{slug}"
    figure_dir = PROJECT_ROOT / f"figures/02-cell_subtype_integration_clustering/{task}/{slug}"
    table_dir = PROJECT_ROOT / f"tables/02-cell_subtype_integration_clustering/{task}/{slug}"
    pending_h5ad_dir = h5ad_dir.with_name(f"{slug}_pending_validation")
    pending_figure_dir = figure_dir.with_name(f"{slug}_pending_validation")
    pending_table_dir = table_dir.with_name(f"{slug}_pending_validation")
    output_name = f"adata_{cell_abbrev}_selected_clustered.h5ad"
    output_h5ad = h5ad_dir / output_name
    pending_h5ad = pending_h5ad_dir / output_name

    if not source_h5ad.is_file():
        raise FileNotFoundError(source_h5ad)
    for path in [
        h5ad_dir,
        figure_dir,
        table_dir,
        pending_h5ad_dir,
        pending_figure_dir,
        pending_table_dir,
    ]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {target_label} clustering artifact: {path}")

    started = time.time()
    adata, initial_run = execute_clean_lineage_clustering(
        source_h5ad, target_label, fixed_resolution
    )
    initial_key = str(initial_run["raw_cluster_column"])
    initial_counts = adata.obs[initial_key].astype(str).value_counts().sort_index()
    if initial_counts.empty:
        raise ValueError(f"Initial {target_label} Leiden result is empty")
    initial_n_clusters = int(initial_counts.size)

    final_resolution = fixed_resolution
    final_run = initial_run
    raw_cluster_column = initial_key
    fixed_resolution_source = f"user_requested_fixed_{resolution_token(fixed_resolution)}"
    adjustment_reason = f"user_requested_fixed_resolution_{resolution_token(fixed_resolution)}"
    backend_runs = [initial_run]
    if initial_n_clusters > CLUSTER_MAX_INITIAL_CLUSTERS and not CLUSTER_FIXED_BY_USER:
        del adata
        release_gpu_arrays()
        adata, final_run = execute_clean_lineage_clustering(
            source_h5ad, target_label, CLUSTER_ADJUSTED_RESOLUTION
        )
        backend_runs.append(final_run)
        final_resolution = CLUSTER_ADJUSTED_RESOLUTION
        raw_cluster_column = str(final_run["raw_cluster_column"])
        adjustment_reason = (
            f"initial_resolution_{CLUSTER_INITIAL_RESOLUTION}_produced_"
            f"{initial_n_clusters}_clusters_gt_{CLUSTER_MAX_INITIAL_CLUSTERS}; "
            f"discarded_mutable_run_reloaded_clean_harmony_and_reran_once_at_"
            f"{CLUSTER_ADJUSTED_RESOLUTION}"
        )

    final_counts = adata.obs[raw_cluster_column].astype(str).value_counts().sort_index()
    if final_counts.empty:
        raise ValueError(f"Accepted {target_label} Leiden result is empty")
    if "cell_subtype" in adata.obs or "functional_state" in adata.obs:
        raise ValueError("Direct clustering must stop before biological subtype annotation")
    adata.uns["lineage_direct_clustering_contract"] = {
        "target_leiden_coarse": target_label,
        "source_h5ad": str(source_h5ad),
        "n_pcs": CLUSTER_N_PCS,
        "n_neighbors": CLUSTER_N_NEIGHBORS,
        "initial_resolution": fixed_resolution,
        "initial_n_clusters": initial_n_clusters,
        "accepted_resolution": final_resolution,
        "accepted_n_clusters": int(final_counts.size),
        "raw_cluster_column": raw_cluster_column,
        "adaptive_rule": f"disabled by user-requested fixed resolution {fixed_resolution}",
        "resolution_selection_source": fixed_resolution_source,
        "candidate_grid_run": False,
        "manual_selection_gate": False,
        "random_seed": SEED,
    }

    for parent in [pending_h5ad_dir.parent, pending_figure_dir.parent, pending_table_dir.parent]:
        parent.mkdir(parents=True, exist_ok=True)
    pending_h5ad_dir.mkdir(parents=False, exist_ok=False)
    pending_figure_dir.mkdir(parents=False, exist_ok=False)
    pending_table_dir.mkdir(parents=False, exist_ok=False)

    sc.set_figure_params(vector_friendly=True, dpi=150, figsize=(3, 3))
    sc.settings.figdir = pending_figure_dir
    sc.pl.umap(adata, color=raw_cluster_column, show=False, save="_raw_clusters.pdf")
    pending_figure = pending_figure_dir / "umap_raw_clusters.pdf"
    if (
        not pending_figure.is_file()
        or pending_figure.stat().st_size == 0
        or pending_figure.read_bytes()[:4] != b"%PDF"
    ):
        raise ValueError(f"Invalid raw-cluster PDF for {target_label}")

    adata.write_h5ad(pending_h5ad, compression="gzip", compression_opts=1)
    check = sc.read_h5ad(pending_h5ad, backed="r")
    validation_errors: list[str] = []
    if check.shape != adata.shape:
        validation_errors.append("shape mismatch")
    if check.raw is None or check.raw.n_vars != adata.raw.n_vars:
        validation_errors.append("raw expression snapshot missing or mismatched")
    if raw_cluster_column not in check.obs:
        validation_errors.append("accepted raw cluster column missing")
    if "X_umap" not in check.obsm or "neighbors" not in check.uns:
        validation_errors.append("neighbors or UMAP missing")
    if "cell_subtype" in check.obs or "functional_state" in check.obs:
        validation_errors.append("biological subtype columns present before annotation")
    if not check.obs_names.is_unique:
        validation_errors.append("obs_names are not unique")
    check.file.close()
    if validation_errors:
        raise ValueError(f"Pending {target_label} clustering validation failed: {validation_errors}")

    initial_counts.rename("n_cells").rename_axis("cluster").reset_index().to_csv(
        pending_table_dir / "initial_cluster_counts.csv", index=False
    )
    final_counts.rename("n_cells").rename_axis("cluster").reset_index().to_csv(
        pending_table_dir / "selected_cluster_counts.csv", index=False
    )
    parameters = {
        "target_leiden_coarse": target_label,
        "lineage_slug": slug,
        "cell_abbrev_for_filename": cell_abbrev,
        "source_h5ad": str(source_h5ad),
        "selected_h5ad": str(output_h5ad),
        "n_obs": int(adata.n_obs),
        "n_vars_hvg": int(adata.n_vars),
        "raw_n_vars": int(adata.raw.n_vars),
        "n_pcs": CLUSTER_N_PCS,
        "n_neighbors": CLUSTER_N_NEIGHBORS,
        "initial_resolution": fixed_resolution,
        "initial_n_clusters": initial_n_clusters,
        "final_resolution": final_resolution,
        "final_n_clusters": int(final_counts.size),
        "raw_cluster_column": raw_cluster_column,
        "resolution_adjustment_reason": adjustment_reason,
        "final_cluster_count_exceeds_20_after_one_allowed_rerun": bool(
            final_counts.size > CLUSTER_MAX_INITIAL_CLUSTERS
        ),
        "clustering_algorithm": "leiden",
        "integration_source": "Harmony",
        "neighbor_use_rep": INTEGRATED_BASIS,
        "neighbor_algorithm": "brute" if not final_run["fallback_backend"] else "scanpy_umap",
        "backend_package": final_run["backend"],
        "candidate_grid_run": False,
        "manual_candidate_selection_run": False,
        "resolution_selection_source": fixed_resolution_source,
        "random_seed": SEED,
        "graph_elapsed_seconds_all_attempts": sum(
            float(run["elapsed_seconds"]) for run in backend_runs
        ),
        "total_elapsed_seconds": time.time() - started,
        "code_file": str(code_file.resolve()),
    }
    pd.DataFrame([parameters]).to_csv(
        pending_table_dir / "direct_clustering_parameters.csv", index=False
    )
    backend_rows = []
    for index, run in enumerate(backend_runs, start=1):
        backend_rows.append(
            {
                "attempt": index,
                "step": f"neighbors_umap_leiden_res{resolution_token(float(run['resolution']))}",
                "planned_backend": "rapids-singlecell_GPU",
                "attempted_backend": (
                    "rapids-singlecell_GPU_then_scanpy_CPU"
                    if run["fallback_backend"] else "rapids-singlecell_GPU"
                ),
                "status": (
                    "completed_CPU_after_GPU_OOM" if run["fallback_backend"] else "completed"
                ),
                "error_summary": run["error_summary"],
                "fallback_backend": run["fallback_backend"],
                "clean_input_reloaded": bool(run["clean_input_reloaded"]) or index > 1,
                "final_backend": run["backend"],
                "elapsed_seconds": run["elapsed_seconds"],
                "gpu_total_mib": run["gpu_total_mib"],
                "gpu_free_before_mib": run["gpu_free_before_mib"],
                "gpu_free_after_mib": run["gpu_free_after_mib"],
            }
        )
    pd.DataFrame(backend_rows).to_csv(
        pending_table_dir / "gpu_backend_capability_summary.csv", index=False
    )
    (pending_table_dir / "package_versions.txt").write_text(
        "\n".join(package_version_lines(code_file, "rapids_direct_lineage_clustering"))
        + "\n",
        encoding="utf-8",
    )
    (pending_table_dir / "readme.txt").write_text(
        f"Direct subtype clustering for {target_label}; no parameter grid.\n\n"
        f"Input clean Harmony object: {source_h5ad}\n"
        f"User-requested fixed run: PCs={CLUSTER_N_PCS}, neighbors={CLUSTER_N_NEIGHBORS}, "
        f"Leiden resolution={fixed_resolution}, clusters={initial_n_clusters}.\n"
        f"Accepted run: resolution={final_resolution}, clusters={final_counts.size}, "
        f"column={raw_cluster_column}.\n"
        f"Resolution source: user-requested fixed {fixed_resolution}; adaptive rerun disabled.\n"
        f"Output: {output_h5ad}\n"
        "Only the raw numeric cluster label is created here. DEG review and biological "
        "subtype annotation belong to task 05.\n",
        encoding="utf-8",
    )
    validation = {
        **parameters,
        "output_h5ad_readable": True,
        "obs_names_unique": bool(adata.obs_names.is_unique),
        "raw_cluster_column_present": raw_cluster_column in adata.obs,
        "neighbor_graph_present": "neighbors" in adata.uns,
        "umap_present": "X_umap" in adata.obsm,
        "raw_expression_present": adata.raw is not None,
        "biological_subtype_columns_absent": (
            "cell_subtype" not in adata.obs and "functional_state" not in adata.obs
        ),
        "candidate_grid_run": False,
        "raw_cluster_pdf_valid": True,
        "validation_errors": validation_errors,
    }
    (pending_table_dir / "direct_clustering_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    pending_h5ad_dir.replace(h5ad_dir)
    pending_figure_dir.replace(figure_dir)
    pending_table_dir.replace(table_dir)
    validation["output_h5ad_readable"] = output_h5ad.is_file()
    print(json.dumps(validation, indent=2, ensure_ascii=False))


def run_raw_cluster_deg(
    target_label: str, slug: str, cell_abbrev: str, code_file: Path
) -> None:
    clustered_h5ad = (
        PROJECT_ROOT
        / f"h5ad/02-cell_subtype_integration_clustering/03-subtype-clustering/{slug}/"
        / f"adata_{cell_abbrev}_selected_clustered.h5ad"
    )
    table_dir = (
        PROJECT_ROOT
        / f"tables/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/{slug}"
    )
    parameter_file = table_dir / "raw_cluster_deg_parameters.csv"
    validation_file = table_dir / "raw_cluster_deg_validation.json"
    version_file = table_dir / "raw_cluster_deg_package_versions.txt"
    readme_file = table_dir / "readme.txt"
    if not clustered_h5ad.is_file():
        raise FileNotFoundError(clustered_h5ad)
    for path in [parameter_file, validation_file, version_file]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {target_label} DEG artifact: {path}")

    started = time.time()
    adata = sc.read_h5ad(clustered_h5ad)
    contract = adata.uns.get("lineage_direct_clustering_contract", {})
    raw_cluster_column = str(contract.get("raw_cluster_column", ""))
    n_pcs = int(contract.get("n_pcs", CLUSTER_N_PCS))
    n_neighbors = int(contract.get("n_neighbors", CLUSTER_N_NEIGHBORS))
    resolution = float(contract.get("accepted_resolution", np.nan))
    if not raw_cluster_column or raw_cluster_column not in adata.obs:
        raise ValueError(f"Accepted raw cluster column is missing for {target_label}")
    if not np.isfinite(resolution):
        raise ValueError(f"Accepted resolution is missing for {target_label}")
    if adata.raw is None:
        raise ValueError(f"Accepted clustered {target_label} object lacks adata.raw")
    if "cell_subtype" in adata.obs or "functional_state" in adata.obs:
        raise ValueError("Raw-cluster DEG export must precede biological subtype annotation")
    groups = list(adata.obs[raw_cluster_column].cat.categories)
    if not groups:
        raise ValueError(f"No raw clusters found for {target_label}")

    parameter_token = (
        f"{raw_cluster_column}_pcs{n_pcs}_nn{n_neighbors}_res{resolution_token(resolution)}"
    )
    deg_dir = table_dir / f"degs_{parameter_token}"
    pending_deg_dir = table_dir / f"degs_{parameter_token}_pending_validation"
    for path in [deg_dir, pending_deg_dir]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {target_label} DEG directory: {path}")
    table_dir.mkdir(parents=True, exist_ok=True)
    pending_deg_dir.mkdir(parents=False, exist_ok=False)

    sc.tl.rank_genes_groups(
        adata,
        groupby=raw_cluster_column,
        method="t-test",
        use_raw=True,
        n_genes=adata.raw.n_vars,
    )
    result = adata.uns["rank_genes_groups"]
    canonical_columns = ["gene", "score", "logfoldchanges", "pvals", "pvals_adj"]
    file_rows = []
    for group in groups:
        group_text = str(group)
        frame = pd.DataFrame(
            {
                "gene": result["names"][group_text],
                "score": result["scores"][group_text],
                "logfoldchanges": result["logfoldchanges"][group_text],
                "pvals": result["pvals"][group_text],
                "pvals_adj": result["pvals_adj"][group_text],
            }
        )
        output_csv = pending_deg_dir / f"{group_text}_degs_{parameter_token}.csv"
        frame.to_csv(output_csv, index=False)
        file_rows.append(
            {
                "cluster": group_text,
                "n_cells": int(adata.obs[raw_cluster_column].astype(str).eq(group_text).sum()),
                "n_deg_rows": int(frame.shape[0]),
                "output_csv": str(deg_dir / output_csv.name),
                "canonical_columns_exact": list(frame.columns) == canonical_columns,
                "genes_unique": bool(frame["gene"].is_unique),
                "genes_nonmissing": bool(frame["gene"].notna().all()),
            }
        )
    file_report = pd.DataFrame(file_rows)
    expected_rows = int(adata.raw.n_vars)
    validation_errors = []
    if file_report.shape[0] != len(groups):
        validation_errors.append("group/file count mismatch")
    if not file_report["n_deg_rows"].eq(expected_rows).all():
        validation_errors.append("one or more DEG CSVs are not full length")
    if not file_report["canonical_columns_exact"].all():
        validation_errors.append("one or more DEG CSVs lack canonical columns")
    if not file_report["genes_unique"].all() or not file_report["genes_nonmissing"].all():
        validation_errors.append("one or more DEG CSVs have invalid gene identifiers")
    if validation_errors:
        raise ValueError(f"Pending {target_label} raw-cluster DEG validation failed: {validation_errors}")
    pending_deg_dir.replace(deg_dir)

    file_report.to_csv(table_dir / "raw_cluster_deg_file_manifest.csv", index=False)
    parameters = {
        "step": "raw_cluster_full_deg_export",
        "target_leiden_coarse": target_label,
        "input_h5ad": str(clustered_h5ad),
        "groupby": raw_cluster_column,
        "method": "t-test",
        "use_raw": True,
        "expression_state": "normalize_total_1e4_log1p_all_shared_genes",
        "n_obs": int(adata.n_obs),
        "n_vars_hvg_X": int(adata.n_vars),
        "n_vars_raw": expected_rows,
        "n_groups": len(groups),
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "backend": "scanpy_CPU",
        "random_seed": SEED,
        "output_directory": str(deg_dir),
        "elapsed_seconds": time.time() - started,
        "code_file": str(code_file.resolve()),
    }
    pd.DataFrame([parameters]).to_csv(parameter_file, index=False)
    version_file.write_text(
        "\n".join(package_version_lines(code_file, "scanpy_CPU_raw_cluster_DEG")) + "\n",
        encoding="utf-8",
    )
    readme_mode = "a" if readme_file.exists() else "w"
    with readme_file.open(readme_mode, encoding="utf-8") as handle:
        if readme_mode == "a":
            handle.write("\n")
        handle.write(
            f"Raw-cluster full DEG export for {target_label}.\n\n"
            f"Input: {clustered_h5ad}\n"
            f"Groupby: {raw_cluster_column}\n"
            "Method: Scanpy t-test with use_raw=True.\n"
            f"Output: {deg_dir}\n"
            f"Each of {len(groups)} cluster CSVs contains all {expected_rows} raw genes.\n"
            "No top-only or combined DEG CSV was created.\n"
        )
    validation = {
        **parameters,
        "expected_group_count": len(groups),
        "observed_csv_count": int(len(list(deg_dir.glob("*.csv")))),
        "all_csvs_full_length": bool(file_report["n_deg_rows"].eq(expected_rows).all()),
        "all_csvs_canonical_columns": bool(file_report["canonical_columns_exact"].all()),
        "all_csv_gene_lists_unique": bool(file_report["genes_unique"].all()),
        "top_only_deg_files_created": False,
        "combined_deg_csv_created": False,
        "biological_subtype_columns_absent": (
            "cell_subtype" not in adata.obs and "functional_state" not in adata.obs
        ),
        "validation_errors": validation_errors,
    }
    validation_file.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


def run_extraction(target_label: str, slug: str, code_file: Path) -> None:
    h5ad_dir = (
        PROJECT_ROOT
        / f"h5ad/02-cell_subtype_integration_clustering/01-subtype-extraction/{slug}"
    )
    pending_h5ad_dir = h5ad_dir.with_name(f"{slug}_pending_validation")
    output_h5ad = h5ad_dir / f"adata_{slug}_normalized_log_all_genes.h5ad"
    pending_h5ad = pending_h5ad_dir / output_h5ad.name
    table_dir = (
        PROJECT_ROOT
        / f"tables/02-cell_subtype_integration_clustering/01-subtype-extraction/{slug}"
    )
    if not ANNOTATION_H5AD.is_file() or not EXPRESSION_H5AD.is_file():
        raise FileNotFoundError("Consistent annotation or combined normalized/log source is missing")
    for path in [h5ad_dir, pending_h5ad_dir, table_dir]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {target_label} extraction artifact: {path}")

    started = time.time()
    annotation = sc.read_h5ad(ANNOTATION_H5AD, backed="r")
    required_annotation = [
        "leiden_coarse", "cell_type", "best_rank_type_global",
        "broad_score_rank_consistent",
    ]
    missing = [column for column in required_annotation if column not in annotation.obs]
    if missing:
        raise KeyError(f"Annotation source lacks columns: {missing}")
    eligible_mask = (
        annotation.obs["leiden_coarse"].astype(str).eq(target_label)
        & annotation.obs["best_rank_type_global"].astype(str).eq(target_label)
        & annotation.obs["broad_score_rank_consistent"].astype(bool)
    )
    eligible_ids = annotation.obs_names[eligible_mask].copy()
    transfer_columns = [
        column
        for column in annotation.obs.columns
        if column in required_annotation
        or column.endswith("_score")
        or column.endswith("_score_rank_pct")
        or column in [
            "highest_score_type", "broad_annotation_source",
            "broad_annotation_confidence",
        ]
    ]
    annotation_transfer = annotation.obs.loc[eligible_ids, transfer_columns].copy()
    annotation_n_obs = int(annotation.n_obs)
    annotation.file.close()
    del annotation
    if len(eligible_ids) == 0 or not eligible_ids.is_unique:
        raise ValueError(f"No unique eligible IDs were selected for {target_label}")

    source = sc.read_h5ad(EXPRESSION_H5AD)
    if source.raw is not None or len(source.layers) != 0:
        raise ValueError("Combined normalized/log source has unexpected raw/layers state")
    if not source.obs_names.is_unique or not source.var_names.is_unique:
        raise ValueError("Combined normalized/log source IDs are not unique")
    if set(source.obs["series"].astype(str).unique()) != {"UCSC", "GSE229413", "GSE278688"}:
        raise ValueError("Combined normalized/log source does not contain all three expected series")
    missing_ids = eligible_ids.difference(source.obs_names)
    if len(missing_ids):
        raise KeyError(f"Eligible IDs absent from expression source: {len(missing_ids)}")
    lineage = source[eligible_ids].copy()
    del source
    if not lineage.obs_names.equals(annotation_transfer.index):
        raise ValueError("Expression subset order differs from annotation transfer order")
    for column in transfer_columns:
        lineage.obs[column] = annotation_transfer[column].to_numpy()
    if not lineage.obs["leiden_coarse"].astype(str).eq(target_label).all():
        raise ValueError(f"Extracted object contains cells outside {target_label}")
    if "neighbors" in lineage.uns or "X_umap" in lineage.obsm:
        raise ValueError("Clean lineage extraction unexpectedly contains a graph or UMAP")
    lineage.raw = lineage.copy()
    lineage.uns["lineage_extraction_contract"] = {
        "annotation_source": str(ANNOTATION_H5AD),
        "expression_source": str(EXPRESSION_H5AD),
        "target_leiden_coarse": target_label,
        "eligibility_rule": "leiden_coarse and best_rank_type_global match; rank-consistent true",
        "expression_state": "three-series post-QC normalize_total_1e4 log1p shared-gene atlas",
        "project_specific_base_reason": (
            "literal adata_qc contains GSE counts only; UCSC is source-QC normalized/log and joins "
            "after GSE QC plus normalization"
        ),
        "UCSC_renormalized": False,
        "raw_assignment": "all-gene normalized/log subset before lineage-specific HVG selection",
        "counts_layer_present": False,
        "random_seed": SEED,
    }

    pending_h5ad_dir.mkdir(parents=True, exist_ok=False)
    lineage.write_h5ad(pending_h5ad, compression="gzip", compression_opts=1)
    pending = sc.read_h5ad(pending_h5ad, backed="r")
    errors = []
    if pending.shape != lineage.shape:
        errors.append("shape mismatch")
    if pending.raw is None or pending.raw.n_vars != lineage.n_vars:
        errors.append("all-gene raw snapshot missing or mismatched")
    if set(required_annotation).difference(pending.obs.columns):
        errors.append("transferred annotation columns missing")
    if len(pending.layers) != 0:
        errors.append("unexpected layer present")
    pending.file.close()
    if errors:
        raise ValueError(f"Pending {target_label} extraction failed validation: {errors}")
    pending_h5ad_dir.replace(h5ad_dir)

    table_dir.mkdir(parents=True, exist_ok=False)
    selected_cells = lineage.obs[
        [
            "original_barcode", "sample", "series", "status", "leiden_coarse",
            "best_rank_type_global", "broad_score_rank_consistent",
        ]
    ].copy()
    selected_cells.insert(0, "cell_id", selected_cells.index)
    selected_cells.to_csv(table_dir / f"selected_{slug}_cells.csv", index=False)
    summary = (
        lineage.obs.groupby(["series", "sample", "status"], observed=True)
        .size().rename(f"n_{slug}_cells").reset_index()
    )
    summary.to_csv(table_dir / f"{slug}_extraction_counts_by_sample.csv", index=False)
    parameters = {
        "step": f"score_rank_consistent_{slug}_extraction",
        "target_leiden_coarse": target_label,
        "annotation_source_h5ad": str(ANNOTATION_H5AD),
        "expression_source_h5ad": str(EXPRESSION_H5AD),
        "output_h5ad": str(output_h5ad),
        "annotation_source_n_obs": annotation_n_obs,
        "n_obs_lineage": int(lineage.n_obs),
        "n_vars_all_gene_normalized_log": int(lineage.n_vars),
        "n_samples": int(lineage.obs["sample"].nunique()),
        "n_series": int(lineage.obs["series"].nunique()),
        "status_values": ";".join(sorted(map(str, lineage.obs["status"].unique()))),
        "UCSC_renormalized": False,
        "counts_layer_present": False,
        "raw_n_vars": int(lineage.raw.n_vars),
        "random_seed": SEED,
        "elapsed_seconds": time.time() - started,
        "code_file": str(code_file.resolve()),
    }
    pd.DataFrame([parameters]).to_csv(table_dir / "extraction_parameters.csv", index=False)
    (table_dir / "package_versions.txt").write_text(
        "\n".join(package_version_lines(code_file, "rapids_with_scanpy_extraction")) + "\n",
        encoding="utf-8",
    )
    (table_dir / "readme.txt").write_text(
        f"Lineage extraction for {target_label}.\n\n"
        f"Eligible IDs: {ANNOTATION_H5AD}\n"
        f"Expression base: {EXPRESSION_H5AD}\n"
        "The expression base is the audited three-series post-QC/log shared-gene object. "
        "The literal adata_qc contains only GSE raw-count cohorts, while UCSC enters as "
        "source-QC normalized/log expression and is not renormalized.\n"
        f"Output: {output_h5ad}\n",
        encoding="utf-8",
    )
    validation = {
        **parameters,
        "eligible_ids_unique": True,
        "missing_eligible_ids_in_expression_source": 0,
        "all_cells_target_lineage": True,
        "all_cells_score_rank_consistent": bool(lineage.obs["broad_score_rank_consistent"].all()),
        "output_h5ad_readable": output_h5ad.is_file(),
    }
    (table_dir / "extraction_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


def run_harmony(target_label: str, slug: str, code_file: Path) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    cp.random.seed(SEED)
    input_h5ad = (
        PROJECT_ROOT
        / f"h5ad/02-cell_subtype_integration_clustering/01-subtype-extraction/{slug}/adata_{slug}_normalized_log_all_genes.h5ad"
    )
    h5ad_dir = (
        PROJECT_ROOT
        / f"h5ad/02-cell_subtype_integration_clustering/02-subtype-harmony/{slug}"
    )
    pending_h5ad_dir = h5ad_dir.with_name(f"{slug}_pending_validation")
    pre_harmony_h5ad = h5ad_dir / f"adata_{slug}_pre_harmony_pca.h5ad"
    output_h5ad = h5ad_dir / f"adata_{slug}_harmony.h5ad"
    pending_pre_harmony = pending_h5ad_dir / pre_harmony_h5ad.name
    pending_output = pending_h5ad_dir / output_h5ad.name
    table_dir = (
        PROJECT_ROOT
        / f"tables/02-cell_subtype_integration_clustering/02-subtype-harmony/{slug}"
    )
    if not input_h5ad.is_file():
        raise FileNotFoundError(input_h5ad)
    for path in [h5ad_dir, pending_h5ad_dir, table_dir]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {target_label} Harmony artifact: {path}")

    started = time.time()
    adata = sc.read_h5ad(input_h5ad)
    n_obs, n_vars_all = adata.shape
    required_obs = {"sample", "series", "status", "original_barcode", *REGRESSION_KEYS}
    missing_obs = required_obs.difference(adata.obs.columns)
    if missing_obs:
        raise KeyError(f"Extracted {target_label} object lacks obs columns: {missing_obs}")
    if adata.raw is None or adata.raw.n_vars != n_vars_all:
        raise ValueError("Extracted lineage lacks the all-gene normalized/log raw snapshot")
    if len(adata.layers) != 0:
        raise ValueError("Mixed-source normalized/log lineage must not fabricate a counts layer")
    for key in REGRESSION_KEYS:
        values = pd.to_numeric(adata.obs[key], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Regression covariate {key} contains non-finite values")
        adata.obs[key] = values

    timings: dict[str, float] = {}
    rsc.get.anndata_to_GPU(adata)
    step_started = time.time()
    rsc.pp.highly_variable_genes(
        adata, n_top_genes=N_TOP_GENES, flavor="seurat", batch_key="series"
    )
    release_gpu_arrays(adata)
    timings["highly_variable_genes"] = time.time() - step_started
    n_hvg = int(adata.var["highly_variable"].sum())
    if n_hvg != N_TOP_GENES:
        raise ValueError(f"Expected {N_TOP_GENES} {target_label} HVGs, observed {n_hvg}")
    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    if adata.raw is None or adata.raw.n_vars != n_vars_all:
        raise ValueError("All-gene raw snapshot was lost during lineage HVG subsetting")

    rsc.get.anndata_to_GPU(adata)
    step_started = time.time()
    rsc.pp.regress_out(
        adata, keys=REGRESSION_KEYS, batchsize=REGRESSION_BATCHSIZE, verbose=False
    )
    timings["regress_out"] = time.time() - step_started
    step_started = time.time()
    rsc.pp.scale(adata, zero_center=True, max_value=SCALE_MAX_VALUE)
    timings["scale"] = time.time() - step_started
    step_started = time.time()
    rsc.tl.pca(adata, n_comps=PCA_N_COMPS, random_state=SEED, dtype="float32")
    timings["pca"] = time.time() - step_started
    release_gpu_arrays(adata)
    if adata.obsm[PCA_BASIS].shape != (n_obs, PCA_N_COMPS):
        raise ValueError("Lineage PCA has an unexpected shape")
    if INTEGRATED_BASIS in adata.obsm or "neighbors" in adata.uns or "X_umap" in adata.obsm:
        raise ValueError("Clean pre-Harmony lineage contains downstream results")
    adata.uns["lineage_pre_harmony_contract"] = {
        "target_leiden_coarse": target_label,
        "input_h5ad": str(input_h5ad),
        "n_top_genes": N_TOP_GENES,
        "hvg_flavor": "seurat",
        "hvg_batch_key": "series",
        "regression_keys": REGRESSION_KEYS,
        "scale_max_value": SCALE_MAX_VALUE,
        "pca_n_comps": PCA_N_COMPS,
        "random_seed": SEED,
    }

    pending_h5ad_dir.mkdir(parents=True, exist_ok=False)
    adata.write_h5ad(pending_pre_harmony, compression="gzip", compression_opts=1)
    del adata
    adata = sc.read_h5ad(pending_pre_harmony)
    step_started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rsc.pp.harmony_integrate(
            adata,
            key=BATCH_KEY,
            basis=PCA_BASIS,
            adjusted_basis=INTEGRATED_BASIS,
            max_iter_harmony=HARMONY_MAX_ITER,
            max_iter_clustering=HARMONY_MAX_CLUSTERING_ITER,
            tol_harmony=1e-4,
            tol_clustering=1e-5,
            random_state=SEED,
            verbose=True,
        )
    timings["harmony_integrate"] = time.time() - step_started
    warning_messages = [str(item.message) for item in caught]
    release_gpu_arrays(adata)
    nonconvergence = [
        message for message in warning_messages if "did not converge" in message.lower()
    ]
    if nonconvergence:
        raise RuntimeError(
            f"{target_label} Harmony did not converge at max_iter={HARMONY_MAX_ITER}: {nonconvergence}"
        )
    if INTEGRATED_BASIS not in adata.obsm:
        raise ValueError("Harmony did not create X_pca_inte")
    if adata.obsm[INTEGRATED_BASIS].shape != adata.obsm[PCA_BASIS].shape:
        raise ValueError("PCA and Harmony embeddings differ in shape")
    forbidden = [
        column for column in adata.obs if column.startswith(("leiden_res", "louvain_res"))
    ]
    if forbidden or "neighbors" in adata.uns or "X_umap" in adata.obsm:
        raise ValueError("Lineage Harmony handoff contains clustering/UMAP results")
    adata.uns["lineage_integration_contract"] = {
        "target_leiden_coarse": target_label,
        "input_expression_state": "three-series all-gene normalized/log; UCSC not renormalized",
        "hvg_n_top_genes": N_TOP_GENES,
        "hvg_flavor": "seurat",
        "hvg_batch_key": "series",
        "raw_assignment_point": "extraction before lineage HVG selection",
        "regression_keys": REGRESSION_KEYS,
        "scale_max_value": SCALE_MAX_VALUE,
        "pca_n_comps": PCA_N_COMPS,
        "harmony_batch_key": BATCH_KEY,
        "harmony_max_iter": HARMONY_MAX_ITER,
        "harmony_max_clustering_iter": HARMONY_MAX_CLUSTERING_ITER,
        "pca_basis": PCA_BASIS,
        "integrated_basis": INTEGRATED_BASIS,
        "random_seed": SEED,
    }
    adata.write_h5ad(pending_output, compression="gzip", compression_opts=1)

    pending_errors = []
    for path, harmony_required in [(pending_pre_harmony, False), (pending_output, True)]:
        pending = sc.read_h5ad(path, backed="r")
        if pending.shape != (n_obs, N_TOP_GENES):
            pending_errors.append(f"shape mismatch: {path.name}")
        if pending.raw is None or pending.raw.n_vars != n_vars_all:
            pending_errors.append(f"raw mismatch: {path.name}")
        if PCA_BASIS not in pending.obsm:
            pending_errors.append(f"PCA missing: {path.name}")
        if harmony_required != (INTEGRATED_BASIS in pending.obsm):
            pending_errors.append(f"Harmony presence mismatch: {path.name}")
        if "neighbors" in pending.uns or "X_umap" in pending.obsm:
            pending_errors.append(f"downstream graph present: {path.name}")
        pending.file.close()
    if pending_errors:
        raise ValueError(f"Pending {target_label} Harmony validation failed: {pending_errors}")
    pending_h5ad_dir.replace(h5ad_dir)

    table_dir.mkdir(parents=True, exist_ok=False)
    common = {
        "target_leiden_coarse": target_label,
        "input_h5ad": str(input_h5ad),
        "output_h5ad": str(output_h5ad),
        "n_obs": n_obs,
        "n_vars_all_gene_raw": n_vars_all,
        "n_vars_hvg": N_TOP_GENES,
        "random_seed": SEED,
        "code_file": str(code_file.resolve()),
    }
    parameter_rows = [
        {**common, "step": "highly_variable_genes", "n_top_genes": N_TOP_GENES, "flavor": "seurat", "batch_key": "series", "elapsed_seconds": timings["highly_variable_genes"]},
        {**common, "step": "raw_preservation_and_hvg_subset", "raw_assignment_point": "extraction_before_HVG_selection", "raw_n_vars": n_vars_all, "elapsed_seconds": 0.0},
        {**common, "step": "regress_out", "regression_keys": ";".join(REGRESSION_KEYS), "batchsize": REGRESSION_BATCHSIZE, "elapsed_seconds": timings["regress_out"]},
        {**common, "step": "scale", "zero_center": True, "max_value": SCALE_MAX_VALUE, "elapsed_seconds": timings["scale"]},
        {**common, "step": "pca", "n_comps": PCA_N_COMPS, "basis": PCA_BASIS, "dtype": "float32", "elapsed_seconds": timings["pca"]},
        {**common, "step": "harmony_integrate", "key": BATCH_KEY, "basis": PCA_BASIS, "adjusted_basis": INTEGRATED_BASIS, "max_iter_harmony": HARMONY_MAX_ITER, "max_iter_clustering": HARMONY_MAX_CLUSTERING_ITER, "converged": True, "warning_messages": " | ".join(warning_messages), "elapsed_seconds": timings["harmony_integrate"]},
    ]
    pd.DataFrame(parameter_rows).to_csv(table_dir / f"{slug}_harmony_parameters.csv", index=False)
    pd.DataFrame(
        [{
            "attempt": 1,
            "status": "completed",
            "error_summary": "",
            "biological_method_changed": False,
            "harmony_converged": True,
        }]
    ).to_csv(table_dir / "execution_attempts.csv", index=False)
    pd.DataFrame(
        [{
            "step": step,
            "planned_backend": "rapids-singlecell_GPU",
            "attempted_backend": "rapids-singlecell_GPU",
            "status": "completed",
            "error_summary": "",
            "fallback_backend": "",
            "clean_input_reloaded": step == "harmony_integrate",
            "final_backend_for_rerun": "rapids-singlecell_GPU",
        } for step in ["highly_variable_genes", "regress_out", "scale", "pca", "harmony_integrate"]]
    ).to_csv(table_dir / "gpu_backend_capability_summary.csv", index=False)
    (table_dir / "package_versions.txt").write_text(
        "\n".join(package_version_lines(code_file, "rapids")) + "\n", encoding="utf-8"
    )
    (table_dir / "readme.txt").write_text(
        f"Lineage-specific HVG, regression, PCA, and Harmony for {target_label}.\n\n"
        f"Input: {input_h5ad}\n"
        f"Pre-Harmony PCA: {pre_harmony_h5ad}\n"
        f"Clean Harmony template: {output_h5ad}\n"
        "No neighbors, UMAP, Leiden, or Louvain output is created in this task.\n",
        encoding="utf-8",
    )
    validation = {
        **common,
        "pre_harmony_h5ad": str(pre_harmony_h5ad),
        "pca_shape": list(adata.obsm[PCA_BASIS].shape),
        "harmony_shape": list(adata.obsm[INTEGRATED_BASIS].shape),
        "harmony_converged": True,
        "harmony_max_iter": HARMONY_MAX_ITER,
        "harmony_max_clustering_iter": HARMONY_MAX_CLUSTERING_ITER,
        "neighbor_graph_present": "neighbors" in adata.uns,
        "umap_present": "X_umap" in adata.obsm,
        "cluster_columns_present": forbidden,
        "counts_layer_present": "counts" in adata.layers,
        "UCSC_renormalized": False,
        "obs_names_unique": bool(adata.obs_names.is_unique),
        "total_elapsed_seconds": time.time() - started,
        "output_h5ad_readable": output_h5ad.is_file(),
    }
    (table_dir / f"{slug}_harmony_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))

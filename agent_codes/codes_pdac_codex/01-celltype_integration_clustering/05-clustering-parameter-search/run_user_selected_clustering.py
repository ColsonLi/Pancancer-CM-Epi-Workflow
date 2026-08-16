#!/usr/bin/env python3
"""Rerun one explicitly user-selected completed clustering candidate from Harmony."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cpsparse
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc


SEED = 42
INTEGRATED_BASIS = "X_pca_inte"
PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
SOURCE_H5AD = PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_harmony.h5ad"
TABLE_ROOT = PROJECT_ROOT / "tables/01-celltype_integration_clustering/05-clustering-parameter-search"
FIGURE_ROOT = PROJECT_ROOT / "figures/01-celltype_integration_clustering/05-clustering-parameter-search"
H5AD_PARENT = PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/05-clustering-parameter-search"
MANIFEST_PATH = TABLE_ROOT / "clustering_grid_manifest.csv"
COMPLETION_PATH = TABLE_ROOT / "clustering_grid_completion_check.csv"

SELECTED_H5AD_DIR = H5AD_PARENT / "selected"
PENDING_H5AD_DIR = H5AD_PARENT / "selected_pending_validation"
SELECTED_H5AD = SELECTED_H5AD_DIR / "adata_inte.h5ad"
PENDING_H5AD = PENDING_H5AD_DIR / "adata_inte.h5ad"
SELECTED_FIGURE_DIR = FIGURE_ROOT / "selected"
PENDING_FIGURE_DIR = FIGURE_ROOT / "selected_pending_validation"
SELECTED_TABLE = TABLE_ROOT / "selected_clustering.csv"
PARAMETER_TABLE = TABLE_ROOT / "selected_clustering_parameters.csv"
CLUSTER_COUNT_TABLE = TABLE_ROOT / "selected_cluster_counts.csv"
COUNT_COMPARISON_TABLE = TABLE_ROOT / "selected_vs_candidate_cluster_count_comparison.csv"
VALIDATION_PATH = TABLE_ROOT / "selected_clustering_validation.json"
BACKEND_TABLE = TABLE_ROOT / "gpu_backend_capability_summary_selected.csv"
PACKAGE_VERSIONS = TABLE_ROOT / "package_versions_selected.txt"

LABEL_RE = re.compile(r"^pcs-(?P<pcs>[0-9]+)_nn-(?P<nn>[0-9]+)_res-(?P<res>[0-9]+p[0-9]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument(
        "--manual-selection-note",
        default="",
        help="Required for an actual run; record the user's explicit selection statement.",
    )
    parser.add_argument(
        "--confirm-user-selected",
        action="store_true",
        help="Required for an actual run; asserts that the user explicitly selected this exact candidate.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the candidate and output boundary without creating selected outputs.",
    )
    return parser.parse_args()


def require_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite selected-clustering artifact: {path}")


def parse_label(label: str) -> tuple[int, int, float, str]:
    match = LABEL_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"Invalid candidate label: {label}")
    n_pcs = int(match.group("pcs"))
    n_neighbors = int(match.group("nn"))
    resolution = float(match.group("res").replace("p", "."))
    cluster_key = f"leiden_res{match.group('res')}"
    return n_pcs, n_neighbors, resolution, cluster_key


def validate_candidate(label: str) -> tuple[pd.Series, pd.Series, int, int, float, str]:
    if not SOURCE_H5AD.is_file() or not MANIFEST_PATH.is_file() or not COMPLETION_PATH.is_file():
        raise FileNotFoundError("Harmony source or completed grid audit is missing")
    completion = pd.read_csv(COMPLETION_PATH)
    if completion.shape[0] != 1 or not bool(completion.iloc[0]["manual_selection_ready"]):
        raise ValueError("Clustering grid is not ready for manual selection")
    if int(completion.iloc[0]["failed_candidates"]) != 0 or int(completion.iloc[0]["missing_candidates"]) != 0:
        raise ValueError("Clustering grid has failed or missing candidates")

    n_pcs, n_neighbors, resolution, cluster_key = parse_label(label)
    manifest = pd.read_csv(MANIFEST_PATH)
    rows = manifest.loc[manifest["candidate_label"].eq(label)]
    if rows.shape[0] != 1:
        raise ValueError(f"Candidate label is absent or non-unique in manifest: {label}")
    row = rows.iloc[0]
    if row["status"] != "completed" or not bool(row["completed"]):
        raise ValueError(f"Candidate was not completed and cannot be selected: {label}")
    if (int(row["n_pcs"]), int(row["n_neighbors"])) != (n_pcs, n_neighbors):
        raise ValueError("Candidate label disagrees with manifest PC/NN values")
    if not np.isclose(float(row["resolution"]), resolution):
        raise ValueError("Candidate label disagrees with manifest resolution")

    candidate_parameters_path = TABLE_ROOT / label / "candidate_parameters.csv"
    candidate_parameters = pd.read_csv(candidate_parameters_path)
    if candidate_parameters.shape[0] != 1:
        raise ValueError("Candidate parameter table must contain exactly one row")
    parameters = candidate_parameters.iloc[0]
    expected = (int(parameters["n_pcs"]), int(parameters["n_neighbors"]), float(parameters["resolution"]))
    if expected[:2] != (n_pcs, n_neighbors) or not np.isclose(expected[2], resolution):
        raise ValueError("Candidate parameter table disagrees with label")
    if parameters["raw_cluster_column"] != cluster_key:
        raise ValueError("Candidate raw cluster column disagrees with label")
    if Path(parameters["source_h5ad"]).resolve() != SOURCE_H5AD.resolve():
        raise ValueError("Candidate was not generated from the canonical Harmony source")
    return row, parameters, n_pcs, n_neighbors, resolution, cluster_key


def release_gpu_arrays(adata: sc.AnnData) -> None:
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
        "install_manager=uv",
        "index_url=https://pypi.tuna.tsinghua.edu.cn/simple",
        f"random_seed={SEED}",
        f"code_file={Path(__file__).resolve()}",
    ]
    lines.extend(f"{package}={importlib.metadata.version(package)}" for package in packages)
    return lines


def main() -> None:
    args = parse_args()
    row, candidate_parameters, n_pcs, n_neighbors, resolution, cluster_key = validate_candidate(
        args.candidate_label
    )
    outputs = [
        SELECTED_H5AD_DIR,
        PENDING_H5AD_DIR,
        SELECTED_FIGURE_DIR,
        PENDING_FIGURE_DIR,
        SELECTED_TABLE,
        PARAMETER_TABLE,
        CLUSTER_COUNT_TABLE,
        COUNT_COMPARISON_TABLE,
        VALIDATION_PATH,
        BACKEND_TABLE,
        PACKAGE_VERSIONS,
    ]
    for path in outputs:
        require_absent(path)

    preflight = {
        "candidate_label": args.candidate_label,
        "manifest_status": row["status"],
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "raw_cluster_column": cluster_key,
        "expected_n_clusters": int(candidate_parameters["n_clusters"]),
        "source_h5ad": str(SOURCE_H5AD),
        "selected_outputs_absent": True,
        "preflight_only": bool(args.preflight_only),
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return
    if not args.confirm_user_selected:
        raise PermissionError(
            "Actual selected rerun requires --confirm-user-selected after the user explicitly chooses this exact label"
        )
    if not args.manual_selection_note.strip():
        raise ValueError("Actual selected rerun requires a non-empty --manual-selection-note")

    random.seed(SEED)
    np.random.seed(SEED)
    cp.random.seed(SEED)
    package_lines = package_version_lines()
    started = time.time()
    adata = sc.read_h5ad(SOURCE_H5AD)
    if INTEGRATED_BASIS not in adata.obsm or adata.obsm[INTEGRATED_BASIS].shape[1] < n_pcs:
        raise ValueError("Canonical Harmony source lacks the selected integrated dimensions")
    if "neighbors" in adata.uns or "X_umap" in adata.obsm:
        raise ValueError("Canonical Harmony source is not clean before selected rerun")
    existing_cluster_columns = [
        column for column in adata.obs if column.startswith(("leiden", "louvain"))
    ]
    if existing_cluster_columns or "leiden_coarse" in adata.obs:
        raise ValueError(f"Canonical Harmony source already has clustering columns: {existing_cluster_columns}")

    free_before, total_vram = cp.cuda.runtime.memGetInfo()
    graph_started = time.time()
    rsc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=INTEGRATED_BASIS,
        random_state=SEED,
        algorithm="brute",
    )
    rsc.tl.umap(adata, random_state=SEED)
    rsc.tl.leiden(adata, resolution=resolution, key_added=cluster_key, random_state=SEED)
    graph_elapsed = time.time() - graph_started
    release_gpu_arrays(adata)

    actual_counts = (
        adata.obs[cluster_key].astype(str).value_counts().sort_index().rename("n_cells").reset_index()
    )
    actual_counts.columns = ["cluster", "n_cells"]
    expected_counts = pd.read_csv(TABLE_ROOT / args.candidate_label / "cluster_counts.csv")
    expected_counts["cluster"] = expected_counts["cluster"].astype(str)
    expected_counts = expected_counts.sort_values("cluster").reset_index(drop=True)
    counts_match_by_label = actual_counts.equals(expected_counts)
    count_multiset_match = sorted(actual_counts["n_cells"].tolist()) == sorted(
        expected_counts["n_cells"].tolist()
    )
    count_comparison = expected_counts.merge(
        actual_counts,
        on="cluster",
        how="outer",
        suffixes=("_candidate_grid", "_clean_selected_rerun"),
    )
    count_comparison["delta_clean_minus_candidate"] = (
        count_comparison["n_cells_clean_selected_rerun"]
        - count_comparison["n_cells_candidate_grid"]
    )
    if "leiden_coarse" in adata.obs:
        raise ValueError("Selected clustering must not create leiden_coarse")

    selected_relative = "h5ad/01-celltype_integration_clustering/05-clustering-parameter-search/selected/adata_inte.h5ad"
    selected_at = datetime.now().astimezone().isoformat()
    adata.uns["selected_clustering_contract"] = {
        "candidate_label": args.candidate_label,
        "selection_source": "explicit_user_reply",
        "manual_selection_note": args.manual_selection_note.strip(),
        "selected_at": selected_at,
        "clean_source_h5ad": str(SOURCE_H5AD),
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "neighbor_use_rep": INTEGRATED_BASIS,
        "neighbor_algorithm": "brute",
        "clustering_algorithm": "leiden",
        "raw_cluster_column": cluster_key,
        "random_seed": SEED,
        "backend": "rapids-singlecell_GPU",
        "candidate_grid_n_clusters": int(expected_counts.shape[0]),
        "clean_selected_rerun_n_clusters": int(actual_counts.shape[0]),
        "cluster_counts_match_candidate_by_label": bool(counts_match_by_label),
        "cluster_size_multiset_matches_candidate": bool(count_multiset_match),
        "selected_rerun_is_authoritative": True,
        "candidate_h5ad_was_not_saved_by_grid_contract": True,
        "reproducibility_note": (
            "RAPIDS Leiden may produce a different partition across clean process reruns "
            "despite an identical recorded seed; selected parameters, not transient candidate "
            "membership, define the formal rerun"
        ),
    }

    PENDING_H5AD_DIR.mkdir(parents=True, exist_ok=False)
    PENDING_FIGURE_DIR.mkdir(parents=True, exist_ok=False)
    adata.write_h5ad(PENDING_H5AD, compression="gzip", compression_opts=1)
    pending = sc.read_h5ad(PENDING_H5AD, backed="r")
    pending_errors = []
    if pending.shape != adata.shape:
        pending_errors.append("shape mismatch")
    if pending.raw is None or pending.raw.n_vars != adata.raw.n_vars:
        pending_errors.append("raw all-gene snapshot missing or mismatched")
    if INTEGRATED_BASIS not in pending.obsm or "X_umap" not in pending.obsm:
        pending_errors.append("Harmony or UMAP embedding missing")
    if "neighbors" not in pending.uns or cluster_key not in pending.obs:
        pending_errors.append("neighbors or selected raw cluster column missing")
    if "leiden_coarse" in pending.obs:
        pending_errors.append("leiden_coarse was created before broad annotation")
    pending.file.close()
    del pending
    if pending_errors:
        raise ValueError(f"Pending selected H5AD failed validation: {pending_errors}")

    sc.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
    sc.settings.figdir = PENDING_FIGURE_DIR
    sc.pl.umap(
        adata,
        color=cluster_key,
        show=False,
        save="_selected_raw_clusters.pdf",
    )
    selected_figure_pending = PENDING_FIGURE_DIR / "umap_selected_raw_clusters.pdf"
    if not selected_figure_pending.is_file() or selected_figure_pending.stat().st_size == 0:
        raise FileNotFoundError("Selected UMAP PDF was not written")

    # Promote validated directories before writing the root-level selection
    # records. Candidate-grid outputs remain untouched.
    PENDING_H5AD_DIR.replace(SELECTED_H5AD_DIR)
    PENDING_FIGURE_DIR.replace(SELECTED_FIGURE_DIR)
    total_elapsed = time.time() - started
    selected_record = {
        "candidate_label": args.candidate_label,
        "selected_h5ad": selected_relative,
        "selected_h5ad_absolute": str(SELECTED_H5AD),
        "final_raw_cluster_column": cluster_key,
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "clustering_algorithm": "leiden",
        "integration_source": "Harmony",
        "manual_selection_note": args.manual_selection_note.strip(),
        "selection_source": "explicit_user_reply",
        "selected_at": selected_at,
        "candidate_manifest_status": row["status"],
        "candidate_completed": bool(row["completed"]),
    }
    pd.DataFrame([selected_record]).to_csv(SELECTED_TABLE, index=False)
    parameters = {
        **selected_record,
        "source_h5ad": str(SOURCE_H5AD),
        "neighbor_use_rep": INTEGRATED_BASIS,
        "neighbor_algorithm": "brute",
        "umap_random_state": SEED,
        "leiden_random_state": SEED,
        "random_seed": SEED,
        "backend_package": "rapids-singlecell_GPU",
        "gpu_free_bytes_before_graph": int(free_before),
        "gpu_total_bytes": int(total_vram),
        "graph_umap_leiden_elapsed_seconds": graph_elapsed,
        "total_elapsed_seconds": total_elapsed,
        "code_file": str(Path(__file__).resolve()),
        "selected_figure": str(SELECTED_FIGURE_DIR / "umap_selected_raw_clusters.pdf"),
        "candidate_grid_outputs_preserved": True,
        "leiden_coarse_created": False,
        "candidate_grid_n_clusters": int(expected_counts.shape[0]),
        "clean_selected_rerun_n_clusters": int(actual_counts.shape[0]),
        "cluster_counts_match_candidate_by_label": bool(counts_match_by_label),
        "cluster_size_multiset_matches_candidate": bool(count_multiset_match),
        "notes_fallbacks": (
            "none; clean selected rerun is authoritative; candidate-grid partition was transient "
            "and RAPIDS Leiden did not reproduce its cluster counts exactly"
            if not counts_match_by_label
            else "none"
        ),
    }
    pd.DataFrame([parameters]).to_csv(PARAMETER_TABLE, index=False)
    actual_counts.to_csv(CLUSTER_COUNT_TABLE, index=False)
    count_comparison.to_csv(COUNT_COMPARISON_TABLE, index=False)
    pd.DataFrame(
        [
            {
                "step": "selected_clustering_rerun",
                "planned_backend": "rapids-singlecell_GPU",
                "attempted_backend": "rapids-singlecell_GPU",
                "status": "completed",
                "error_summary": "",
                "fallback_backend": "",
                "clean_input_reloaded": True,
                "final_backend_for_rerun": "rapids-singlecell_GPU",
            }
        ]
    ).to_csv(BACKEND_TABLE, index=False)
    PACKAGE_VERSIONS.write_text("\n".join(package_lines) + "\n", encoding="utf-8")
    validation = {
        **preflight,
        "preflight_only": False,
        "user_selection_confirmed": True,
        "manual_selection_note": args.manual_selection_note.strip(),
        "selected_h5ad": str(SELECTED_H5AD),
        "selected_figure": str(SELECTED_FIGURE_DIR / "umap_selected_raw_clusters.pdf"),
        "n_obs": adata.n_obs,
        "n_vars_HVG": adata.n_vars,
        "raw_n_vars_all_gene_normalized_log": adata.raw.n_vars,
        "candidate_grid_n_clusters": int(expected_counts.shape[0]),
        "n_clusters": int(actual_counts.shape[0]),
        "cluster_counts_match_candidate_exactly": bool(counts_match_by_label),
        "cluster_size_multiset_matches_candidate": bool(count_multiset_match),
        "selected_parameters_match_candidate_exactly": True,
        "selected_clean_rerun_is_authoritative": True,
        "candidate_h5ad_saved": False,
        "count_comparison_table": str(COUNT_COMPARISON_TABLE),
        "reproducibility_note": (
            "RAPIDS Leiden clean-process rerun differed from the transient candidate partition "
            "despite the same seed; no candidate H5AD existed to preserve membership"
            if not counts_match_by_label
            else "candidate and selected clean rerun cluster counts match exactly"
        ),
        "neighbor_graph_present": "neighbors" in adata.uns,
        "umap_present": "X_umap" in adata.obsm,
        "raw_cluster_column_present": cluster_key in adata.obs,
        "leiden_coarse_present": "leiden_coarse" in adata.obs,
        "candidate_grid_outputs_preserved": True,
        "random_seed": SEED,
    }
    VALIDATION_PATH.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

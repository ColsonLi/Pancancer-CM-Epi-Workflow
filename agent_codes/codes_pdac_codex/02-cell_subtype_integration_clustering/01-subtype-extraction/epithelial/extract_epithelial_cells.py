#!/usr/bin/env python3
"""Extract score/rank-consistent epithelial IDs from the normalized all-cell atlas."""

from __future__ import annotations

import importlib.metadata
import json
import sys
import time
from pathlib import Path

import pandas as pd
import scanpy as sc


SEED = 42
TARGET_LABEL = "Epithelial Cells"
PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
ANNOTATION_H5AD = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/07-score-rank-qc/adata_anno_score_genes_rank_consistent.h5ad"
)
EXPRESSION_H5AD = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_normalized_log_all_genes.h5ad"
)
H5AD_DIR = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/01-subtype-extraction/epithelial"
)
PENDING_H5AD_DIR = (
    PROJECT_ROOT
    / "h5ad/02-cell_subtype_integration_clustering/01-subtype-extraction/epithelial_pending_validation"
)
OUTPUT_H5AD = H5AD_DIR / "adata_epithelial_normalized_log_all_genes.h5ad"
PENDING_H5AD = PENDING_H5AD_DIR / OUTPUT_H5AD.name
TABLE_DIR = (
    PROJECT_ROOT
    / "tables/02-cell_subtype_integration_clustering/01-subtype-extraction/epithelial"
)


def package_version_lines() -> list[str]:
    packages = ["scanpy", "anndata", "numpy", "pandas", "scipy", "h5py"]
    lines = [
        "environment_category=rapids_with_scanpy_extraction",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"random_seed={SEED}",
        f"code_files={Path(__file__).resolve()}",
    ]
    lines.extend(f"{package}={importlib.metadata.version(package)}" for package in packages)
    return lines


def main() -> None:
    if not ANNOTATION_H5AD.is_file() or not EXPRESSION_H5AD.is_file():
        raise FileNotFoundError("Consistent annotation source or combined normalized expression source is missing")
    for path in [H5AD_DIR, PENDING_H5AD_DIR, TABLE_DIR]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite epithelial extraction artifact: {path}")

    started = time.time()
    annotation = sc.read_h5ad(ANNOTATION_H5AD, backed="r")
    required_annotation = [
        "leiden_coarse",
        "cell_type",
        "best_rank_type_global",
        "broad_score_rank_consistent",
    ]
    missing = [column for column in required_annotation if column not in annotation.obs]
    if missing:
        raise KeyError(f"Annotation source lacks columns: {missing}")
    eligible_mask = (
        annotation.obs["leiden_coarse"].astype(str).eq(TARGET_LABEL)
        & annotation.obs["best_rank_type_global"].astype(str).eq(TARGET_LABEL)
        & annotation.obs["broad_score_rank_consistent"].astype(bool)
    )
    eligible_ids = annotation.obs_names[eligible_mask].copy()
    transfer_columns = [
        column
        for column in annotation.obs.columns
        if column
        in required_annotation
        or column.endswith("_score")
        or column.endswith("_score_rank_pct")
        or column
        in ["highest_score_type", "broad_annotation_source", "broad_annotation_confidence"]
    ]
    annotation_transfer = annotation.obs.loc[eligible_ids, transfer_columns].copy()
    annotation_n_obs = int(annotation.n_obs)
    annotation.file.close()
    del annotation
    if len(eligible_ids) == 0:
        raise ValueError("No score/rank-consistent epithelial cells were selected")

    source = sc.read_h5ad(EXPRESSION_H5AD)
    if source.raw is not None or len(source.layers) != 0:
        raise ValueError("Combined normalized expression source has unexpected raw/layers state")
    if not source.obs_names.is_unique or not source.var_names.is_unique:
        raise ValueError("Combined normalized expression source IDs are not unique")
    missing_ids = eligible_ids.difference(source.obs_names)
    if len(missing_ids):
        raise KeyError(f"Eligible annotation IDs absent from expression source: {len(missing_ids)}")
    epi = source[eligible_ids].copy()
    del source
    if not epi.obs_names.equals(annotation_transfer.index):
        raise ValueError("Expression subset order differs from annotation transfer order")
    for column in transfer_columns:
        epi.obs[column] = annotation_transfer[column].to_numpy()
    if not epi.obs["leiden_coarse"].astype(str).eq(TARGET_LABEL).all():
        raise ValueError("Extracted object contains a non-epithelial broad label")
    epi.raw = epi.copy()
    epi.uns["epithelial_extraction_contract"] = {
        "annotation_source": str(ANNOTATION_H5AD),
        "expression_source": str(EXPRESSION_H5AD),
        "target_leiden_coarse": TARGET_LABEL,
        "eligibility_rule": "leiden_coarse and best_rank_type_global both equal Epithelial Cells",
        "expression_state": "already normalize_total_1e4_log1p for non-UCSC; existing matching normalized/log UCSC",
        "UCSC_renormalized": False,
        "raw_assignment": "all_gene_normalized_log_subset_before_subtype_HVG_selection",
        "counts_layer_present": False,
        "random_seed": SEED,
    }

    PENDING_H5AD_DIR.mkdir(parents=True, exist_ok=False)
    epi.write_h5ad(PENDING_H5AD, compression="gzip", compression_opts=1)
    pending = sc.read_h5ad(PENDING_H5AD, backed="r")
    errors = []
    if pending.shape != epi.shape:
        errors.append("shape mismatch")
    if pending.raw is None or pending.raw.n_vars != epi.n_vars:
        errors.append("all-gene raw snapshot missing or mismatched")
    if set(required_annotation).difference(pending.obs.columns):
        errors.append("transferred annotation columns missing")
    if len(pending.layers) != 0:
        errors.append("fabricated or unexpected layer present")
    pending.file.close()
    del pending
    if errors:
        raise ValueError(f"Pending epithelial extraction failed validation: {errors}")
    PENDING_H5AD_DIR.replace(H5AD_DIR)

    TABLE_DIR.mkdir(parents=True, exist_ok=False)
    selected_cells = epi.obs[
        [
            "original_barcode",
            "sample",
            "series",
            "status",
            "leiden_coarse",
            "best_rank_type_global",
            "broad_score_rank_consistent",
        ]
    ].copy()
    selected_cells.insert(0, "cell_id", selected_cells.index)
    selected_cells.to_csv(TABLE_DIR / "selected_epithelial_cells.csv", index=False)
    summary = (
        epi.obs.groupby(["series", "sample", "status"], observed=True)
        .size()
        .rename("n_epithelial_cells")
        .reset_index()
    )
    summary.to_csv(TABLE_DIR / "epithelial_extraction_counts_by_sample.csv", index=False)
    elapsed = time.time() - started
    parameters = {
        "step": "score_rank_consistent_epithelial_extraction",
        "annotation_source_h5ad": str(ANNOTATION_H5AD),
        "expression_source_h5ad": str(EXPRESSION_H5AD),
        "output_h5ad": str(OUTPUT_H5AD),
        "annotation_source_n_obs": annotation_n_obs,
        "n_obs_epithelial": int(epi.n_obs),
        "n_vars_all_gene_normalized_log": int(epi.n_vars),
        "n_samples": int(epi.obs["sample"].nunique()),
        "n_series": int(epi.obs["series"].nunique()),
        "status_values": ";".join(sorted(map(str, epi.obs["status"].unique()))),
        "UCSC_renormalized": False,
        "counts_layer_present": False,
        "raw_n_vars": int(epi.raw.n_vars),
        "random_seed": SEED,
        "elapsed_seconds": elapsed,
        "code_file": str(Path(__file__).resolve()),
    }
    pd.DataFrame([parameters]).to_csv(TABLE_DIR / "extraction_parameters.csv", index=False)
    (TABLE_DIR / "package_versions.txt").write_text(
        "\n".join(package_version_lines()) + "\n", encoding="utf-8"
    )
    validation = {
        **parameters,
        "eligible_ids_unique": bool(eligible_ids.is_unique),
        "missing_eligible_ids_in_expression_source": 0,
        "all_cells_epithelial": True,
        "all_cells_score_rank_consistent": bool(epi.obs["broad_score_rank_consistent"].all()),
        "output_h5ad_readable": OUTPUT_H5AD.is_file(),
    }
    (TABLE_DIR / "extraction_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

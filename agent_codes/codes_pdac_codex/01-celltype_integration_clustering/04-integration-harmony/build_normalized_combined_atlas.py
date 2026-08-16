#!/usr/bin/env python3
"""Normalize eligible count data, then merge it with the pre-normalized UCSC atlas."""

from __future__ import annotations

import gzip
import json
import random
import re
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
from scipy import sparse
from scipy.io import mmread


SEED = 42
TARGET_SUM = 10_000.0
random.seed(SEED)
np.random.seed(SEED)
cp.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
SOURCE_ROOT = Path("/mnt/disk18t/lr_xcy/riku/external_atlas_for_cm-epi_validation/PDAC")
INPUT_COUNTS_H5AD = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/03-qc/adata_qc.h5ad"
)
OUTPUT_H5AD = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_normalized_log_all_genes.h5ad"
)
TABLE_DIR = (
    PROJECT_ROOT
    / "tables/01-celltype_integration_clustering/04-integration-harmony"
)
AUDIT_TABLE_DIR = (
    PROJECT_ROOT
    / "tables/01-celltype_integration_clustering/01-input-metadata-audit"
)
INCLUDED_MANIFEST = AUDIT_TABLE_DIR / "included/metadata.csv"
UCSC_MATRIX = SOURCE_ROOT / "UCSC/matrix.mtx.gz"
UCSC_FEATURES = SOURCE_ROOT / "UCSC/features.tsv.gz"
UCSC_BARCODES = SOURCE_ROOT / "UCSC/barcodes.tsv.gz"
UCSC_CELL_META = SOURCE_ROOT / "UCSC/metadata_filtered.tsv"

ENSEMBL_GENE_RE = re.compile(r"^ENS[A-Z]*G[0-9]+(?:\.[0-9]+)?$")


def require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def assert_normalized_log(matrix: sparse.spmatrix, label: str) -> dict[str, object]:
    if not sparse.issparse(matrix):
        raise TypeError(f"{label}: expected sparse expression")
    values = matrix.data
    if values.size and (not np.isfinite(values).all() or float(values.min()) < 0):
        raise ValueError(f"{label}: expression contains negative or non-finite values")
    noninteger = int(np.count_nonzero(values != np.floor(values)))
    fraction = float(noninteger / values.size) if values.size else 0.0
    if fraction < 0.50:
        raise ValueError(
            f"{label}: only {fraction:.3%} noninteger nonzeros; normalized/log contract is doubtful"
        )
    return {
        "object": label,
        "n_obs": int(matrix.shape[0]),
        "n_vars": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "minimum_nonzero": float(values.min()) if values.size else 0.0,
        "maximum_nonzero": float(values.max()) if values.size else 0.0,
        "noninteger_nnz": noninteger,
        "noninteger_nnz_fraction": fraction,
        "normalized_log_contract_supported": True,
    }


def gene_audit(index: pd.Index, object_layer: str, input_path: Path) -> dict[str, object]:
    values = pd.Index(index).astype(str)
    is_ensembl = values.to_series(index=range(len(values))).str.match(ENSEMBL_GENE_RE)
    fraction = float(is_ensembl.mean()) if len(values) else 0.0
    kind = "gene_id" if fraction >= 0.90 else (
        "gene_name" if fraction <= 0.10 else "mixed_or_unknown"
    )
    return {
        "input_h5ad": str(input_path),
        "object_layer": object_layer,
        "var_names_type": kind,
        "n_features": len(values),
        "n_unique": int(values.nunique()),
        "n_duplicates": int(values.duplicated().sum()),
        "ensembl_like_n": int(is_ensembl.sum()),
        "ensembl_like_fraction": fraction,
        "candidate_gene_id_columns": "",
        "candidate_gene_name_columns": "var_names;gene_name",
        "selected_gene_id_source": "",
        "selected_gene_name_source": "var_names",
        "action_taken": "retained_unique_gene_names_and_inner_join",
    }


def normalize_counts_on_gpu(adata: ad.AnnData) -> tuple[ad.AnnData, dict[str, object]]:
    if "counts" not in adata.layers:
        raise ValueError("QC count atlas lacks layers['counts']")
    if not sparse.issparse(adata.X):
        raise TypeError("QC count atlas X must be sparse")
    if np.count_nonzero(adata.X.data != np.floor(adata.X.data)):
        raise ValueError("QC count atlas X is not integer-valued before normalization")

    counts_layer = adata.layers["counts"]
    del adata.layers["counts"]
    backend = {
        "step": "normalize_total_and_log1p_non_UCSC_counts",
        "planned_backend": "rapids-singlecell_GPU",
        "attempted_backend": "rapids-singlecell_GPU",
        "status": "running",
        "error_summary": "",
        "fallback_backend": "",
        "clean_input_reloaded": False,
        "final_backend_for_rerun": "rapids-singlecell_GPU",
    }
    try:
        rsc.get.anndata_to_GPU(adata)
        rsc.pp.normalize_total(adata, target_sum=TARGET_SUM)
        rsc.pp.log1p(adata)
        rsc.get.anndata_to_CPU(adata)
        backend["status"] = "completed"
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        backend["status"] = "failed"
        backend["error_summary"] = message[:1000]
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        finally:
            pass
        if "out of memory" not in message.lower() and "outofmemory" not in message.lower():
            raise
        backend["fallback_backend"] = "scanpy_CPU_after_GPU_OOM"
        backend["final_backend_for_rerun"] = "scanpy_CPU_after_GPU_OOM"
        clean = sc.read_h5ad(INPUT_COUNTS_H5AD)
        counts_layer = clean.layers["counts"]
        del clean.layers["counts"]
        sc.pp.normalize_total(clean, target_sum=TARGET_SUM)
        sc.pp.log1p(clean)
        adata = clean
        backend["status"] = "completed_after_GPU_OOM_CPU_fallback"
        backend["clean_input_reloaded"] = True
    adata.layers["counts"] = counts_layer
    return adata, backend


def load_ucsc() -> tuple[ad.AnnData, dict[str, object]]:
    features = pd.read_csv(UCSC_FEATURES, sep="\t", header=None, dtype=str)[0]
    barcodes = pd.read_csv(UCSC_BARCODES, sep="\t", header=None, dtype=str)[0]
    metadata = pd.read_csv(UCSC_CELL_META, sep="\t", dtype=str)
    metadata["Cell_ID"] = metadata["Cell_ID"].astype(str)
    if features.duplicated().any():
        raise ValueError("UCSC features are duplicated")
    if barcodes.duplicated().any():
        raise ValueError("UCSC barcodes are duplicated")
    barcode_to_position = pd.Series(np.arange(len(barcodes), dtype=np.int64), index=barcodes)
    if not metadata["Cell_ID"].isin(barcode_to_position.index).all():
        raise ValueError("Filtered UCSC metadata contains barcodes absent from the matrix")
    selected_positions = barcode_to_position.loc[metadata["Cell_ID"]].to_numpy()
    if len(np.unique(selected_positions)) != len(selected_positions):
        raise ValueError("Filtered UCSC metadata contains duplicated cell IDs")

    with gzip.open(UCSC_MATRIX, "rb") as handle:
        full_matrix = mmread(handle)
    if full_matrix.shape != (len(features), len(barcodes)):
        raise ValueError(
            f"UCSC matrix shape {full_matrix.shape} does not match "
            f"features/barcodes {(len(features), len(barcodes))}"
        )
    full_csc = full_matrix.tocsc()
    del full_matrix
    selected = full_csc[:, selected_positions].transpose().tocsr().astype(np.float32)
    del full_csc

    obs = metadata.set_index("Cell_ID", drop=False).copy()
    included = pd.read_csv(INCLUDED_MANIFEST, low_memory=False)
    sample_manifest = included.loc[included["dataset"].eq("UCSC")].copy()
    if sample_manifest["sample_id"].duplicated().any():
        raise ValueError("UCSC included sample manifest has duplicate sample IDs")
    manifest_lookup = sample_manifest.set_index("sample_id", drop=False)
    sample_ids = obs["Sample_ID"].astype(str)
    if not sample_ids.isin(manifest_lookup.index.astype(str)).all():
        missing = sorted(set(sample_ids) - set(manifest_lookup.index.astype(str)))
        raise ValueError(f"Filtered UCSC cells include ineligible/unmapped samples: {missing}")

    mapped = manifest_lookup.loc[sample_ids].reset_index(drop=True)
    obs["original_barcode"] = obs["Cell_ID"].astype(str)
    obs["sample"] = sample_ids.to_numpy()
    obs["sample_label"] = mapped["sample_id"].astype(str).to_numpy()
    obs["series"] = "UCSC"
    obs["status"] = mapped["status"].astype(str).to_numpy()
    obs["patient_id"] = mapped["patient_id"].astype(str).to_numpy()
    obs["sample_type"] = mapped["sample_type"].fillna("").astype(str).to_numpy()
    obs["tissue"] = mapped["tissue"].fillna("").astype(str).to_numpy()
    obs["source_matrix_path"] = str(UCSC_MATRIX)
    obs["source_expression_kind"] = "QC_completed_normalize_total_1e4_log1p"
    obs["source_qc_state"] = "source_QC_completed"
    obs["source_publication"] = obs["Publication"].astype(str)
    obs["source_level_1"] = obs["Level_1"].astype(str)
    obs["source_level_2"] = obs["Level_2"].astype(str)
    obs["pct_counts_MT"] = pd.to_numeric(obs["pct_counts_mt"], errors="raise")
    obs["n_genes_by_counts"] = pd.to_numeric(obs["n_genes_by_counts"], errors="raise")
    obs["total_counts"] = pd.to_numeric(obs["total_counts"], errors="raise")
    obs.index = pd.Index(
        obs["sample"].astype(str) + "#" + obs["original_barcode"].astype(str),
        name="merged_cell_id",
    )
    var = pd.DataFrame(
        {"gene_name": features.astype(str).to_numpy()},
        index=pd.Index(features.astype(str), name=None),
    )
    adata = ad.AnnData(X=selected, obs=obs, var=var)
    audit = {
        "matrix_rows_features": len(features),
        "matrix_columns_all_cells": len(barcodes),
        "matrix_nnz_all_cells": 1_347_605_434,
        "filtered_metadata_cells": metadata.shape[0],
        "selected_matrix_cells": adata.n_obs,
        "selected_matrix_genes": adata.n_vars,
        "selected_matrix_nnz": int(adata.X.nnz),
        "filtered_barcode_order_used": True,
        "filtered_samples": int(adata.obs["sample"].nunique()),
        "excluded_samples_present": False,
        "source_matrix_role": "QC_completed_normalize_total_1e4_log1p",
    }
    return adata, audit


def main() -> None:
    required = [INPUT_COUNTS_H5AD, INCLUDED_MANIFEST, UCSC_MATRIX, UCSC_FEATURES, UCSC_BARCODES, UCSC_CELL_META]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    outputs = [
        OUTPUT_H5AD,
        TABLE_DIR / "expression_matrix_provenance.csv",
        TABLE_DIR / "expression_value_audit.csv",
        TABLE_DIR / "ucsc_filtered_matrix_contract.csv",
        TABLE_DIR / "combined_sample_summary.csv",
        TABLE_DIR / "combined_cell_barcode_mapping.csv.gz",
        TABLE_DIR / "gene_identifier_audit.csv",
        TABLE_DIR / "gene_identifier_examples.csv",
        TABLE_DIR / "01_normalize_total_parameters.csv",
        TABLE_DIR / "02_log1p_parameters.csv",
        TABLE_DIR / "gpu_backend_normalize_ucsc_merge.csv",
        TABLE_DIR / "combined_atlas_validation.json",
    ]
    for path in outputs:
        require_new(path)
    OUTPUT_H5AD.parent.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    non_ucsc = sc.read_h5ad(INPUT_COUNTS_H5AD)
    n_obs_before = non_ucsc.n_obs
    n_vars_before = non_ucsc.n_vars
    non_ucsc, backend = normalize_counts_on_gpu(non_ucsc)
    non_ucsc_audit = assert_normalized_log(non_ucsc.X, "non_UCSC_QC_normalize_total_1e4_log1p")
    counts_layer = non_ucsc.layers["counts"]
    if np.count_nonzero(counts_layer.data != np.floor(counts_layer.data)):
        raise ValueError("Non-UCSC counts layer changed during normalization")
    del non_ucsc.layers["counts"]

    ucsc, ucsc_contract = load_ucsc()
    ucsc_audit = assert_normalized_log(ucsc.X, "UCSC_source_QC_normalize_total_1e4_log1p")
    common_genes = non_ucsc.var_names.intersection(ucsc.var_names, sort=False)
    if not len(common_genes):
        raise RuntimeError("No shared genes between normalized non-UCSC and UCSC atlases")
    combined = ad.concat(
        [non_ucsc[:, common_genes], ucsc[:, common_genes]],
        join="inner",
        merge="first",
        index_unique=None,
    )
    combined.X = combined.X.tocsr().astype(np.float32)
    combined.var["gene_name"] = combined.var_names.astype(str)
    if not combined.obs_names.is_unique:
        raise ValueError("Combined atlas cell IDs are not unique")
    if "counts" in combined.layers:
        raise ValueError("A partial/invalid combined counts layer must not be created")
    required_obs = {"sample", "series", "status", "original_barcode"}
    if not required_obs.issubset(combined.obs.columns):
        raise ValueError(f"Combined atlas lacks obs columns: {required_obs - set(combined.obs.columns)}")
    combined_audit = assert_normalized_log(combined.X, "combined_normalized_log_atlas")

    combined.uns["matrix_provenance"] = {
        "X": "all-cell normalized/log expression",
        "non_UCSC_transform": "project QC counts -> normalize_total(target_sum=10000) -> log1p",
        "UCSC_transform": "source-provided QC-completed normalize_total(10000) -> log1p",
        "counts_layer_present": False,
        "counts_layer_absence_reason": "UCSC raw counts were not supplied; no fabricated partial/full count layer",
        "non_UCSC_raw_counts_h5ad": str(INPUT_COUNTS_H5AD),
        "gene_join": "inner",
        "random_seed": SEED,
    }
    combined.uns["UCSC_raw_counts_available"] = False
    combined.uns["non_UCSC_raw_counts_available_separately"] = True
    combined.uns["non_UCSC_raw_counts_h5ad"] = str(INPUT_COUNTS_H5AD)
    for column in combined.obs.select_dtypes(include=["object"]).columns:
        if column != "original_barcode":
            combined.obs[column] = combined.obs[column].astype("category")

    provenance = pd.DataFrame(
        [
            {
                "cell_scope": "non_UCSC",
                "series": "GSE229413;GSE278688",
                "input_matrix": str(INPUT_COUNTS_H5AD),
                "input_role": "post_QC_raw_nonnegative_integer_counts",
                "operation": "normalize_total(target_sum=10000);log1p",
                "combined_X_role": "normalized_log_expression",
                "raw_counts_preserved_at": str(INPUT_COUNTS_H5AD),
                "combined_counts_layer": False,
            },
            {
                "cell_scope": "UCSC",
                "series": "UCSC",
                "input_matrix": str(UCSC_MATRIX),
                "input_role": "source_QC_completed_normalize_total_10000_log1p",
                "operation": "none_no_renormalization",
                "combined_X_role": "normalized_log_expression",
                "raw_counts_preserved_at": "not_available_from_source",
                "combined_counts_layer": False,
            },
        ]
    )
    provenance.to_csv(TABLE_DIR / "expression_matrix_provenance.csv", index=False)
    pd.DataFrame([ucsc_contract]).to_csv(TABLE_DIR / "ucsc_filtered_matrix_contract.csv", index=False)
    sample_summary = (
        combined.obs.groupby(["series", "sample", "status"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    sample_summary.to_csv(TABLE_DIR / "combined_sample_summary.csv", index=False)
    cell_map = combined.obs[["sample", "original_barcode", "source_matrix_path"]].copy()
    cell_map.insert(0, "merged_cell_id", combined.obs_names.astype(str))
    cell_map.to_csv(TABLE_DIR / "combined_cell_barcode_mapping.csv.gz", index=False, compression="gzip")
    pd.DataFrame([non_ucsc_audit, ucsc_audit, combined_audit]).to_csv(
        TABLE_DIR / "expression_value_audit.csv", index=False
    )

    audits = [
        gene_audit(non_ucsc.var_names, "non_UCSC_normalized_log:var_names", INPUT_COUNTS_H5AD),
        gene_audit(ucsc.var_names, "UCSC_normalized_log:var_names", UCSC_MATRIX),
        gene_audit(combined.var_names, "combined_normalized_log:var_names", OUTPUT_H5AD),
    ]
    pd.DataFrame(audits).to_csv(TABLE_DIR / "gene_identifier_audit.csv", index=False)
    example_rows = []
    for label, index in [
        ("non_UCSC_normalized_log:var_names", non_ucsc.var_names),
        ("UCSC_normalized_log:var_names", ucsc.var_names),
        ("combined_normalized_log:var_names", combined.var_names),
    ]:
        example_rows.extend(
            {"object_layer": label, "example_order": order + 1, "var_name": value}
            for order, value in enumerate(pd.Index(index).astype(str)[:20])
        )
    pd.DataFrame(example_rows).to_csv(TABLE_DIR / "gene_identifier_examples.csv", index=False)
    common_parameters = {
        "input_h5ad": str(INPUT_COUNTS_H5AD),
        "output_h5ad_or_object": str(OUTPUT_H5AD),
        "n_obs_before": n_obs_before,
        "n_obs_after": combined.n_obs,
        "n_vars_before": n_vars_before,
        "n_vars_after": combined.n_vars,
        "UCSC_renormalized": False,
        "gene_join": "inner",
        "backend_package": "rapids-singlecell",
        "code_file": str(Path(__file__).resolve()),
        "random_seed": SEED,
        "notes": "UCSC is never labeled or stored as counts",
    }
    pd.DataFrame(
        [{**common_parameters, "step": "normalize_total_non_UCSC", "target_sum": TARGET_SUM}]
    ).to_csv(TABLE_DIR / "01_normalize_total_parameters.csv", index=False)
    pd.DataFrame(
        [{**common_parameters, "step": "log1p_non_UCSC", "log1p": True}]
    ).to_csv(TABLE_DIR / "02_log1p_parameters.csv", index=False)
    pd.DataFrame([backend]).to_csv(TABLE_DIR / "gpu_backend_normalize_ucsc_merge.csv", index=False)

    combined.write_h5ad(OUTPUT_H5AD, compression="gzip", compression_opts=1)
    validation = {
        "output_h5ad": str(OUTPUT_H5AD),
        "n_obs": combined.n_obs,
        "n_vars": combined.n_vars,
        "nnz": int(combined.X.nnz),
        "n_samples": int(combined.obs["sample"].nunique()),
        "sample_counts_by_series": {
            str(key): int(value)
            for key, value in combined.obs.groupby("series", observed=True)["sample"].nunique().items()
        },
        "status_cell_counts": {
            str(key): int(value) for key, value in combined.obs["status"].value_counts().items()
        },
        "counts_layer_present": "counts" in combined.layers,
        "non_UCSC_raw_counts_h5ad": str(INPUT_COUNTS_H5AD),
        "UCSC_renormalized": False,
        "obs_names_unique": bool(combined.obs_names.is_unique),
        "random_seed": SEED,
    }
    (TABLE_DIR / "combined_atlas_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

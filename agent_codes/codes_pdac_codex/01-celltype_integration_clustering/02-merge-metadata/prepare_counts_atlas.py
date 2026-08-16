#!/usr/bin/env python3
"""Build the pre-QC raw-count atlas from eligible non-UCSC PDAC samples.

UCSC is intentionally excluded here because its matrix is already QC-completed,
normalize_total(1e4), log1p expression rather than raw counts.  It is joined only
after this counts atlas has completed QC and normalization/log transformation.
"""

from __future__ import annotations

import gzip
import importlib.metadata
import json
import random
import re
import shutil
import sys
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmread


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
SOURCE_ROOT = Path("/mnt/disk18t/lr_xcy/riku/external_atlas_for_cm-epi_validation/PDAC")
AUDIT_TABLE_DIR = (
    PROJECT_ROOT
    / "tables/01-celltype_integration_clustering/01-input-metadata-audit"
)
RECEIPT_DIR = PROJECT_ROOT / "inputs/geo-official-20260806"
STAGING_DIR = PROJECT_ROOT / "inputs/01-singlecell-counts-staging/GSE229413-filtered-10x"
TABLE_DIR = (
    PROJECT_ROOT
    / "tables/01-celltype_integration_clustering/02-merge-metadata"
)
H5AD_DIR = (
    PROJECT_ROOT
    / "h5ad/01-celltype_integration_clustering/02-merge-metadata"
)
OUTPUT_H5AD = H5AD_DIR / "adata_merge.h5ad"

GSE229413_TAR = SOURCE_ROOT / "GSE229413/GSE229413_RAW.tar"
GSE278688_MATRIX = RECEIPT_DIR / "GSE278688_sc_matrix.mtx.gz"
GSE278688_FEATURES = RECEIPT_DIR / "GSE278688_sc_features.tsv.gz"
GSE278688_BARCODES = RECEIPT_DIR / "GSE278688_sc_barcodes.tsv.gz"
GSE278688_CELL_META = RECEIPT_DIR / "GSE278688_sc_metadata.csv.gz"
INCLUDED_MANIFEST = AUDIT_TABLE_DIR / "included/metadata.csv"
FULL_MANIFEST = AUDIT_TABLE_DIR / "sample_metadata.csv"

ENSEMBL_GENE_RE = re.compile(r"^ENS[A-Z]*G[0-9]+(?:\.[0-9]+)?$")


def require_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))


def require_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {path}. Use a versioned path."
        )


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def normalize_patient(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", clean_text(value)).upper()


def assert_integer_counts(matrix: sparse.spmatrix, label: str) -> dict[str, object]:
    if not sparse.issparse(matrix):
        raise TypeError(f"{label}: expected a sparse count matrix")
    data = matrix.data
    if data.size and (not np.isfinite(data).all() or float(data.min()) < 0):
        raise ValueError(f"{label}: counts contain non-finite or negative values")
    noninteger = int(np.count_nonzero(data != np.floor(data)))
    if noninteger:
        raise ValueError(f"{label}: found {noninteger} non-integer nonzero values")
    return {
        "object": label,
        "n_obs": int(matrix.shape[0]),
        "n_vars": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "minimum_nonzero": float(data.min()) if data.size else 0.0,
        "maximum_nonzero": float(data.max()) if data.size else 0.0,
        "noninteger_nnz": noninteger,
        "nonnegative_integer_counts": True,
    }


def gene_identifier_row(
    index: pd.Index,
    object_label: str,
    candidate_id_columns: list[str],
    candidate_name_columns: list[str],
    action: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    values = pd.Index(index).astype(str)
    is_ensembl = values.to_series(index=range(len(values))).str.match(ENSEMBL_GENE_RE)
    fraction = float(is_ensembl.mean()) if len(values) else 0.0
    kind = "gene_id" if fraction >= 0.90 else (
        "gene_name" if fraction <= 0.10 else "mixed_or_unknown"
    )
    row = {
        "input_h5ad": "not_yet_materialized",
        "object_layer": object_label,
        "var_names_type": kind,
        "n_features": len(values),
        "n_unique": int(values.nunique()),
        "n_duplicates": int(values.duplicated().sum()),
        "ensembl_like_n": int(is_ensembl.sum()),
        "ensembl_like_fraction": fraction,
        "candidate_gene_id_columns": ";".join(candidate_id_columns),
        "candidate_gene_name_columns": ";".join(candidate_name_columns),
        "selected_gene_id_source": "gene_ids" if "gene_ids" in candidate_id_columns else "",
        "selected_gene_name_source": "var_names",
        "action_taken": action,
    }
    examples = [
        {"object_layer": object_label, "example_order": i + 1, "var_name": name}
        for i, name in enumerate(values[:20])
    ]
    return row, examples


def collapse_duplicate_gene_symbols(
    adata: ad.AnnData,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Sum count columns sharing a gene symbol and retain a reversible map."""
    symbols = pd.Index(adata.var_names.astype(str), name="gene_name")
    codes, unique_symbols = pd.factorize(symbols, sort=False)
    group_sizes = pd.Series(codes).value_counts().sort_index().to_numpy()
    gene_ids = (
        adata.var["gene_ids"].astype(str).to_numpy()
        if "gene_ids" in adata.var.columns
        else np.repeat("", adata.n_vars)
    )
    mapping = pd.DataFrame(
        {
            "source_feature_order": np.arange(adata.n_vars) + 1,
            "source_gene_id": gene_ids,
            "gene_name": symbols.to_numpy(),
            "duplicate_group_size": group_sizes[codes],
        }
    )
    mapping["action"] = np.where(
        mapping["duplicate_group_size"].gt(1),
        "summed_counts_across_duplicate_gene_symbol",
        "kept_one_to_one",
    )

    if len(unique_symbols) == adata.n_vars:
        out = adata.copy()
        out.var["gene_name"] = out.var_names.astype(str)
        return out, mapping

    projector = sparse.csr_matrix(
        (
            np.ones(adata.n_vars, dtype=np.float32),
            (np.arange(adata.n_vars, dtype=np.int64), codes.astype(np.int64)),
        ),
        shape=(adata.n_vars, len(unique_symbols)),
    )
    collapsed_x = (adata.X.tocsr() @ projector).tocsr()
    first_positions = np.full(len(unique_symbols), adata.n_vars, dtype=np.int64)
    np.minimum.at(first_positions, codes, np.arange(adata.n_vars, dtype=np.int64))
    gene_id_groups = gene_ids[first_positions].astype(object)
    for duplicate_code in np.flatnonzero(group_sizes > 1):
        gene_id_groups[duplicate_code] = ";".join(
            pd.unique(gene_ids[codes == duplicate_code].astype(str))
        )
    var = pd.DataFrame(
        {
            "gene_name": unique_symbols.astype(str),
            "source_gene_ids": gene_id_groups,
        },
        index=pd.Index(unique_symbols.astype(str), name=None),
    )
    out = ad.AnnData(X=collapsed_x, obs=adata.obs.copy(), var=var)
    return out, mapping


def annotate_sample(adata: ad.AnnData, row: pd.Series, source_path: str) -> ad.AnnData:
    out = adata.copy()
    raw_barcodes = pd.Index(out.obs_names.astype(str))
    gsm = clean_text(row["gsm"])
    out.obs["original_barcode"] = raw_barcodes.to_numpy()
    out.obs["sample"] = gsm
    out.obs["sample_label"] = clean_text(row["sample_id"])
    out.obs["series"] = clean_text(row["dataset"])
    out.obs["status"] = clean_text(row["status"])
    out.obs["patient_id"] = clean_text(row["patient_id"])
    out.obs["sample_type"] = clean_text(row["sample_type"])
    out.obs["tissue"] = clean_text(row["tissue"])
    out.obs["source_matrix_path"] = source_path
    out.obs["source_expression_kind"] = "raw_nonnegative_integer_counts"
    out.obs["source_qc_state"] = "cellranger_or_geo_filtered_counts_pre_project_qc"
    out.obs_names = pd.Index(gsm + "#" + raw_barcodes.astype(str), name="merged_cell_id")
    return out


def extract_gse229413_members(manifest: pd.DataFrame) -> tuple[dict[str, Path], pd.DataFrame]:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    rows: list[dict[str, object]] = []
    with tarfile.open(GSE229413_TAR, "r") as archive:
        members = archive.getmembers()
        for _, sample_row in manifest.iterrows():
            gsm = clean_text(sample_row["gsm"])
            hits = [
                member
                for member in members
                if Path(member.name).name.startswith(gsm + "_")
                and Path(member.name).name.endswith("_filtered_feature_bc_matrix.h5")
                and member.isfile()
            ]
            if len(hits) != 1:
                raise ValueError(f"{gsm}: expected one filtered 10x H5 member, found {len(hits)}")
            member = hits[0]
            destination = STAGING_DIR / Path(member.name).name
            if destination.exists():
                if destination.stat().st_size != member.size:
                    raise ValueError(f"Existing staged file has wrong size: {destination}")
                action = "reused_exact_size_staged_file"
            else:
                part = destination.with_suffix(destination.suffix + ".part")
                source = archive.extractfile(member)
                if source is None:
                    raise OSError(f"Could not open tar member: {member.name}")
                with source, part.open("wb") as target:
                    shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
                if part.stat().st_size != member.size:
                    raise OSError(f"Incomplete extraction: {member.name}")
                part.replace(destination)
                action = "extracted_from_validated_source_tar"
            extracted[gsm] = destination
            rows.append(
                {
                    "gsm": gsm,
                    "biological_sample_id": clean_text(sample_row["sample_id"]),
                    "tar_path": str(GSE229413_TAR),
                    "tar_member": member.name,
                    "member_size_bytes": member.size,
                    "staged_path": str(destination),
                    "action": action,
                }
            )
    return extracted, pd.DataFrame(rows)


def load_gse229413(
    manifest: pd.DataFrame,
    matrix_audits: list[dict[str, object]],
    gene_audits: list[dict[str, object]],
    gene_examples: list[dict[str, object]],
) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths, extraction_manifest = extract_gse229413_members(manifest)
    sample_objects: list[ad.AnnData] = []
    summary_rows: list[dict[str, object]] = []
    canonical_feature_mapping: pd.DataFrame | None = None

    for _, row in manifest.sort_values("gsm").iterrows():
        gsm = clean_text(row["gsm"])
        path = paths[gsm]
        sample = sc.read_10x_h5(path, gex_only=True)
        pre_row, pre_examples = gene_identifier_row(
            sample.var_names,
            f"GSE229413:{gsm}:var_names_before_duplicate_collapse",
            [column for column in ["gene_ids"] if column in sample.var.columns],
            ["var_names"],
            "audited_before_explicit_duplicate_gene_symbol_collapse",
        )
        gene_audits.append(pre_row)
        gene_examples.extend(pre_examples)
        matrix_audits.append(assert_integer_counts(sample.X, f"GSE229413:{gsm}:source"))

        sample, mapping = collapse_duplicate_gene_symbols(sample)
        if canonical_feature_mapping is None:
            canonical_feature_mapping = mapping
        else:
            keys = ["source_gene_id", "gene_name", "duplicate_group_size", "action"]
            if not canonical_feature_mapping[keys].equals(mapping[keys]):
                raise ValueError(f"{gsm}: feature mapping differs from the first GSE229413 sample")
        if sample.var_names.has_duplicates:
            raise ValueError(f"{gsm}: duplicate gene symbols remain after explicit collapse")
        sample = annotate_sample(sample, row, str(path))
        sample_objects.append(sample)
        summary_rows.append(
            {
                "series": "GSE229413",
                "sample": gsm,
                "sample_label": clean_text(row["sample_id"]),
                "status": clean_text(row["status"]),
                "n_cells": sample.n_obs,
                "n_genes": sample.n_vars,
                "source_matrix_path": str(path),
            }
        )

    if canonical_feature_mapping is None:
        raise RuntimeError("No GSE229413 samples were loaded")
    merged = ad.concat(sample_objects, join="inner", merge="same", index_unique=None)
    merged.var["gene_name"] = merged.var_names.astype(str)
    merged.X = merged.X.tocsr().astype(np.float32)
    matrix_audits.append(assert_integer_counts(merged.X, "GSE229413:dataset_merged"))
    return merged, extraction_manifest, pd.DataFrame(summary_rows), canonical_feature_mapping


def load_gse278688(
    full_manifest: pd.DataFrame,
    included_manifest: pd.DataFrame,
    matrix_audits: list[dict[str, object]],
    gene_audits: list[dict[str, object]],
    gene_examples: list[dict[str, object]],
) -> tuple[ad.AnnData, pd.DataFrame]:
    features = pd.read_csv(GSE278688_FEATURES, sep="\t", header=None, dtype=str)
    if features.shape[1] < 2:
        raise ValueError("GSE278688 features file lacks a gene-name column")
    features.columns = ["gene_id", "gene_name", "feature_type"][: features.shape[1]]
    if features["gene_name"].duplicated().any():
        raise ValueError("GSE278688 gene names are duplicated; explicit collapse is required")
    barcodes = pd.read_csv(GSE278688_BARCODES, sep="\t", header=None, dtype=str)[0]
    cell_meta = pd.read_csv(GSE278688_CELL_META, index_col=0)
    cell_meta.index = cell_meta.index.astype(str)
    if not pd.Index(barcodes).equals(cell_meta.index):
        raise ValueError("GSE278688 barcode order does not exactly match cell metadata")

    with gzip.open(GSE278688_MATRIX, "rb") as handle:
        source_matrix = mmread(handle)
    if source_matrix.shape != (features.shape[0], len(barcodes)):
        raise ValueError(
            f"GSE278688 matrix shape {source_matrix.shape} does not match "
            f"features/barcodes {(features.shape[0], len(barcodes))}"
        )
    matrix = source_matrix.transpose().tocsr().astype(np.float32)
    del source_matrix
    matrix_audits.append(assert_integer_counts(matrix, "GSE278688:source_all_cells"))

    obs = cell_meta.copy()
    obs["original_barcode"] = pd.Index(barcodes).to_numpy()
    obs["patient_key"] = obs["patients"].map(normalize_patient)
    obs["status_key"] = np.select(
        [obs["tissue"].astype(str).eq("Tumor"), obs["tissue"].astype(str).eq("Adjacent_normal")],
        ["tumor", "normal-like"],
        default="excluded_other_tissue",
    )

    series_manifest = full_manifest.loc[full_manifest["dataset"].eq("GSE278688")].copy()
    series_manifest["patient_key"] = series_manifest["patient_id"].map(normalize_patient)
    map_rows = series_manifest.loc[
        series_manifest["status"].isin(["tumor", "normal-like"])
        & series_manifest["sample_type"].isin(["PT", "AT"]),
        ["patient_key", "status", "gsm", "sample_id", "patient_id", "sample_type", "tissue"],
    ].copy()
    if map_rows.duplicated(["patient_key", "status"]).any():
        raise ValueError("GSE278688 patient/status keys are not unique in the sample manifest")
    map_rows = map_rows.rename(columns={"status": "status_key"})
    obs = obs.reset_index(names="cell_barcode").merge(
        map_rows,
        on=["patient_key", "status_key"],
        how="left",
        validate="many_to_one",
    ).set_index("cell_barcode")

    included_gsms = set(included_manifest["gsm"].dropna().astype(str))
    eligible = obs["gsm"].astype(str).isin(included_gsms)
    all_data = ad.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(
            {
                "gene_id": features["gene_id"].to_numpy(),
                "gene_name": features["gene_name"].to_numpy(),
                "feature_type": features.get("feature_type", pd.Series("", index=features.index)).to_numpy(),
            },
            index=pd.Index(features["gene_name"].astype(str), name=None),
        ),
    )
    subset = all_data[eligible.to_numpy()].copy()
    del all_data, matrix
    if subset.n_obs == 0:
        raise RuntimeError("No eligible GSE278688 cells remain after manifest mapping")

    sample_objects: list[ad.AnnData] = []
    summary_rows: list[dict[str, object]] = []
    manifest_index = included_manifest.set_index("gsm", drop=False)
    for gsm in sorted(included_gsms):
        sample_mask = subset.obs["gsm"].astype(str).eq(gsm).to_numpy()
        sample = subset[sample_mask].copy()
        if sample.n_obs == 0:
            raise ValueError(f"Included GSE278688 sample has zero mapped cells: {gsm}")
        row = manifest_index.loc[gsm]
        raw_barcodes = sample.obs.index.astype(str)
        sample.obs["sample"] = gsm
        sample.obs["sample_label"] = clean_text(row["sample_id"])
        sample.obs["series"] = "GSE278688"
        sample.obs["status"] = clean_text(row["status"])
        sample.obs["patient_id"] = clean_text(row["patient_id"])
        sample.obs["sample_type"] = clean_text(row["sample_type"])
        sample.obs["tissue"] = clean_text(row["tissue"])
        sample.obs["source_matrix_path"] = str(GSE278688_MATRIX)
        sample.obs["source_expression_kind"] = "raw_nonnegative_integer_counts"
        sample.obs["source_qc_state"] = "geo_filtered_counts_pre_project_qc"
        sample.obs_names = pd.Index(gsm + "#" + raw_barcodes, name="merged_cell_id")
        sample_objects.append(sample)
        summary_rows.append(
            {
                "series": "GSE278688",
                "sample": gsm,
                "sample_label": clean_text(row["sample_id"]),
                "status": clean_text(row["status"]),
                "n_cells": sample.n_obs,
                "n_genes": sample.n_vars,
                "source_matrix_path": str(GSE278688_MATRIX),
            }
        )

    merged = ad.concat(sample_objects, join="inner", merge="same", index_unique=None)
    merged.X = merged.X.tocsr().astype(np.float32)
    matrix_audits.append(assert_integer_counts(merged.X, "GSE278688:eligible_dataset_merged"))
    gene_row, examples = gene_identifier_row(
        merged.var_names,
        "GSE278688:eligible_dataset_merged:var_names",
        [column for column in ["gene_id"] if column in merged.var.columns],
        ["gene_name"],
        "retained_unique_geo_gene_names",
    )
    gene_audits.append(gene_row)
    gene_examples.extend(examples)
    return merged, pd.DataFrame(summary_rows)


def main() -> None:
    require_inputs(
        [
            INCLUDED_MANIFEST,
            FULL_MANIFEST,
            GSE229413_TAR,
            GSE278688_MATRIX,
            GSE278688_FEATURES,
            GSE278688_BARCODES,
            GSE278688_CELL_META,
        ]
    )
    require_new_output(OUTPUT_H5AD)
    for table_name in [
        "cell_barcode_mapping.csv.gz",
        "sample_merge_summary.csv",
        "matrix_integer_count_audit.csv",
        "gene_identifier_audit.csv",
        "gene_identifier_examples.csv",
        "gse229413_tar_extraction_manifest.csv",
        "gse229413_gene_id_to_gene_name_mapping.csv",
        "merge_parameters.csv",
        "merge_validation.json",
        "package_versions.txt",
    ]:
        require_new_output(TABLE_DIR / table_name)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    H5AD_DIR.mkdir(parents=True, exist_ok=True)

    included = pd.read_csv(INCLUDED_MANIFEST, low_memory=False)
    full = pd.read_csv(FULL_MANIFEST, low_memory=False)
    non_ucsc = included.loc[
        included["modality"].eq("single_cell") & ~included["dataset"].eq("UCSC")
    ].copy()
    expected = {"GSE229413": 30, "GSE278688": 12}
    observed = non_ucsc.groupby("dataset")["gsm"].nunique().to_dict()
    if observed != expected:
        raise ValueError(f"Eligible non-UCSC sample counts changed: {observed} != {expected}")

    matrix_audits: list[dict[str, object]] = []
    gene_audits: list[dict[str, object]] = []
    gene_examples: list[dict[str, object]] = []

    gse229413, extraction, summary_229, feature_mapping = load_gse229413(
        non_ucsc.loc[non_ucsc["dataset"].eq("GSE229413")],
        matrix_audits,
        gene_audits,
        gene_examples,
    )
    post_row, post_examples = gene_identifier_row(
        gse229413.var_names,
        "GSE229413:dataset_merged:var_names_after_duplicate_collapse",
        [column for column in ["source_gene_ids"] if column in gse229413.var.columns],
        ["gene_name"],
        "duplicate_gene_symbols_explicitly_collapsed_by_summing_counts",
    )
    gene_audits.append(post_row)
    gene_examples.extend(post_examples)

    gse278688, summary_278 = load_gse278688(
        full,
        non_ucsc.loc[non_ucsc["dataset"].eq("GSE278688")],
        matrix_audits,
        gene_audits,
        gene_examples,
    )

    common_genes = gse229413.var_names.intersection(gse278688.var_names, sort=False)
    if len(common_genes) == 0:
        raise RuntimeError("No shared genes between GSE229413 and GSE278688")
    atlas = ad.concat(
        [gse229413[:, common_genes], gse278688[:, common_genes]],
        join="inner",
        merge="first",
        index_unique=None,
    )
    atlas.X = atlas.X.tocsr().astype(np.float32)
    atlas.var["gene_name"] = atlas.var_names.astype(str)
    if not atlas.obs_names.is_unique:
        raise ValueError("Merged cell IDs are not globally unique")
    required_obs = {"sample", "series", "status", "original_barcode"}
    if not required_obs.issubset(atlas.obs.columns):
        raise ValueError(f"Missing required merged obs columns: {required_obs - set(atlas.obs.columns)}")
    matrix_audits.append(assert_integer_counts(atlas.X, "non_UCSC_cross_dataset_merged"))

    atlas.layers["counts"] = atlas.X.copy()
    atlas.uns["matrix_provenance"] = {
        "X": "raw nonnegative integer counts from eligible GSE229413 and GSE278688 cells",
        "layers_counts": "exact copy of X before project QC",
        "UCSC_included": False,
        "UCSC_exclusion_reason": (
            "UCSC is QC-completed normalize_total(1e4)+log1p and is joined only after "
            "the count atlas completes QC and matching normalization/log transformation"
        ),
        "within_dataset_join": "inner",
        "cross_dataset_join": "inner",
        "random_seed": SEED,
    }
    atlas.uns["sample_manifest_path"] = str(INCLUDED_MANIFEST)
    atlas.uns["source_datasets"] = ["GSE229413", "GSE278688"]
    for column in atlas.obs.select_dtypes(include=["object"]).columns:
        if column != "original_barcode":
            atlas.obs[column] = atlas.obs[column].astype("category")

    merged_gene_row, merged_examples = gene_identifier_row(
        atlas.var_names,
        "non_UCSC_cross_dataset_merged:var_names",
        [column for column in ["gene_id", "source_gene_ids"] if column in atlas.var.columns],
        ["gene_name"],
        "retained_inner_join_shared_unique_gene_names",
    )
    merged_gene_row["input_h5ad"] = str(OUTPUT_H5AD)
    gene_audits.append(merged_gene_row)
    gene_examples.extend(merged_examples)

    cell_map = atlas.obs[["sample", "original_barcode", "source_matrix_path"]].copy()
    cell_map.insert(0, "merged_cell_id", atlas.obs_names.astype(str))
    cell_map.to_csv(TABLE_DIR / "cell_barcode_mapping.csv.gz", index=False, compression="gzip")
    sample_summary = pd.concat([summary_229, summary_278], ignore_index=True)
    sample_summary.to_csv(TABLE_DIR / "sample_merge_summary.csv", index=False)
    pd.DataFrame(matrix_audits).to_csv(TABLE_DIR / "matrix_integer_count_audit.csv", index=False)
    pd.DataFrame(gene_audits).to_csv(TABLE_DIR / "gene_identifier_audit.csv", index=False)
    pd.DataFrame(gene_examples).to_csv(TABLE_DIR / "gene_identifier_examples.csv", index=False)
    extraction.to_csv(TABLE_DIR / "gse229413_tar_extraction_manifest.csv", index=False)
    feature_mapping.to_csv(
        TABLE_DIR / "gse229413_gene_id_to_gene_name_mapping.csv", index=False
    )

    parameters = pd.DataFrame(
        [
            {
                "step": "02-merge-metadata",
                "input_h5ad": "none_raw_matrices",
                "output_h5ad_or_object": str(OUTPUT_H5AD),
                "n_obs_before": int(gse229413.n_obs + gse278688.n_obs),
                "n_obs_after": atlas.n_obs,
                "n_vars_before": f"GSE229413={gse229413.n_vars};GSE278688={gse278688.n_vars}",
                "n_vars_after": atlas.n_vars,
                "within_dataset_join": "inner",
                "cross_dataset_join": "inner",
                "UCSC_present": False,
                "matrix_role": "raw_counts_in_X_and_layers_counts",
                "backend_package": f"scanpy={sc.__version__};anndata={ad.__version__}",
                "code_file": str(Path(__file__).resolve()),
                "random_seed": SEED,
                "notes": "UCSC deferred until counts QC and normalize_total(1e4)+log1p",
            }
        ]
    )
    parameters.to_csv(TABLE_DIR / "merge_parameters.csv", index=False)

    package_names = ["scanpy", "anndata", "numpy", "pandas", "scipy", "h5py"]
    package_lines = [
        f"environment_category=main",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"random_seed={SEED}",
        "install_manager=uv",
        "index_url=https://pypi.tuna.tsinghua.edu.cn/simple",
    ]
    for package_name in package_names:
        package_lines.append(
            f"{package_name}={importlib.metadata.version(package_name)}"
        )
    (TABLE_DIR / "package_versions.txt").write_text(
        "\n".join(package_lines) + "\n", encoding="utf-8"
    )

    atlas.write_h5ad(OUTPUT_H5AD, compression="gzip", compression_opts=1)
    validation = {
        "output_h5ad": str(OUTPUT_H5AD),
        "n_obs": atlas.n_obs,
        "n_vars": atlas.n_vars,
        "nnz": int(atlas.X.nnz),
        "n_samples": int(atlas.obs["sample"].nunique()),
        "sample_count_by_series": {
            str(key): int(value)
            for key, value in atlas.obs.groupby("series", observed=True)["sample"].nunique().items()
        },
        "status_cell_counts": {
            str(key): int(value) for key, value in atlas.obs["status"].value_counts().items()
        },
        "required_obs_present": sorted(required_obs),
        "obs_names_unique": bool(atlas.obs_names.is_unique),
        "counts_layer_exact_shape": list(atlas.layers["counts"].shape),
        "UCSC_present": False,
        "seed": SEED,
    }
    (TABLE_DIR / "merge_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

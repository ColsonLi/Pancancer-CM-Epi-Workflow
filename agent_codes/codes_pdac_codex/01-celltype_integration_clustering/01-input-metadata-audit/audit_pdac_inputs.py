#!/usr/bin/env python3
"""Audit PDAC uploads and consolidate sample-level inclusion metadata.

This script is intentionally read-only with respect to DATA_ROOT.  It writes
only provenance/readiness tables beneath the active PDAC analysis root.
"""

from __future__ import annotations

import csv
import gzip
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


SEED = 42
DATA_ROOT = Path(
    "/mnt/disk18t/lr_xcy/riku/external_atlas_for_cm-epi_validation/PDAC"
)
ANALYSIS_ROOT = Path(__file__).resolve().parents[3]
TABLE_ROOT = (
    ANALYSIS_ROOT
    / "tables"
    / "01-celltype_integration_clustering"
    / "01-input-metadata-audit"
)


SAMPLE_METADATA_FILES = (
    DATA_ROOT / "UCSC" / "metadata_samples.csv",
    DATA_ROOT / "GSE229413" / "GSE229413_metadata_samples.csv",
    DATA_ROOT / "GSE278688" / "GSE278688_metadata_samples.csv",
    DATA_ROOT / "GSE278687_st" / "GSE278687_metadata_samples.csv",
)

READINESS_TARGETS = (
    ("UCSC", "single_cell_normalized_log", DATA_ROOT / "UCSC" / "matrix.mtx.gz"),
    ("UCSC", "single_cell_normalized_log", DATA_ROOT / "UCSC" / "features.tsv.gz"),
    ("UCSC", "single_cell_normalized_log", DATA_ROOT / "UCSC" / "barcodes.tsv.gz"),
    ("UCSC", "single_cell_metadata", DATA_ROOT / "UCSC" / "metadata_filtered.tsv"),
    ("GSE229413", "single_cell_raw_counts", DATA_ROOT / "GSE229413" / "GSE229413_RAW.tar"),
    ("GSE229413", "single_cell_raw_counts_optional", DATA_ROOT / "GSE229413" / "GSE229413_donor_samples.zip"),
    ("GSE229413", "single_cell_raw_counts_optional", DATA_ROOT / "GSE229413" / "GSE229413_tumor_samples.zip"),
    ("GSE278688", "single_cell_raw_counts", DATA_ROOT / "GSE278688" / "GSE278688_sc_matrix.mtx.gz"),
    ("GSE278688", "single_cell_raw_counts", DATA_ROOT / "GSE278688" / "GSE278688_sc_features.tsv.gz"),
    ("GSE278688", "single_cell_raw_counts", DATA_ROOT / "GSE278688" / "GSE278688_sc_barcodes.tsv.gz"),
    ("GSE278688", "single_cell_metadata", DATA_ROOT / "GSE278688" / "GSE278688_sc_metadata.csv.gz"),
    ("GSE278687_st", "spatial", DATA_ROOT / "GSE278687_st" / "GSE278687_RAW.tar"),
)


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def first_nonempty(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = str(row.get(name, "") or "").strip()
        if value:
            return value
    return ""


def yes(value: object) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1", "y"}


def normalized_dataset(path: Path, row: dict[str, str]) -> str:
    return first_nonempty(row, "Dataset") or path.parent.name


def modality_for(path: Path, row: dict[str, str]) -> str:
    assay = first_nonempty(row, "Assay").lower()
    if path.parent.name.endswith("_st") or "spatial" in assay or "visium" in assay:
        return "spatial"
    return "single_cell"


def metastasis_token(text: str) -> bool:
    value = text.lower()
    positive_phrases = (
        "explicit lymph-node metastasis",
        "explicit lymph node metastasis",
        "explicit distant metastasis",
        "matched liver metastasis",
        "metastatic lesion",
        "patient explicitly staged metastatic",
        "with distant metastasis",
    )
    if any(term in value for term in positive_phrases):
        return True
    if re.search(r"(?:^|[^a-z0-9])m1(?:[^a-z0-9]|$)", value):
        return True
    if re.search(r"(?:^|[^a-z0-9])n(?:1|2|3|\+)(?:[^a-z0-9]|$)", value):
        return True
    if re.search(r"stage\s*iv(?:[^a-z]|$)", value):
        return True
    return False


def forbidden_site_token(text: str) -> str:
    value = text.lower()
    if "lymph node" in value or "lymph-node" in value:
        return "lymph_node_sample"
    if "thrombus" in value or "tumor thrombus" in value:
        return "thrombus_sample"
    if "peripheral blood" in value or re.search(r"(?:^|\W)blood(?:$|\W)", value):
        return "blood_sample"
    return ""


def consolidate_metadata() -> list[dict[str, object]]:
    staged: list[dict[str, object]] = []
    patient_evidence: dict[tuple[str, str], list[str]] = defaultdict(list)

    for source_path in SAMPLE_METADATA_FILES:
        for source_row_number, row in enumerate(read_csv(source_path), start=2):
            dataset = normalized_dataset(source_path, row)
            patient_id = first_nonempty(row, "Patient_ID")
            sample_id = first_nonempty(row, "Sample_ID", "GSM", "Sample_title")
            if not patient_id:
                # UCSC metadata already encodes patient-level curation.  Retain the
                # stable sample ID rather than inventing a cross-study patient key.
                patient_id = sample_id
            evidence_fields = [
                first_nonempty(row, "pTNM"),
                first_nonempty(row, "Clinical_stage", "Stage"),
                first_nonempty(row, "N_status"),
                first_nonempty(row, "M_status"),
                first_nonempty(row, "Exclusion_reason"),
                first_nonempty(row, "Evidence_note"),
            ]
            evidence_text = " | ".join(x for x in evidence_fields if x)
            if metastasis_token(evidence_text):
                patient_evidence[(dataset, patient_id)].append(evidence_text)

            site_text = " | ".join(
                x
                for x in (
                    first_nonempty(row, "Sample_type", "Type"),
                    first_nonempty(row, "Tissue", "Location"),
                    first_nonempty(row, "Sample_title"),
                    first_nonempty(row, "Exclusion_reason"),
                )
                if x
            )
            staged.append(
                {
                    "dataset": dataset,
                    "modality": modality_for(source_path, row),
                    "patient_id": patient_id,
                    "sample_id": sample_id,
                    "gsm": first_nonempty(row, "GSM"),
                    "sample_type": first_nonempty(row, "Sample_type", "Type"),
                    "tissue": first_nonempty(row, "Tissue", "Location"),
                    "assay": first_nonempty(row, "Assay"),
                    "n_cells_reported": first_nonempty(row, "n_cells"),
                    "ptnm": first_nonempty(row, "pTNM"),
                    "clinical_stage": first_nonempty(row, "Clinical_stage", "Stage"),
                    "n_status": first_nonempty(row, "N_status"),
                    "m_status": first_nonempty(row, "M_status"),
                    "lineage_sorting": first_nonempty(row, "Lineage_sorting"),
                    "source_use": first_nonempty(row, "Use"),
                    "source_exclusion_reason": first_nonempty(row, "Exclusion_reason"),
                    "source_evidence_note": first_nonempty(row, "Evidence_note"),
                    "source_url": first_nonempty(row, "Source_URL"),
                    "source_metadata_csv": str(source_path),
                    "source_row_number": source_row_number,
                    "site_rule_flag": forbidden_site_token(site_text),
                }
            )

    output: list[dict[str, object]] = []
    for row in staged:
        patient_key = (str(row["dataset"]), str(row["patient_id"]))
        patient_metastasis = patient_key in patient_evidence
        source_included = yes(row["source_use"])
        reasons: list[str] = []
        if patient_metastasis:
            reasons.append("patient_has_explicit_metastasis_evidence")
        if row["site_rule_flag"]:
            reasons.append(str(row["site_rule_flag"]))
        if not source_included:
            source_reason = str(row["source_exclusion_reason"] or "source_metadata_marks_excluded")
            reasons.append(f"source_excluded:{source_reason}")

        include = source_included and not patient_metastasis and not row["site_rule_flag"]
        row["patient_metastasis_evidence"] = patient_metastasis
        row["patient_metastasis_evidence_text"] = " || ".join(patient_evidence.get(patient_key, []))
        row["include"] = include
        row["decision"] = "include" if include else "exclude"
        row["decision_reason"] = ";".join(reasons) if reasons else "eligible_by_curated_metadata_and_user_rules"
        row["status"] = (
            "normal-like"
            if re.search(r"normal|adjacent|donor", f"{row['sample_type']} {row['tissue']}", re.I)
            else "tumor"
        )
        output.append(row)
    return output


def sniff_file(path: Path) -> tuple[str, bool, str]:
    if not path.exists():
        return "missing", False, "file_not_found"
    with path.open("rb") as handle:
        head = handle.read(4096)
    stripped = head.lstrip().lower()
    if stripped.startswith(b"<!doctype html") or stripped.startswith(b"<html"):
        return "html_error_page", False, "download_returned_html_instead_of_expected_data"
    if path.suffix == ".zip":
        valid = zipfile.is_zipfile(path)
        return "zip", valid, "ok" if valid else "invalid_zip"
    if path.suffix == ".tar":
        valid = tarfile.is_tarfile(path)
        return "tar", valid, "ok" if valid else "invalid_tar"
    if path.suffix == ".gz":
        valid = head.startswith(b"\x1f\x8b")
        return "gzip", valid, "ok_magic_only_crc_test_separate" if valid else "invalid_gzip_magic"
    return "plain", path.stat().st_size > 0, "ok" if path.stat().st_size > 0 else "empty_file"


def readiness_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, role, path in READINESS_TARGETS:
        kind, valid, note = sniff_file(path)
        rows.append(
            {
                "dataset": dataset,
                "role": role,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "mtime_ns": path.stat().st_mtime_ns if path.exists() else "",
                "detected_type": kind,
                "format_valid": valid,
                "readiness_note": note,
            }
        )
    required_by_dataset: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if not str(row["role"]).endswith("_optional"):
            required_by_dataset[str(row["dataset"])].append(bool(row["format_valid"]))
    for row in rows:
        row["dataset_ready"] = all(required_by_dataset[str(row["dataset"])])
    return rows


def count_gzip_lines(path: Path) -> int:
    count = 0
    last = b""
    with gzip.open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            count += chunk.count(b"\n")
            last = chunk[-1:]
    return count + (1 if last and last != b"\n" else 0)


def matrix_market_header(path: Path) -> tuple[str, int, int, int, list[float]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as handle:
        banner = handle.readline().strip()
        line = handle.readline()
        while line.startswith("%"):
            line = handle.readline()
        n_rows, n_cols, nnz = map(int, line.split())
        values: list[float] = []
        for _ in range(100):
            entry = handle.readline()
            if not entry:
                break
            values.append(float(entry.split()[2]))
    return banner, n_rows, n_cols, nnz, values


def ucsc_contract_rows() -> list[dict[str, object]]:
    root = DATA_ROOT / "UCSC"
    banner, n_rows, n_cols, nnz, values = matrix_market_header(root / "matrix.mtx.gz")
    feature_lines = count_gzip_lines(root / "features.tsv.gz")
    barcode_lines = count_gzip_lines(root / "barcodes.tsv.gz")
    filtered_metadata = read_csv(root / "metadata_filtered.tsv", delimiter="\t")
    sample_metadata = read_csv(root / "metadata_samples.csv")
    included_samples = {r["Sample_ID"] for r in sample_metadata if yes(r.get("Use", ""))}
    metadata_samples = {r["Sample_ID"] for r in filtered_metadata}
    values_noninteger = sum(abs(value - round(value)) > 1e-8 for value in values)
    return [
        {
            "matrix_banner": banner,
            "matrix_n_features": n_rows,
            "features_file_rows": feature_lines,
            "features_match_matrix": n_rows == feature_lines,
            "matrix_n_barcodes": n_cols,
            "barcodes_file_rows": barcode_lines,
            "barcodes_match_matrix": n_cols == barcode_lines,
            "matrix_nnz": nnz,
            "sampled_values_n": len(values),
            "sampled_noninteger_values_n": values_noninteger,
            "matrix_state": "normalize_total_1e4_log1p" if values_noninteger else "ambiguous",
            "filtered_metadata_cells": len(filtered_metadata),
            "filtered_metadata_samples": len(metadata_samples),
            "included_sample_manifest_n": len(included_samples),
            "metadata_samples_all_in_included_manifest": metadata_samples <= included_samples,
            "excluded_samples_absent_from_filtered_metadata": metadata_samples.isdisjoint(
                {r["Sample_ID"] for r in sample_metadata if not yes(r.get("Use", ""))}
            ),
        }
    ]


def gse229413_archive_rows() -> list[dict[str, object]]:
    tar_path = DATA_ROOT / "GSE229413" / "GSE229413_RAW.tar"
    rows = read_csv(DATA_ROOT / "GSE229413" / "GSE229413_metadata_samples.csv")
    expected_gsm = {r["GSM"] for r in rows if yes(r.get("Use", "")) and r.get("GSM")}
    with tarfile.open(tar_path, "r") as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]
    filtered_h5 = [name for name in members if name.endswith("_filtered_feature_bc_matrix.h5")]
    raw_h5 = [name for name in members if name.endswith("_raw_feature_bc_matrix.h5")]
    observed_filtered_gsm = {Path(name).name.split("_", 1)[0] for name in filtered_h5}
    return [
        {
            "tar_path": str(tar_path),
            "archive_files": len(members),
            "filtered_feature_bc_matrix_h5": len(filtered_h5),
            "raw_feature_bc_matrix_h5": len(raw_h5),
            "included_metadata_gsm": len(expected_gsm),
            "included_gsm_with_filtered_h5": len(expected_gsm & observed_filtered_gsm),
            "included_gsm_missing_filtered_h5": ";".join(sorted(expected_gsm - observed_filtered_gsm)),
            "filtered_h5_not_in_included_manifest": ";".join(sorted(observed_filtered_gsm - expected_gsm)),
        }
    ]


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    metadata = consolidate_metadata()
    metadata_fields = [
        "dataset", "modality", "patient_id", "sample_id", "gsm", "status",
        "sample_type", "tissue", "assay", "n_cells_reported", "ptnm",
        "clinical_stage", "n_status", "m_status", "lineage_sorting",
        "source_use", "source_exclusion_reason", "source_evidence_note",
        "source_url", "patient_metastasis_evidence",
        "patient_metastasis_evidence_text", "site_rule_flag", "include",
        "decision", "decision_reason", "source_metadata_csv", "source_row_number",
    ]
    write_csv(TABLE_ROOT / "sample_metadata.csv", metadata, metadata_fields)
    included = [row for row in metadata if row["include"]]
    excluded = [row for row in metadata if not row["include"]]
    write_csv(TABLE_ROOT / "included" / "metadata.csv", included, metadata_fields)
    write_csv(TABLE_ROOT / "excluded" / "metadata.csv", excluded, metadata_fields)

    readiness = readiness_rows()
    write_csv(
        TABLE_ROOT / "input_file_readiness.csv",
        readiness,
        [
            "dataset", "role", "path", "exists", "size_bytes", "mtime_ns",
            "detected_type", "format_valid", "readiness_note", "dataset_ready",
        ],
    )
    ucsc = ucsc_contract_rows()
    write_csv(TABLE_ROOT / "ucsc_matrix_contract.csv", ucsc, list(ucsc[0]))
    archive = gse229413_archive_rows()
    write_csv(TABLE_ROOT / "gse229413_archive_contract.csv", archive, list(archive[0]))

    summary_rows: list[dict[str, object]] = []
    for (dataset, modality), group in sorted(
        ((key, list(values)) for key, values in _group(metadata, "dataset", "modality")),
        key=lambda item: item[0],
    ):
        decisions = Counter(str(row["decision"]) for row in group)
        summary_rows.append(
            {
                "dataset": dataset,
                "modality": modality,
                "n_samples": len(group),
                "n_included": decisions.get("include", 0),
                "n_excluded": decisions.get("exclude", 0),
            }
        )
    write_csv(
        TABLE_ROOT / "sample_decision_summary.csv",
        summary_rows,
        ["dataset", "modality", "n_samples", "n_included", "n_excluded"],
    )

    readme = f"""PDAC input and metadata audit

Input root: {DATA_ROOT}
Code: {Path(__file__).resolve()}
Random seed contract: {SEED} (no stochastic operation is used in this audit)

sample_metadata.csv is the canonical combined sample manifest.  The exact
included and excluded subsets are also materialized as included/metadata.csv
and excluded/metadata.csv, respectively.  Existing curated Use/Exclusion_reason
fields are preserved and are never used to promote a source-excluded sample.
The audit additionally enforces patient-level explicit metastasis evidence and
sample-site exclusions for lymph node, thrombus, and blood specimens.

UCSC is treated as QC-completed normalize_total(1e4)+log1p expression.  It must
not be relabeled as counts.  Other single-cell inputs are expected to contain
raw nonnegative integer counts and will be merged/QC-normalized before the
normalized/log UCSC subset is concatenated.

input_file_readiness.csv records format-level readiness only.  Gzip CRC checks
are executed separately and should be added to the run provenance before Block
1 starts.  HTML error pages are hard failures, not valid tiny input files.
"""
    (TABLE_ROOT / "readme.txt").write_text(readme, encoding="utf-8")


def _group(rows: list[dict[str, object]], *keys: str):
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    return grouped.items()


if __name__ == "__main__":
    main()

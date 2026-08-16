#!/usr/bin/env python3
"""Validate official GEO replacements and preserve a versioned receipt snapshot."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import tarfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
SOURCE_ROOT = Path("/mnt/disk18t/lr_xcy/riku/external_atlas_for_cm-epi_validation/PDAC")
RECEIPT_DIR = PROJECT_ROOT / "inputs/geo-official-20260806"
TABLE_PARENT = (
    PROJECT_ROOT
    / "tables/01-celltype_integration_clustering/01-input-metadata-audit"
)
TIMEZONE = ZoneInfo("Asia/Shanghai")

FILES = {
    "GSE278688_sc_matrix.mtx.gz": {
        "expected_bytes": 1_093_755_688,
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278688/suppl/GSE278688_sc_matrix.mtx.gz",
        "kind": "gzip_matrix_market",
        "source_placeholder": SOURCE_ROOT / "GSE278688/GSE278688_sc_matrix.mtx.gz",
    },
    "GSE278688_sc_features.tsv.gz": {
        "expected_bytes": 221_703,
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278688/suppl/GSE278688_sc_features.tsv.gz",
        "kind": "gzip_tsv",
        "source_placeholder": SOURCE_ROOT / "GSE278688/GSE278688_sc_features.tsv.gz",
    },
    "GSE278688_sc_barcodes.tsv.gz": {
        "expected_bytes": 911_362,
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278688/suppl/GSE278688_sc_barcodes.tsv.gz",
        "kind": "gzip_tsv",
        "source_placeholder": SOURCE_ROOT / "GSE278688/GSE278688_sc_barcodes.tsv.gz",
    },
    "GSE278688_sc_metadata.csv.gz": {
        "expected_bytes": 1_166_884,
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278688/suppl/GSE278688_sc_metadata.csv.gz",
        "kind": "gzip_csv",
        "source_placeholder": SOURCE_ROOT / "GSE278688/GSE278688_sc_metadata.csv.gz",
    },
    "GSE278687_RAW.tar": {
        "expected_bytes": 1_117_061_120,
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278687/suppl/GSE278687_RAW.tar",
        "kind": "tar",
        "source_placeholder": SOURCE_ROOT / "GSE278687_st/GSE278687_RAW.tar",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def gzip_crc_ok(path: Path) -> bool:
    with gzip.open(path, "rb") as handle:
        while handle.read(16 * 1024 * 1024):
            pass
    return True


def classify_small_source(path: Path) -> str:
    if not path.is_file():
        return "missing"
    with path.open("rb") as handle:
        prefix = handle.read(512).lower()
    if b"<!doctype html" in prefix or b"<html" in prefix:
        return "html_error_page"
    if prefix.startswith(b"\x1f\x8b"):
        return "gzip"
    if tarfile.is_tarfile(path):
        return "tar"
    return "unknown"


def read_matrix_market_header(path: Path) -> tuple[str, tuple[int, int, int]]:
    with gzip.open(path, "rt", encoding="ascii") as handle:
        banner = handle.readline().strip()
        line = handle.readline()
        while line.startswith("%"):
            line = handle.readline()
        dims = tuple(int(value) for value in line.split())
    if len(dims) != 3:
        raise ValueError(f"Unexpected Matrix Market dimensions: {dims}")
    return banner, dims


def main() -> None:
    now = datetime.now(TIMEZONE)
    snapshot_label = now.strftime("geo-receipt-%Y%m%dT%H%M%S%z")
    table_dir = TABLE_PARENT / snapshot_label
    if table_dir.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot: {table_dir}")
    table_dir.mkdir(parents=True)

    receipt_rows = []
    placeholder_rows = []
    tar_rows = []
    all_ready = True

    for name, contract in FILES.items():
        path = RECEIPT_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_size = path.stat().st_size
        exact_size = observed_size == contract["expected_bytes"]
        digest = sha256(path)
        crc_ok: bool | str = "not_applicable"
        tar_ok: bool | str = "not_applicable"
        n_tar_members: int | str = "not_applicable"
        if str(contract["kind"]).startswith("gzip"):
            crc_ok = gzip_crc_ok(path)
        elif contract["kind"] == "tar":
            with tarfile.open(path, "r") as archive:
                members = archive.getmembers()
                n_tar_members = len(members)
                tar_rows.extend(
                    {
                        "member_order": index + 1,
                        "member_name": member.name,
                        "member_size_bytes": member.size,
                        "member_type": "file" if member.isfile() else "directory_or_other",
                    }
                    for index, member in enumerate(members)
                )
            tar_ok = True
        file_ready = bool(exact_size and (crc_ok is True or tar_ok is True))
        all_ready = all_ready and file_ready
        receipt_rows.append(
            {
                "filename": name,
                "path": str(path),
                "source_url": contract["url"],
                "file_kind": contract["kind"],
                "expected_bytes": contract["expected_bytes"],
                "observed_bytes": observed_size,
                "exact_content_length": exact_size,
                "sha256": digest,
                "gzip_crc_ok": crc_ok,
                "tar_readable": tar_ok,
                "tar_member_count": n_tar_members,
                "mtime_iso": datetime.fromtimestamp(path.stat().st_mtime, TIMEZONE).isoformat(),
                "ready": file_ready,
            }
        )

        placeholder = Path(contract["source_placeholder"])
        placeholder_rows.append(
            {
                "filename": name,
                "original_upload_path": str(placeholder),
                "exists": placeholder.exists(),
                "size_bytes": placeholder.stat().st_size if placeholder.exists() else "",
                "sha256": sha256(placeholder) if placeholder.exists() else "",
                "detected_type": classify_small_source(placeholder),
                "modified": False,
            }
        )

    matrix_banner, matrix_dims = read_matrix_market_header(
        RECEIPT_DIR / "GSE278688_sc_matrix.mtx.gz"
    )
    n_features = sum(
        1 for _ in gzip.open(RECEIPT_DIR / "GSE278688_sc_features.tsv.gz", "rt")
    )
    n_barcodes = sum(
        1 for _ in gzip.open(RECEIPT_DIR / "GSE278688_sc_barcodes.tsv.gz", "rt")
    )
    metadata = pd.read_csv(RECEIPT_DIR / "GSE278688_sc_metadata.csv.gz", index_col=0)
    dimensions_match = matrix_dims[:2] == (n_features, n_barcodes)
    metadata_matches = metadata.shape[0] == n_barcodes
    all_ready = all_ready and dimensions_match and metadata_matches

    pd.DataFrame(receipt_rows).to_csv(table_dir / "geo_receipt_manifest.csv", index=False)
    pd.DataFrame(placeholder_rows).to_csv(
        table_dir / "original_upload_placeholder_snapshot.csv", index=False
    )
    pd.DataFrame(tar_rows).to_csv(
        table_dir / "gse278687_tar_member_inventory.csv.gz",
        index=False,
        compression="gzip",
        quoting=csv.QUOTE_MINIMAL,
    )
    matrix_contract = {
        "matrix_market_banner": matrix_banner,
        "matrix_n_features": matrix_dims[0],
        "matrix_n_barcodes": matrix_dims[1],
        "matrix_nnz": matrix_dims[2],
        "features_rows": n_features,
        "barcodes_rows": n_barcodes,
        "metadata_rows": metadata.shape[0],
        "metadata_columns": metadata.shape[1],
        "matrix_features_barcodes_match": dimensions_match,
        "metadata_barcodes_match": metadata_matches,
    }
    pd.DataFrame([matrix_contract]).to_csv(
        table_dir / "gse278688_matrix_contract.csv", index=False
    )
    summary = {
        "snapshot_time": now.isoformat(),
        "snapshot_directory": str(table_dir),
        "official_receipt_directory": str(RECEIPT_DIR),
        "original_upload_directory_modified": False,
        "n_files_expected": len(FILES),
        "n_files_ready": sum(bool(row["ready"]) for row in receipt_rows),
        "gse278688_dimensions": matrix_contract,
        "all_inputs_ready": bool(all_ready),
        "random_seed": SEED,
        "code_file": str(Path(__file__).resolve()),
    }
    (table_dir / "readiness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not all_ready:
        raise RuntimeError("Official GEO receipt validation did not pass")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge only curated-XLSX-v5 included scRNA-seq samples as raw integer counts.

The script reads directly from the five GEO RAW tar archives and the shared
GSE234129 sparse matrix.  Each sample is first written to a resumable h5ad
checkpoint. Samples are inner-joined within each GEO series first, then the
five series-level objects are inner-joined. This preserves only genes present at
both hierarchy levels and records each intersection explicitly.
Nothing is normalized, logged, scaled, or downsampled here.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import platform
import random
import re
import tarfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import BinaryIO, Iterable

import anndata as ad
import numpy as np
import pandas as pd
from anndata.experimental import concat_on_disk
from scipy import sparse
from scipy.io import mmread


SEED = 42
SCRNA_SERIES = (
    "GSE150290",
    "GSE163558",
    "GSE183904",
    "GSE234129",
    "GSE270680",
)
EXPECTED_SAMPLE_COUNTS = {
    "GSE150290": 47,
    "GSE163558": 2,
    "GSE183904": 31,
    "GSE234129": 2,
    "GSE270680": 47,
}
ROOT = Path(__file__).resolve().parents[4]
WF = ROOT / "epi-cm-core-workflow"
DEFAULT_MANIFEST = (
    WF
    / "tables/01-celltype_integration_clustering/01-sample-manifest/"
    "all_datasets_curated_xlsx_inclusion_v5.csv"
)
DEFAULT_DATA_ROOT = Path(
    "/mnt/disk18t/lr_xcy/riku/external_atlas_for_cm-epi_validation/"
    "Gastric_cancer"
)
DEFAULT_H5AD_DIR = WF / "h5ad/01-celltype_integration_clustering/02-merge-metadata"
DEFAULT_TABLE_DIR = WF / "tables/01-celltype_integration_clustering/02-merge-metadata"

REQUIRED_MANIFEST_COLUMNS = {
    "series",
    "modality",
    "sample",
    "sample_title",
    "patient",
    "tissue_class",
    "status",
    "used",
    "local_data_present",
    "source_matrix_path",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_row_sha256(row: pd.Series) -> str:
    payload = {str(key): str(value) for key, value in sorted(row.items())}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    truth = {"true", "1", "yes", "y", "used", "include", "included"}
    return series.astype(str).str.strip().str.lower().isin(truth)


def atomic_write_h5ad(adata: ad.AnnData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.h5ad")
    if tmp.exists():
        tmp.unlink()
    adata.write_h5ad(tmp, compression="gzip")
    os.replace(tmp, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("anndata", "scanpy", "numpy", "pandas", "scipy", "h5py"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Curated XLSX v5 manifest is missing: {path}")
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest.columns))
    if missing:
        raise KeyError(f"Manifest lacks required columns: {missing}")
    duplicated = manifest.duplicated(["series", "sample"], keep=False)
    if duplicated.any():
        raise ValueError(
            "Manifest series+sample keys are not unique: "
            + str(manifest.loc[duplicated, ["series", "sample"]].to_dict("records")[:10])
        )
    used = manifest[
        as_bool(manifest["used"])
        & manifest["series"].isin(SCRNA_SERIES)
        & manifest["modality"].str.lower().str.contains("scrna", regex=False)
    ].copy()
    if used.empty:
        raise ValueError("Curated manifest has no used scRNA-seq samples")
    if not as_bool(used["local_data_present"]).all():
        bad = used.loc[~as_bool(used["local_data_present"]), ["series", "sample"]]
        raise FileNotFoundError(f"Used samples lack local data: {bad.to_dict('records')}")
    if not set(used["status"]).issubset({"normal", "tumor"}):
        raise ValueError("Used scRNA-seq status must be exactly normal or tumor")
    observed_counts = used.groupby("series", observed=True).size().to_dict()
    if observed_counts != EXPECTED_SAMPLE_COUNTS:
        raise ValueError(
            f"Curated manifest sample counts disagree: expected={EXPECTED_SAMPLE_COUNTS}, "
            f"observed={observed_counts}"
        )
    if len(used) != 129 or used["sample"].nunique() != 129:
        raise ValueError("Curated manifest must contain 129 unique included GSMs")
    used["manifest_row_sha256"] = used.apply(canonical_row_sha256, axis=1)
    return used.sort_values(["series", "sample"], kind="stable").reset_index(drop=True)


def clean_gene_names(values: Iterable[object]) -> np.ndarray:
    names = pd.Index([str(value).strip() for value in values], dtype="object")
    valid = (~names.isna()) & (names != "") & (names.str.lower() != "nan")
    if not bool(valid.all()):
        raise ValueError("Feature table contains blank gene symbols")
    return names.to_numpy(dtype=str)


def collapse_duplicate_genes(
    matrix: sparse.spmatrix, gene_names: Iterable[object]
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Sum duplicate gene-symbol columns without changing integer counts."""
    genes = clean_gene_names(gene_names)
    codes, unique = pd.factorize(genes, sort=True)
    x = sparse.csr_matrix(matrix)
    if len(unique) != len(genes):
        projector = sparse.csr_matrix(
            (
                np.ones(len(codes), dtype=np.int8),
                (np.arange(len(codes), dtype=np.int64), codes.astype(np.int64)),
            ),
            shape=(len(codes), len(unique)),
        )
        x = (x @ projector).tocsr()
    return x, np.asarray(unique, dtype=str)


def enforce_integer_counts(matrix: sparse.spmatrix, context: str) -> sparse.csr_matrix:
    x = sparse.csr_matrix(matrix)
    x.sum_duplicates()
    x.eliminate_zeros()
    if x.data.size:
        if not np.isfinite(x.data).all() or float(x.data.min()) < 0:
            raise ValueError(f"{context}: count matrix contains negative/non-finite values")
        if not np.equal(x.data, np.rint(x.data)).all():
            raise ValueError(f"{context}: count matrix is not integer-valued")
        if float(x.data.max()) > np.iinfo(np.int32).max:
            raise OverflowError(f"{context}: count exceeds int32")
    return x.astype(np.int32, copy=False)


def metadata_obs(row: pd.Series, barcodes: Iterable[object]) -> pd.DataFrame:
    original = pd.Index([str(value).strip() for value in barcodes], dtype="object")
    if original.has_duplicates or (original == "").any():
        raise ValueError(f"{row['series']} {row['sample']}: invalid/duplicate barcodes")
    sample = str(row["sample"])
    series = str(row["series"])
    obs_names = pd.Index([f"{series}|{sample}|{barcode}" for barcode in original])
    if obs_names.has_duplicates:
        raise ValueError(f"{series} {sample}: constructed obs_names are duplicated")
    obs = pd.DataFrame(index=obs_names)
    obs["sample"] = sample
    obs["series"] = series
    obs["status"] = str(row["status"])
    obs["patient"] = str(row["patient"])
    obs["tissue_class"] = str(row["tissue_class"])
    obs["original_barcode"] = original.to_numpy()
    return obs


def make_adata(
    matrix_cells_by_genes: sparse.spmatrix,
    genes: Iterable[object],
    barcodes: Iterable[object],
    row: pd.Series,
    source_format: str,
) -> ad.AnnData:
    barcodes = list(barcodes)
    genes = np.asarray(list(genes) if not isinstance(genes, np.ndarray) else genes, dtype=str)
    x = enforce_integer_counts(matrix_cells_by_genes, f"{row['series']} {row['sample']}")
    if x.shape != (len(barcodes), len(genes)):
        raise ValueError(
            f"Matrix dimensions disagree for {row['series']} {row['sample']}: "
            f"matrix={x.shape}, barcodes={len(barcodes)}, genes={len(genes)}"
        )
    x, unique_genes = collapse_duplicate_genes(x, genes)
    if x.shape[0] != len(barcodes):
        raise ValueError(f"Cell/barcode dimension mismatch for {row['series']} {row['sample']}")
    obs = metadata_obs(row, barcodes)
    var = pd.DataFrame(index=pd.Index(unique_genes, name="gene"))
    var["gene_symbol"] = unique_genes
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.uns["sample_provenance"] = {
        "series": str(row["series"]),
        "sample": str(row["sample"]),
        "sample_title": str(row["sample_title"]),
        "manifest_row_sha256": str(row["manifest_row_sha256"]),
        "source_matrix_path": str(row["source_matrix_path"]),
        "source_format": source_format,
        "raw_integer_counts": True,
        "downsampled": False,
        "seed": SEED,
    }
    return adata


def gzip_text_from_tar(archive: tarfile.TarFile, member: str) -> io.TextIOWrapper:
    raw = archive.extractfile(member)
    if raw is None:
        raise FileNotFoundError(f"Cannot extract {member}")
    return io.TextIOWrapper(gzip.GzipFile(fileobj=raw), encoding="utf-8")


def read_tsv_member(archive: tarfile.TarFile, member: str) -> list[list[str]]:
    with gzip_text_from_tar(archive, member) as handle:
        return [row for row in csv.reader(handle, delimiter="\t") if row]


def choose_members(archive: tarfile.TarFile, row: pd.Series) -> tuple[str, list[str]]:
    files = [member.name for member in archive.getmembers() if member.isfile()]
    gsm = str(row["sample"])
    matched = [name for name in files if gsm in Path(name).name]
    if not matched:
        raise FileNotFoundError(f"No RAW member matches {row['series']} {gsm}")

    # Submitter-provided count tables are already restricted to called cells.
    dense = [
        name
        for name in matched
        if name.endswith((".txt.gz", ".csv.gz"))
        and ("CountMatrix" in name or name.endswith(".csv.gz"))
    ]
    if str(row["series"]) == "GSE150290":
        dense = [name for name in matched if name.endswith(".txt.gz")]
    if dense:
        return "dense_gene_by_cell", [sorted(dense)[0]]

    matrices = [name for name in matched if name.endswith("_matrix.mtx.gz")]
    for matrix in sorted(matrices):
        stem = matrix[: -len("_matrix.mtx.gz")]
        barcodes = stem + "_barcodes.tsv.gz"
        features = stem + "_features.tsv.gz"
        if barcodes in files and features in files:
            return "10x_mtx_triplet", [matrix, features, barcodes]
    raise FileNotFoundError(
        f"No supported called-cell count table/10x triplet for {row['series']} {gsm}; "
        f"matched={matched}"
    )


def read_dense_member(
    archive: tarfile.TarFile, member: str, row: pd.Series
) -> ad.AnnData:
    delimiter = "," if member.endswith(".csv.gz") else "\t"
    with gzip_text_from_tar(archive, member) as handle:
        frame = pd.read_csv(handle, sep=delimiter, index_col=0, low_memory=False)
    genes = frame.index.astype(str).to_numpy()
    barcodes = frame.columns.astype(str).to_numpy()
    values = frame.to_numpy(dtype=np.float32, copy=False)
    matrix = sparse.csr_matrix(values.T)
    del frame, values
    return make_adata(matrix, genes, barcodes, row, f"dense_count_table:{member}")


def read_10x_members(
    archive: tarfile.TarFile, members: list[str], row: pd.Series
) -> ad.AnnData:
    matrix_member, feature_member, barcode_member = members
    raw = archive.extractfile(matrix_member)
    if raw is None:
        raise FileNotFoundError(matrix_member)
    with gzip.GzipFile(fileobj=raw) as handle:
        matrix = mmread(handle).tocsr()
    feature_rows = read_tsv_member(archive, feature_member)
    barcode_rows = read_tsv_member(archive, barcode_member)
    genes = [fields[1] if len(fields) > 1 and fields[1] else fields[0] for fields in feature_rows]
    barcodes = [fields[0] for fields in barcode_rows]
    if matrix.shape != (len(genes), len(barcodes)):
        raise ValueError(
            f"10x dimensions disagree for {row['series']} {row['sample']}: "
            f"matrix={matrix.shape}, genes={len(genes)}, barcodes={len(barcodes)}"
        )
    return make_adata(
        matrix.T.tocsr(), genes, barcodes, row, f"10x_mtx_triplet:{matrix_member}"
    )


def load_archived_sample(archive: tarfile.TarFile, row: pd.Series) -> ad.AnnData:
    source_format, members = choose_members(archive, row)
    if source_format == "10x_mtx_triplet":
        return read_10x_members(archive, members, row)
    return read_dense_member(archive, members[0], row)


def normalize_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def load_gse234_shared(data_dir: Path) -> tuple[sparse.csr_matrix, np.ndarray, pd.DataFrame]:
    paths = {
        "matrix": data_dir / "GSE234129_count_matrix.mtx.gz",
        "features": data_dir / "GSE234129_features.tsv.gz",
        "barcodes": data_dir / "GSE234129_barcodes.tsv.gz",
        "meta": data_dir / "GSE234129_meta.tsv.gz",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"GSE234129 shared files missing: {missing}")
    with gzip.open(paths["matrix"], "rb") as handle:
        matrix = mmread(handle).tocsr()
    features = pd.read_csv(paths["features"], sep="\t", header=None, dtype=str)
    barcodes = pd.read_csv(paths["barcodes"], sep="\t", header=None, dtype=str)[0]
    meta = pd.read_csv(paths["meta"], sep="\t", dtype=str, keep_default_na=False)
    if "cell_barcodes" not in meta or "sample" not in meta:
        raise KeyError("GSE234129 meta must contain cell_barcodes and sample")
    if meta["cell_barcodes"].duplicated().any() or barcodes.duplicated().any():
        raise ValueError("GSE234129 cell barcodes are not unique")
    if matrix.shape != (len(features), len(barcodes)):
        raise ValueError(
            f"GSE234129 shared dimensions disagree: matrix={matrix.shape}, "
            f"features={len(features)}, barcodes={len(barcodes)}"
        )
    meta = meta.set_index("cell_barcodes").reindex(barcodes)
    if meta["sample"].eq("").any() or meta["sample"].isna().any():
        raise ValueError("GSE234129 metadata does not align one-to-one to barcodes")
    gene_col = 1 if features.shape[1] > 1 else 0
    genes = features.iloc[:, gene_col].astype(str).to_numpy()
    return matrix.T.tocsr(), genes, meta


def gse234_row_mask(meta: pd.DataFrame, row: pd.Series) -> np.ndarray:
    candidates = {
        normalize_key(row["sample_title"]),
        normalize_key(row["sample"]),
    }
    candidates.discard("")
    keys = meta["sample"].map(normalize_key)
    matched_values = sorted(set(meta.loc[keys.isin(candidates), "sample"]))
    if len(matched_values) != 1:
        raise ValueError(
            f"GSE234129 {row['sample']}: expected exactly one shared-meta sample for "
            f"sample_title={row['sample_title']!r}; matched={matched_values}"
        )
    return meta["sample"].eq(matched_values[0]).to_numpy()


def checkpoint_path(checkpoint_dir: Path, row: pd.Series) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{row['series']}__{row['sample']}")
    return checkpoint_dir / f"{safe}.h5ad"


def checkpoint_is_current(path: Path, row: pd.Series) -> bool:
    if not path.is_file():
        return False
    try:
        prior = ad.read_h5ad(path, backed="r")
        provenance = prior.uns.get("sample_provenance", {})
        okay = (
            str(provenance.get("manifest_row_sha256", ""))
            == str(row["manifest_row_sha256"])
            and prior.n_obs > 0
            and prior.n_vars > 0
            and prior.obs_names.is_unique
        )
        prior.file.close()
        return bool(okay)
    except Exception:
        return False


def validate_merged(adata: ad.AnnData, used: pd.DataFrame) -> dict[str, object]:
    required_obs = [
        "sample",
        "series",
        "status",
        "patient",
        "tissue_class",
        "original_barcode",
    ]
    missing = [column for column in required_obs if column not in adata.obs]
    if missing:
        raise KeyError(f"Merged h5ad lacks obs columns: {missing}")
    if not adata.obs_names.is_unique:
        raise ValueError("Merged obs_names are not unique")
    if adata.raw is not None or adata.layers:
        raise ValueError("Merge output must contain only raw X counts, without raw/layers")
    if not sparse.isspmatrix_csr(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    x = enforce_integer_counts(adata.X, "merged")
    adata.X = x
    observed = set(zip(adata.obs["series"].astype(str), adata.obs["sample"].astype(str)))
    expected = set(zip(used["series"], used["sample"]))
    if observed != expected:
        raise ValueError(
            f"Merged sample keys disagree with manifest: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    if set(adata.obs["status"].astype(str)) - {"normal", "tumor"}:
        raise ValueError("Merged status contains values other than normal/tumor")
    expected_prefix = (
        adata.obs["series"].astype(str)
        + "|"
        + adata.obs["sample"].astype(str)
        + "|"
        + adata.obs["original_barcode"].astype(str)
    )
    if not np.array_equal(expected_prefix.to_numpy(), adata.obs_names.to_numpy()):
        raise ValueError("obs_names do not follow series|sample|original_barcode")
    return {
        "all_passed": True,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_nonzero": int(adata.X.nnz),
        "n_samples": len(observed),
        "n_series": len(set(adata.obs["series"].astype(str))),
        "status_counts": adata.obs["status"].astype(str).value_counts().to_dict(),
        "raw_integer_counts": True,
        "downsampled": False,
        "obs_names_unique": True,
    }


def intersect_checkpoint_genes(paths: list[Path]) -> tuple[set[str], list[int]]:
    intersection: set[str] | None = None
    counts: list[int] = []
    for path in paths:
        item = ad.read_h5ad(path, backed="r")
        genes = set(item.var_names.astype(str))
        counts.append(len(genes))
        intersection = genes if intersection is None else intersection & genes
        item.file.close()
    return (intersection or set()), counts


def inner_concat_on_disk(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + ".part.h5ad")
    if tmp.exists():
        tmp.unlink()
    concat_on_disk(
        paths,
        tmp,
        axis=0,
        join="inner",
        merge="first",
        max_loaded_elems=100_000_000,
    )
    os.replace(tmp, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--h5ad-dir", type=Path, default=DEFAULT_H5AD_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse only per-sample checkpoints whose canonical row hash matches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    started = time.time()
    output = args.h5ad_dir / "adata_merge.h5ad"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite completed merge: {output}")
    used = read_manifest(args.manifest)
    args.h5ad_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.h5ad_dir / "sample_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for series in SCRNA_SERIES:
        subset = used[used["series"].eq(series)]
        if subset.empty or series == "GSE234129":
            continue
        tar_path = args.data_root / series / f"{series}_RAW.tar"
        if not tar_path.is_file():
            raise FileNotFoundError(tar_path)
        with tarfile.open(tar_path, "r:*") as archive:
            for _, row in subset.iterrows():
                checkpoint = checkpoint_path(checkpoint_dir, row)
                if args.resume and checkpoint_is_current(checkpoint, row):
                    prior = ad.read_h5ad(checkpoint, backed="r")
                    provenance = prior.uns.get("sample_provenance", {})
                    audit_rows.append(
                        {
                            "series": series,
                            "sample": row["sample"],
                            "state": "reused_checkpoint",
                            "n_cells": prior.n_obs,
                            "n_genes": prior.n_vars,
                            "checkpoint": str(checkpoint.relative_to(WF)),
                            "manifest_row_sha256": row["manifest_row_sha256"],
                            "source_format": provenance.get("source_format", ""),
                            "source_matrix_path": row["source_matrix_path"],
                        }
                    )
                    prior.file.close()
                    continue
                if checkpoint.exists():
                    checkpoint.unlink()
                sample_started = time.time()
                try:
                    sample_adata = load_archived_sample(archive, row)
                    atomic_write_h5ad(sample_adata, checkpoint)
                    provenance = sample_adata.uns["sample_provenance"]
                    audit_rows.append(
                        {
                            "series": series,
                            "sample": row["sample"],
                            "state": "completed",
                            "n_cells": sample_adata.n_obs,
                            "n_genes": sample_adata.n_vars,
                            "checkpoint": str(checkpoint.relative_to(WF)),
                            "manifest_row_sha256": row["manifest_row_sha256"],
                            "source_format": provenance["source_format"],
                            "source_matrix_path": row["source_matrix_path"],
                            "elapsed_seconds": time.time() - sample_started,
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "series": series,
                            "sample": row["sample"],
                            "error_type": type(exc).__name__,
                            "error": repr(exc),
                        }
                    )

    subset234 = used[used["series"].eq("GSE234129")]
    pending234 = [
        row
        for _, row in subset234.iterrows()
        if not (args.resume and checkpoint_is_current(checkpoint_path(checkpoint_dir, row), row))
    ]
    for _, row in subset234.iterrows():
        checkpoint = checkpoint_path(checkpoint_dir, row)
        if args.resume and checkpoint_is_current(checkpoint, row):
            prior = ad.read_h5ad(checkpoint, backed="r")
            provenance = prior.uns.get("sample_provenance", {})
            audit_rows.append(
                {
                    "series": "GSE234129",
                    "sample": row["sample"],
                    "state": "reused_checkpoint",
                    "n_cells": prior.n_obs,
                    "n_genes": prior.n_vars,
                    "checkpoint": str(checkpoint.relative_to(WF)),
                    "manifest_row_sha256": row["manifest_row_sha256"],
                    "source_format": provenance.get("source_format", ""),
                    "source_matrix_path": row["source_matrix_path"],
                }
            )
            prior.file.close()
    if pending234:
        shared_started = time.time()
        try:
            shared_x, shared_genes, shared_meta = load_gse234_shared(
                args.data_root / "GSE234129"
            )
            for row in pending234:
                checkpoint = checkpoint_path(checkpoint_dir, row)
                if checkpoint.exists():
                    checkpoint.unlink()
                sample_started = time.time()
                try:
                    mask = gse234_row_mask(shared_meta, row)
                    barcodes = shared_meta.index.to_numpy()[mask]
                    sample_adata = make_adata(
                        shared_x[mask],
                        shared_genes,
                        barcodes,
                        row,
                        "shared_10x_mtx_subset_by_GSE234129_meta.sample",
                    )
                    atomic_write_h5ad(sample_adata, checkpoint)
                    provenance = sample_adata.uns["sample_provenance"]
                    audit_rows.append(
                        {
                            "series": "GSE234129",
                            "sample": row["sample"],
                            "state": "completed",
                            "n_cells": sample_adata.n_obs,
                            "n_genes": sample_adata.n_vars,
                            "checkpoint": str(checkpoint.relative_to(WF)),
                            "manifest_row_sha256": row["manifest_row_sha256"],
                            "source_format": provenance["source_format"],
                            "source_matrix_path": row["source_matrix_path"],
                            "elapsed_seconds": time.time() - sample_started,
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "series": "GSE234129",
                            "sample": row["sample"],
                            "error_type": type(exc).__name__,
                            "error": repr(exc),
                        }
                    )
            audit_rows.append(
                {
                    "series": "GSE234129",
                    "sample": "__shared_matrix__",
                    "state": "shared_matrix_loaded",
                    "n_cells": shared_x.shape[0],
                    "n_genes": shared_x.shape[1],
                    "elapsed_seconds": time.time() - shared_started,
                }
            )
        except Exception as exc:
            for row in pending234:
                failures.append(
                    {
                        "series": "GSE234129",
                        "sample": row["sample"],
                        "error_type": type(exc).__name__,
                        "error": repr(exc),
                    }
                )

    audit = pd.DataFrame(audit_rows)
    failed = pd.DataFrame(
        failures, columns=["series", "sample", "error_type", "error"]
    ).drop_duplicates()
    atomic_write_csv(audit, args.table_dir / "sample_merge_audit.csv")
    atomic_write_csv(failed, args.table_dir / "failed_samples.csv")
    if not failed.empty:
        raise RuntimeError(
            f"{len(failed)} used samples failed; see {args.table_dir / 'failed_samples.csv'}"
        )

    ordered_checkpoints = [checkpoint_path(checkpoint_dir, row) for _, row in used.iterrows()]
    missing_checkpoints = [str(path) for path in ordered_checkpoints if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(f"Missing used-sample checkpoints: {missing_checkpoints}")
    gene_join_audit: list[dict[str, object]] = []
    series_dir = args.h5ad_dir / "series_checkpoints"
    series_paths: list[Path] = []
    for series in SCRNA_SERIES:
        rows = used[used["series"].eq(series)]
        paths = [checkpoint_path(checkpoint_dir, row) for _, row in rows.iterrows()]
        if not paths:
            raise RuntimeError(f"Curated XLSX v5 has no used sample for required series {series}")
        intersection, input_gene_counts = intersect_checkpoint_genes(paths)
        if not intersection:
            raise RuntimeError(f"Within-series gene intersection is empty for {series}")
        series_path = series_dir / f"{series}.h5ad"
        inner_concat_on_disk(paths, series_path)
        check = ad.read_h5ad(series_path, backed="r")
        if check.n_vars != len(intersection):
            raise RuntimeError(
                f"{series} inner join returned {check.n_vars}, expected {len(intersection)} genes"
            )
        n_cells = check.n_obs
        check.file.close()
        series_paths.append(series_path)
        gene_join_audit.append(
            {
                "join_level": "within_series_samples",
                "series": series,
                "n_inputs": len(paths),
                "input_gene_count_min": min(input_gene_counts),
                "input_gene_count_max": max(input_gene_counts),
                "intersection_gene_count": len(intersection),
                "output_cells": n_cells,
                "join": "inner",
                "output": str(series_path.relative_to(WF)),
            }
        )

    global_intersection, series_gene_counts = intersect_checkpoint_genes(series_paths)
    if not global_intersection:
        raise RuntimeError("Five-series gene intersection is empty")
    tmp_output = output.with_name(output.stem + ".concat-part.h5ad")
    inner_concat_on_disk(series_paths, tmp_output)
    merged = ad.read_h5ad(tmp_output)
    if merged.n_vars != len(global_intersection):
        raise RuntimeError(
            f"Between-series inner join returned {merged.n_vars}, "
            f"expected {len(global_intersection)} genes"
        )
    gene_join_audit.append(
        {
            "join_level": "between_five_series",
            "series": "ALL_FIVE",
            "n_inputs": len(series_paths),
            "input_gene_count_min": min(series_gene_counts),
            "input_gene_count_max": max(series_gene_counts),
            "intersection_gene_count": len(global_intersection),
            "output_cells": merged.n_obs,
            "join": "inner",
            "output": str(output.relative_to(WF)),
        }
    )
    atomic_write_csv(pd.DataFrame(gene_join_audit), args.table_dir / "gene_join_hierarchy_audit.csv")
    validation = validate_merged(merged, used)
    merged.uns["merge_parameters"] = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "included_series": list(SCRNA_SERIES),
        "included_rows": int(len(used)),
        "join": "inner_within_series_then_inner_between_five_series_by_gene_symbol",
        "duplicate_gene_rule": "sum_within_sample",
        "cell_id_rule": "series|sample|original_barcode",
        "raw_integer_counts": True,
        "normalization": False,
        "downsampled": False,
        "seed": SEED,
    }
    atomic_write_h5ad(merged, output)
    tmp_output.unlink()
    validation.update(
        {
            "manifest_sha256": sha256(args.manifest),
            "output_sha256": sha256(output),
            "elapsed_seconds": time.time() - started,
        }
    )
    atomic_write_json(validation, args.table_dir / "merge_validation.json")
    parameters = {
        "seed": SEED,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "data_root": str(args.data_root),
        "output": str(output),
        "resume": bool(args.resume),
        "raw_integer_counts": True,
        "downsampled": False,
        "normalization": False,
        "log_transform": False,
        "join": "inner_within_series_then_inner_between_series",
        "duplicate_gene_rule": "sum",
        "n_used_samples": int(len(used)),
    }
    atomic_write_json(parameters, args.table_dir / "merge_parameters.json")
    atomic_write_json(package_versions(), args.table_dir / "package_versions.json")
    source_rows = []
    for series in SCRNA_SERIES:
        data_dir = args.data_root / series
        for path in sorted(data_dir.glob("*")):
            if path.is_file():
                stat = path.stat()
                source_rows.append(
                    {
                        "series": series,
                        "path": str(path),
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
    atomic_write_csv(pd.DataFrame(source_rows), args.table_dir / "source_file_inventory.csv")
    (args.table_dir / "readme.txt").write_text(
        "adata_merge.h5ad contains only curated-XLSX-v5 used scRNA-seq samples.\n"
        "X is non-negative raw integer counts; no normalization, log transform, "
        "gene selection, cell filtering, or downsampling was performed.\n"
        "Genes are inner-joined across samples within each series, then inner-joined "
        "across the five series; see gene_join_hierarchy_audit.csv.\n"
        "Per-sample checkpoints are reusable only when the complete curated-manifest "
        "manifest-row SHA256 is unchanged. Any sample read failure stops the merge.\n"
    )


if __name__ == "__main__":
    main()

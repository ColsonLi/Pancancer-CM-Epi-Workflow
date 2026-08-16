#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[4]
MANIFEST = (
    ROOT
    / "epi-cm-core-workflow/tables/01-celltype_integration_clustering/01-sample-manifest/all_datasets_all_samples_inclusion_v2.csv"
)
TABLE_DIR = ROOT / "epi-cm-core-workflow/tables/01-celltype_integration_clustering/02-merge-metadata"
H5AD_DIR = ROOT / "epi-cm-core-workflow/h5ad/01-celltype_integration_clustering/02-merge-metadata"
H5AD_OUT = H5AD_DIR / "adata_merge.h5ad"

GC_BASE = Path(
    "/mnt/disk18t/lr_xcy/riku/external_atlas_for_cm-epi_validation/Gastric_cancer"
)
ARCHIVES = {
    series: GC_BASE / series / f"{series}_RAW.tar"
    for series in ["GSE150290", "GSE163558", "GSE183904", "GSE270680"]
}
SERIES_ORDER = ["GSE150290", "GSE163558", "GSE183904", "GSE234129", "GSE270680"]

gene_audit_rows: list[dict[str, object]] = []
duplicate_rows: list[dict[str, object]] = []
input_rows: list[dict[str, object]] = []
sample_rows: list[dict[str, object]] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_new(path: Path, rows: list[dict[str, object]], columns: list[str] | None = None) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if columns is None:
            raise ValueError(f"No rows and no columns supplied for {path}")
        fields = columns
    else:
        fields = columns or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def feature_kind(gene_ids: list[str], gene_symbols: list[str]) -> str:
    if gene_ids == gene_symbols:
        return "gene_symbol_only"
    ensembl_fraction = np.mean([bool(re.match(r"^ENSG\d+", value)) for value in gene_ids])
    return "ensembl_id_plus_gene_symbol" if ensembl_fraction > 0.9 else "id_plus_gene_symbol"


def aggregate_duplicate_symbols(
    matrix: sparse.spmatrix,
    gene_ids: list[str],
    gene_symbols: list[str],
    *,
    series: str,
    sample: str,
) -> tuple[sparse.csr_matrix, list[str]]:
    if matrix.shape[1] != len(gene_symbols) or len(gene_ids) != len(gene_symbols):
        raise ValueError(
            f"{series}/{sample}: matrix vars={matrix.shape[1]}, ids={len(gene_ids)}, symbols={len(gene_symbols)}"
        )
    clean_symbols = [symbol.strip() if symbol.strip() else gene_id for gene_id, symbol in zip(gene_ids, gene_symbols)]
    codes, unique = pd.factorize(pd.Index(clean_symbols), sort=False)
    counts = np.bincount(codes, minlength=len(unique))
    duplicated = {str(unique[i]): int(counts[i]) for i in np.flatnonzero(counts > 1)}

    ids_by_symbol: dict[str, list[str]] = defaultdict(list)
    for gene_id, symbol in zip(gene_ids, clean_symbols):
        ids_by_symbol[symbol].append(gene_id)
    for symbol, n_features in duplicated.items():
        duplicate_rows.append(
            {
                "series": series,
                "sample": sample,
                "gene_symbol": symbol,
                "n_source_features": n_features,
                "source_gene_ids": ";".join(ids_by_symbol[symbol]),
                "aggregation": "sum_counts",
            }
        )

    matrix = matrix.tocsr().astype(np.float32)
    if len(unique) != len(clean_symbols):
        projection = sparse.csr_matrix(
            (np.ones(len(codes), dtype=np.float32), (np.arange(len(codes)), codes)),
            shape=(len(codes), len(unique)),
        )
        matrix = (matrix @ projection).tocsr()

    gene_audit_rows.append(
        {
            "series": series,
            "sample": sample,
            "identifier_schema": feature_kind(gene_ids, gene_symbols),
            "n_features_input": len(gene_symbols),
            "n_unique_gene_symbols": len(unique),
            "n_duplicate_symbol_groups": len(duplicated),
            "n_features_collapsed_by_sum": len(gene_symbols) - len(unique),
            "var_names_make_unique_called": False,
        }
    )
    return matrix, [str(item) for item in unique]


def build_adata(
    matrix: sparse.spmatrix,
    gene_ids: list[str],
    gene_symbols: list[str],
    barcodes: list[str],
    manifest_row: pd.Series,
    selected_source: str,
) -> AnnData:
    series = str(manifest_row["series"])
    sample = str(manifest_row["sample"])
    if matrix.shape[0] != len(barcodes):
        raise ValueError(f"{series}/{sample}: cells={matrix.shape[0]}, barcodes={len(barcodes)}")
    if len(set(barcodes)) != len(barcodes):
        raise ValueError(f"{series}/{sample}: duplicated source barcodes")
    matrix, symbols = aggregate_duplicate_symbols(
        matrix, gene_ids, gene_symbols, series=series, sample=sample
    )
    obs = pd.DataFrame(index=pd.Index([f"{series}|{sample}|{bc}" for bc in barcodes], name="cell_id"))
    obs["original_barcode"] = barcodes
    for column in [
        "sample",
        "sample_title",
        "series",
        "patient",
        "stage",
        "tissue_class",
        "status",
        "treatment_context",
        "source_matrix_path",
    ]:
        obs[column] = str(manifest_row[column]) if pd.notna(manifest_row[column]) else ""
    obs["source_matrix_path_selected"] = selected_source
    adata = AnnData(matrix, obs=obs, var=pd.DataFrame(index=pd.Index(symbols, name="gene_symbol")))
    if not adata.var_names.is_unique:
        raise RuntimeError(f"{series}/{sample}: symbol aggregation did not yield unique var_names")
    sample_rows.append(
        {
            "series": series,
            "sample": sample,
            "sample_title": manifest_row["sample_title"],
            "patient": manifest_row["patient"],
            "status": manifest_row["status"],
            "n_cells": adata.n_obs,
            "n_genes_after_explicit_symbol_aggregation": adata.n_vars,
            "nnz": adata.X.nnz,
            "sum_counts": float(adata.X.sum()),
            "selected_source": selected_source,
        }
    )
    input_rows.append(
        {
            "series": series,
            "sample": sample,
            "selected_source": selected_source,
            "manifest_source_candidates": manifest_row["source_matrix_path"],
            "selection_reason": (
                "author-provided cell-by-gene filtered count table; nested raw droplet matrix not used"
                if series == "GSE150290"
                else "author-provided per-sample count matrix"
                if series in {"GSE163558", "GSE183904", "GSE270680"}
                else "combined author count matrix filtered by exact cell-level sample metadata"
            ),
        }
    )
    return adata


def read_dense_member(
    archive: Path,
    member: str,
    delimiter: str,
) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    with tarfile.open(archive, "r") as tf:
        raw = tf.extractfile(member)
        if raw is None:
            raise FileNotFoundError(f"{archive}::{member}")
        with gzip.GzipFile(fileobj=raw) as gz, io.TextIOWrapper(gz, encoding="utf-8") as text:
            first_line = text.readline()
            header = next(csv.reader([first_line], delimiter=delimiter))
            barcodes = header[1:] if header and header[0] == "" else header
            names = ["gene"] + barcodes
            matrices = []
            genes: list[str] = []
            for chunk in pd.read_csv(
                text,
                sep=delimiter,
                header=None,
                names=names,
                chunksize=256,
                quotechar='"',
            ):
                genes.extend(chunk.iloc[:, 0].astype(str).tolist())
                values = chunk.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
                matrices.append(sparse.csr_matrix(values))
    gene_by_cell = sparse.vstack(matrices, format="csr")
    return gene_by_cell.T.tocsr(), genes, barcodes


def read_10x_tar(
    archive: Path, sample: str
) -> tuple[sparse.csr_matrix, list[str], list[str], list[str], str]:
    with tarfile.open(archive, "r") as tf:
        names = [name for name in tf.getnames() if sample in Path(name).name]
        features_member = next(name for name in names if "features.tsv.gz" in name)
        barcodes_member = next(name for name in names if "barcodes.tsv.gz" in name)
        matrix_member = next(name for name in names if "matrix.mtx.gz" in name)

        feature_raw = tf.extractfile(features_member)
        barcode_raw = tf.extractfile(barcodes_member)
        matrix_raw = tf.extractfile(matrix_member)
        if feature_raw is None or barcode_raw is None or matrix_raw is None:
            raise FileNotFoundError(f"Missing 10x member for {sample} in {archive}")
        with gzip.GzipFile(fileobj=feature_raw) as handle:
            features = [line.decode().rstrip("\n").split("\t") for line in handle]
        with gzip.GzipFile(fileobj=barcode_raw) as handle:
            barcodes = [line.decode().strip() for line in handle]
        with gzip.GzipFile(fileobj=matrix_raw) as handle:
            matrix = mmread(handle).T.tocsr().astype(np.float32)
    gene_ids = [row[0] for row in features]
    gene_symbols = [row[1] if len(row) > 1 else row[0] for row in features]
    selected = ";".join(
        f"{archive}::{member}" for member in [matrix_member, barcodes_member, features_member]
    )
    return matrix, gene_ids, gene_symbols, barcodes, selected


def read_gse234129() -> tuple[sparse.csr_matrix, list[str], list[str], list[str], pd.DataFrame, str]:
    folder = GC_BASE / "GSE234129"
    matrix_path = folder / "GSE234129_count_matrix.mtx.gz"
    barcode_path = folder / "GSE234129_barcodes.tsv.gz"
    feature_path = folder / "GSE234129_features.tsv.gz"
    meta_path = folder / "GSE234129_meta.tsv.gz"
    with gzip.open(matrix_path, "rb") as handle:
        matrix = mmread(handle).T.tocsr().astype(np.float32)
    barcodes = pd.read_csv(barcode_path, sep="\t", header=None)[0].astype(str).tolist()
    symbols = pd.read_csv(feature_path, sep="\t", header=None)[0].astype(str).tolist()
    meta = pd.read_csv(meta_path, sep="\t", dtype=str)
    if barcodes != meta["cell_barcodes"].tolist():
        raise RuntimeError("GSE234129 barcode file and metadata are not in identical order")
    if matrix.shape != (len(barcodes), len(symbols)):
        raise RuntimeError("GSE234129 matrix dimensions do not match barcode/feature files")
    selected = ";".join(map(str, [matrix_path, barcode_path, feature_path, meta_path]))
    return matrix, symbols, symbols, barcodes, meta, selected


def main() -> None:
    if H5AD_OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {H5AD_OUT}")
    if TABLE_DIR.exists() and any(TABLE_DIR.iterdir()):
        raise FileExistsError(f"Refusing to mix with existing merge audit artifacts: {TABLE_DIR}")

    manifest = pd.read_csv(MANIFEST)
    used = manifest[(manifest["used"] == True) & (manifest["modality"] == "scRNA-seq")].copy()  # noqa: E712
    if len(used) != 57:
        raise RuntimeError(f"Expected 57 v2 scRNA-seq samples, found {len(used)}")
    if set(used["series"]) != set(SERIES_ORDER):
        raise RuntimeError(f"Unexpected used series: {sorted(set(used['series']))}")

    gse234_cache = read_gse234129()
    series_objects: dict[str, AnnData] = {}
    retention_rows: list[dict[str, object]] = []

    for series in SERIES_ORDER:
        sample_objects = []
        subset = used[used["series"] == series].sort_values("sample")
        for _, manifest_row in subset.iterrows():
            sample = str(manifest_row["sample"])
            print(f"reading {series}/{sample}", flush=True)
            if series == "GSE150290":
                with tarfile.open(ARCHIVES[series], "r") as tf:
                    member = next(
                        name for name in tf.getnames() if sample in Path(name).name and name.endswith(".txt.gz")
                    )
                matrix, symbols, barcodes = read_dense_member(ARCHIVES[series], member, "\t")
                gene_ids = symbols
                selected = f"{ARCHIVES[series]}::{member}"
            elif series in {"GSE163558", "GSE270680"}:
                matrix, gene_ids, symbols, barcodes, selected = read_10x_tar(
                    ARCHIVES[series], sample
                )
            elif series == "GSE183904":
                with tarfile.open(ARCHIVES[series], "r") as tf:
                    member = next(
                        name for name in tf.getnames() if sample in Path(name).name and name.endswith(".csv.gz")
                    )
                matrix, symbols, barcodes = read_dense_member(ARCHIVES[series], member, ",")
                gene_ids = symbols
                selected = f"{ARCHIVES[series]}::{member}"
            elif series == "GSE234129":
                full_matrix, gene_ids, symbols, all_barcodes, meta, selected = gse234_cache
                mask = meta["sample"].eq(str(manifest_row["sample_title"])).to_numpy()
                if not mask.any():
                    raise RuntimeError(f"No GSE234129 cells found for {manifest_row['sample_title']}")
                matrix = full_matrix[mask].copy()
                barcodes = list(np.asarray(all_barcodes)[mask])
            else:
                raise AssertionError(series)
            sample_objects.append(
                build_adata(matrix, gene_ids, symbols, barcodes, manifest_row, selected)
            )

        series_input_genes = [obj.n_vars for obj in sample_objects]
        series_adata = ad.concat(
            sample_objects,
            axis=0,
            join="inner",
            merge="same",
            index_unique=None,
        )
        series_adata.X = series_adata.X.tocsr().astype(np.float32)
        series_objects[series] = series_adata
        retention_rows.append(
            {
                "level": "within_series_inner_join",
                "group": series,
                "n_samples_or_series": len(sample_objects),
                "min_input_genes": min(series_input_genes),
                "max_input_genes": max(series_input_genes),
                "n_genes_retained": series_adata.n_vars,
                "n_cells": series_adata.n_obs,
            }
        )
        print(f"merged series {series}: {series_adata.shape}", flush=True)

    final = ad.concat(
        [series_objects[series] for series in SERIES_ORDER],
        axis=0,
        join="inner",
        merge="same",
        index_unique=None,
    )
    final.X = final.X.tocsr().astype(np.float32)
    retention_rows.append(
        {
            "level": "across_series_inner_join",
            "group": "all_scRNA-seq_series",
            "n_samples_or_series": len(SERIES_ORDER),
            "min_input_genes": min(obj.n_vars for obj in series_objects.values()),
            "max_input_genes": max(obj.n_vars for obj in series_objects.values()),
            "n_genes_retained": final.n_vars,
            "n_cells": final.n_obs,
        }
    )

    if not final.obs_names.is_unique or not final.var_names.is_unique:
        raise RuntimeError("Merged obs_names or var_names are not unique")
    if final.X.data.size and not np.allclose(final.X.data, np.round(final.X.data)):
        raise RuntimeError("Merged matrix contains non-integer values; expected raw counts")
    if set(final.obs["sample"]) != set(used["sample"]):
        raise RuntimeError("Merged sample set differs from v2 used sample set")

    final.uns["matrix_semantics"] = "raw unnormalized integer UMI/read counts in X"
    final.uns["gene_join"] = "explicit duplicate-symbol sum within sample, then inner join within and across series"
    final.uns["sample_manifest"] = str(MANIFEST)
    final.uns["n_used_scrna_samples"] = int(len(used))

    H5AD_DIR.mkdir(parents=True, exist_ok=True)
    temporary = H5AD_OUT.with_suffix(".tmp.h5ad")
    if temporary.exists():
        temporary.unlink()
    final.write_h5ad(temporary, compression="gzip")
    os.replace(temporary, H5AD_OUT)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_new(TABLE_DIR / "matrix_input_selection_by_sample.csv", input_rows)
    write_csv_new(TABLE_DIR / "sample_matrix_summary.csv", sample_rows)
    write_csv_new(TABLE_DIR / "gene_schema_audit_by_sample.csv", gene_audit_rows)
    write_csv_new(
        TABLE_DIR / "duplicate_gene_symbol_aggregation.csv",
        duplicate_rows,
        ["series", "sample", "gene_symbol", "n_source_features", "source_gene_ids", "aggregation"],
    )
    write_csv_new(TABLE_DIR / "gene_inner_join_retention.csv", retention_rows)
    write_csv_new(
        TABLE_DIR / "inner_join_gene_symbols.csv",
        [{"gene_symbol": gene} for gene in final.var_names],
    )
    provenance_columns = [
        "original_barcode",
        "sample",
        "sample_title",
        "series",
        "patient",
        "stage",
        "tissue_class",
        "status",
        "treatment_context",
        "source_matrix_path",
        "source_matrix_path_selected",
    ]
    final.obs[provenance_columns].to_csv(
        TABLE_DIR / "merged_cell_provenance.csv.gz", compression="gzip"
    )
    validation = [
        {"check": "manifest_used_scrna_samples", "passed": len(used) == 57, "value": len(used)},
        {"check": "all_manifest_samples_present", "passed": set(final.obs["sample"]) == set(used["sample"]), "value": final.obs["sample"].nunique()},
        {"check": "obs_names_unique", "passed": final.obs_names.is_unique, "value": final.n_obs},
        {"check": "var_names_unique_after_explicit_aggregation", "passed": final.var_names.is_unique, "value": final.n_vars},
        {"check": "raw_counts_are_integer_valued", "passed": bool(np.allclose(final.X.data, np.round(final.X.data))), "value": final.X.nnz},
        {"check": "required_provenance_columns_present", "passed": set(["sample", "series", "status", "original_barcode", "source_matrix_path"]).issubset(final.obs), "value": len(final.obs.columns)},
        {"check": "h5ad_written", "passed": H5AD_OUT.exists(), "value": H5AD_OUT.stat().st_size},
    ]
    write_csv_new(TABLE_DIR / "merge_validation.csv", validation)
    write_csv_new(
        TABLE_DIR / "adata_merge_artifact.csv",
        [
            {
                "file": str(H5AD_OUT),
                "size_bytes": H5AD_OUT.stat().st_size,
                "sha256": sha256(H5AD_OUT),
                "n_cells": final.n_obs,
                "n_genes": final.n_vars,
                "nnz": final.X.nnz,
                "n_samples": final.obs["sample"].nunique(),
                "n_series": final.obs["series"].nunique(),
            }
        ],
    )
    if not all(bool(row["passed"]) for row in validation):
        raise RuntimeError("One or more merge validation checks failed")
    print(f"wrote {H5AD_OUT}: {final.shape}, samples={final.obs['sample'].nunique()}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build canonical normalized/log Visium AnnData inputs for 14 PDAC samples.

The GEO ``*_spatial.tar.gz`` files are double-gzipped.  This runner reads the
second gzip stream in memory, joins the true Space Ranger pixel/array
coordinates to the filtered 10x count matrix, preserves integer counts in a
layer, then applies normalize_total(1e4) + log1p to ``X``.
"""

from __future__ import annotations

import gzip
import importlib.metadata
import io
import json
import random
import sys
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
INPUT_MANIFEST = (
    PROJECT_ROOT
    / "tables/04-spatial-validation-optional/01-input-staging/spatial_input_manifest.csv"
)
OUT_DIR = PROJECT_ROOT / "h5ad/04-spatial-validation-optional/02-build-spatial-h5ad"
TABLE_DIR = PROJECT_ROOT / "tables/04-spatial-validation-optional/02-build-spatial-h5ad"


def require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def read_inner_member(archive_path: Path, member_name: str) -> bytes:
    """Read a named member from gzip -> gzip -> POSIX tar without extraction."""
    with gzip.open(archive_path, "rb") as first_gzip:
        with tarfile.open(fileobj=first_gzip, mode="r:gz") as inner_tar:
            member = inner_tar.getmember(member_name)
            source = inner_tar.extractfile(member)
            if source is None:
                raise OSError(f"Cannot read {member_name} from {archive_path}")
            return source.read()


def collapse_duplicate_symbols(adata: ad.AnnData) -> tuple[ad.AnnData, int]:
    """Sum count columns sharing one gene symbol, preserving first-seen order."""
    names = pd.Index(adata.var_names.astype(str))
    unique_names = pd.Index(pd.unique(names), dtype=str)
    n_collapsed = int(len(names) - len(unique_names))
    if n_collapsed == 0:
        return adata, 0

    lookup = pd.Series(np.arange(len(unique_names), dtype=np.int64), index=unique_names)
    codes = lookup.loc[names].to_numpy(dtype=np.int64)
    mapper = sparse.csr_matrix(
        (
            np.ones(len(names), dtype=np.float32),
            (np.arange(len(names), dtype=np.int64), codes),
        ),
        shape=(len(names), len(unique_names)),
    )
    X = sparse.csr_matrix(adata.X) @ mapper
    var = pd.DataFrame(index=unique_names)
    gene_id_table = pd.DataFrame(
        {
            "gene_symbol": names.to_numpy(),
            "gene_id": adata.var["gene_ids"].astype(str).to_numpy(),
        }
    )
    collapsed_ids = gene_id_table.groupby("gene_symbol", sort=False)["gene_id"].agg(
        lambda values: ";".join(pd.unique(values).tolist())
    )
    var["gene_ids"] = collapsed_ids.reindex(unique_names).to_numpy()
    var["feature_types"] = "Gene Expression"
    out = ad.AnnData(X=X.tocsr(), obs=adata.obs.copy(), var=var)
    return out, n_collapsed


def load_coordinates(archive_path: Path) -> tuple[pd.DataFrame, dict]:
    positions = pd.read_csv(
        io.BytesIO(read_inner_member(archive_path, "spatial/tissue_positions.csv"))
    )
    expected = {
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    }
    if not expected.issubset(positions.columns):
        raise ValueError(f"Missing Visium coordinate columns in {archive_path}")
    if positions["barcode"].duplicated().any():
        raise ValueError(f"Duplicate coordinate barcodes in {archive_path}")
    scale = json.loads(
        read_inner_member(archive_path, "spatial/scalefactors_json.json").decode("utf-8")
    )
    return positions.set_index("barcode"), scale


def main() -> None:
    if not INPUT_MANIFEST.is_file():
        raise FileNotFoundError(INPUT_MANIFEST)
    audit_path = TABLE_DIR / "spatial_h5ad_build_audit.csv"
    validation_path = TABLE_DIR / "spatial_h5ad_validation.json"
    versions_path = TABLE_DIR / "package_versions.txt"
    readme_path = TABLE_DIR / "readme.txt"
    for path in [audit_path, validation_path, versions_path, readme_path]:
        require_new(path)

    manifest = pd.read_csv(INPUT_MANIFEST)
    if manifest["sample"].nunique() != 14 or len(manifest) != 28:
        raise ValueError("Expected exactly two staged inputs for each of 14 samples")
    piv = manifest.pivot(index="sample", columns="matrix_or_spatial_role", values="staged_path")
    expected_roles = {"filtered_feature_bc_matrix_h5", "spatial_tar_gz"}
    if not expected_roles.issubset(piv.columns):
        raise ValueError(f"Missing staged roles: {expected_roles - set(piv.columns)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for sample, paths in piv.sort_index().iterrows():
        sample = str(sample)
        matrix_path = Path(paths["filtered_feature_bc_matrix_h5"])
        archive_path = Path(paths["spatial_tar_gz"])
        out_path = OUT_DIR / f"{sample}_spatial_normalized_log.h5ad"
        require_new(out_path)

        raw = sc.read_10x_h5(matrix_path, gex_only=True)
        input_genes = raw.n_vars
        raw, n_duplicate_columns_collapsed = collapse_duplicate_symbols(raw)
        raw.X = sparse.csr_matrix(raw.X)
        if raw.X.nnz == 0 or np.min(raw.X.data) < 0 or not np.allclose(raw.X.data, np.rint(raw.X.data)):
            raise ValueError(f"{sample}: filtered 10x matrix is not nonnegative integer counts")

        positions, scalefactors = load_coordinates(archive_path)
        missing = pd.Index(raw.obs_names.astype(str)).difference(positions.index)
        if len(missing):
            raise ValueError(f"{sample}: {len(missing)} matrix barcodes lack spatial coordinates")
        pos = positions.loc[raw.obs_names.astype(str)].copy()
        original_barcodes = raw.obs_names.astype(str).to_numpy()
        raw.obs = pos
        raw.obs["sample"] = sample
        raw.obs["original_barcode"] = original_barcodes
        raw.obs_names = pd.Index([f"{sample}#{bc}" for bc in original_barcodes])
        raw.obs_names.name = None
        raw.obsm["spatial"] = raw.obs[
            ["pxl_col_in_fullres", "pxl_row_in_fullres"]
        ].to_numpy(dtype=np.float64)
        raw.uns["visium_scalefactors"] = scalefactors
        raw.uns["spatial_coordinate_contract"] = {
            "x": "pxl_col_in_fullres",
            "y": "pxl_row_in_fullres",
            "source": str(archive_path) + "::spatial/tissue_positions.csv",
            "container_format": "gzip_then_gzip_then_posix_tar",
        }
        raw.uns["normalization_contract"] = {
            "source_matrix": str(matrix_path),
            "counts_layer": "raw filtered 10x integer counts after duplicate-symbol summation",
            "X": "normalize_total(target_sum=1e4) then log1p",
            "target_sum": 10000.0,
        }

        raw.layers["counts"] = raw.X.copy()
        count_totals = np.asarray(raw.X.sum(axis=1)).ravel()
        sc.pp.normalize_total(raw, target_sum=1e4)
        sc.pp.log1p(raw)
        raw.write_h5ad(out_path, compression="gzip")

        rows.append(
            {
                "sample": sample,
                "matrix_path": str(matrix_path),
                "spatial_archive_path": str(archive_path),
                "output_h5ad": str(out_path),
                "n_spots": raw.n_obs,
                "n_in_tissue": int(raw.obs["in_tissue"].astype(int).sum()),
                "all_filtered_spots_in_tissue": bool(raw.obs["in_tissue"].astype(int).eq(1).all()),
                "input_gene_columns": input_genes,
                "output_unique_gene_symbols": raw.n_vars,
                "duplicate_gene_columns_collapsed": n_duplicate_columns_collapsed,
                "counts_are_integer": True,
                "min_counts_per_spot": float(count_totals.min()),
                "median_counts_per_spot": float(np.median(count_totals)),
                "max_counts_per_spot": float(count_totals.max()),
                "has_real_spatial_coordinates": "spatial" in raw.obsm,
                "counts_layer_present": "counts" in raw.layers,
                "normalization": "normalize_total_1e4_log1p",
            }
        )
        print(f"built {sample}: {raw.n_obs:,} spots x {raw.n_vars:,} genes", flush=True)

    audit = pd.DataFrame(rows)
    audit.to_csv(audit_path, index=False)
    validation = {
        "n_samples": int(audit["sample"].nunique()),
        "n_h5ad": int(audit["output_h5ad"].nunique()),
        "total_spots": int(audit["n_spots"].sum()),
        "all_counts_integer_before_normalization": bool(audit["counts_are_integer"].all()),
        "all_counts_layers_present": bool(audit["counts_layer_present"].all()),
        "all_real_spatial_coordinates_present": bool(audit["has_real_spatial_coordinates"].all()),
        "all_normalized_log": bool(audit["normalization"].eq("normalize_total_1e4_log1p").all()),
        "random_seed": SEED,
    }
    if validation["n_samples"] != 14 or not all(
        validation[key]
        for key in [
            "all_counts_integer_before_normalization",
            "all_counts_layers_present",
            "all_real_spatial_coordinates_present",
            "all_normalized_log",
        ]
    ):
        raise RuntimeError(f"Spatial H5AD validation failed: {validation}")
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    versions_path.write_text(
        "\n".join(
            [
                "environment_category=rapids",
                f"python_executable={sys.executable}",
                f"python_version={sys.version.replace(chr(10), ' ')}",
                f"scanpy={importlib.metadata.version('scanpy')}",
                f"anndata={importlib.metadata.version('anndata')}",
                f"numpy={importlib.metadata.version('numpy')}",
                f"pandas={importlib.metadata.version('pandas')}",
                f"scipy={importlib.metadata.version('scipy')}",
                f"random_seed={SEED}",
                f"code_file={Path(__file__).resolve()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    readme_path.write_text(
        "Each output is a clean per-sample Visium expression object. X is per-sample "
        "normalize_total(target_sum=1e4)+log1p expression; layers['counts'] is the "
        "corresponding raw filtered 10x count matrix after duplicate gene symbols were "
        "summed. obsm['spatial'] stores true full-resolution image x/y pixel coordinates. "
        "No CM or epithelial score has been added at this stage.\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()

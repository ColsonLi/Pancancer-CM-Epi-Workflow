#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import random
import re
import time
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
from scipy import sparse

SEED = 42
RUN = "from-scratch-scvi-latent30-v1"
N_HVG = 3000
TARGET_SUM = 10000.0
HVG_SPAN = 0.8

random.seed(SEED)
np.random.seed(SEED)

WF = Path(__file__).resolve().parents[4]
INPUT = WF / "h5ad/01-celltype_integration_clustering/03-qc/adata_qc.h5ad"
OUTPUT_DIR = WF / f"h5ad/01-celltype_integration_clustering_scVI/04-integration-scvi/{RUN}"
OUTPUT = OUTPUT_DIR / "adata_scvi_input.h5ad"
TABLE = WF / f"tables/01-celltype_integration_clustering_scVI/04-integration-scvi/{RUN}"
STAGE_H5AD = OUTPUT_DIR.with_name(f".{RUN}-prepare.staging")
STAGE_TABLE = TABLE.with_name(f".{RUN}-prepare.staging")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_existing() -> None:
    existing = [str(path) for path in (OUTPUT_DIR, TABLE, STAGE_H5AD, STAGE_TABLE) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing scVI preparation overwrite: {existing}")


def matrix_audit(matrix) -> dict[str, object]:
    if sparse.issparse(matrix):
        values = matrix.data
        matrix_type = type(matrix).__name__
        nnz = int(matrix.nnz)
    else:
        values = np.asarray(matrix).ravel()
        matrix_type = type(matrix).__name__
        nnz = int(np.count_nonzero(values))
    finite = bool(np.isfinite(values).all())
    minimum = float(values.min()) if values.size else 0.0
    maximum = float(values.max()) if values.size else 0.0
    integer_like = bool(np.allclose(values, np.rint(values), atol=1e-6))
    return {
        "matrix_type": matrix_type,
        "dtype": str(matrix.dtype),
        "n_rows": int(matrix.shape[0]),
        "n_columns": int(matrix.shape[1]),
        "nnz": nnz,
        "finite": finite,
        "minimum": minimum,
        "maximum": maximum,
        "nonnegative": minimum >= 0,
        "integer_like": integer_like,
    }


def main() -> None:
    started = time.time()
    refuse_existing()
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    if not cp.cuda.is_available():
        raise RuntimeError("CUDA is required for scVI input preparation")

    adata_counts = ad.read_h5ad(INPUT)
    required_obs = {"sample", "series", "status", "original_barcode"}
    missing = sorted(required_obs - set(adata_counts.obs.columns))
    if missing or not adata_counts.obs_names.is_unique:
        raise RuntimeError(f"QC metadata/cell-ID contract failed: missing={missing}")
    input_ids = adata_counts.obs_names.copy()
    audit = matrix_audit(adata_counts.X)
    if not (audit["finite"] and audit["nonnegative"] and audit["integer_like"]):
        raise RuntimeError(f"QC .X is not valid raw counts: {audit}")
    counts_dtype = str(adata_counts.X.dtype)

    # RAPIDS requires a supported floating working dtype. Values remain exact
    # integer-like counts; the on-disk QC object is read-only and unchanged.
    adata_counts.X = adata_counts.X.astype(np.float32, copy=False)
    free_before, total_gpu = cp.cuda.runtime.memGetInfo()
    rsc.get.anndata_to_GPU(adata_counts)
    rsc.pp.highly_variable_genes(
        adata_counts,
        n_top_genes=min(N_HVG, adata_counts.n_vars),
        flavor="seurat_v3",
        batch_key="sample",
        span=HVG_SPAN,
    )
    rsc.get.anndata_to_CPU(adata_counts, convert_all=True)
    hvg_mask = adata_counts.var["highly_variable"].to_numpy(dtype=bool)
    actual_hvg = int(hvg_mask.sum())
    if actual_hvg != min(N_HVG, adata_counts.n_vars):
        raise RuntimeError(f"Unexpected HVG count: {actual_hvg}")

    adata_log = adata_counts.copy()
    rsc.get.anndata_to_GPU(adata_log)
    rsc.pp.normalize_total(adata_log, target_sum=TARGET_SUM)
    rsc.pp.log1p(adata_log)
    rsc.get.anndata_to_CPU(adata_log, convert_all=True)
    free_after, _ = cp.cuda.runtime.memGetInfo()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

    adata_scvi = adata_counts[:, hvg_mask].copy()
    # Counts were float-cast only for RAPIDS compatibility; they remain exact
    # integer-like raw values and are explicitly recorded in the counts layer.
    adata_scvi.layers["counts"] = adata_scvi.X.copy()
    adata_scvi.raw = adata_log
    if not adata_scvi.obs_names.equals(input_ids) or not adata_scvi.obs_names.equals(adata_scvi.raw.obs_names):
        raise RuntimeError("Cell IDs/order differ between counts, raw, and scVI object")
    output_count_audit = matrix_audit(adata_scvi.layers["counts"])
    if not (
        output_count_audit["finite"]
        and output_count_audit["nonnegative"]
        and output_count_audit["integer_like"]
    ):
        raise RuntimeError(f"Prepared layers[counts] audit failed: {output_count_audit}")
    if adata_scvi.raw.n_vars != adata_log.n_vars:
        raise RuntimeError("adata.raw did not retain all genes")

    adata_scvi.uns["scvi_input_preparation"] = {
        "source_h5ad": str(INPUT.relative_to(WF)),
        "counts_source": "validated QC .X",
        "counts_source_dtype": counts_dtype,
        "counts_layer": "counts",
        "hvg_flavor": "seurat_v3",
        "hvg_batch_key": "sample",
        "hvg_span": HVG_SPAN,
        "n_top_genes": actual_hvg,
        "raw_expression": "full-gene normalize_total(target_sum=1e4) then log1p",
        "random_seed": SEED,
        "downsampled": False,
    }

    STAGE_H5AD.mkdir(parents=True, exist_ok=False)
    STAGE_TABLE.mkdir(parents=True, exist_ok=False)
    staged = STAGE_H5AD / OUTPUT.name
    adata_scvi.write_h5ad(staged, compression="gzip")

    pd.DataFrame(
        [
            {
                "input_h5ad": str(INPUT.relative_to(WF)),
                "selected_count_source": "X",
                "output_layer": "counts",
                "n_obs": adata_scvi.n_obs,
                "n_genes_input": adata_counts.n_vars,
                "n_genes_scvi": adata_scvi.n_vars,
                "source_dtype": counts_dtype,
                **{f"input_{key}": value for key, value in audit.items()},
                **{f"output_{key}": value for key, value in output_count_audit.items()},
                "action": "copy validated QC X into explicit layers[counts] after HVG subset",
            }
        ]
    ).to_csv(STAGE_TABLE / "01_scvi_input_counts_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "input_h5ad": str(INPUT.relative_to(WF)),
                "output_h5ad": str(OUTPUT.relative_to(WF)),
                "flavor": "seurat_v3",
                "batch_key": "sample",
                "span": HVG_SPAN,
                "n_top_genes_requested": N_HVG,
                "n_top_genes_actual": actual_hvg,
                "count_source": "validated QC X",
                "backend": "rapids-singlecell GPU",
                "random_seed": SEED,
                "downsampled": False,
            }
        ]
    ).to_csv(STAGE_TABLE / "02_scvi_hvg_or_gene_selection_parameters.csv", index=False)
    pd.DataFrame({"gene": adata_scvi.var_names.astype(str)}).to_csv(
        STAGE_TABLE / "02_scvi_selected_hvgs.csv", index=False
    )

    values = pd.Index(adata_scvi.var_names.astype(str))
    ens = values.to_series(index=range(len(values))).str.match(r"^ENS[A-Z]*G[0-9]+(?:\.[0-9]+)?$")
    pd.DataFrame(
        [
            {
                "input_h5ad": str(INPUT.relative_to(WF)),
                "object_layer": "var_names",
                "var_names_type": "gene_id" if ens.mean() >= 0.9 else "gene_name" if ens.mean() <= 0.1 else "mixed_or_unknown",
                "n_features": len(values),
                "n_unique": int(values.nunique()),
                "n_duplicates": int(values.duplicated().sum()),
                "ensembl_like_n": int(ens.sum()),
                "ensembl_like_fraction": float(ens.mean()),
                "candidate_gene_id_columns": ";".join(c for c in adata_scvi.var.columns if re.search("id|ensembl", c, re.I)),
                "candidate_gene_name_columns": ";".join(c for c in adata_scvi.var.columns if re.search("name|symbol", c, re.I)),
                "selected_gene_id_source": "",
                "selected_gene_name_source": "var_names",
                "action_taken": "none",
            },
            {
                "input_h5ad": str(INPUT.relative_to(WF)),
                "object_layer": "raw.var_names",
                "var_names_type": "gene_id" if ens.mean() >= 0.9 else "gene_name" if ens.mean() <= 0.1 else "mixed_or_unknown",
                "n_features": adata_scvi.raw.n_vars,
                "n_unique": int(pd.Index(adata_scvi.raw.var_names).nunique()),
                "n_duplicates": int(pd.Index(adata_scvi.raw.var_names).duplicated().sum()),
                "ensembl_like_n": int(pd.Index(adata_scvi.raw.var_names.astype(str)).to_series(index=range(adata_scvi.raw.n_vars)).str.match(r"^ENS[A-Z]*G[0-9]+(?:\.[0-9]+)?$").sum()),
                "ensembl_like_fraction": float(pd.Index(adata_scvi.raw.var_names.astype(str)).to_series(index=range(adata_scvi.raw.n_vars)).str.match(r"^ENS[A-Z]*G[0-9]+(?:\.[0-9]+)?$").mean()),
                "candidate_gene_id_columns": "",
                "candidate_gene_name_columns": "",
                "selected_gene_id_source": "",
                "selected_gene_name_source": "raw.var_names",
                "action_taken": "none",
            },
        ]
    ).to_csv(STAGE_TABLE / "gene_identifier_audit.csv", index=False)
    pd.DataFrame(
        {
            "object_layer": ["var_names"] * min(20, adata_scvi.n_vars),
            "example": adata_scvi.var_names.astype(str)[:20],
        }
    ).to_csv(STAGE_TABLE / "gene_identifier_examples.csv", index=False)

    completion = {
        "all_passed": True,
        "n_obs": adata_scvi.n_obs,
        "n_vars_scvi": adata_scvi.n_vars,
        "raw_n_vars": adata_scvi.raw.n_vars,
        "counts_layer_present": "counts" in adata_scvi.layers,
        "counts_finite_nonnegative_integer_like": True,
        "cell_order_exact": adata_scvi.obs_names.equals(adata_scvi.raw.obs_names),
        "gpu_total_bytes": int(total_gpu),
        "gpu_free_before_bytes": int(free_before),
        "gpu_free_after_bytes": int(free_after),
        "output_sha256": sha256(staged),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (STAGE_TABLE / "scvi_input_preparation_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    versions = {name: version(name) for name in ("anndata", "scanpy", "rapids-singlecell", "cupy-cuda12x", "numpy", "pandas")}
    versions.update({"python": platform.python_version(), "seed": SEED, "environment": "uv_envs/rapids/.venv"})
    (STAGE_TABLE / "package_versions.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in versions.items()) + "\n", encoding="utf-8"
    )
    (STAGE_TABLE / "readme.txt").write_text(
        "Prepared the full-cell scVI input from the clean QC raw-count object.\n"
        "Validated QC X was the raw-count source and was copied to explicit layers[counts].\n"
        "scVI uses 3000 sample-aware seurat_v3 HVGs with LOESS span=0.8; per-sample diagnostics showed that the default span=0.3 and span=0.5 were incompatible with the two smallest retained samples (near-singular LOESS fits), while span=0.8 passed all 129 samples.\n"
        "adata.raw retains all genes after normalize_total(1e4)/log1p for DEG.\n"
        "No cells were sampled or removed, and the existing Harmony branch was not read.\n",
        encoding="utf-8",
    )

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    os.replace(STAGE_TABLE, TABLE)
    os.replace(STAGE_H5AD, OUTPUT_DIR)
    print(json.dumps(completion, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent full-cell GPU preprocessing and Harmony for one lineage."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block02_common import (  # noqa: E402
    MODULE, SEED, WORKFLOW_ROOT, atomic_write_h5ad, cell_id_digest,
    partial_h5ad_path, task_root,
)

HVG_MIN_BATCH_CELLS = 5


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--lineage", required=True)
    p.add_argument("--n-top-genes", type=int, default=3000); p.add_argument("--batch-key", default="sample")
    return p.parse_args()


def main() -> None:
    a = args()
    inp = task_root("h5ad", "01-lineage-selection") / a.lineage / f"adata_{a.lineage}_qc_counts.h5ad"
    out = task_root("h5ad", "02-subtype-harmony") / a.lineage / f"adata_{a.lineage}_harmony.h5ad"
    table = task_root("tables", "02-subtype-harmony") / a.lineage / "harmony_parameters.json"
    if out.exists() and table.exists():
        x = ad.read_h5ad(out, backed="r"); meta = json.loads(table.read_text())
        ok = "X_pca_inte" in x.obsm and x.raw is not None and meta["cell_id_sha256"] == cell_id_digest(x.obs_names)
        x.file.close()
        if ok: print(f"validated_existing={out}"); return
        raise RuntimeError(f"Existing Harmony output failed validation: {out}")
    if out.exists() or table.exists() or partial_h5ad_path(out).exists():
        raise FileExistsError("Partial Harmony output exists; refusing overwrite")
    adata = ad.read_h5ad(inp)
    if a.batch_key not in adata.obs: raise KeyError(f"Missing Harmony key obs[{a.batch_key!r}]")
    # Lineage subsetting preserves the full-atlas categorical level set.  Empty
    # sample categories are interpreted as zero-cell batches by RAPIDS HVG and
    # make dispersion binning fail with ``Cannot cut empty array``.  Remove only
    # unused category levels; cell rows and the original sample values are
    # unchanged, and both HVG and Harmony still use the exact sample key.
    batch_categories_before = (
        len(adata.obs[a.batch_key].cat.categories)
        if isinstance(adata.obs[a.batch_key].dtype, pd.CategoricalDtype)
        else int(adata.obs[a.batch_key].astype(str).nunique())
    )
    if not isinstance(adata.obs[a.batch_key].dtype, pd.CategoricalDtype):
        adata.obs[a.batch_key] = pd.Categorical(adata.obs[a.batch_key].astype(str))
    adata.obs[a.batch_key] = adata.obs[a.batch_key].cat.remove_unused_categories()
    batch_categories_after = len(adata.obs[a.batch_key].cat.categories)
    if (adata.obs[a.batch_key].value_counts(sort=False) <= 0).any():
        raise RuntimeError("Unused batch categories remain after categorical cleanup")
    batch_counts = adata.obs[a.batch_key].value_counts(sort=False)
    rare_batches = batch_counts[batch_counts < HVG_MIN_BATCH_CELLS].index.astype(str).tolist()
    hvg_batch_key = f"__hvg_{a.batch_key}"
    hvg_values = adata.obs[a.batch_key].astype(str)
    if rare_batches:
        hvg_values = hvg_values.where(
            ~hvg_values.isin(rare_batches), "__rare_sample_pool__"
        )
    adata.obs[hvg_batch_key] = pd.Categorical(hvg_values)
    adata.obs[hvg_batch_key] = adata.obs[hvg_batch_key].cat.remove_unused_categories()
    hvg_batch_counts = adata.obs[hvg_batch_key].value_counts(sort=False)
    if (hvg_batch_counts < 2).any():
        raise RuntimeError(
            f"HVG batch pooling still contains a <2-cell group: {hvg_batch_counts.to_dict()}"
        )
    for key in ("total_counts", "pct_counts_MT"):
        if key not in adata.obs:
            alt = "pct_counts_mt" if key == "pct_counts_MT" else None
            if alt and alt in adata.obs: adata.obs[key] = adata.obs[alt].values
            else: raise KeyError(f"Missing regression covariate obs[{key!r}]")
    import cupy as cp
    import rapids_singlecell as rsc
    input_x_dtype = str(adata.X.dtype)
    # The source lineage h5ad remains raw integer counts on disk.  RAPIDS/CuPy
    # sparse matrices require a supported floating dtype, so only this freshly
    # loaded working copy is converted.
    adata.X = adata.X.astype(np.float32, copy=False)
    gpu_working_x_dtype = str(adata.X.dtype)
    if gpu_working_x_dtype != "float32":
        raise RuntimeError(f"Failed to prepare float32 GPU working matrix: {gpu_working_x_dtype}")
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.normalize_total(adata, target_sum=1e4); rsc.pp.log1p(adata)
    rsc.pp.highly_variable_genes(adata, n_top_genes=min(a.n_top_genes, adata.n_vars),
                                 flavor="seurat", batch_key=hvg_batch_key)
    rsc.get.anndata_to_CPU(adata, convert_all=True)
    del adata.obs[hvg_batch_key]
    adata.raw = adata.copy(); adata = adata[:, adata.var.highly_variable].copy()
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.regress_out(adata, keys=["total_counts", "pct_counts_MT"], batchsize=100000)
    rsc.pp.scale(adata, max_value=10); rsc.tl.pca(adata, n_comps=min(50, adata.n_vars - 1, adata.n_obs - 1), random_state=SEED)
    clean_pca = adata.obsm["X_pca"].copy(); attempts = []; accepted = False
    for max_iter in (10, 30, 60, 100):
        adata.obsm["X_pca"] = clean_pca.copy()
        if "X_pca_inte" in adata.obsm: del adata.obsm["X_pca_inte"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rsc.pp.harmony_integrate(adata, key=a.batch_key, basis="X_pca", adjusted_basis="X_pca_inte",
                correction_method="original", use_gemm=False, max_iter_harmony=max_iter,
                random_state=SEED, verbose=True)
        messages = [str(item.message) for item in caught]
        failed = any("did not converge" in message.lower() for message in messages)
        attempts.append({"max_iter_harmony": max_iter, "warnings": messages, "accepted": not failed})
        if not failed: accepted = True; break
    if not accepted: raise RuntimeError("Lineage Harmony did not converge from clean PCA by max_iter=100")
    rsc.get.anndata_to_CPU(adata, convert_all=True); cp.get_default_memory_pool().free_all_blocks(); gc.collect()
    if adata.obsm["X_pca_inte"].shape[0] != adata.n_obs: raise AssertionError("Harmony row mismatch")
    table.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "lineage": a.lineage, "input_h5ad": str(inp), "output_h5ad": str(out),
        "n_cells": adata.n_obs, "n_hvg": adata.n_vars, "raw_n_genes": adata.raw.n_vars,
        "cell_id_sha256": cell_id_digest(adata.obs_names), "batch_key": a.batch_key,
        "normalize_target": 10000, "hvg_flavor": "seurat", "n_top_genes_requested": a.n_top_genes,
        "regress_keys": ["total_counts", "pct_counts_MT"], "n_pcs": int(adata.obsm["X_pca_inte"].shape[1]),
        "hvg_batch_key": a.batch_key,
        "hvg_effective_batch_rule": f"sample batches with <{HVG_MIN_BATCH_CELLS} lineage cells pooled for HVG only",
        "hvg_rare_sample_batches_pooled": rare_batches,
        "hvg_n_rare_sample_batches_pooled": len(rare_batches),
        "hvg_n_cells_in_rare_sample_pool": int(batch_counts[batch_counts < HVG_MIN_BATCH_CELLS].sum()),
        "hvg_effective_batch_count": int(len(hvg_batch_counts)),
        "harmony_uses_original_unpooled_sample_key": True,
        "harmony_attempts": attempts,
        "backend": "rapids_singlecell_gpu", "random_seed": SEED, "downsampled": False,
        "input_x_dtype": input_x_dtype, "gpu_working_x_dtype": gpu_working_x_dtype,
        "source_lineage_h5ad_unchanged": True,
        "batch_categories_before_cleanup": batch_categories_before,
        "batch_categories_after_cleanup": batch_categories_after,
        "unused_batch_categories_removed": batch_categories_before - batch_categories_after,
        "batch_values_or_cells_changed": False,
    }
    table.write_text(json.dumps(meta, indent=2) + "\n")
    # Publish the h5ad atomically and last.  A failed write leaves a named
    # partial that a future invocation refuses rather than overwriting.
    atomic_write_h5ad(adata, out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__": main()

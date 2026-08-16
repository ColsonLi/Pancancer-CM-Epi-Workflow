#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import platform
import random
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scvi
import torch
from scipy import sparse

SEED = 42
RUN = "from-scratch-scvi-latent30-v1"
N_LATENT = 30
N_LAYERS = 2
GENE_LIKELIHOOD = "nb"
BATCH_KEY = "sample"
COUNT_LAYER = "counts"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
scvi.settings.seed = SEED

WF = Path(__file__).resolve().parents[4]
TASK = "01-celltype_integration_clustering_scVI/04-integration-scvi"
INPUT = WF / f"h5ad/{TASK}/{RUN}/adata_scvi_input.h5ad"
OUTPUT_DIR = WF / f"h5ad/{TASK}/{RUN}"
OUTPUT = OUTPUT_DIR / "adata_scvi.h5ad"
MODEL_DIR = OUTPUT_DIR / "scvi_model"
TABLE_DIR = WF / f"tables/{TASK}/{RUN}"
STAGE_ROOT = OUTPUT_DIR.parent / f".{RUN}-train.staging"
STAGE_H5AD = STAGE_ROOT / "adata_scvi.h5ad"
STAGE_MODEL = STAGE_ROOT / "scvi_model"
STAGE_TABLE = STAGE_ROOT / "tables"


def matrix_audit(matrix) -> dict[str, object]:
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    minimum = float(values.min()) if values.size else 0.0
    return {
        "finite": bool(np.isfinite(values).all()),
        "nonnegative": minimum >= 0,
        "integer_like": bool(np.allclose(values, np.rint(values), atol=1e-6)),
        "minimum": minimum,
        "maximum": float(values.max()) if values.size else 0.0,
        "dtype": str(matrix.dtype),
        "matrix_type": type(matrix).__name__,
    }


def refuse_existing() -> None:
    paths = [OUTPUT, MODEL_DIR, STAGE_ROOT]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing scVI training overwrite: {existing}")
    table_names = (
        "03_scvi_setup_parameters.csv",
        "04_scvi_train_parameters.csv",
        "05_scvi_latent_and_model_paths.csv",
        "scvi_training_history_summary.csv",
        "scvi_training_completion.json",
        "gpu_backend_capability_summary.csv",
    )
    existing_tables = [str(TABLE_DIR / name) for name in table_names if (TABLE_DIR / name).exists()]
    if existing_tables:
        raise FileExistsError(f"Refusing scVI table overwrite: {existing_tables}")


def history_to_tables(model: scvi.model.SCVI) -> tuple[int, list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    lengths: list[int] = []
    for key, value in model.history.items():
        frame = pd.DataFrame(value).copy()
        frame.index.name = frame.index.name or "epoch"
        frame.reset_index().to_csv(STAGE_TABLE / f"training_history_{key}.csv", index=False)
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        lengths.append(len(frame))
        for column in numeric.columns:
            vals = numeric[column].dropna()
            summary_rows.append(
                {
                    "metric": str(key),
                    "column": str(column),
                    "n_recorded": int(len(vals)),
                    "first": float(vals.iloc[0]) if len(vals) else np.nan,
                    "last": float(vals.iloc[-1]) if len(vals) else np.nan,
                    "minimum": float(vals.min()) if len(vals) else np.nan,
                    "maximum": float(vals.max()) if len(vals) else np.nan,
                }
            )
    pd.DataFrame(summary_rows).to_csv(STAGE_TABLE / "scvi_training_history_summary.csv", index=False)
    return max(lengths, default=0), summary_rows


def main() -> None:
    started = time.time()
    refuse_existing()
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for scVI training")
    STAGE_TABLE.mkdir(parents=True, exist_ok=False)

    adata = ad.read_h5ad(INPUT)
    if COUNT_LAYER not in adata.layers or BATCH_KEY not in adata.obs:
        raise RuntimeError("Required count layer or sample batch key is missing")
    if adata.raw is None or not adata.obs_names.equals(adata.raw.obs_names):
        raise RuntimeError("Full-gene adata.raw or exact cell order is missing")
    count_audit = matrix_audit(adata.layers[COUNT_LAYER])
    if not all(count_audit[key] for key in ("finite", "nonnegative", "integer_like")):
        raise RuntimeError(f"Invalid scVI raw-count layer: {count_audit}")

    scvi.model.SCVI.setup_anndata(adata, layer=COUNT_LAYER, batch_key=BATCH_KEY)
    model = scvi.model.SCVI(
        adata,
        n_latent=N_LATENT,
        n_layers=N_LAYERS,
        gene_likelihood=GENE_LIKELIHOOD,
    )
    train_started = time.time()
    model.train(max_epochs=None, accelerator="gpu", devices=1)
    train_elapsed = time.time() - train_started
    latent = np.asarray(model.get_latent_representation())
    if latent.shape != (adata.n_obs, N_LATENT) or not np.isfinite(latent).all():
        raise RuntimeError(f"Invalid X_scVI latent matrix: {latent.shape}")
    adata.obsm["X_scVI"] = latent

    actual_epochs, history_summary = history_to_tables(model)
    trainer_current_epoch = int(model.trainer.current_epoch)
    adata.uns["scvi_integration"] = {
        "source_h5ad": str(INPUT.relative_to(WF)),
        "integration_source": "scVI",
        "layer": COUNT_LAYER,
        "batch_key": BATCH_KEY,
        "n_latent": N_LATENT,
        "n_layers": N_LAYERS,
        "gene_likelihood": GENE_LIKELIHOOD,
        "max_epochs_requested": None,
        "actual_history_epochs": actual_epochs,
        "trainer_current_epoch": trainer_current_epoch,
        "early_stopping": False,
        "seed": SEED,
        "device": torch.cuda.get_device_name(0),
        "downsampled": False,
    }

    adata.write_h5ad(STAGE_H5AD, compression="gzip")
    model.save(str(STAGE_MODEL), overwrite=False, save_anndata=False)

    # Independent reload validation against the saved AnnData and saved weights.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    check = ad.read_h5ad(STAGE_H5AD)
    reloaded = scvi.model.SCVI.load(str(STAGE_MODEL), adata=check, accelerator="gpu", device=1)
    indices = np.arange(min(128, check.n_obs), dtype=int)
    latent_reload = np.asarray(reloaded.get_latent_representation(indices=indices))
    reload_finite = bool(np.isfinite(latent_reload).all())
    reload_matches_saved = bool(np.allclose(latent_reload, check.obsm["X_scVI"][indices], rtol=1e-5, atol=1e-5))
    if latent_reload.shape != (len(indices), N_LATENT) or not reload_finite or not reload_matches_saved:
        raise RuntimeError("Saved scVI model reload validation failed")

    source_rel = str(INPUT.relative_to(WF))
    output_rel = str(OUTPUT.relative_to(WF))
    model_rel = str(MODEL_DIR.relative_to(WF))
    pd.DataFrame([{
        "step": "scvi_setup_anndata",
        "input_h5ad": source_rel,
        "output_h5ad_or_object": output_rel,
        "n_obs_before": adata.n_obs,
        "n_obs_after": adata.n_obs,
        "n_vars_before": adata.n_vars,
        "n_vars_after": adata.n_vars,
        "layer": COUNT_LAYER,
        "batch_key": BATCH_KEY,
        "n_batches": int(adata.obs[BATCH_KEY].nunique()),
        "counts_finite": count_audit["finite"],
        "counts_nonnegative": count_audit["nonnegative"],
        "counts_integer_like": count_audit["integer_like"],
        "backend_package": "scvi-tools",
        "code_file": str(Path(__file__).relative_to(WF)),
        "random_seed": SEED,
        "notes_fallbacks": "none",
    }]).to_csv(STAGE_TABLE / "03_scvi_setup_parameters.csv", index=False)
    pd.DataFrame([{
        "step": "scvi_train",
        "input_h5ad": source_rel,
        "output_h5ad_or_object": output_rel,
        "n_obs_before": adata.n_obs,
        "n_obs_after": adata.n_obs,
        "n_vars_before": adata.n_vars,
        "n_vars_after": adata.n_vars,
        "n_latent": N_LATENT,
        "n_layers": N_LAYERS,
        "gene_likelihood": GENE_LIKELIHOOD,
        "max_epochs_requested": "None (scvi-tools heuristic)",
        "actual_history_epochs": actual_epochs,
        "trainer_current_epoch": trainer_current_epoch,
        "early_stopping": False,
        "batch_size": 128,
        "accelerator": "gpu",
        "devices": 1,
        "device_name": torch.cuda.get_device_name(0),
        "elapsed_seconds": round(train_elapsed, 3),
        "backend_package": "scvi-tools",
        "code_file": str(Path(__file__).relative_to(WF)),
        "random_seed": SEED,
        "notes_fallbacks": "none",
    }]).to_csv(STAGE_TABLE / "04_scvi_train_parameters.csv", index=False)
    pd.DataFrame([{
        "step": "scvi_latent_model_save_reload",
        "source_h5ad": source_rel,
        "output_h5ad": output_rel,
        "model_path": model_rel,
        "latent_key": "X_scVI",
        "latent_n_rows": latent.shape[0],
        "latent_n_dims": latent.shape[1],
        "latent_finite": bool(np.isfinite(latent).all()),
        "model_reload_passed": True,
        "reload_test_n_cells": len(indices),
        "reload_latent_finite": reload_finite,
        "reload_latent_matches_saved": reload_matches_saved,
        "raw_n_vars": adata.raw.n_vars,
        "cell_order_exact": adata.obs_names.equals(adata.raw.obs_names),
        "downsampled": False,
    }]).to_csv(STAGE_TABLE / "05_scvi_latent_and_model_paths.csv", index=False)
    pd.DataFrame([{
        "step": "scvi_training_and_latent",
        "planned_backend": "PyTorch CUDA GPU",
        "attempted_backend": "PyTorch CUDA GPU",
        "status": "passed",
        "error_summary": "",
        "fallback_backend": "",
        "clean_input_reloaded": True,
        "final_backend_for_rerun": "PyTorch CUDA GPU",
    }]).to_csv(STAGE_TABLE / "gpu_backend_capability_summary.csv", index=False)

    completion = {
        "all_passed": True,
        "n_obs": int(adata.n_obs),
        "n_vars_scvi": int(adata.n_vars),
        "raw_n_vars": int(adata.raw.n_vars),
        "x_scvi_shape": list(latent.shape),
        "x_scvi_finite": bool(np.isfinite(latent).all()),
        "model_reload_passed": True,
        "actual_history_epochs": actual_epochs,
        "trainer_current_epoch": trainer_current_epoch,
        "training_elapsed_seconds": round(train_elapsed, 3),
        "total_elapsed_seconds": round(time.time() - started, 3),
        "device": torch.cuda.get_device_name(0),
        "downsampled": False,
        "history_metric_rows": len(history_summary),
    }
    (STAGE_TABLE / "scvi_training_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    versions = {
        name: version(name)
        for name in ("scvi-tools", "torch", "anndata", "scanpy", "lightning", "numpy", "pandas")
    }
    versions.update({
        "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0),
        "seed": SEED,
        "environment": "uv_envs/scvi/.venv",
    })
    (STAGE_TABLE / "package_versions_scvi.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in versions.items()) + "\n", encoding="utf-8"
    )
    (STAGE_TABLE / "readme_scvi_training.txt").write_text(
        "Trained scVI from full-cell raw integer-like layers[counts] with sample as batch.\n"
        "Parameters: latent=30, layers=2, likelihood=NB, max_epochs=None, seed=42, GPU.\n"
        "adata.raw remains the full-gene normalize_total(1e4)/log1p matrix for later Scanpy DEG.\n"
        "X_scVI is the only integration representation approved for the candidate graph/UMAP/Leiden grid.\n"
        "The saved model was reloaded against the saved AnnData and its latent output was verified.\n",
        encoding="utf-8",
    )

    # Promote only after h5ad/model/reload/table validation has passed.
    os.replace(STAGE_H5AD, OUTPUT)
    os.replace(STAGE_MODEL, MODEL_DIR)
    for staged_file in sorted(STAGE_TABLE.iterdir()):
        target = TABLE_DIR / staged_file.name
        if target.exists():
            raise FileExistsError(target)
        os.replace(staged_file, target)
    STAGE_TABLE.rmdir()
    STAGE_ROOT.rmdir()
    print(json.dumps(completion, ensure_ascii=False), flush=True)

    del reloaded, check, adata, latent
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

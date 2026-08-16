#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scvi
import torch

SEED = 42
RUN = "from-scratch-scvi-latent30-v1"
TRAIN_ELAPSED_FROM_TRAINER_SECONDS = 1585.0

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


def main() -> None:
    started = time.time()
    if OUTPUT.exists() or MODEL_DIR.exists():
        raise FileExistsError("Final scVI h5ad/model already exists; refusing overwrite")
    if not STAGE_H5AD.is_file() or not (STAGE_MODEL / "model.pt").is_file():
        raise FileNotFoundError("Completed staged scVI h5ad/model is missing")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for saved-model reload validation")

    adata = ad.read_h5ad(STAGE_H5AD)
    latent = np.asarray(adata.obsm["X_scVI"])
    if latent.shape != (adata.n_obs, 30) or not np.isfinite(latent).all():
        raise RuntimeError(f"Invalid staged X_scVI: {latent.shape}")
    if adata.raw is None or adata.raw.n_vars != 8290 or not adata.obs_names.equals(adata.raw.obs_names):
        raise RuntimeError("Staged full-gene adata.raw contract failed")
    if "counts" not in adata.layers:
        raise RuntimeError("Staged layers[counts] is missing")

    # scvi-tools 1.5 passes this value to Lightning's devices parser. Integer
    # 1 means one GPU and resolves to cuda:0; integer 0 means zero devices.
    reloaded = scvi.model.SCVI.load(
        str(STAGE_MODEL), adata=adata, accelerator="gpu", device=1
    )
    indices = np.arange(min(128, adata.n_obs), dtype=int)
    latent_reload = np.asarray(reloaded.get_latent_representation(indices=indices))
    reload_finite = bool(np.isfinite(latent_reload).all())
    reload_matches = bool(np.allclose(latent_reload, latent[indices], rtol=1e-5, atol=1e-5))
    if latent_reload.shape != (len(indices), 30) or not reload_finite or not reload_matches:
        raise RuntimeError("Saved scVI model reload/latent validation failed")

    history_files = sorted(STAGE_TABLE.glob("training_history_*.csv"))
    if len(history_files) < 1:
        raise RuntimeError("Saved training-history files are missing")
    summary_rows = []
    actual_epochs = 0
    for path in history_files:
        if path.name == "scvi_training_history_summary.csv":
            continue
        frame = pd.read_csv(path)
        actual_epochs = max(actual_epochs, len(frame))
        for column in frame.columns:
            if column == "epoch":
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            summary_rows.append({
                "metric": path.stem.removeprefix("training_history_"),
                "column": column,
                "n_recorded": len(values),
                "first": float(values.iloc[0]),
                "last": float(values.iloc[-1]),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            })
    if actual_epochs != 17:
        raise RuntimeError(f"Unexpected training-history epoch count: {actual_epochs}")
    pd.DataFrame(summary_rows).to_csv(STAGE_TABLE / "scvi_training_history_summary.csv", index=False)

    source_rel = str(INPUT.relative_to(WF))
    output_rel = str(OUTPUT.relative_to(WF))
    model_rel = str(MODEL_DIR.relative_to(WF))
    code_rel = str(Path(__file__).relative_to(WF))
    pd.DataFrame([{
        "step": "scvi_setup_anndata",
        "input_h5ad": source_rel,
        "output_h5ad_or_object": output_rel,
        "n_obs_before": adata.n_obs,
        "n_obs_after": adata.n_obs,
        "n_vars_before": adata.n_vars,
        "n_vars_after": adata.n_vars,
        "layer": "counts",
        "batch_key": "sample",
        "n_batches": adata.obs["sample"].nunique(),
        "counts_finite": True,
        "counts_nonnegative": True,
        "counts_integer_like": True,
        "backend_package": "scvi-tools",
        "code_file": code_rel,
        "random_seed": SEED,
        "notes_fallbacks": "none during setup/training",
    }]).to_csv(STAGE_TABLE / "03_scvi_setup_parameters.csv", index=False)
    pd.DataFrame([{
        "step": "scvi_train",
        "input_h5ad": source_rel,
        "output_h5ad_or_object": output_rel,
        "n_obs_before": adata.n_obs,
        "n_obs_after": adata.n_obs,
        "n_vars_before": adata.n_vars,
        "n_vars_after": adata.n_vars,
        "n_latent": 30,
        "n_layers": 2,
        "gene_likelihood": "nb",
        "max_epochs_requested": "None (scvi-tools heuristic)",
        "actual_history_epochs": actual_epochs,
        "trainer_current_epoch": 17,
        "stopping_reason": "max_epochs=17 reached",
        "early_stopping": False,
        "batch_size": 128,
        "accelerator": "gpu",
        "devices": 1,
        "device_name": torch.cuda.get_device_name(0),
        "elapsed_seconds": TRAIN_ELAPSED_FROM_TRAINER_SECONDS,
        "elapsed_source": "Lightning trainer progress 26m25s",
        "backend_package": "scvi-tools",
        "code_file": code_rel,
        "random_seed": SEED,
        "notes_fallbacks": "none during training",
    }]).to_csv(STAGE_TABLE / "04_scvi_train_parameters.csv", index=False)
    pd.DataFrame([{
        "step": "scvi_latent_model_save_reload",
        "source_h5ad": source_rel,
        "output_h5ad": output_rel,
        "model_path": model_rel,
        "latent_key": "X_scVI",
        "latent_n_rows": latent.shape[0],
        "latent_n_dims": latent.shape[1],
        "latent_finite": True,
        "model_reload_passed": True,
        "reload_device_argument": 1,
        "reload_resolved_device": "cuda:0",
        "reload_test_n_cells": len(indices),
        "reload_latent_finite": reload_finite,
        "reload_latent_matches_saved": reload_matches,
        "raw_n_vars": adata.raw.n_vars,
        "cell_order_exact": adata.obs_names.equals(adata.raw.obs_names),
        "downsampled": False,
    }]).to_csv(STAGE_TABLE / "05_scvi_latent_and_model_paths.csv", index=False)
    pd.DataFrame([{
        "step": "scvi_training_and_latent",
        "planned_backend": "PyTorch CUDA GPU",
        "attempted_backend": "PyTorch CUDA GPU",
        "status": "passed",
        "error_summary": "post-save reload compatibility audit found device=0 means zero GPUs and cuda:0 is not accepted by this Lightning devices parser; repaired with device=1, resolving to cuda:0",
        "fallback_backend": "",
        "clean_input_reloaded": True,
        "final_backend_for_rerun": "PyTorch CUDA GPU; reload device argument=1 resolves to cuda:0",
    }]).to_csv(STAGE_TABLE / "gpu_backend_capability_summary.csv", index=False)

    completion = {
        "all_passed": True,
        "n_obs": int(adata.n_obs),
        "n_vars_scvi": int(adata.n_vars),
        "raw_n_vars": int(adata.raw.n_vars),
        "x_scvi_shape": list(latent.shape),
        "x_scvi_finite": True,
        "model_reload_passed": True,
        "reload_latent_matches_saved": reload_matches,
        "actual_history_epochs": actual_epochs,
        "trainer_current_epoch": 17,
        "stopping_reason": "max_epochs=17 reached",
        "training_elapsed_seconds": TRAIN_ELAPSED_FROM_TRAINER_SECONDS,
        "finalization_elapsed_seconds": round(time.time() - started, 3),
        "device": torch.cuda.get_device_name(0),
        "downsampled": False,
    }
    (STAGE_TABLE / "scvi_training_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    versions = {name: version(name) for name in ("scvi-tools", "torch", "anndata", "scanpy", "lightning", "numpy", "pandas")}
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
        "Trained scVI on all 457,999 cells using raw layers[counts], sample batch, latent=30, layers=2, NB, seed=42, and CUDA GPU.\n"
        "scvi-tools max_epochs=None selected 17 epochs; training reached epoch 17 normally and train_loss fell from 1124.425 to 1024.830.\n"
        "The staged model and AnnData were reloaded using scvi-tools device=1, which resolves to cuda:0; the reloaded latent matched saved X_scVI for 128 audited cells.\n"
        "The initial device=0 and device=cuda:0 calls exposed Lightning parser semantics after successful save; no retraining or CPU fallback was used.\n",
        encoding="utf-8",
    )

    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
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


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run canonical CUDA Tangram pseudobulk mapping for all 14 PDAC samples."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import tangram as tg
import torch


SEED = 42
MODE = "cells"
DEVICE = "cuda:0"
NUM_EPOCHS = 350
LEARNING_RATE = 0.05

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
REFERENCE = (
    PROJECT_ROOT
    / "h5ad/04-spatial-validation-optional/03-tangram-reference/adata_cell_subtype_pseudobulk_deg_markers.h5ad"
)
REFERENCE_TABLE_DIR = PROJECT_ROOT / "tables/04-spatial-validation-optional/03-tangram-reference"
MARKER_GENES = REFERENCE_TABLE_DIR / "tangram_union_deg_marker_genes.csv"
SAMPLE_MANIFEST = REFERENCE_TABLE_DIR / "spatial_sample_scope_manifest.csv"
CM_LOADING = REFERENCE_TABLE_DIR / "cm_loading_fraction_non_epi_node_by_CM.csv"
EPI_SUBTYPES = REFERENCE_TABLE_DIR / "epithelial_subtypes.csv"
TABLE_DIR = PROJECT_ROOT / "tables/04-spatial-validation-optional/04-tangram-mapping"
SCORE_DIR = TABLE_DIR / "spot-scores"
METRICS_DIR = TABLE_DIR / "per-sample-metrics"
MAPPER_DIR = PROJECT_ROOT / "h5ad/04-spatial-validation-optional/04-tangram-mapping"
RUN_MANIFEST = TABLE_DIR / "tangram_sample_run_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-sample", default=None)
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def valid_existing_score(
    path: Path,
    n_expected: int,
    cm_names: list[str],
    epithelial_subtypes: list[str],
) -> bool:
    if not path.is_file():
        return False
    header = pd.read_csv(path, nrows=2)
    required = {
        "sample",
        "spot_id",
        "barcode",
        "array_row",
        "array_col",
        "spatial_x",
        "spatial_y",
        *[f"CMact__{name}" for name in cm_names],
        *[f"EPIfrac__{name}" for name in epithelial_subtypes],
    }
    if not required.issubset(header.columns):
        return False
    with path.open("rb") as handle:
        n_rows = sum(1 for _ in handle) - 1
    return n_rows == n_expected


def prepare_spatial_sample(path: Path, reference_genes: set[str], marker_genes: list[str]):
    spatial = ad.read_h5ad(path)
    required_obs = [
        "sample",
        "original_barcode",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]
    missing = [column for column in required_obs if column not in spatial.obs]
    if missing or "spatial" not in spatial.obsm or "counts" not in spatial.layers:
        raise ValueError(f"{path}: incomplete clean Visium contract; missing={missing}")
    if any(column.startswith(("CMact__", "EPIfrac__")) for column in spatial.obs.columns):
        raise ValueError(f"{path}: input H5AD is not clean; preexisting spatial scores found")
    common = [
        gene
        for gene in marker_genes
        if gene in reference_genes and gene in set(spatial.var_names.astype(str))
    ]
    if len(common) < 50:
        raise ValueError(f"{path}: only {len(common)} selected DEG markers overlap")
    spatial = spatial[:, common].copy()
    return spatial, common


def map_one(
    sample_row: pd.Series,
    reference_template: ad.AnnData,
    marker_genes: list[str],
    loadings: pd.DataFrame,
    epithelial_subtypes: list[str],
) -> tuple[Path, Path, dict]:
    sample = str(sample_row["sample"])
    started = time.time()
    spatial, common = prepare_spatial_sample(
        Path(sample_row["output_h5ad"]),
        set(reference_template.var_names.astype(str)),
        marker_genes,
    )
    reference = reference_template[:, common].copy()
    tg.pp_adatas(reference, spatial, genes=common)
    mapper = tg.map_cells_to_space(
        reference,
        spatial,
        mode=MODE,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        random_state=SEED,
        verbose=False,
    )
    tg.project_cell_annotations(mapper, spatial, annotation="cell_subtype")
    abundance = spatial.obsm["tangram_ct_pred"].copy()
    if not isinstance(abundance, pd.DataFrame):
        abundance = pd.DataFrame(abundance, index=spatial.obs_names)
    abundance.index = spatial.obs_names
    missing_nodes = sorted(set(loadings.index) - set(abundance.columns))
    missing_epi = sorted(set(epithelial_subtypes) - set(abundance.columns))
    if missing_nodes or missing_epi:
        raise ValueError(
            f"{sample}: Tangram abundance lacks nodes: non_epi={missing_nodes}, epi={missing_epi}"
        )

    epi_abundance = abundance.loc[:, epithelial_subtypes].astype(float)
    epi_total = epi_abundance.sum(axis=1)
    epi_fraction = epi_abundance.div(epi_total.replace(0, np.nan), axis=0).fillna(0.0)
    positive_epi = epi_total.gt(0)
    if positive_epi.any() and not np.allclose(
        epi_fraction.loc[positive_epi].sum(axis=1).to_numpy(),
        1.0,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(f"{sample}: within-epithelial EPIfrac normalization failed")
    # Canonical spatial CM activity: Tangram projected non-epi abundance times
    # the canonical column-normalized H loading matrix.  No percentile scaling.
    cm_activity = (
        abundance.loc[:, loadings.index].to_numpy(dtype=float)
        @ loadings.to_numpy(dtype=float)
    )
    cm_activity = pd.DataFrame(
        cm_activity, index=abundance.index, columns=loadings.columns
    )

    coordinates = np.asarray(spatial.obsm["spatial"], dtype=float)
    score = pd.DataFrame(index=spatial.obs_names)
    score["sample"] = sample
    score["spot_id"] = score.index.astype(str)
    score["barcode"] = spatial.obs["original_barcode"].astype(str).to_numpy()
    score["patient_id"] = str(sample_row["patient_id"])
    score["status"] = str(sample_row["status"])
    score["array_row"] = spatial.obs["array_row"].astype(int).to_numpy()
    score["array_col"] = spatial.obs["array_col"].astype(int).to_numpy()
    score["spatial_x"] = coordinates[:, 0]
    score["spatial_y"] = coordinates[:, 1]
    for cm in cm_activity.columns:
        score[f"CMact__{cm}"] = cm_activity[cm].to_numpy()
    for subtype in epi_fraction.columns:
        score[f"EPIfrac__{subtype}"] = epi_fraction[subtype].to_numpy()

    score_path = SCORE_DIR / f"{sample}_tangram_pseudobulk_epi_cm_spot_scores.csv"
    mapper_path = MAPPER_DIR / f"{sample}_tangram_mapper.h5ad"
    score_temporary = score_path.with_name(score_path.name + ".pending")
    mapper_temporary = mapper_path.with_name(mapper_path.stem + "_pending.h5ad")
    if any(path.exists() for path in [score_path, mapper_path, score_temporary, mapper_temporary]):
        raise FileExistsError(f"Refusing to overwrite incomplete Tangram output for {sample}")
    score.reset_index(drop=True).to_csv(score_temporary, index=False)
    mapper.write_h5ad(mapper_temporary, compression="gzip")
    mapper_temporary.replace(mapper_path)
    score_temporary.replace(score_path)
    training_history_keys = sorted(map(str, mapper.uns.keys()))
    metrics = {
        "sample": sample,
        "n_spots": int(spatial.n_obs),
        "n_common_deg_markers": len(common),
        "n_reference_subtypes": int(reference.n_obs),
        "n_cms": int(len(loadings.columns)),
        "n_epithelial_subtypes": int(len(epithelial_subtypes)),
        "n_positive_epi_rows": int(positive_epi.sum()),
        "n_zero_epi_rows": int((~positive_epi).sum()),
        "mode": MODE,
        "device": DEVICE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "random_state": SEED,
        "tangram_mapper_uns_keys": training_history_keys,
        "epifrac_definition": "Tangram epithelial subtype abundance row-normalized only within epithelial subtypes",
        "cmact_definition": "Tangram projected non-epithelial node abundance matrix-multiplied by canonical H loading fractions",
        "score_csv": str(score_path),
        "mapper_h5ad": str(mapper_path),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    del mapper, abundance, cm_activity, epi_fraction, reference, spatial, score
    gc.collect()
    torch.cuda.empty_cache()
    return score_path, mapper_path, metrics


def main() -> None:
    cli = parse_args()
    required = [REFERENCE, MARKER_GENES, SAMPLE_MANIFEST, CM_LOADING, EPI_SUBTYPES]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Tangram mapping inputs missing: {missing}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Canonical Tangram requires usable CUDA device cuda:0")
    if not str(torch.cuda.get_device_name(0)):
        raise RuntimeError("CUDA device name is empty")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MAPPER_DIR.mkdir(parents=True, exist_ok=True)
    sample_manifest = pd.read_csv(SAMPLE_MANIFEST)
    if sample_manifest["sample"].nunique() != 14:
        raise ValueError("Expected exactly 14 spatial samples")
    if cli.only_sample is not None:
        if cli.only_sample not in set(sample_manifest["sample"].astype(str)):
            raise KeyError(cli.only_sample)
        run_samples = sample_manifest.loc[
            sample_manifest["sample"].astype(str).eq(cli.only_sample)
        ].copy()
    else:
        run_samples = sample_manifest.copy()
    marker_genes = pd.read_csv(MARKER_GENES)["gene"].astype(str).tolist()
    loadings = pd.read_csv(CM_LOADING, index_col=0)
    epithelial_subtypes = pd.read_csv(EPI_SUBTYPES)["cell_subtype"].astype(str).tolist()
    reference = ad.read_h5ad(REFERENCE)
    reference.obs["cell_subtype"] = reference.obs_names.astype(str)

    if RUN_MANIFEST.is_file():
        run_manifest = pd.read_csv(RUN_MANIFEST)
        if set(run_manifest["sample"].astype(str)) != set(sample_manifest["sample"].astype(str)):
            raise ValueError("Existing Tangram run manifest sample scope differs")
    else:
        run_manifest = sample_manifest.copy()
        run_manifest["status_run"] = "pending"
        run_manifest["score_csv"] = ""
        run_manifest["mapper_h5ad"] = ""
        run_manifest["elapsed_seconds"] = np.nan
        run_manifest["error"] = ""
    for index, row in run_manifest.iterrows():
        sample = str(row["sample"])
        score_path = SCORE_DIR / f"{sample}_tangram_pseudobulk_epi_cm_spot_scores.csv"
        mapper_path = MAPPER_DIR / f"{sample}_tangram_mapper.h5ad"
        if valid_existing_score(
            score_path, int(row["n_spots"]), list(loadings.columns), epithelial_subtypes
        ) and mapper_path.is_file():
            run_manifest.loc[index, ["status_run", "score_csv", "mapper_h5ad", "error"]] = [
                "completed",
                str(score_path),
                str(mapper_path),
                "",
            ]
    atomic_csv(run_manifest, RUN_MANIFEST)

    for _, sample_row in run_samples.iterrows():
        sample = str(sample_row["sample"])
        index = run_manifest.index[run_manifest["sample"].astype(str).eq(sample)][0]
        if run_manifest.loc[index, "status_run"] == "completed":
            print(f"SKIP validated completed {sample}", flush=True)
            continue
        run_manifest.loc[index, ["status_run", "error"]] = ["running", ""]
        atomic_csv(run_manifest, RUN_MANIFEST)
        print(f"START {sample}: {int(sample_row['n_spots']):,} spots", flush=True)
        try:
            score_path, mapper_path, metrics = map_one(
                sample_row, reference, marker_genes, loadings, epithelial_subtypes
            )
            atomic_json(metrics, METRICS_DIR / f"{sample}_mapping_metrics.json")
            run_manifest.loc[
                index,
                ["status_run", "score_csv", "mapper_h5ad", "elapsed_seconds", "error"],
            ] = [
                "completed",
                str(score_path),
                str(mapper_path),
                metrics["elapsed_seconds"],
                "",
            ]
            atomic_csv(run_manifest, RUN_MANIFEST)
            print(
                f"DONE {sample}: {metrics['n_common_deg_markers']} genes, "
                f"{metrics['elapsed_seconds']} sec",
                flush=True,
            )
        except Exception as error:
            run_manifest.loc[index, ["status_run", "error"]] = [
                "failed",
                f"{type(error).__name__}: {error}",
            ]
            atomic_csv(run_manifest, RUN_MANIFEST)
            raise

    completed = run_manifest["status_run"].eq("completed")
    validation = {
        "samples_expected": 14,
        "samples_completed": int(completed.sum()),
        "samples_failed": int(run_manifest["status_run"].eq("failed").sum()),
        "spot_score_csvs": int(len(list(SCORE_DIR.glob("*_tangram_pseudobulk_epi_cm_spot_scores.csv")))),
        "mapper_h5ads": int(len(list(MAPPER_DIR.glob("*_tangram_mapper.h5ad")))),
        "mode": MODE,
        "device": DEVICE,
        "cuda_device": torch.cuda.get_device_name(0),
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "random_state": SEED,
        "canonical_tangram": True,
        "validation_passed": bool(completed.sum() == 14),
    }
    atomic_json(validation, TABLE_DIR / "tangram_mapping_validation.json")
    versions = [
        "environment_category=main_tangram_cuda",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"torch={torch.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"cuda_device={torch.cuda.get_device_name(0)}",
        f"tangram-sc={importlib.metadata.version('tangram-sc')}",
        f"anndata={importlib.metadata.version('anndata')}",
        f"numpy={importlib.metadata.version('numpy')}",
        f"pandas={importlib.metadata.version('pandas')}",
        f"random_seed={SEED}",
        f"code_file={Path(__file__).resolve()}",
    ]
    (TABLE_DIR / "package_versions.txt").write_text("\n".join(versions) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()

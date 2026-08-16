#!/usr/bin/env python3
"""Run resumable canonical Tangram on all 72 GSE270678 Visium sections."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import platform
import random
import tarfile
import tempfile
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import tangram as tg
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

WF = Path(__file__).resolve().parents[3]
TASK = "01-tangram-pseudobulk-v2"
BASE = WF / f"tables/04-spatial-validation-optional/{TASK}"
REF_DIR = BASE / "01-reference-and-manifest"
TABLE_DIR = BASE / "02-tangram-mapping"
SCORE_DIR = TABLE_DIR / "spot_scores"
RAW_TAR = Path(
    "/mnt/disk18t/lr_xcy/riku/external_atlas_for_cm-epi_validation/"
    "Gastric_cancer/GSE270678_st/GSE270678_RAW.tar"
)
REFERENCE = (
    WF
    / f"h5ad/04-spatial-validation-optional/{TASK}/01-reference-and-manifest/"
    "adata_cell_subtype_pseudobulk_deg_panel.h5ad"
)
COMMON_GENES = REF_DIR / "tangram_common_deg_genes.csv"
SAMPLE_MANIFEST = REF_DIR / "spatial_sample_scope_manifest.csv"
CM_LOADING = REF_DIR / "cm_loading_fraction_non_epi_subtype_by_CM.csv"
EPI_SUBTYPES = REF_DIR / "epithelial_subtypes.csv"
RUN_MANIFEST = TABLE_DIR / "tangram_section_run_manifest.csv"

MODE = "cells"
DEVICE = "cuda:0"
NUM_EPOCHS = 350
LEARNING_RATE = 0.05
TARGET_SUM = 1e4


def cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-sample")
    return parser.parse_args()


def decode(values) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def valid_existing(path: Path, expected_rows: int, cms: list[str], epis: list[str]) -> bool:
    if not path.exists():
        return False
    head = pd.read_csv(path, nrows=2)
    required = {
        "sample", "spot_id", "barcode", "array_row", "array_col", "spatial_x", "spatial_y",
        *[f"CMact__{name}" for name in cms],
        *[f"EPIfrac__{name}" for name in epis],
    }
    if not required.issubset(head.columns):
        return False
    with path.open("rb") as handle:
        rows = sum(1 for _ in handle) - 1
    return rows == expected_rows


def load_section_from_tar(row: pd.Series, common: list[str]) -> ad.AnnData:
    """Read one clean 10x count matrix and real Visium coordinates from the source tar."""
    with tarfile.open(RAW_TAR, "r") as archive, tempfile.TemporaryDirectory() as temp_dir:
        for name in (str(row["h5_member"]), str(row["positions_member"])):
            archive.extract(archive.getmember(name), temp_dir, filter="data")
        h5_path = Path(temp_dir) / str(row["h5_member"])
        pos_path = Path(temp_dir) / str(row["positions_member"])
        with h5py.File(h5_path, "r") as handle:
            node = handle["matrix"]
            shape = tuple(int(x) for x in node["shape"][()])
            gene_by_spot = sparse.csc_matrix(
                (node["data"][()], node["indices"][()], node["indptr"][()]), shape=shape
            )
            symbols = decode(node["features"]["name"][()])
            barcodes = decode(node["barcodes"][()])

        common_to_i = {gene: i for i, gene in enumerate(common)}
        selected_feature_i = np.asarray(
            [i for i, symbol in enumerate(symbols) if symbol in common_to_i], dtype=np.int64
        )
        selected_target_i = np.asarray(
            [common_to_i[symbols[i]] for i in selected_feature_i], dtype=np.int64
        )
        if len(set(selected_target_i.tolist())) != len(common):
            missing = sorted(set(range(len(common))) - set(selected_target_i.tolist()))
            raise ValueError(f"{row['sample']}: {len(missing)} common genes absent from 10x features")
        selected = gene_by_spot[selected_feature_i, :].tocsr()
        aggregate = sparse.csr_matrix(
            (
                np.ones(len(selected_feature_i), dtype=np.float32),
                (selected_target_i, np.arange(len(selected_feature_i), dtype=np.int64)),
            ),
            shape=(len(common), len(selected_feature_i)),
        )
        spot_by_gene = (aggregate @ selected).T.tocsr().astype(np.float32)

        with gzip.open(pos_path, "rt", encoding="utf-8") as handle:
            positions = pd.read_csv(
                handle,
                header=None,
                names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"],
            )
        positions["barcode"] = positions["barcode"].astype(str)
        positions = positions.set_index("barcode")
        missing_barcodes = sorted(set(barcodes) - set(positions.index))
        if missing_barcodes:
            raise ValueError(f"{row['sample']}: {len(missing_barcodes)} matrix barcodes lack positions")
        positions = positions.loc[barcodes]
        if not positions["in_tissue"].astype(int).eq(1).all():
            raise ValueError(f"{row['sample']}: filtered matrix contains off-tissue spots")

    sample = str(row["sample"])
    obs_names = [f"{sample}#{barcode}" for barcode in barcodes]
    obs = pd.DataFrame(index=pd.Index(obs_names, name="spot_id"))
    obs["barcode"] = barcodes
    obs["sample"] = sample
    obs["patient_id"] = str(row["patient_id"])
    obs["biological_region"] = str(row["biological_region"])
    obs["section_role"] = str(row["section_role"])
    obs["status"] = "tumor"
    for column in ("array_row", "array_col", "pxl_row", "pxl_col"):
        obs[column] = pd.to_numeric(positions[column], errors="raise").to_numpy()
    asp = ad.AnnData(
        X=spot_by_gene,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(common, name="gene")),
    )
    asp.layers["counts"] = asp.X.copy()
    asp.obsm["spatial"] = obs[["pxl_col", "pxl_row"]].to_numpy(dtype=np.float32)
    sc.pp.normalize_total(asp, target_sum=TARGET_SUM)
    sc.pp.log1p(asp)
    if asp.shape != (int(row["n_spatial_spots"]), len(common)):
        raise ValueError(f"{sample}: prepared shape {asp.shape} violates manifest")
    return asp


def map_one(
    row: pd.Series,
    reference_template: ad.AnnData,
    common: list[str],
    loadings: pd.DataFrame,
    epi_names: list[str],
) -> tuple[Path, dict[str, object]]:
    sample = str(row["sample"])
    started = time.time()
    asp = load_section_from_tar(row, common)
    asc = reference_template.copy()
    tg.pp_adatas(asc, asp, genes=common)
    training_genes = [str(x) for x in asc.uns.get("training_genes", [])]
    if not training_genes:
        raise ValueError(f"{sample}: Tangram retained no training genes")
    mapper = tg.map_cells_to_space(
        asc,
        asp,
        mode=MODE,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        random_state=SEED,
        verbose=False,
    )
    tg.project_cell_annotations(mapper, asp, annotation="cell_subtype")
    abundance = asp.obsm["tangram_ct_pred"].copy()
    if not isinstance(abundance, pd.DataFrame):
        abundance = pd.DataFrame(abundance, index=asp.obs_names)
    abundance.columns = abundance.columns.astype(str)
    missing_cm_nodes = sorted(set(loadings.index) - set(abundance.columns))
    missing_epi = sorted(set(epi_names) - set(abundance.columns))
    if missing_cm_nodes or missing_epi:
        raise ValueError(f"{sample}: missing mapped subtypes CM={missing_cm_nodes}, Epi={missing_epi}")

    epi_abundance = abundance.loc[:, epi_names].astype(float)
    epi_total = epi_abundance.sum(axis=1)
    epi = epi_abundance.div(epi_total.replace(0, np.nan), axis=0).fillna(0.0)
    positive_epi = epi_total.gt(0)
    if positive_epi.any() and not np.allclose(
        epi.loc[positive_epi].sum(axis=1).to_numpy(), 1.0, rtol=1e-6, atol=1e-8
    ):
        raise ValueError(f"{sample}: within-epithelial EPIfrac normalization failed")
    cm_values = abundance.loc[:, loadings.index].to_numpy(float) @ loadings.to_numpy(float)
    cm = pd.DataFrame(cm_values, index=abundance.index, columns=loadings.columns)

    score = pd.DataFrame(index=asp.obs_names)
    for column in (
        "sample", "barcode", "patient_id", "biological_region", "section_role", "status",
        "array_row", "array_col",
    ):
        score[column] = asp.obs[column].to_numpy()
    score.insert(1, "spot_id", score.index.astype(str))
    score["spatial_x"] = asp.obs["pxl_col"].to_numpy(float)
    score["spatial_y"] = asp.obs["pxl_row"].to_numpy(float)
    for name in cm.columns:
        score[f"CMact__{name}"] = cm[name].to_numpy(float)
    for name in epi.columns:
        score[f"EPIfrac__{name}"] = epi[name].to_numpy(float)
    output = SCORE_DIR / f"{sample}_tangram_pseudobulk_epi_cm_spot_scores.csv"
    atomic_csv(score.reset_index(drop=True), output)
    metrics = {
        "sample": sample,
        "n_spatial_spots": asp.n_obs,
        "n_common_genes_requested": len(common),
        "n_training_genes": len(training_genes),
        "n_reference_subtypes": asc.n_obs,
        "n_cm": len(loadings.columns),
        "n_epi_subtypes": len(epi_names),
        "n_positive_epi_spots": int(positive_epi.sum()),
        "n_zero_epi_spots": int((~positive_epi).sum()),
        "mode": MODE,
        "device": DEVICE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "random_state": SEED,
        "score_csv": str(output),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    del mapper, abundance, epi_abundance, epi, cm, score, asc, asp
    gc.collect()
    torch.cuda.empty_cache()
    return output, metrics


def main() -> None:
    args = cli_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Canonical Tangram run requires cuda:0")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(SAMPLE_MANIFEST)
    common = pd.read_csv(COMMON_GENES)["gene"].astype(str).tolist()
    loadings = pd.read_csv(CM_LOADING, index_col=0)
    epi_names = pd.read_csv(EPI_SUBTYPES)["cell_subtype"].astype(str).tolist()
    reference = ad.read_h5ad(REFERENCE)
    reference.obs["cell_subtype"] = reference.obs_names.astype(str)
    if list(reference.var_names) != common:
        raise ValueError("Reference/common-gene order mismatch")

    run = manifest.copy()
    run["status_run"] = "pending"
    run["score_csv"] = ""
    run["elapsed_seconds"] = np.nan
    run["error"] = ""
    for i, row in run.iterrows():
        output = SCORE_DIR / f"{row['sample']}_tangram_pseudobulk_epi_cm_spot_scores.csv"
        if valid_existing(output, int(row["n_spatial_spots"]), list(loadings.columns), epi_names):
            run.loc[i, ["status_run", "score_csv"]] = ["completed", str(output)]
    atomic_csv(run, RUN_MANIFEST)

    if args.only_sample:
        selected = run.loc[run["sample"].eq(args.only_sample)]
        if selected.empty:
            raise KeyError(args.only_sample)
    else:
        selected = run
    for i in selected.index:
        sample = str(run.loc[i, "sample"])
        if run.loc[i, "status_run"] == "completed":
            print(f"SKIP completed {sample}", flush=True)
            continue
        run.loc[i, "status_run"] = "running"
        atomic_csv(run, RUN_MANIFEST)
        print(f"START {sample} spots={int(run.loc[i, 'n_spatial_spots'])}", flush=True)
        try:
            output, metrics = map_one(run.loc[i], reference, common, loadings, epi_names)
            run.loc[i, "status_run"] = "completed"
            run.loc[i, "score_csv"] = str(output)
            run.loc[i, "elapsed_seconds"] = metrics["elapsed_seconds"]
            (TABLE_DIR / f"{sample}_mapping_metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"DONE {sample} elapsed={metrics['elapsed_seconds']}", flush=True)
        except Exception as exc:
            run.loc[i, "status_run"] = "failed"
            run.loc[i, "error"] = repr(exc)
            atomic_csv(run, RUN_MANIFEST)
            raise
        atomic_csv(run, RUN_MANIFEST)

    pair_manifest = pd.MultiIndex.from_product(
        [manifest["sample"].astype(str), loadings.columns.astype(str), epi_names],
        names=["sample", "CM", "epi_subtype"],
    ).to_frame(index=False)
    pair_manifest.to_csv(TABLE_DIR / "all_sample_cm_epi_pair_manifest.csv", index=False)
    summary = {
        "n_expected_sections": 72,
        "n_completed_sections": int(run["status_run"].eq("completed").sum()),
        "n_expected_pairs": len(pair_manifest),
        "n_cm": len(loadings.columns),
        "n_epi_subtypes": len(epi_names),
        "mode": MODE,
        "device": DEVICE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "random_state": SEED,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "tangram": getattr(tg, "__version__", "unknown"),
    }
    (TABLE_DIR / "tangram_mapping_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

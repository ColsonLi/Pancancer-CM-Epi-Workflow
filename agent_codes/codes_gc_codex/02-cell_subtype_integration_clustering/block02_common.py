#!/usr/bin/env python3
"""Shared, dependency-light contracts for gastric-cancer Block 02."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
WORKFLOW_ROOT = CODE_DIR.parents[1]
MODULE = "02-cell_subtype_integration_clustering"
RUN_VERSION = os.environ.get("GC_RUN_VERSION", "").strip()
SEED = 42
PCS_VALUES = tuple(range(10, 51, 5))
NN_VALUES = tuple(range(10, 51, 5))
RES_VALUES = tuple(round(x / 10, 1) for x in range(1, 11))
EXPECTED_CANDIDATES = len(PCS_VALUES) * len(NN_VALUES) * len(RES_VALUES)
BROAD_COLUMN_CANDIDATES = ("leiden_coarse", "cell_type", "broad_cell_type", "celltype_broad")


def task_root(category: str, task: str) -> Path:
    root = WORKFLOW_ROOT / category / MODULE / task
    return root / RUN_VERSION if RUN_VERSION else root


CANONICAL_MARKERS = {
    "epithelial": {
        "pit_mucous": {"MUC5AC", "GKN1", "GKN2", "TFF1", "TFF2", "KLF4"},
        "mucous_neck": {"MUC6", "TFF2", "AQP5", "WFDC2", "LYZ"},
        "chief": {"PGA3", "PGA4", "PGA5", "PGC", "LIPF", "GIF"},
        "parietal": {"ATP4A", "ATP4B", "KCNQ1", "KCNE2", "CLIC6"},
        "intestinal_like": {"MUC2", "TFF3", "FABP1", "FABP2", "KRT20", "CDX2"},
        "enteroendocrine": {"CHGA", "CHGB", "SCG5", "NEUROD1", "PCSK1N"},
        "tuft": {"POU2F3", "TRPM5", "SOX9", "AVIL", "IL17RB"},
        "basal_tumor_like": {"KRT17", "KRT5", "KRT14", "TP63", "SFN"},
        "epithelial": {"EPCAM", "KRT8", "KRT18", "KRT19", "KRT7"},
    },
    "t_nk": {
        "cd4_naive": {"CCR7", "LTB", "IL7R", "MAL", "TCF7", "LEF1"},
        "cd4_memory": {"IL7R", "LTB", "MALAT1", "LTST1", "CXCR4"},
        "treg": {"FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2"},
        "cd8_cytotoxic": {"CD8A", "CD8B", "CCL5", "NKG7", "GZMK", "GZMH"},
        "exhausted_t": {"PDCD1", "HAVCR2", "LAG3", "TOX", "CXCL13", "TIGIT"},
        "nk": {"NKG7", "GNLY", "PRF1", "KLRD1", "FCGR3A", "XCL1", "XCL2"},
        "t_cell": {"CD3D", "CD3E", "CD3G", "TRBC1", "TRBC2"},
    },
    "b_plasma": {
        "naive_b": {"MS4A1", "CD79A", "TCL1A", "IGHM", "IGHD", "HVCN1"},
        "memory_b": {"CD79A", "CD37", "CD27", "AIM2", "BANK1"},
        "germinal_center_b": {"BCL6", "AICDA", "RGS13", "MEF2B", "CD83"},
        "plasma": {"MZB1", "JCHAIN", "SDC1", "DERL3", "XBP1", "IGHG1"},
        "b_cell": {"CD79A", "CD79B", "CD74", "HLA-DRA", "MS4A1", "CD37"},
    },
    "myeloid": {
        "classical_monocyte": {"S100A8", "S100A9", "LYZ", "CTSS", "FCN1", "VCAN"},
        "nonclassical_monocyte": {"LST1", "FCGR3A", "MS4A7", "IFITM3", "LILRB1"},
        "macrophage": {"C1QA", "C1QB", "C1QC", "APOE", "CD68", "CTSB"},
        "spp1_macrophage": {"SPP1", "APOC1", "TREM2", "GPNMB", "CTSL"},
        "dendritic": {"FCER1A", "CD1C", "CLEC10A", "CST3", "HLA-DQA1"},
        "dc_lamp3": {"LAMP3", "CCR7", "FSCN1", "CCL19", "CCL22"},
        "neutrophil": {"CSF3R", "FCGR3B", "CXCR2", "FPR1", "MNDA"},
        "myeloid": {"LYZ", "LST1", "TYROBP", "FCER1G", "AIF1"},
    },
    "stromal": {
        "fibroblast": {"COL1A1", "COL1A2", "DCN", "COL3A1", "LUM", "PDGFRA"},
        "myofibroblast": {"ACTA2", "TAGLN", "MYL9", "TPM2", "COL1A1"},
        "pericyte": {"RGS5", "CSPG4", "MCAM", "PDGFRB", "ABCC9", "KCNJ8"},
        "smooth_muscle": {"MYH11", "ACTG2", "CNN1", "DES", "TAGLN"},
        "telocyte": {"PDGFRA", "COL15A1", "FOXL1", "WNT2B", "BMP4"},
    },
    "endothelial": {
        "capillary": {"CA4", "RGCC", "VWF", "PECAM1", "EMCN", "KDR"},
        "arterial": {"EFNB2", "GJA5", "SOX17", "FBLN5", "HEY1"},
        "venous": {"ACKR1", "SELP", "VWF", "NR2F2", "VCAM1"},
        "lymphatic": {"PROX1", "PDPN", "LYVE1", "FLT4", "CCL21"},
        "tip_angiogenic": {"ESM1", "KDR", "ANGPT2", "APLNR", "PGF"},
        "endothelial": {"PECAM1", "VWF", "KDR", "ENG", "EMCN"},
    },
    "mast": {"mast": {"TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2", "HDC"}},
    "schwann": {"schwann": {"S100B", "SOX10", "PLP1", "S100A1", "MPZ", "NGFR"}},
    "cycling": {"cycling": {"MKI67", "TOP2A", "UBE2C", "CENPF", "STMN1", "PCNA"}},
}

STATE_MARKERS = {
    "cycling": {"MKI67", "TOP2A", "UBE2C", "CENPF", "STMN1", "PCNA"},
    "interferon_response": {"ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "OAS1", "IFI6"},
    "stress_response": {"FOS", "JUN", "JUNB", "DUSP1", "HSPA1A", "HSPA1B"},
    "hypoxia": {"CA9", "VEGFA", "NDRG1", "ENO1", "LDHA", "BNIP3"},
    "antigen_presentation": {"HLA-DRA", "HLA-DRB1", "HLA-DPA1", "CD74", "B2M"},
    "cytotoxic": {"NKG7", "GNLY", "PRF1", "GZMB", "GZMH", "CCL5"},
    "emt": {"VIM", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "FN1"},
}


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return value or "lineage"


def deterministic_unique_tokens(items: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Return stable globally unique tokens keyed by stable item IDs.

    A token that is already unique is retained.  Every member of a duplicated
    token group receives a readable ``__<item-slug>`` suffix, with a stable
    numeric suffix only if that still collides with another final token.
    """
    pairs = sorted((str(key), str(token)) for key, token in items)
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("deterministic_unique_tokens keys must be unique")
    token_counts: dict[str, int] = {}
    for _, token in pairs:
        token_counts[token] = token_counts.get(token, 0) + 1
    result: dict[str, str] = {}
    used: set[str] = set()
    for key, token in pairs:
        candidate = token if token_counts[token] == 1 else f"{token}__{slugify(key)}"
        base = candidate
        ordinal = 2
        while candidate in used:
            candidate = f"{base}__{ordinal}"
            ordinal += 1
        result[key] = candidate
        used.add(candidate)
    return result


def partial_h5ad_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.partial{path.suffix}")


def refuse_h5ad_residual(path: Path) -> None:
    partial = partial_h5ad_path(path)
    if path.exists() or partial.exists():
        raise FileExistsError(f"Refusing completed/residual h5ad overwrite: {path} / {partial}")


def atomic_write_h5ad(adata, path: Path) -> None:
    """Write a new h5ad through a same-directory partial, never overwriting."""
    refuse_h5ad_residual(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_h5ad_path(path)
    adata.write_h5ad(partial)
    os.replace(partial, path)


def prefix_for(label: str, slug: str) -> str:
    text = f"{label} {slug}".lower()
    rules = (
        ("epithelial", "Epi"), ("plasma", "Plasma"), ("myeloid", "Mye"),
        ("macroph", "Mye"), ("mast", "Mast"), ("t cell", "T"), ("t_cell", "T"), ("nk", "NK"),
        ("b cell", "B"), ("b_cell", "B"), ("endothel", "Endo"),
        ("fibro", "Fibro"), ("strom", "Stromal"), ("pericy", "Peri"),
        ("smooth", "SMC"), ("schwann", "Schwann"),
        ("cycling", "Cycling"),
    )
    for needle, prefix in rules:
        if needle in text:
            return prefix
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", label) if p]
    return "".join(p[0].upper() for p in parts)[:10] or "Cell"


def marker_family(label: str) -> str:
    text = label.lower()
    if "cycling" in text or "prolifer" in text: return "cycling"
    if "epithelial" in text: return "epithelial"
    if "mast" in text: return "mast"
    if any(x in text for x in ("t cell", "t_cell", "nk", "lymphoid")): return "t_nk"
    if any(x in text for x in ("b cell", "b_cell", "plasma")): return "b_plasma"
    if any(x in text for x in ("myeloid", "monocyte", "macroph", "dendritic", "neutro")): return "myeloid"
    if any(x in text for x in ("strom", "fibro", "pericy", "smooth", "mesench")): return "stromal"
    if "endothel" in text: return "endothelial"
    if "schwann" in text or "glial" in text: return "schwann"
    return ""


def cell_id_digest(ids: Iterable[str]) -> str:
    h = hashlib.sha256()
    for value in ids:
        h.update(str(value).encode("utf-8")); h.update(b"\0")
    return h.hexdigest()


def candidate_id(n_pcs: int, n_neighbors: int, resolution: float) -> str:
    return f"pcs{n_pcs}_nn{n_neighbors}_res{resolution:.1f}".replace(".", "p")


def candidate_path(root: Path, lineage: str, n_pcs: int, n_neighbors: int, resolution: float) -> Path:
    return root / lineage / f"pcs{n_pcs}_nn{n_neighbors}" / f"{candidate_id(n_pcs, n_neighbors, resolution)}.npz"


def terminal_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["metadata"].item()))


def save_candidate_once(path: Path, labels: np.ndarray | None, metadata: dict) -> None:
    """Write an immutable terminal candidate; existing files are never replaced."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.asarray([], dtype=np.int16) if labels is None else np.asarray(labels, dtype=np.int16)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, labels=payload, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    tmp.replace(path)


def normalized_entropy(values: np.ndarray) -> float:
    _, counts = np.unique(values.astype(str), return_counts=True)
    if counts.size <= 1: return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(counts.size))


def weighted_cluster_mixing(labels: np.ndarray, values: np.ndarray) -> float:
    total = len(labels)
    if total == 0 or len(np.unique(values.astype(str))) <= 1: return 1.0
    score = 0.0
    for cluster in np.unique(labels):
        mask = labels == cluster
        score += float(mask.sum() / total) * normalized_entropy(values[mask])
    return float(score)


def assert_grid_contract() -> None:
    assert PCS_VALUES == (10, 15, 20, 25, 30, 35, 40, 45, 50)
    assert NN_VALUES == (10, 15, 20, 25, 30, 35, 40, 45, 50)
    assert RES_VALUES == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    assert EXPECTED_CANDIDATES == 810

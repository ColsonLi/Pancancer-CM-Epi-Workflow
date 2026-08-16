#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
from anndata import AnnData


ROOT = Path(__file__).resolve().parents[4]
GRID_VARIANT = os.environ.get("GC_CLUSTER_GRID_VARIANT", "default")
if GRID_VARIANT == "expanded-v2":
    GRID_DIR = "05-clustering-parameter-search-expanded-v2"
    TEST_N_PCS = 50
    TEST_N_NEIGHBORS = 50
    TEST_RESOLUTION = 1.0
elif GRID_VARIANT == "default":
    GRID_DIR = "05-clustering-parameter-search"
    TEST_N_PCS = 10
    TEST_N_NEIGHBORS = 10
    TEST_RESOLUTION = 0.2
else:
    raise ValueError(f"Unsupported GC_CLUSTER_GRID_VARIANT: {GRID_VARIANT}")
OUTPUT = ROOT / f"epi-cm-core-workflow/tables/01-celltype_integration_clustering/{GRID_DIR}/gpu_preflight.csv"
if OUTPUT.exists():
    raise FileExistsError(f"Refusing to overwrite existing artifact: {OUTPUT}")

rng = np.random.default_rng(42)
embedding = rng.normal(size=(5000, max(50, TEST_N_PCS))).astype(np.float32)
adata = AnnData(
    X=embedding.copy(),
    obs=pd.DataFrame(index=[f"cell_{i}" for i in range(len(embedding))]),
    obsm={"X_pca_inte": embedding.copy()},
)
rsc.get.anndata_to_GPU(adata)
rsc.pp.neighbors(
    adata,
    n_neighbors=TEST_N_NEIGHBORS,
    n_pcs=TEST_N_PCS,
    use_rep="X_pca_inte",
    algorithm="cagra",
    random_state=42,
)
rsc.tl.umap(adata, random_state=42)
cluster_key = f"leiden_res{str(TEST_RESOLUTION).replace('.', 'p')}"
rsc.tl.leiden(adata, resolution=TEST_RESOLUTION, key_added=cluster_key, random_state=42)

coords = adata.obsm["X_umap"]
if isinstance(coords, cp.ndarray):
    coords = cp.asnumpy(coords)
checks = pd.DataFrame(
    [
        {"check": "neighbors_connectivities", "passed": "connectivities" in adata.obsp and adata.obsp["connectivities"].shape == (5000, 5000), "observed": str(adata.obsp["connectivities"].shape)},
        {"check": "umap_finite", "passed": coords.shape == (5000, 2) and np.isfinite(coords).all(), "observed": str(coords.shape)},
        {"check": "leiden_complete", "passed": cluster_key in adata.obs and not adata.obs[cluster_key].isna().any(), "observed": adata.obs[cluster_key].nunique()},
    ]
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
checks.to_csv(OUTPUT, index=False)
if not checks["passed"].all():
    raise RuntimeError("Clustering GPU preflight failed")
print(checks.to_string(index=False))

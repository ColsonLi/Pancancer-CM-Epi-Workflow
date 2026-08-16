#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
from scipy import sparse

SEED = 42
ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "epi-cm-core-workflow/tables/01-celltype_integration_clustering/04-integration-harmony/gpu_preflight_attempt02.csv"


def main() -> None:
    np.random.seed(SEED)
    x = sparse.csr_matrix(np.random.poisson(1, (120, 200)).astype(np.float32))
    a = ad.AnnData(x)
    a.obs["sample"] = pd.Categorical(np.repeat(["s1", "s2", "s3"], 40))
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.var_names = [f"g{i}" for i in range(a.n_vars)]
    rows: list[dict[str, object]] = []
    try:
        rsc.get.anndata_to_GPU(a)
        rsc.pp.normalize_total(a, target_sum=1e4)
        rsc.pp.log1p(a)
        rsc.pp.highly_variable_genes(a, n_top_genes=100, flavor="seurat", batch_key="sample")
        a = a[:, a.var["highly_variable"]].copy()
        rsc.pp.scale(a, max_value=10)
        rsc.tl.pca(a, n_comps=20, random_state=SEED)
        rsc.pp.harmony_integrate(
            a, key="sample", basis="X_pca", adjusted_basis="X_pca_inte",
            correction_method="original", use_gemm=False, max_iter_harmony=30,
            random_state=SEED,
        )
        # Harmony returns its adjusted embedding on CPU in this package version,
        # even though the preceding RAPIDS operations execute on GPU.
        finite = bool(np.isfinite(np.asarray(a.obsm["X_pca_inte"])).all())
        rows.append({"check": "rapids_harmony_gpu", "passed": finite, "detail": str(a.obsm["X_pca_inte"].shape)})
    except Exception as exc:
        rows.append({"check": "rapids_harmony_gpu", "passed": False, "detail": repr(exc)})
    rows.append({"check": "cuda_device", "passed": cp.cuda.runtime.getDeviceCount() > 0,
                 "detail": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    OUT.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    if not all(bool(r["passed"]) for r in rows):
        raise RuntimeError(rows)


if __name__ == "__main__":
    main()

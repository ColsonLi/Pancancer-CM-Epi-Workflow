#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

SEED = 42
MIN_ARI = 0.98
MAX_AUTOMATIC_RANK = 20
ROOT = Path(__file__).resolve().parents[4]
WF = ROOT / "epi-cm-core-workflow"
SOURCE = WF / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_harmony.h5ad"
TABLE = WF / "tables/01-celltype_integration_clustering/05-clustering-parameter-search"
RANKING = TABLE / "automatic_candidate_ranking.csv"
OUTPUT = TABLE / "clean_rerun_reproducibility_screen_rank3to20.csv"


def token(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    ranking = pd.read_csv(RANKING).sort_values("automatic_rank")
    candidates = ranking[
        ranking["automatic_rank"].between(3, MAX_AUTOMATIC_RANK)
    ].copy()
    if candidates.empty:
        raise RuntimeError("No ranked candidates available for reproducibility screen")

    full = ad.read_h5ad(SOURCE, backed="r")
    try:
        n_obs = full.n_obs
        obs = full.obs[["sample", "series", "status"]].copy()
        ids = full.obs_names.copy()
        rep = np.asarray(full.obsm["X_pca_inte"], dtype=np.float32)
    finally:
        full.file.close()

    rows = []
    for candidate in candidates.itertuples(index=False):
        pcs = int(candidate.n_pcs)
        nn = int(candidate.n_neighbors)
        res = float(candidate.resolution)
        rank = int(candidate.automatic_rank)
        key = f"leiden_res{token(res)}"
        checkpoint = (
            TABLE
            / f"pcs-{pcs}_nn-{nn}_res-0p1-1p0"
            / f"labels_{key}.npz"
        )
        with np.load(checkpoint, allow_pickle=False) as z:
            expected = z["labels"].astype(str)

        run = ad.AnnData(
            sparse.csr_matrix((n_obs, 1), dtype=np.float32), obs=obs.copy()
        )
        run.obs_names = ids.copy()
        run.obsm["X_pca_inte"] = rep
        rsc.get.anndata_to_GPU(run)
        rsc.pp.neighbors(
            run, n_neighbors=nn, n_pcs=pcs, use_rep="X_pca_inte", random_state=SEED
        )
        # Match the grid's exact graph -> UMAP -> Leiden execution order.
        rsc.tl.umap(run, random_state=SEED)
        rsc.tl.leiden(run, resolution=res, key_added=key, random_state=SEED)
        observed = run.obs[key].astype(str).to_numpy()
        ari = float(adjusted_rand_score(expected, observed))
        grid_k = int(pd.Series(expected).nunique())
        rerun_k = int(pd.Series(observed).nunique())
        accepted = bool(ari >= MIN_ARI and grid_k == rerun_k)
        rows.append({
            "automatic_rank": rank,
            "n_pcs": pcs,
            "n_neighbors": nn,
            "resolution": res,
            "grid_n_clusters": grid_k,
            "clean_rerun_n_clusters": rerun_k,
            "clean_rerun_ari": ari,
            "minimum_required_ari": MIN_ARI,
            "cluster_count_equal": grid_k == rerun_k,
            "accepted_for_broad_evidence_test": accepted,
            "full_cell": True,
            "downsampled": False,
            "basis_dtype": "float32",
            "backend": "rapids-singlecell GPU",
        })
        atomic_csv(pd.DataFrame(rows), OUTPUT)
        print(
            f"rank={rank} pcs={pcs} nn={nn} res={res:.1f} "
            f"ARI={ari:.9f} grid_k={grid_k} rerun_k={rerun_k} accepted={accepted}",
            flush=True,
        )
        del run
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        if accepted:
            return
    raise RuntimeError(
        f"No reproducible candidate found through automatic rank {MAX_AUTOMATIC_RANK}"
    )


if __name__ == "__main__":
    main()

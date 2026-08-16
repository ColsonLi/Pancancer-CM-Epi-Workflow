#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
from scipy import sparse

SEED = 42
LATENT_DIMS = 30
N_NEIGHBORS = (10, 20, 30, 40)
RESOLUTIONS = (0.2, 0.3, 0.4, 0.5)

WF = Path(__file__).resolve().parents[4]
SOURCE = WF / "h5ad/01-celltype_integration_clustering_scVI/04-integration-scvi/from-scratch-scvi-latent30-v1/adata_scvi.h5ad"
OLD_MANIFEST = WF / "tables/01-celltype_integration_clustering_scVI/05-clustering-parameter-search/from-scratch-scvi-latent30-v1/candidate_review_manifest.csv"
VERSION = "from-scratch-scvi-latent30-v1-scanpy-default-v2"
FIG_ROOT = WF / f"figures/01-celltype_integration_clustering_scVI/05-clustering-parameter-search/{VERSION}"
TABLE_ROOT = WF / f"tables/01-celltype_integration_clustering_scVI/05-clustering-parameter-search/{VERSION}"


def rt(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def name(nn: int, res: float) -> str:
    return f"latent-{LATENT_DIMS}_nn-{nn}_res-{rt(res)}"


def save_default_umap(adata: ad.AnnData, color: str | list[str], output_dir: Path, suffix: str) -> Path:
    # Use Scanpy's own saving interface and plotting defaults. In particular,
    # do not pass size, legend_loc, legend_fontsize, title, palette, figsize,
    # alpha, return_fig, or any Matplotlib savefig arguments.
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = output_dir / f"umap_{suffix}.pdf"
    if expected.exists():
        raise FileExistsError(expected)
    sc.settings.figdir = output_dir
    sc.pl.umap(adata, color=color, show=False, save=f"_{suffix}.pdf")
    if not expected.is_file():
        raise RuntimeError(f"Scanpy did not create expected output: {expected}")
    return expected


def main() -> None:
    started = time.time()
    if FIG_ROOT.exists() or TABLE_ROOT.exists():
        raise FileExistsError("Default-Scanpy redraw output already exists")
    if not SOURCE.is_file() or not OLD_MANIFEST.is_file():
        raise FileNotFoundError("scVI source or completed candidate manifest is missing")
    if not cp.cuda.is_available():
        raise RuntimeError("CUDA is required to reconstruct the saved grid UMAPs")

    source = ad.read_h5ad(SOURCE)
    latent = np.asarray(source.obsm["X_scVI"], dtype=np.float32)
    obs = source.obs[["sample", "series", "status"]].copy()
    if latent.shape != (source.n_obs, LATENT_DIMS) or not np.isfinite(latent).all():
        raise RuntimeError(f"Invalid X_scVI: {latent.shape}")
    n_obs = source.n_obs
    del source
    gc.collect()

    old = pd.read_csv(OLD_MANIFEST).set_index("candidate")
    stage_fig = FIG_ROOT.with_name(f".{VERSION}.staging")
    stage_table = TABLE_ROOT.with_name(f".{VERSION}.staging")
    stage_fig.mkdir(parents=True, exist_ok=False)
    stage_table.mkdir(parents=True, exist_ok=False)
    rows = []

    for nn in N_NEIGHBORS:
        work = ad.AnnData(X=sparse.csr_matrix((n_obs, 1), dtype=np.float32), obs=obs.copy())
        work.obsm["X_scVI"] = latent.copy()
        rsc.get.anndata_to_GPU(work, convert_all=True)
        rsc.pp.neighbors(
            work,
            use_rep="X_scVI",
            n_neighbors=nn,
            random_state=SEED,
            algorithm="cagra",
        )
        rsc.tl.umap(work, random_state=SEED)
        for res in RESOLUTIONS:
            key = f"scvi_leiden_res{rt(res)}"
            rsc.tl.leiden(work, resolution=res, key_added=key, random_state=SEED)
        rsc.get.anndata_to_CPU(work, convert_all=True)

        # Batch plots also use the unchanged Scanpy plotting and save interface.
        aggregate = f"latent-{LATENT_DIMS}_nn-{nn}_res-all"
        save_default_umap(work, ["series", "status"], stage_fig / aggregate, f"{aggregate}_series_status")
        save_default_umap(work, "sample", stage_fig / aggregate, f"{aggregate}_sample")

        for res in RESOLUTIONS:
            candidate = name(nn, res)
            key = f"scvi_leiden_res{rt(res)}"
            n_clusters = int(work.obs[key].nunique())
            expected = int(old.loc[candidate, "n_clusters"])
            if n_clusters != expected:
                raise RuntimeError(
                    f"Deterministic redraw cluster count mismatch for {candidate}: {n_clusters} != {expected}"
                )
            save_default_umap(work, key, stage_fig / candidate, candidate)
            rows.append({
                "candidate": candidate,
                "source_h5ad": str(SOURCE.relative_to(WF)),
                "integration_source": "scVI",
                "use_rep": "X_scVI",
                "latent_n_dims_used": LATENT_DIMS,
                "n_neighbors": nn,
                "resolution": res,
                "cluster_key": key,
                "n_clusters": n_clusters,
                "plot_interface": "scanpy.sc.pl.umap",
                "point_size_argument": "None (Scanpy automatic default)",
                "automatic_point_size": 120000.0 / n_obs,
                "manual_title": False,
                "manual_legend_position": False,
                "manual_legend_fontsize": False,
                "candidate_h5ad_saved": False,
                "output_pdf": str((FIG_ROOT / candidate / f"umap_{candidate}.pdf").relative_to(WF)),
            })
        del work
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()

    manifest = pd.DataFrame(rows).sort_values(["n_neighbors", "resolution"])
    if len(manifest) != 16:
        raise RuntimeError("Expected 16 redrawn candidates")
    manifest.to_csv(stage_table / "redraw_manifest.csv", index=False)
    completion = {
        "all_passed": True,
        "completed_candidate_pdfs": 16,
        "completed_batch_pdfs": 8,
        "n_obs": n_obs,
        "scanpy_default_point_size": 120000.0 / n_obs,
        "manual_point_size_removed": True,
        "cluster_counts_match_original_grid": True,
        "candidate_h5ad_saved": False,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (stage_table / "redraw_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    (stage_table / "readme.txt").write_text(
        "Redrew the completed scVI grid with the default Scanpy UMAP plotting interface.\n"
        "Figures are saved by sc.pl.umap(..., show=False, save='_name.pdf'); Matplotlib savefig is not used.\n"
        "The candidate calls do not provide size, legend_loc, legend_fontsize, title, palette, figsize, alpha, or return_fig.\n"
        "Scanpy therefore uses its automatic point size, 120000/n_obs = 0.262009.\n"
        "The graphs, UMAP coordinates, and Leiden labels were deterministically reconstructed with the original parameters; all cluster counts match.\n"
        "No candidate h5ad was saved and the previous figures were preserved.\n",
        encoding="utf-8",
    )
    FIG_ROOT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage_fig, FIG_ROOT)
    os.replace(stage_table, TABLE_ROOT)
    print(json.dumps(completion, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

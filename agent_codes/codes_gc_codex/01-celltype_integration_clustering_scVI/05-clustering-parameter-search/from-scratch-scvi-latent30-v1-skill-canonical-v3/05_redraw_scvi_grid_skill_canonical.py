#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import random
import time
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

SEED = 42
LATENT_DIMS = 30
N_NEIGHBORS = (10, 20, 30, 40)
RESOLUTIONS = (0.2, 0.3, 0.4, 0.5)
VERSION = "from-scratch-scvi-latent30-v1"
STAGE_TAG = "from-scratch-scvi-latent30-v1-skill-overwrite-random-init"

random.seed(SEED)
np.random.seed(SEED)

WF = Path(__file__).resolve().parents[4]
SOURCE = WF / "h5ad/01-celltype_integration_clustering_scVI/04-integration-scvi/from-scratch-scvi-latent30-v1/adata_scvi.h5ad"
FIG_ROOT = WF / f"figures/01-celltype_integration_clustering_scVI/05-clustering-parameter-search/{VERSION}"
TABLE_ROOT = WF / f"tables/01-celltype_integration_clustering_scVI/05-clustering-parameter-search/{VERSION}"
STAGE_FIG = FIG_ROOT.with_name(f".{STAGE_TAG}.staging")
STAGE_TABLE = TABLE_ROOT.with_name(f".{STAGE_TAG}.staging")

# Updated skill contract for high-cell-count Scanpy embeddings.
sc.settings.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
sc.settings.autoshow = False


def res_text(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def candidate_name(nn: int, res: float) -> str:
    return f"latent-{LATENT_DIMS}_nn-{nn}_res-{res_text(res)}"


def scanpy_umap_save(adata: ad.AnnData, color: str | list[str], directory: Path, suffix: str, **kwargs) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"umap_{suffix}.pdf"
    if output.exists():
        raise FileExistsError(output)
    sc.settings.figdir = directory
    # Canonical skill route: Scanpy owns both drawing and saving.
    sc.pl.umap(adata, color=color, save=f"_{suffix}.pdf", **kwargs)
    if not output.is_file():
        raise RuntimeError(f"Scanpy did not create expected PDF: {output}")
    return output


def batch_metrics(adata: ad.AnnData, cluster_key: str, nn: int) -> pd.DataFrame:
    graph = adata.obsp["distances"].tocsr()
    total = int(graph.nnz)
    arrays = {column: adata.obs[column].astype(str).to_numpy() for column in ("sample", "series", "status")}
    same = {column: 0 for column in arrays}
    for start in range(0, adata.n_obs, 50000):
        stop = min(start + 50000, adata.n_obs)
        left = np.repeat(np.arange(start, stop), np.diff(graph.indptr[start : stop + 1]))
        right = graph.indices[graph.indptr[start] : graph.indptr[stop]]
        for column, values in arrays.items():
            same[column] += int(np.count_nonzero(values[left] == values[right]))
    rows = [
        {
            "n_neighbors": nn,
            "metric": f"same_{column}_neighbor_edge_fraction",
            "value": same[column] / total,
            "n_directed_edges": total,
            "interpretation": "descriptive only; lower implies more local mixing but biology and cohort composition confound this metric",
        }
        for column in arrays
    ]
    clusters = adata.obs[cluster_key].astype(str).to_numpy()
    for column, values in arrays.items():
        rows.extend([
            {
                "n_neighbors": nn,
                "metric": f"cluster_vs_{column}_adjusted_rand_index",
                "value": adjusted_rand_score(clusters, values),
                "n_directed_edges": np.nan,
                "interpretation": "descriptive cluster-batch association; closer to zero means weaker adjusted agreement",
            },
            {
                "n_neighbors": nn,
                "metric": f"cluster_vs_{column}_normalized_mutual_information",
                "value": normalized_mutual_info_score(clusters, values),
                "n_directed_edges": np.nan,
                "interpretation": "descriptive cluster-batch association; lower means weaker association",
            },
        ])
    return pd.DataFrame(rows)


def promote_tree(stage: Path, final: Path) -> None:
    for staged_file in sorted(path for path in stage.rglob("*") if path.is_file()):
        target = final / staged_file.relative_to(stage)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_file, target)
    for directory in sorted((path for path in stage.rglob("*") if path.is_dir()), reverse=True):
        directory.rmdir()
    stage.rmdir()


def main() -> None:
    started = time.time()
    for path in (STAGE_FIG, STAGE_TABLE):
        if path.exists():
            raise FileExistsError(f"Refusing staging reuse: {path}")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if not cp.cuda.is_available():
        raise RuntimeError("CUDA is required to reconstruct the scVI grid")

    source = ad.read_h5ad(SOURCE)
    latent = np.asarray(source.obsm["X_scVI"], dtype=np.float32)
    required_obs = ["sample", "series", "status"]
    missing = [column for column in required_obs if column not in source.obs]
    if missing or latent.shape != (source.n_obs, LATENT_DIMS) or not np.isfinite(latent).all():
        raise RuntimeError(f"Invalid scVI source: missing={missing}, latent={latent.shape}")
    obs = source.obs[required_obs].copy()
    n_obs = source.n_obs
    del source
    gc.collect()

    STAGE_FIG.mkdir(parents=True, exist_ok=False)
    STAGE_TABLE.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []

    for nn in N_NEIGHBORS:
        graph_started = time.time()
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
        # RAPIDS auto/spectral initialization produced extreme coordinates
        # (x=536, y=-1269) that compressed 99% of cells into the plot center.
        # The documented RAPIDS remedy for odd embeddings is random init.
        rsc.tl.umap(work, random_state=SEED, init_pos="random")
        for res in RESOLUTIONS:
            key = f"scvi_leiden_res{res_text(res)}"
            rsc.tl.leiden(work, resolution=res, key_added=key, random_state=SEED)
        rsc.get.anndata_to_CPU(work, convert_all=True)

        aggregate = f"latent-{LATENT_DIMS}_nn-{nn}_res-all"
        diagnostic_dir = STAGE_FIG / aggregate
        scanpy_umap_save(
            work,
            ["series", "status"],
            diagnostic_dir,
            f"{aggregate}_series_status",
            ncols=2,
            wspace=0.4,
        )
        scanpy_umap_save(
            work,
            "sample",
            diagnostic_dir,
            f"{aggregate}_sample",
            legend_fontsize=4,
        )

        for res in RESOLUTIONS:
            candidate = candidate_name(nn, res)
            key = f"scvi_leiden_res{res_text(res)}"
            n_clusters = int(work.obs[key].nunique())
            figure = scanpy_umap_save(work, key, STAGE_FIG / candidate, candidate)
            counts = (
                work.obs[key]
                .astype(str)
                .value_counts()
                .rename_axis("cluster")
                .reset_index(name="n_cells")
            )
            counts["fraction"] = counts["n_cells"] / n_obs
            candidate_table = STAGE_TABLE / candidate
            candidate_table.mkdir(parents=True, exist_ok=False)
            counts.to_csv(candidate_table / "cluster_counts.csv", index=False)
            batch_metrics(work, key, nn).to_csv(candidate_table / "batch_association_metrics.csv", index=False)
            row = {
                "candidate": candidate,
                "source_h5ad": str(SOURCE.relative_to(WF)),
                "integration_source": "scVI",
                "use_rep": "X_scVI",
                "latent_n_dims_used": LATENT_DIMS,
                "n_neighbors": nn,
                "neighbor_algorithm": "cagra",
                "resolution": res,
                "clustering_algorithm": "leiden",
                "cluster_key": key,
                "n_clusters": n_clusters,
                "seed": SEED,
                "umap_init_pos": "random",
                "umap_init_reason": "RAPIDS auto/spectral diagnostic produced extreme coordinates; random init removed outliers",
                "backend": "rapids-singlecell GPU",
                "plot_interface": "sc.pl.umap",
                "plot_save_interface": "sc.settings.figdir + save suffix",
                "figure_size_inches": "3x3",
                "dpi": 150,
                "vector_friendly": True,
                "point_size_argument": "not supplied; Scanpy default",
                "frameon_argument": "not supplied; Scanpy default",
                "custom_title": False,
                "candidate_h5ad_saved": False,
                "figure_pdf": str((FIG_ROOT / candidate / figure.name).relative_to(WF)),
                "cluster_count_table": str((TABLE_ROOT / candidate / "cluster_counts.csv").relative_to(WF)),
                "graph_to_outputs_elapsed_seconds": round(time.time() - graph_started, 3),
                "manual_selection_status": "pending",
                "code_file": str(Path(__file__).relative_to(WF)),
            }
            pd.DataFrame([row]).to_csv(candidate_table / "clustering_parameters.csv", index=False)
            rows.append(row)

        del work
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()

    manifest = pd.DataFrame(rows).sort_values(["n_neighbors", "resolution"])
    if len(manifest) != 16 or manifest["candidate"].nunique() != 16:
        raise RuntimeError("Expected 16 unique canonical candidates")
    manifest.to_csv(STAGE_TABLE / "candidate_review_manifest.csv", index=False)
    completion = {
        "all_passed": True,
        "canonical_plotting_route": True,
        "source_h5ad": str(SOURCE.relative_to(WF)),
        "n_obs": n_obs,
        "completed_candidates": 16,
        "completed_candidate_pdfs": 16,
        "completed_batch_diagnostic_pdfs": 8,
        "vector_friendly": True,
        "figure_size_inches": [3, 3],
        "dpi": 150,
        "scanpy_save_argument_used": True,
        "matplotlib_savefig_used": False,
        "manual_point_size_used": False,
        "umap_init_pos": "random",
        "spectral_init_extreme_coordinate_diagnostic": True,
        "candidate_h5ad_saved": False,
        "manual_selection_pending": True,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (STAGE_TABLE / "canonical_redraw_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    (STAGE_TABLE / "grid_completion.json").write_text(
        json.dumps({
            "all_passed": True,
            "source_h5ad": str(SOURCE.relative_to(WF)),
            "integration_source": "scVI",
            "use_rep": "X_scVI",
            "latent_n_dims_used": LATENT_DIMS,
            "expected_candidates": 16,
            "completed_candidates": 16,
            "n_obs": n_obs,
            "downsampled": False,
            "candidate_h5ad_saved": False,
            "manual_selection_pending": True,
            "umap_init_pos": "random",
            "vector_friendly": True,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    (STAGE_TABLE / "readme.txt").write_text(
        "Canonical scVI grid redraw following the updated project skill.\n"
        "Before plotting: sc.settings.set_figure_params(figsize=(3,3), dpi=150, vector_friendly=True).\n"
        "Every UMAP uses sc.settings.figdir plus sc.pl.umap(..., save='_name.pdf').\n"
        "No Matplotlib axes, return_fig, fig.savefig, manual point size, frame removal, or custom candidate title is used.\n"
        "Dense point collections are rasterized inside PDF while text, legends, and axes remain vector-editable.\n"
        "UMAP uses init_pos=random with seed=42 because RAPIDS auto/spectral initialization produced extreme coordinates that visually compressed the atlas.\n"
        "The user authorized replacement of the original grid figures/tables; this original directory now contains the current canonical outputs.\n",
        encoding="utf-8",
    )

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    promote_tree(STAGE_FIG, FIG_ROOT)
    promote_tree(STAGE_TABLE, TABLE_ROOT)
    print(json.dumps(completion, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

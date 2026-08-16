#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

SEED = 42
RUN = "from-scratch-scvi-latent30-v1"
LATENT_DIMS = 30
N_NEIGHBORS = (10, 20, 30, 40)
RESOLUTIONS = (0.2, 0.3, 0.4, 0.5)
KNN_ALGORITHM = "cagra"

random.seed(SEED)
np.random.seed(SEED)

WF = Path(__file__).resolve().parents[4]
SOURCE = WF / f"h5ad/01-celltype_integration_clustering_scVI/04-integration-scvi/{RUN}/adata_scvi.h5ad"
FIG_ROOT = WF / f"figures/01-celltype_integration_clustering_scVI/05-clustering-parameter-search/{RUN}"
TABLE_ROOT = WF / f"tables/01-celltype_integration_clustering_scVI/05-clustering-parameter-search/{RUN}"
CODE = Path(__file__).resolve()


def res_text(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def candidate_name(nn: int, res: float) -> str:
    return f"latent-{LATENT_DIMS}_nn-{nn}_res-{res_text(res)}"


def candidate_complete(nn: int, res: float) -> bool:
    name = candidate_name(nn, res)
    return (
        (FIG_ROOT / name / f"umap_{name}.pdf").is_file()
        and (TABLE_ROOT / name / "clustering_parameters.csv").is_file()
        and (TABLE_ROOT / name / "cluster_counts.csv").is_file()
        and (TABLE_ROOT / name / "batch_association_metrics.csv").is_file()
    )


def promote_file(staged: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, final)


def edge_mixing_rows(adata: ad.AnnData, nn: int) -> list[dict[str, object]]:
    graph = adata.obsp["distances"].tocsr()
    total = int(graph.nnz)
    if total == 0:
        raise RuntimeError("Neighbor distance graph is empty")
    same = {column: 0 for column in ("sample", "series", "status")}
    for start in range(0, adata.n_obs, 50000):
        stop = min(start + 50000, adata.n_obs)
        left = np.repeat(np.arange(start, stop), np.diff(graph.indptr[start : stop + 1]))
        right = graph.indices[graph.indptr[start] : graph.indptr[stop]]
        for column in same:
            values = adata.obs[column].astype(str).to_numpy()
            same[column] += int(np.count_nonzero(values[left] == values[right]))
    return [
        {
            "n_neighbors": nn,
            "metric": f"same_{column}_neighbor_edge_fraction",
            "value": same[column] / total,
            "n_directed_edges": total,
            "interpretation": "descriptive only; lower implies more local mixing but biology and cohort composition confound this metric",
        }
        for column in same
    ]


def save_batch_figures(adata: ad.AnnData, nn: int, staging: Path) -> None:
    aggregate = f"latent-{LATENT_DIMS}_nn-{nn}_res-all"
    batch_dir = staging / aggregate
    batch_dir.mkdir(parents=True, exist_ok=False)
    fig = sc.pl.umap(
        adata,
        color=["series", "status"],
        ncols=2,
        show=False,
        return_fig=True,
        frameon=False,
        size=1.0,
    )
    fig.savefig(batch_dir / f"umap_{aggregate}_series_status.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    fig = sc.pl.umap(
        adata,
        color="sample",
        legend_loc=None,
        show=False,
        return_fig=True,
        frameon=False,
        size=1.0,
        title=f"scVI latent 30; nn={nn}; sample",
    )
    fig.savefig(batch_dir / f"umap_{aggregate}_sample.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    started = time.time()
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if not cp.cuda.is_available():
        raise RuntimeError("CUDA is required for the RAPIDS scVI clustering grid")
    manifest_path = TABLE_ROOT / "candidate_review_manifest.csv"
    completion_path = TABLE_ROOT / "grid_completion.json"
    if manifest_path.exists() or completion_path.exists():
        raise FileExistsError("Completed grid manifest already exists; refusing overwrite")

    source = ad.read_h5ad(SOURCE)
    if "X_scVI" not in source.obsm:
        raise KeyError("X_scVI is missing")
    latent = np.asarray(source.obsm["X_scVI"], dtype=np.float32)
    required_obs = ["sample", "series", "status"]
    missing = [column for column in required_obs if column not in source.obs]
    if missing or latent.shape != (source.n_obs, LATENT_DIMS) or not np.isfinite(latent).all():
        raise RuntimeError(f"scVI source validation failed: missing={missing}, latent={latent.shape}")
    obs = source.obs[required_obs].copy()
    n_obs = source.n_obs
    del source
    gc.collect()

    manifest_rows: list[dict[str, object]] = []
    graph_rows: list[dict[str, object]] = []
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    for nn in N_NEIGHBORS:
        missing_res = [res for res in RESOLUTIONS if not candidate_complete(nn, res)]
        if not missing_res:
            for res in RESOLUTIONS:
                row = pd.read_csv(TABLE_ROOT / candidate_name(nn, res) / "clustering_parameters.csv").iloc[0].to_dict()
                manifest_rows.append(row)
            continue

        graph_started = time.time()
        work = ad.AnnData(X=sparse.csr_matrix((n_obs, 1), dtype=np.float32), obs=obs.copy())
        work.obsm["X_scVI"] = latent.copy()
        rsc.get.anndata_to_GPU(work, convert_all=True)
        rsc.pp.neighbors(
            work,
            use_rep="X_scVI",
            n_neighbors=nn,
            random_state=SEED,
            algorithm=KNN_ALGORITHM,
        )
        rsc.tl.umap(work, random_state=SEED)
        clustering_meta: dict[float, tuple[str, str, int]] = {}
        for res in missing_res:
            rt = res_text(res)
            key = f"scvi_leiden_res{rt}"
            rsc.tl.leiden(work, resolution=res, key_added=key, random_state=SEED)
            n_clusters = int(work.obs[key].nunique())
            algorithm = "leiden"
            if n_clusters > 100:
                del work.obs[key]
                key = f"scvi_louvain_res{rt}"
                rsc.tl.louvain(work, resolution=res, key_added=key, random_state=SEED)
                n_clusters = int(work.obs[key].nunique())
                algorithm = "louvain"
            clustering_meta[res] = (algorithm, key, n_clusters)
        rsc.get.anndata_to_CPU(work, convert_all=True)

        graph_mixing = edge_mixing_rows(work, nn)
        stage_fig = FIG_ROOT / f".nn-{nn}.staging"
        stage_table = TABLE_ROOT / f".nn-{nn}.staging"
        if stage_fig.exists() or stage_table.exists():
            raise FileExistsError(f"Staging path exists for nn={nn}")
        stage_fig.mkdir(parents=True)
        stage_table.mkdir(parents=True)
        save_batch_figures(work, nn, stage_fig)

        for res in missing_res:
            name = candidate_name(nn, res)
            algorithm, key, n_clusters = clustering_meta[res]
            fig_dir = stage_fig / name
            tab_dir = stage_table / name
            fig_dir.mkdir(parents=True, exist_ok=False)
            tab_dir.mkdir(parents=True, exist_ok=False)

            fig = sc.pl.umap(
                work,
                color=key,
                show=False,
                return_fig=True,
                frameon=False,
                size=1.0,
                legend_loc="on data",
                legend_fontsize=6,
                title=f"scVI latent 30; nn={nn}; {algorithm} res={res:g}",
            )
            fig.savefig(fig_dir / f"umap_{name}.pdf", bbox_inches="tight", dpi=300)
            plt.close(fig)

            counts = work.obs[key].astype(str).value_counts().rename_axis("cluster").reset_index(name="n_cells")
            counts["fraction"] = counts["n_cells"] / n_obs
            counts.to_csv(tab_dir / "cluster_counts.csv", index=False)
            batch_rows = list(graph_mixing)
            for batch_column in ("sample", "series", "status"):
                batch_rows.extend([
                    {
                        "n_neighbors": nn,
                        "metric": f"cluster_vs_{batch_column}_adjusted_rand_index",
                        "value": adjusted_rand_score(work.obs[key].astype(str), work.obs[batch_column].astype(str)),
                        "n_directed_edges": np.nan,
                        "interpretation": "descriptive cluster-batch association; closer to zero means weaker adjusted agreement",
                    },
                    {
                        "n_neighbors": nn,
                        "metric": f"cluster_vs_{batch_column}_normalized_mutual_information",
                        "value": normalized_mutual_info_score(work.obs[key].astype(str), work.obs[batch_column].astype(str)),
                        "n_directed_edges": np.nan,
                        "interpretation": "descriptive cluster-batch association; lower means weaker association",
                    },
                ])
            pd.DataFrame(batch_rows).to_csv(tab_dir / "batch_association_metrics.csv", index=False)

            row = {
                "candidate": name,
                "source_h5ad": str(SOURCE.relative_to(WF)),
                "integration_source": "scVI",
                "use_rep": "X_scVI",
                "latent_n_dims_used": LATENT_DIMS,
                "n_neighbors": nn,
                "neighbor_algorithm": KNN_ALGORITHM,
                "umap_random_state": SEED,
                "clustering_algorithm": algorithm,
                "resolution": res,
                "cluster_key": key,
                "n_clusters": n_clusters,
                "seed": SEED,
                "backend": "rapids-singlecell GPU",
                "candidate_h5ad_saved": False,
                "candidate_object_deleted_after_outputs": True,
                "figure_pdf": str((FIG_ROOT / name / f"umap_{name}.pdf").relative_to(WF)),
                "cluster_count_table": str((TABLE_ROOT / name / "cluster_counts.csv").relative_to(WF)),
                "batch_metric_table": str((TABLE_ROOT / name / "batch_association_metrics.csv").relative_to(WF)),
                "code_file": str(CODE.relative_to(WF)),
                "review_notes": "manual selection pending",
            }
            pd.DataFrame([row]).to_csv(tab_dir / "clustering_parameters.csv", index=False)
            manifest_rows.append(row)

        aggregate = f"latent-{LATENT_DIMS}_nn-{nn}_res-all"
        promote_file(stage_fig / aggregate / f"umap_{aggregate}_series_status.pdf", FIG_ROOT / aggregate / f"umap_{aggregate}_series_status.pdf")
        promote_file(stage_fig / aggregate / f"umap_{aggregate}_sample.pdf", FIG_ROOT / aggregate / f"umap_{aggregate}_sample.pdf")
        (stage_fig / aggregate).rmdir()
        for res in missing_res:
            name = candidate_name(nn, res)
            for filename in (f"umap_{name}.pdf",):
                promote_file(stage_fig / name / filename, FIG_ROOT / name / filename)
            (stage_fig / name).rmdir()
            for filename in ("clustering_parameters.csv", "cluster_counts.csv", "batch_association_metrics.csv"):
                promote_file(stage_table / name / filename, TABLE_ROOT / name / filename)
            (stage_table / name).rmdir()
        stage_fig.rmdir()
        stage_table.rmdir()
        graph_rows.append({
            "step": f"scvi_neighbors_umap_clustering_nn{nn}",
            "planned_backend": "rapids-singlecell GPU",
            "attempted_backend": "rapids-singlecell GPU",
            "status": "passed",
            "error_summary": "",
            "fallback_backend": "",
            "clean_input_reloaded": True,
            "final_backend_for_rerun": "rapids-singlecell GPU",
            "elapsed_seconds": round(time.time() - graph_started, 3),
        })
        del work
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()

    # Re-read candidate rows so a resumed run produces one authoritative 16-row manifest.
    manifest_rows = []
    for nn in N_NEIGHBORS:
        for res in RESOLUTIONS:
            path = TABLE_ROOT / candidate_name(nn, res) / "clustering_parameters.csv"
            if not path.is_file():
                raise RuntimeError(f"Incomplete grid, missing {path}")
            manifest_rows.append(pd.read_csv(path).iloc[0].to_dict())
    manifest = pd.DataFrame(manifest_rows).sort_values(["n_neighbors", "resolution"])
    if len(manifest) != 16 or manifest["candidate"].nunique() != 16:
        raise RuntimeError("Expected 16 unique completed scVI candidates")
    manifest.to_csv(manifest_path, index=False)
    pd.DataFrame(graph_rows).to_csv(TABLE_ROOT / "gpu_backend_capability_summary.csv", index=False)
    completion = {
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
        "elapsed_seconds": round(time.time() - started, 3),
    }
    completion_path.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    versions = {name: version(name) for name in ("rapids-singlecell", "scanpy", "anndata", "cupy-cuda12x", "numpy", "pandas", "scikit-learn")}
    versions.update({"python": platform.python_version(), "seed": SEED, "environment": "uv_envs/rapids/.venv"})
    (TABLE_ROOT / "package_versions.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in versions.items()) + "\n", encoding="utf-8"
    )
    (TABLE_ROOT / "readme.txt").write_text(
        "Generated the 16-candidate scVI clustering grid from X_scVI only.\n"
        "Grid: neighbors=10,20,30,40 and Leiden resolutions=0.2,0.3,0.4,0.5; seed=42.\n"
        "RAPIDS GPU CAGRA neighbors, GPU UMAP, and GPU Leiden were used; no candidate h5ad was saved.\n"
        "Batch diagnostics (series/status/sample UMAP and full-cell graph/association metrics) were saved.\n"
        "Stop after this grid and wait for the user's manual parameter selection.\n",
        encoding="utf-8",
    )
    print(json.dumps(completion, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

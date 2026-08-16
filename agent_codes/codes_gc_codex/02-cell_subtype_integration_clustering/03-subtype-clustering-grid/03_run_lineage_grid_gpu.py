#!/usr/bin/env python3
"""Immutable/resumable 810-candidate GPU grid for one broad lineage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block02_common import (  # noqa: E402
    EXPECTED_CANDIDATES, MODULE, NN_VALUES, PCS_VALUES, RES_VALUES, SEED, WORKFLOW_ROOT,
    candidate_id, candidate_path, cell_id_digest, save_candidate_once, terminal_metadata,
    task_root, weighted_cluster_mixing,
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--lineage", required=True); return p.parse_args()


def atomic_csv_verified(frame: pd.DataFrame, path: Path, key_columns: list[str]) -> None:
    if path.exists():
        old = pd.read_csv(path)
        if key_columns and not old[key_columns].fillna("").equals(frame[key_columns].fillna("")):
            raise RuntimeError(f"Existing table differs from regenerated content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    if partial.exists(): raise FileExistsError(f"Residual partial table: {partial}")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def atomic_json_once(payload: dict, path: Path) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f"Existing JSON differs from regenerated content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    if partial.exists(): raise FileExistsError(f"Residual partial JSON: {partial}")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(partial, path)


def graph_token(pcs: int, nn: int) -> str:
    return f"pcs{pcs}_nn{nn}_res0p1-1p0"


def graph_output_paths(lineage: str, pcs: int, nn: int) -> dict[str, Path]:
    token = graph_token(pcs, nn)
    table = task_root("tables", "03-subtype-clustering-grid") / lineage / token
    figure = task_root("figures", "03-subtype-clustering-grid") / lineage / token
    return {
        "parameters": table / "graph_resolution_parameters.csv",
        "counts": table / "cluster_counts.csv",
        "pdf": figure / "umap_all_resolutions.pdf",
        "svg": figure / "umap_all_resolutions.svg",
    }


def render_review(
    embedding: np.ndarray,
    labels_by_res: dict[float, np.ndarray],
    skipped_by_res: dict[float, str],
    lineage: str,
    pcs: int,
    nn: int,
    outputs: dict[str, Path],
) -> None:
    if outputs["pdf"].exists() and outputs["svg"].exists():
        return

    review_obs = pd.DataFrame(index=pd.Index([str(i) for i in range(embedding.shape[0])]))
    color_keys: list[str] = []
    for res in RES_VALUES:
        key = f"leiden_res{res:.1f}".replace(".", "p")
        color_keys.append(key)
        labels = labels_by_res.get(res)
        if labels is None:
            reason = skipped_by_res.get(res, "not run")
            review_obs[key] = pd.Categorical([f"skipped: {reason}"] * embedding.shape[0])
        else:
            review_obs[key] = pd.Categorical(labels.astype(str))

    review = ad.AnnData(X=np.empty((embedding.shape[0], 0), dtype=np.float32), obs=review_obs)
    review.obsm["X_umap"] = embedding
    figure_dir = outputs["pdf"].parent
    figure_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.set_figure_params(figsize=(3, 3), dpi=150, vector_friendly=True)
    sc.settings.figdir = figure_dir
    common = dict(color=color_keys, ncols=5, wspace=0.4, frameon=True, show=False)
    if not outputs["pdf"].exists():
        sc.pl.umap(review, save="_all_resolutions.pdf", **common)
    if not outputs["svg"].exists():
        sc.pl.umap(review, save="_all_resolutions.svg", **common)
    if not outputs["pdf"].exists() or not outputs["svg"].exists():
        raise RuntimeError(f"Scanpy did not create both review figures: {figure_dir}")


def load_terminal(path: Path, expected: dict) -> tuple[dict, np.ndarray | None]:
    meta = terminal_metadata(path)
    for key, value in expected.items():
        if meta.get(key) != value: raise RuntimeError(f"Stale terminal {path}: {key}={meta.get(key)!r}, expected {value!r}")
    with np.load(path, allow_pickle=False) as z: labels = z["labels"].copy()
    if meta["status"] == "completed" and labels.size != expected["n_cells"]:
        raise RuntimeError(f"Label length mismatch: {path}")
    if meta["status"].startswith("skipped") and labels.size: raise RuntimeError(f"Skipped result has labels: {path}")
    return meta, labels if labels.size else None


def previous_graph_labels(root: Path, lineage: str, pcs: int, nn: int, res: float, digest: str, n: int) -> dict[str, np.ndarray]:
    arrays = {}
    for axis, p, k in (("nn", pcs, nn - 5), ("pcs", pcs - 5, nn)):
        if p not in PCS_VALUES or k not in NN_VALUES: continue
        path = candidate_path(root, lineage, p, k, res)
        if not path.exists(): continue
        meta, labels = load_terminal(path, {
            "lineage": lineage, "n_pcs": p, "n_neighbors": k, "resolution": res,
            "cell_id_sha256": digest, "n_cells": n,
        })
        if meta["status"] == "completed" and labels is not None: arrays[axis] = labels
    return arrays


def main() -> None:
    a = args()
    inp = task_root("h5ad", "02-subtype-harmony") / a.lineage / f"adata_{a.lineage}_harmony.h5ad"
    root = task_root("tables", "03-subtype-clustering-grid") / "candidate-terminals"
    summary_dir = task_root("tables", "03-subtype-clustering-grid") / a.lineage
    completion = summary_dir / "clustering_grid_completion.json"
    manifest_path = summary_dir / "candidate_manifest.csv"
    source = ad.read_h5ad(inp, backed="r")
    if "X_pca_inte" not in source.obsm: raise KeyError("Missing obsm['X_pca_inte']")
    obs = source.obs.copy(); basis = np.asarray(source.obsm["X_pca_inte"], dtype=np.float32)
    digest = cell_id_digest(source.obs_names); n = source.n_obs; source.file.close()
    if basis.shape[1] < max(PCS_VALUES): raise RuntimeError(f"Harmony has only {basis.shape[1]} PCs; 50 required")
    if completion.exists() and manifest_path.exists():
        done = json.loads(completion.read_text())
        graph_outputs_ok = all(
            path.exists()
            for pcs in PCS_VALUES for nn in NN_VALUES
            for path in graph_output_paths(a.lineage, pcs, nn).values()
        )
        if done.get("terminal_candidates") == EXPECTED_CANDIDATES and done.get("failed_candidates") == 0 and graph_outputs_ok:
            print(f"validated_existing={completion}"); return
        raise RuntimeError(f"Existing completion is invalid; refusing overwrite: {completion}")
    if completion.exists() != manifest_path.exists():
        raise RuntimeError("Residual final grid manifest/completion exists; refusing overwrite")

    import cupy as cp
    import rapids_singlecell as rsc
    search = ad.AnnData(X=np.zeros((n, 1), np.float32), obs=obs)
    search.obsm["X_pca_inte"] = basis
    rsc.get.anndata_to_GPU(search)
    try:
        for pcs in PCS_VALUES:
            for nn in NN_VALUES:
                outputs = graph_output_paths(a.lineage, pcs, nn)
                graph_paths = [candidate_path(root, a.lineage, pcs, nn, res) for res in RES_VALUES]
                if all(path.exists() for path in graph_paths) and all(path.exists() for path in outputs.values()):
                    exceeded = False
                    for res, path in zip(RES_VALUES, graph_paths):
                        meta, _ = load_terminal(path, {
                            "lineage": a.lineage, "n_pcs": pcs, "n_neighbors": nn,
                            "resolution": res, "cell_id_sha256": digest, "n_cells": n,
                        })
                        if exceeded and meta["status"] != "skipped_cluster_gt20":
                            raise RuntimeError(f"Terminal violates >20 skip boundary: {path}")
                        if not exceeded and meta["status"] == "skipped_cluster_gt20":
                            raise RuntimeError(f"Skip terminal lacks preceding >20 boundary: {path}")
                        if meta["status"] == "completed" and int(meta["n_clusters"]) > 20:
                            exceeded = True
                    continue
                run = search.copy()
                rsc.pp.neighbors(run, n_neighbors=nn, n_pcs=pcs, use_rep="X_pca_inte", key_added="grid_neighbors", random_state=SEED)
                rsc.tl.umap(run, neighbors_key="grid_neighbors", key_added="X_umap", random_state=SEED)
                previous_res: np.ndarray | None = None
                exceeded = False
                labels_by_res: dict[float, np.ndarray] = {}
                skipped_by_res: dict[float, str] = {}
                metadata_by_res: dict[float, dict] = {}
                for res in RES_VALUES:
                    path = candidate_path(root, a.lineage, pcs, nn, res)
                    expected = {
                        "lineage": a.lineage, "n_pcs": pcs, "n_neighbors": nn, "resolution": res,
                        "cell_id_sha256": digest, "n_cells": n,
                    }
                    if path.exists():
                        meta, labels = load_terminal(path, expected)
                        if exceeded and meta["status"] != "skipped_cluster_gt20":
                            raise RuntimeError(f"Terminal violates >20 skip boundary: {path}")
                        if not exceeded and meta["status"] == "skipped_cluster_gt20":
                            raise RuntimeError(f"Skip terminal has no preceding >20-cluster boundary: {path}")
                        if meta["status"] == "completed":
                            previous_res = labels
                            if labels is None: raise RuntimeError(f"Completed terminal lacks labels: {path}")
                            labels_by_res[res] = labels
                            if int(meta["n_clusters"]) > 20: exceeded = True
                        else:
                            skipped_by_res[res] = str(meta.get("reason", "skipped_cluster_gt20"))
                        metadata_by_res[res] = meta
                        continue
                    if exceeded:
                        meta = {**expected, "candidate_id": candidate_id(pcs, nn, res),
                            "status": "skipped_cluster_gt20", "reason": "a lower resolution in this graph first exceeded 20 clusters",
                            "backend": "not_run_by_contract", "random_seed": SEED}
                        save_candidate_once(path, None, meta)
                        skipped_by_res[res] = str(meta["reason"]); metadata_by_res[res] = meta
                        continue
                    key1, key2 = "grid_leiden_seed42", "grid_leiden_seed43"
                    rsc.tl.leiden(run, resolution=res, key_added=key1, neighbors_key="grid_neighbors", random_state=SEED)
                    rsc.tl.leiden(run, resolution=res, key_added=key2, neighbors_key="grid_neighbors", random_state=SEED + 1)
                    labels = run.obs[key1].cat.codes.to_numpy(dtype=np.int16)
                    labels2 = run.obs[key2].cat.codes.to_numpy(dtype=np.int16)
                    counts = np.bincount(labels); n_clusters = int(len(counts)); local_ari = adjusted_rand_score(labels, labels2)
                    resolution_ari = adjusted_rand_score(labels, previous_res) if previous_res is not None else np.nan
                    graph_arrays = previous_graph_labels(root, a.lineage, pcs, nn, res, digest, n)
                    axis_aris = {axis: float(adjusted_rand_score(labels, x)) for axis, x in graph_arrays.items()}
                    graph_ari = float(np.mean(list(axis_aris.values()))) if axis_aris else np.nan
                    small_cutoff = max(20, int(np.ceil(n * 0.001)))
                    small_fraction = float(counts[counts < small_cutoff].sum() / n)
                    field = lambda choices: next((c for c in choices if c in obs), None)
                    sample_col = field(("sample", "sample_id", "orig.ident")); series_col = field(("series", "GSE", "gse", "dataset")); status_col = field(("status",))
                    mix = lambda col: weighted_cluster_mixing(labels, obs[col].astype(str).to_numpy()) if col else 1.0
                    meta = {**expected, "candidate_id": candidate_id(pcs, nn, res), "status": "completed",
                        "n_clusters": n_clusters, "min_cluster_size": int(counts.min()), "max_cluster_size": int(counts.max()),
                        "small_cluster_cutoff": small_cutoff, "small_cluster_cell_fraction": small_fraction,
                        "cluster_size_score": 1.0 - small_fraction, "local_seed_ari": float(local_ari),
                        "previous_resolution_ari": None if np.isnan(resolution_ari) else float(resolution_ari),
                        "previous_nn_ari": axis_aris.get("nn"), "previous_pcs_ari": axis_aris.get("pcs"),
                        "neighbor_graph_ari": None if np.isnan(graph_ari) else graph_ari,
                        "sample_mixing": mix(sample_col), "series_mixing": mix(series_col), "status_mixing": mix(status_col),
                        "sample_column": sample_col, "series_column": series_col, "status_column": status_col,
                        "backend": "rapids_singlecell_gpu", "random_seed": SEED, "downsampled": False}
                    save_candidate_once(path, labels, meta); previous_res = labels
                    labels_by_res[res] = labels; metadata_by_res[res] = meta
                    del run.obs[key1], run.obs[key2]
                    if n_clusters > 20: exceeded = True

                if set(metadata_by_res) != set(RES_VALUES):
                    raise RuntimeError(f"Graph did not produce ten terminal states: PCs={pcs}, NN={nn}")
                embedding_gpu = run.obsm["X_umap"]
                embedding = cp.asnumpy(embedding_gpu) if isinstance(embedding_gpu, cp.ndarray) else np.asarray(embedding_gpu)
                if embedding.shape != (n, 2) or not np.isfinite(embedding).all():
                    raise RuntimeError(f"Invalid graph UMAP: PCs={pcs}, NN={nn}, shape={embedding.shape}")

                parameter_rows = []
                count_rows = []
                for res in RES_VALUES:
                    meta = metadata_by_res[res]
                    terminal = candidate_path(root, a.lineage, pcs, nn, res)
                    parameter_rows.append({
                        **meta,
                        "source_harmony_h5ad": str(inp),
                        "harmony_basis": "X_pca_inte",
                        "neighbors_key": "grid_neighbors",
                        "umap_key": "X_umap",
                        "umap_random_seed": SEED,
                        "terminal_path": str(terminal),
                        "review_figure_pdf": str(outputs["pdf"]),
                        "review_figure_svg": str(outputs["svg"]),
                        "candidate_h5ad_saved": False,
                        "candidate_object_created_by_copy": True,
                        "candidate_object_deleted_after_outputs": True,
                        "plotting_interface": "scanpy_sc_pl_umap_save",
                        "vector_friendly": True,
                        "figure_size_inches": "3,3",
                        "figure_dpi": 150,
                        "manual_axes_used": False,
                        "matplotlib_savefig_used": False,
                        "manual_point_size_used": False,
                    })
                    labels = labels_by_res.get(res)
                    if labels is not None:
                        clusters, counts = np.unique(labels, return_counts=True)
                        count_rows.extend({
                            "lineage": a.lineage, "n_pcs": pcs, "n_neighbors": nn,
                            "resolution": res, "cluster": int(cluster), "n_cells": int(count),
                        } for cluster, count in zip(clusters, counts))
                parameter_frame = pd.DataFrame(parameter_rows)
                count_frame = pd.DataFrame(count_rows)
                atomic_csv_verified(parameter_frame, outputs["parameters"], ["candidate_id", "status"])
                atomic_csv_verified(count_frame, outputs["counts"], ["n_pcs", "n_neighbors", "resolution", "cluster", "n_cells"])
                render_review(embedding, labels_by_res, skipped_by_res, a.lineage, pcs, nn, outputs)
                del run; cp.get_default_memory_pool().free_all_blocks()
    finally:
        try: rsc.get.anndata_to_CPU(search)
        except Exception: pass
        cp.get_default_memory_pool().free_all_blocks()

    rows = []
    for pcs in PCS_VALUES:
        for nn in NN_VALUES:
            for res in RES_VALUES:
                path = candidate_path(root, a.lineage, pcs, nn, res)
                if not path.exists(): raise RuntimeError(f"Missing terminal: {path}")
                meta, _ = load_terminal(path, {"lineage": a.lineage, "n_pcs": pcs, "n_neighbors": nn,
                    "resolution": res, "cell_id_sha256": digest, "n_cells": n})
                outputs = graph_output_paths(a.lineage, pcs, nn)
                if not all(output.exists() for output in outputs.values()):
                    raise RuntimeError(f"Missing graph review output: lineage={a.lineage}, PCs={pcs}, NN={nn}")
                rows.append({**meta, "terminal_path": str(path),
                    "graph_parameters_path": str(outputs["parameters"]),
                    "cluster_counts_path": str(outputs["counts"]),
                    "review_figure_pdf": str(outputs["pdf"]),
                    "review_figure_svg": str(outputs["svg"])})
    manifest = pd.DataFrame(rows)
    terminal = len(manifest); completed_n = int((manifest.status == "completed").sum()); skipped = int((manifest.status == "skipped_cluster_gt20").sum())
    if terminal != EXPECTED_CANDIDATES or completed_n + skipped != EXPECTED_CANDIDATES: raise AssertionError("Grid is not 810 terminal")
    summary_dir.mkdir(parents=True, exist_ok=True)
    comparable = ["candidate_id", "status", "n_pcs", "n_neighbors", "resolution", "cell_id_sha256"]
    atomic_csv_verified(manifest, manifest_path, comparable)
    payload = {"lineage": a.lineage, "expected_candidates": EXPECTED_CANDIDATES, "terminal_candidates": terminal,
        "completed_candidates": completed_n, "skipped_cluster_gt20": skipped, "failed_candidates": 0,
        "terminal_accounting_ok": True, "all_graph_review_outputs_exist": True,
        "graph_review_pairs": len(PCS_VALUES) * len(NN_VALUES), "cell_id_sha256": digest}
    atomic_json_once(payload, completion); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()

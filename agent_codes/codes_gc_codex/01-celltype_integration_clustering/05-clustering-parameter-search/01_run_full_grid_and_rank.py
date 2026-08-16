#!/usr/bin/env python3
from __future__ import annotations

import gc
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import seaborn as sns
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

SEED = 42
PCS = list(range(10, 51, 5))
NNS = list(range(10, 51, 5))
RESOLUTIONS = [round(x / 10, 1) for x in range(1, 11)]
MAX_CLUSTERS = 20
ROOT = Path(__file__).resolve().parents[4]
WF = ROOT / "epi-cm-core-workflow"
SOURCE = WF / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_harmony.h5ad"
TABLE = WF / "tables/01-celltype_integration_clustering/05-clustering-parameter-search"
FIGURE = WF / "figures/01-celltype_integration_clustering/05-clustering-parameter-search"
MANIFEST = TABLE / "clustering_grid_manifest.csv"
LOCK_FILE = TABLE / ".grid_run.lock"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def label(pcs: int, nn: int) -> str:
    return f"pcs-{pcs}_nn-{nn}_res-0p1-1p0"


def res_token(resolution: float) -> str:
    return f"{resolution:.1f}".replace(".", "p")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def prepare_mixing_codes(obs: pd.DataFrame) -> dict[str, tuple[np.ndarray, int]]:
    return {
        column: (
            pd.Categorical(obs[column].astype(str)).codes.astype(np.int32, copy=False),
            int(obs[column].astype(str).nunique()),
        )
        for column in ("sample", "series", "status")
    }


def cluster_mixing(
    mixing_codes: dict[str, tuple[np.ndarray, int]], labels: np.ndarray
) -> tuple[float, float, float, float]:
    """Match the former weighted per-cluster entropy using NumPy contingencies."""
    cluster_codes = pd.Categorical(labels.astype(str)).codes.astype(np.int32, copy=False)
    n_clusters = int(cluster_codes.max(initial=-1)) + 1
    total = len(cluster_codes)
    vals: dict[str, float] = {}
    for column, (category_codes, n_categories) in mixing_codes.items():
        counts = np.bincount(
            cluster_codes.astype(np.int64) * n_categories + category_codes,
            minlength=n_clusters * n_categories,
        ).reshape(n_clusters, n_categories).astype(np.float64, copy=False)
        cluster_sizes = counts.sum(axis=1)
        proportions = np.divide(
            counts,
            cluster_sizes[:, None],
            out=np.zeros_like(counts),
            where=cluster_sizes[:, None] > 0,
        )
        log_terms = np.zeros_like(proportions)
        positive = proportions > 0
        log_terms[positive] = proportions[positive] * np.log(proportions[positive])
        entropy = -log_terms.sum(axis=1)
        present = (counts > 0).sum(axis=1)
        valid = present > 1
        entropy[valid] /= np.log(present[valid])
        entropy[~valid] = 0.0
        vals[column] = float(np.sum(cluster_sizes / total * entropy))
    composite = 0.5 * vals["sample"] + 0.3 * vals["series"] + 0.2 * vals["status"]
    return vals["sample"], vals["series"], vals["status"], composite


def size_score(labels: np.ndarray) -> tuple[int, int, float, float]:
    _, counts = np.unique(labels, return_counts=True)
    n = int(counts.sum())
    tiny = counts < max(50, math.ceil(0.001 * n))
    tiny_cell_fraction = float(counts[tiny].sum() / n)
    tiny_cluster_fraction = float(tiny.mean())
    score = max(0.0, 1.0 - 0.5 * tiny_cell_fraction - 0.5 * tiny_cluster_fraction)
    return int(counts.min()), int(counts.max()), tiny_cell_fraction, score


def build_manifest(source_sha: str, n_obs: int) -> pd.DataFrame:
    rows = []
    for pcs in PCS:
        for nn in NNS:
            d = label(pcs, nn)
            for res in RESOLUTIONS:
                tok = res_token(res)
                rows.append({
                    "n_pcs": pcs, "n_neighbors": nn, "resolution": res,
                    "candidate_label": f"pcs-{pcs}_nn-{nn}_res-{tok}", "graph_label": d,
                    "clustering_algorithm": "leiden", "raw_cluster_column": f"leiden_res{tok}",
                    "status": "planned", "completed": False, "n_clusters": np.nan,
                    "reason_if_skipped": "", "trigger_resolution": np.nan,
                    "labels_path": str((TABLE / d / f"labels_leiden_res{tok}.npz").relative_to(WF)),
                    "figure_pdf": str((FIGURE / d / "umap_all_resolutions.pdf").relative_to(WF)),
                    "figure_svg": str((FIGURE / d / "umap_all_resolutions.svg").relative_to(WF)),
                    "source_h5ad": str(SOURCE.relative_to(WF)), "source_sha256": source_sha,
                    "source_n_obs": n_obs, "backend": "rapids-singlecell GPU", "seed": SEED,
                    "sample_entropy": np.nan, "series_entropy": np.nan, "status_entropy": np.nan,
                    "mixing_score": np.nan, "min_cluster_size": np.nan, "max_cluster_size": np.nan,
                    "tiny_cell_fraction": np.nan, "cluster_size_score": np.nan,
                })
    return pd.DataFrame(rows)


def validate_or_create_manifest(source_sha: str, n_obs: int) -> pd.DataFrame:
    expected = build_manifest(source_sha, n_obs)
    if not MANIFEST.exists():
        atomic_csv(expected, MANIFEST)
        return expected
    frame = pd.read_csv(MANIFEST)
    if len(frame) != 810 or set(frame["source_sha256"].astype(str)) != {source_sha} or set(frame["source_n_obs"].astype(int)) != {n_obs}:
        raise RuntimeError("Existing grid manifest belongs to another source or grid")
    keys = ["n_pcs", "n_neighbors", "resolution"]
    if set(map(tuple, frame[keys].itertuples(index=False, name=None))) != set(map(tuple, expected[keys].itertuples(index=False, name=None))):
        raise RuntimeError("Existing grid manifest has the wrong 810 combinations")
    return frame


def load_labels(path: Path, n_obs: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        values = z["labels"].astype(str)
    if len(values) != n_obs:
        raise RuntimeError(f"Checkpoint length mismatch: {path}")
    return values


def load_label_codes(path: Path, n_obs: int) -> np.ndarray:
    """Load a checkpoint as compact integer codes for ARI calculations."""
    with np.load(path, allow_pickle=False) as z:
        values = z["labels"]
    if len(values) != n_obs:
        raise RuntimeError(f"Checkpoint length mismatch: {path}")
    try:
        return values.astype(np.int32, copy=False)
    except (TypeError, ValueError):
        return pd.Categorical(values.astype(str)).codes.astype(np.int32, copy=False)


def atomic_label_checkpoint(path: Path, labels: np.ndarray) -> None:
    """Write pickle-free fixed-width Unicode labels, then atomically publish."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_labels = np.asarray(labels, dtype=np.str_)
    if safe_labels.dtype.kind != "U":
        raise RuntimeError(f"Unsafe label dtype for {path}: {safe_labels.dtype}")
    tmp = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(tmp, labels=safe_labels)
    # Prove the checkpoint can be read with pickle disabled before publishing.
    load_labels(tmp, len(safe_labels))
    os.replace(tmp, path)


def scatter_categorical(
    ax: plt.Axes, embedding: np.ndarray, values: pd.Series, title: str
) -> None:
    categories = sorted(values.astype(str).unique())
    categorical = pd.Categorical(values.astype(str), categories=categories)
    cmap = plt.get_cmap("tab20", max(1, len(categories)))
    ax.scatter(
        embedding[:, 0], embedding[:, 1], c=categorical.codes, cmap=cmap,
        s=0.18, linewidths=0, rasterized=True,
    )
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=3,
               markerfacecolor=cmap(i), markeredgecolor="none", label=category)
        for i, category in enumerate(categories)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=4.5, frameon=False,
              handletextpad=0.2, borderaxespad=0.1)
    ax.set_title(title, fontsize=8)


def render_graph(
    embedding: np.ndarray,
    obs: pd.DataFrame,
    labels_by_res: dict[float, np.ndarray],
    out_dir: Path,
    skipped: dict[float, str],
) -> None:
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), squeeze=False)
    flat = axes.ravel()
    scatter_categorical(flat[0], embedding, obs["series"], "series")
    scatter_categorical(flat[1], embedding, obs["status"], "status")
    for ax, res in zip(flat[2:], RESOLUTIONS):
        if res in labels_by_res:
            lab = labels_by_res[res]
            categories = sorted(np.unique(lab), key=lambda x: int(x) if str(x).isdigit() else str(x))
            code = pd.Categorical(lab, categories=categories).codes
            ax.scatter(embedding[:, 0], embedding[:, 1], c=code, cmap="tab20", s=0.18,
                       linewidths=0, rasterized=True)
            ax.set_title(f"leiden_res{res_token(res)} (k={len(categories)})", fontsize=8)
        else:
            ax.text(0.5, 0.5, skipped.get(res, "not run"), ha="center", va="center", wrap=True, fontsize=7)
            ax.set_title(f"res{res_token(res)} skipped", fontsize=8)
        ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_box_aspect(1)
    for ax in flat[:2]:
        ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
        ax.tick_params(labelsize=6); ax.set_box_aspect(1)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "umap_all_resolutions.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "umap_all_resolutions.svg", bbox_inches="tight")
    plt.close(fig)


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    s = s.fillna(s.median())
    lo, hi = float(s.min()), float(s.max())
    return pd.Series(1.0 if hi == lo else (s - lo) / (hi - lo), index=s.index)


def rank_candidates(manifest: pd.DataFrame, n_obs: int) -> pd.DataFrame:
    eligible = manifest[(manifest["status"] == "completed") & pd.to_numeric(manifest["n_clusters"], errors="coerce").between(8, MAX_CLUSTERS)].copy()
    if eligible.empty:
        raise RuntimeError("No completed candidate with the required 8..20 broad clusters")
    index = {(int(r.n_pcs), int(r.n_neighbors), round(float(r.resolution), 1)): r for r in eligible.itertuples()}

    # Compress all eligible labels once.  With <=20 eligible clusters, uint8 is
    # exact and keeps the full 480k-cell matrix near 200 MB.  This avoids both
    # the multi-GB Unicode cache that killed the first ranking attempt and the
    # repeated NPZ decompression of a tiny rolling cache.
    keys = list(index)
    key_to_position = {key: i for i, key in enumerate(keys)}
    label_matrix = np.empty((len(keys), n_obs), dtype=np.uint8)
    for position, key in enumerate(keys):
        codes = load_label_codes(WF / index[key].labels_path, n_obs)
        if codes.min(initial=0) < 0 or codes.max(initial=0) > np.iinfo(np.uint8).max:
            raise RuntimeError(f"Label codes cannot be represented exactly as uint8: {key}")
        label_matrix[position] = codes.astype(np.uint8, copy=False)
        if (position + 1) % 50 == 0 or position + 1 == len(keys):
            print(f"ranking labels loaded: {position + 1}/{len(keys)}", flush=True)

    def get(k: tuple[int, int, float]) -> np.ndarray:
        return label_matrix[key_to_position[k]]
    adjacent_res = []
    neighbor_graph = []
    for row in eligible.itertuples():
        key = (int(row.n_pcs), int(row.n_neighbors), round(float(row.resolution), 1))
        aris = []
        for dr in [-0.1, 0.1]:
            other = (key[0], key[1], round(key[2] + dr, 1))
            if other in index:
                aris.append(adjusted_rand_score(get(key), get(other)))
        adjacent_res.append(float(np.mean(aris)) if aris else np.nan)
        aris = []
        for dp, dn in [(-5, 0), (5, 0), (0, -5), (0, 5)]:
            other = (key[0] + dp, key[1] + dn, key[2])
            if other in index:
                aris.append(adjusted_rand_score(get(key), get(other)))
        neighbor_graph.append(float(np.mean(aris)) if aris else np.nan)
        if len(adjacent_res) % 50 == 0 or len(adjacent_res) == len(eligible):
            print(f"ranking ARI candidates: {len(adjacent_res)}/{len(eligible)}", flush=True)
    eligible["adjacent_resolution_ari"] = adjacent_res
    eligible["neighbor_graph_ari"] = neighbor_graph
    eligible["batch_mixing_score"] = (
        0.625 * pd.to_numeric(eligible["sample_entropy"], errors="coerce")
        + 0.375 * pd.to_numeric(eligible["series_entropy"], errors="coerce")
    )
    eligible["ari_score_norm"] = minmax(eligible["adjacent_resolution_ari"])
    eligible["size_score_norm"] = minmax(eligible["cluster_size_score"])
    eligible["mixing_score_norm"] = minmax(eligible["batch_mixing_score"])
    eligible["neighbor_score_norm"] = minmax(eligible["neighbor_graph_ari"])
    eligible["composite_score"] = (0.5 * eligible["ari_score_norm"] + 0.2 * eligible["size_score_norm"] +
                                   0.2 * eligible["mixing_score_norm"] + 0.1 * eligible["neighbor_score_norm"])
    eligible = eligible.sort_values(
        ["composite_score", "adjacent_resolution_ari", "cluster_size_score", "batch_mixing_score", "n_clusters", "resolution", "n_pcs", "n_neighbors"],
        ascending=[False, False, False, False, True, True, True, True], na_position="last"
    ).reset_index(drop=True)
    eligible["automatic_rank"] = np.arange(1, len(eligible) + 1)
    return eligible


def main() -> None:
    TABLE.mkdir(parents=True, exist_ok=True)
    # The execution environment can detach a long-running process from its
    # terminal.  Hold an OS lock for the process lifetime so a resumed command
    # can never race the still-running writer.  flock is released by the kernel
    # even after SIGTERM, so there is no stale-lock recovery ambiguity.
    lock_handle = LOCK_FILE.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another clustering-grid writer holds {LOCK_FILE}") from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()}\n")
    lock_handle.flush()
    random.seed(SEED); np.random.seed(SEED)
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source_sha = sha256(SOURCE)
    full = ad.read_h5ad(SOURCE, backed="r")
    try:
        n_obs = full.n_obs
        if "X_pca_inte" not in full.obsm or full.obsm["X_pca_inte"].shape[1] < 50:
            raise RuntimeError("Clean Harmony basis missing")
        if "X_umap" in full.obsm or len(full.obsp):
            raise RuntimeError("Harmony source is not graphless")
        obs = full.obs[["sample", "series", "status"]].copy()
        mixing_codes = prepare_mixing_codes(obs)
        ids = full.obs_names.copy()
        rep = np.asarray(full.obsm["X_pca_inte"], dtype=np.float32)
    finally:
        full.file.close()
    manifest = validate_or_create_manifest(source_sha, n_obs)
    resources = []
    # Overlap one CPU renderer with the next GPU graph.  A single worker keeps
    # Matplotlib serialized; bounding the queue at one later graph prevents
    # embeddings/labels from accumulating in RAM.
    render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grid-render")
    render_futures: list[tuple[str, Future[None]]] = []
    for pcs in PCS:
        for nn in NNS:
            graph_mask = (manifest["n_pcs"].astype(int) == pcs) & (manifest["n_neighbors"].astype(int) == nn)
            graph_rows = manifest.loc[graph_mask].sort_values("resolution")
            graph_name = label(pcs, nn)
            out_fig = FIGURE / graph_name
            params_dir = TABLE / graph_name
            graph_terminal = graph_rows["status"].isin(
                ["completed", "skipped_cluster_gt20"]
            ).all()
            graph_outputs_complete = all(
                path.is_file()
                for path in (
                    out_fig / "umap_all_resolutions.pdf",
                    out_fig / "umap_all_resolutions.svg",
                    params_dir / "graph_parameters.csv",
                )
            )
            if graph_terminal and graph_outputs_complete:
                continue
            started = time.monotonic()
            template = ad.AnnData(sparse.csr_matrix((n_obs, 1), dtype=np.float32), obs=obs.copy())
            template.obs_names = ids.copy()
            template.obsm["X_pca_inte"] = rep
            rsc.get.anndata_to_GPU(template)
            run = template.copy()
            rsc.pp.neighbors(
                run, n_neighbors=nn, n_pcs=pcs, use_rep="X_pca_inte",
                algorithm="cagra", random_state=SEED,
            )
            rsc.tl.umap(run, random_state=SEED)
            labels_by_res: dict[float, np.ndarray] = {}
            skipped: dict[float, str] = {}
            triggered: float | None = None
            for res in RESOLUTIONS:
                row_idx = manifest.index[graph_mask & np.isclose(manifest["resolution"].astype(float), res)]
                if len(row_idx) != 1:
                    raise RuntimeError("Manifest key not unique")
                i = row_idx[0]
                current = str(manifest.at[i, "status"])
                checkpoint = WF / str(manifest.at[i, "labels_path"])
                if current == "completed":
                    labels_np = load_labels(checkpoint, n_obs)
                    labels_by_res[res] = labels_np
                    if int(manifest.at[i, "n_clusters"]) > MAX_CLUSTERS:
                        triggered = res
                    continue
                if current == "skipped_cluster_gt20":
                    skipped[res] = str(manifest.at[i, "reason_if_skipped"])
                    continue
                if triggered is not None:
                    reason = f"higher resolution skipped after res={triggered:.1f} produced >{MAX_CLUSTERS} clusters"
                    manifest.loc[i, ["status", "completed", "reason_if_skipped", "trigger_resolution"]] = ["skipped_cluster_gt20", False, reason, triggered]
                    skipped[res] = reason
                    atomic_csv(manifest, MANIFEST)
                    continue
                if checkpoint.exists():
                    # Recover an atomically written label checkpoint left immediately before
                    # a manifest update; never overwrite an existing completed computation.
                    values = load_labels(checkpoint, n_obs)
                    k = int(pd.Series(values).nunique())
                    se, de, te, mix = cluster_mixing(mixing_codes, values)
                    min_size, max_size, tiny_frac, sz_score = size_score(values)
                    manifest.loc[i, ["status", "completed", "n_clusters", "reason_if_skipped",
                                     "sample_entropy", "series_entropy", "status_entropy", "mixing_score",
                                     "min_cluster_size", "max_cluster_size", "tiny_cell_fraction", "cluster_size_score"]] = [
                        "completed", True, k, "recovered_existing_label_checkpoint", se, de, te, mix,
                        min_size, max_size, tiny_frac, sz_score]
                    atomic_csv(manifest, MANIFEST)
                    labels_by_res[res] = values
                    if k > MAX_CLUSTERS:
                        triggered = res
                    continue
                key = f"leiden_res{res_token(res)}"
                rsc.tl.leiden(run, resolution=res, key_added=key, random_state=SEED)
                values = np.asarray(run.obs[key].astype(str).to_numpy(), dtype=np.str_)
                k = int(pd.Series(values).nunique())
                atomic_label_checkpoint(checkpoint, values)
                se, de, te, mix = cluster_mixing(mixing_codes, values)
                min_size, max_size, tiny_frac, sz_score = size_score(values)
                manifest.loc[i, ["status", "completed", "n_clusters", "reason_if_skipped",
                                 "sample_entropy", "series_entropy", "status_entropy", "mixing_score",
                                 "min_cluster_size", "max_cluster_size", "tiny_cell_fraction", "cluster_size_score"]] = [
                    "completed", True, k, "", se, de, te, mix, min_size, max_size, tiny_frac, sz_score]
                atomic_csv(manifest, MANIFEST)
                labels_by_res[res] = values
                if k > MAX_CLUSTERS:
                    triggered = res
            # Only the 2-D embedding is needed for the candidate figure.
            # Converting the complete 480k-cell graph object (including GPU
            # neighbor matrices) back to CPU is both unnecessary and was
            # observed to terminate at this graph boundary.  Transfer this
            # small array explicitly, then release the GPU graph below.
            embedding_gpu = run.obsm["X_umap"]
            embedding = (
                cp.asnumpy(embedding_gpu)
                if isinstance(embedding_gpu, cp.ndarray)
                else np.asarray(embedding_gpu)
            )
            render_futures.append(
                (
                    graph_name,
                    render_executor.submit(
                        render_graph, embedding, obs, labels_by_res, out_fig, skipped
                    ),
                )
            )
            if len(render_futures) >= 2:
                finished_name, finished = render_futures.pop(0)
                try:
                    finished.result()
                except Exception as exc:
                    raise RuntimeError(f"Rendering failed for {finished_name}") from exc
            params_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"source_h5ad": str(SOURCE.relative_to(WF)), "source_sha256": source_sha,
                           "n_cells": n_obs, "n_pcs": pcs, "n_neighbors": nn,
                           "resolutions": "0.1:1.0:0.1", "max_clusters": MAX_CLUSTERS,
                           "early_stop_rule": "first n_clusters>20 completes; higher resolutions skipped",
                           "candidate_h5ad_saved": False, "graph_only_all_cells": True,
                           "source_template_transferred_to_gpu_before_loop": True,
                           "candidate_object_created_by_copy": True,
                           "candidate_object_deleted_after_outputs": True,
                           "clustering_algorithm": "leiden",
                           "neighbor_algorithm": "cagra",
                           "downsampled": False, "random_seed": SEED, "backend": "rapids-singlecell GPU",
                           "elapsed_seconds": time.monotonic() - started}]).to_csv(params_dir / "graph_parameters.csv", index=False)
            free, total = cp.cuda.runtime.memGetInfo()
            resources.append({"n_pcs": pcs, "n_neighbors": nn, "elapsed_seconds": time.monotonic() - started,
                              "gpu_free_bytes_after": int(free), "gpu_total_bytes": int(total)})
            del run, template; gc.collect(); cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()
    for pending_name, pending in render_futures:
        try:
            pending.result()
        except Exception as exc:
            raise RuntimeError(f"Rendering failed for {pending_name}") from exc
    render_executor.shutdown(wait=True)
    manifest = pd.read_csv(MANIFEST)
    terminal = manifest["status"].isin(["completed", "skipped_cluster_gt20"])
    if len(manifest) != 810 or not terminal.all():
        raise RuntimeError("Grid did not reach 810 terminal rows")
    completed = int((manifest["status"] == "completed").sum())
    skipped_n = int((manifest["status"] == "skipped_cluster_gt20").sum())
    if completed + skipped_n != 810 or manifest[manifest["status"] == "completed"]["n_clusters"].isna().any():
        raise RuntimeError("Invalid terminal accounting")
    expected_graph_names = {label(pcs, nn) for pcs in PCS for nn in NNS}
    pdf_graph_names = {path.parent.name for path in FIGURE.glob("*/umap_all_resolutions.pdf")}
    svg_graph_names = {path.parent.name for path in FIGURE.glob("*/umap_all_resolutions.svg")}
    if pdf_graph_names != expected_graph_names or svg_graph_names != expected_graph_names:
        raise RuntimeError("Grid figures are incomplete or contain unexpected graph directories")
    ranking = rank_candidates(manifest, n_obs)
    final_paths = [
        TABLE / "automatic_candidate_ranking.csv",
        TABLE / "selected_clustering.csv",
        TABLE / "clustering_grid_completion_check.csv",
        TABLE / "gpu_resource_log.csv",
        TABLE / "package_versions.json",
        TABLE / "readme.txt",
    ]
    existing = [str(path) for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite final grid-summary outputs: {existing}")
    atomic_csv(ranking, TABLE / "automatic_candidate_ranking.csv")
    best = ranking.iloc[0]
    selection = pd.DataFrame([{
        "n_pcs": int(best.n_pcs), "n_neighbors": int(best.n_neighbors), "resolution": float(best.resolution),
        "n_clusters": int(best.n_clusters), "raw_cluster_column": f"leiden_res{res_token(float(best.resolution))}",
        "integration_source": "Harmony", "source_h5ad": str(SOURCE.relative_to(WF)), "source_sha256": source_sha,
        "selection_mode": "agent_authorized_automatic", "automatic_rank": 1,
        "composite_score": float(best.composite_score), "selection_note": "User authorized agent to choose best parameters",
        "random_seed": SEED, "downsampled": False,
    }])
    atomic_csv(selection, TABLE / "selected_clustering.csv")
    completion = pd.DataFrame([{"expected_graphs": 81, "terminal_graphs": 81, "expected_rows": 810,
                                "completed_rows": completed, "skipped_cluster_gt20_rows": skipped_n,
                                "failed_rows": 0, "missing_rows": 0, "all_terminal": True,
                                "automatic_selection_ready": True, "selected_n_pcs": int(best.n_pcs),
                                "selected_n_neighbors": int(best.n_neighbors), "selected_resolution": float(best.resolution)}])
    atomic_csv(completion, TABLE / "clustering_grid_completion_check.csv")
    # Reconstruct a complete resource ledger from the immutable per-graph
    # parameter records so a checkpoint-only ranking resume still reports all
    # 81 graphs instead of an empty in-memory list.
    resource_rows = []
    for pcs in PCS:
        for nn in NNS:
            parameter_path = TABLE / label(pcs, nn) / "graph_parameters.csv"
            parameter = pd.read_csv(parameter_path).iloc[0].to_dict()
            resource_rows.append({
                "n_pcs": pcs,
                "n_neighbors": nn,
                "elapsed_seconds": parameter.get("elapsed_seconds", np.nan),
                "backend": parameter.get("backend", "rapids-singlecell GPU"),
                "recovered_from_graph_parameters": True,
            })
    atomic_csv(pd.DataFrame(resource_rows), TABLE / "gpu_resource_log.csv")
    versions = {p: version(p) for p in ["anndata", "rapids-singlecell", "cupy-cuda12x", "numpy", "pandas", "scikit-learn"]}
    versions.update({"python": platform.python_version(), "seed": SEED})
    versions_tmp = TABLE / ".package_versions.json.tmp"
    versions_tmp.write_text(json.dumps(versions, indent=2) + "\n")
    os.replace(versions_tmp, TABLE / "package_versions.json")
    readme_tmp = TABLE / ".readme.txt.tmp"
    readme_tmp.write_text(
        "Full-cell GPU graph grid: PCs/NN 10..50 step5, resolution 0.1..1.0 step0.1.\n"
        "After the first resolution with >20 clusters, higher resolutions for that graph are terminal skipped.\n"
        "Completed checkpoints are reused exactly. No candidate h5ad and no cell downsampling.\n"
        "Automatic ranking uses adjacent-resolution ARI, cluster-size quality, sample/series batch mixing, and neighboring-graph ARI.\n"
        "Status mixing is diagnostic only and is not rewarded because tumor/normal separation may be biological.\n"
    )
    os.replace(readme_tmp, TABLE / "readme.txt")


if __name__ == "__main__":
    main()

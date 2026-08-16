#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import platform
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
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / "epi-cm-core-workflow"
SOURCE = WORKFLOW / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_harmony.h5ad"
GRID_VARIANT = os.environ.get("GC_CLUSTER_GRID_VARIANT", "default")
if GRID_VARIANT == "expanded-v2":
    GRID_DIR = "05-clustering-parameter-search-expanded-v2"
    PCS_VALUES = list(range(10, 51, 5))
    NEIGHBOR_VALUES = list(range(10, 51, 5))
    RESOLUTIONS = [value / 10 for value in range(1, 11)]
elif GRID_VARIANT == "default":
    GRID_DIR = "05-clustering-parameter-search"
    PCS_VALUES = [10, 15, 20, 25]
    NEIGHBOR_VALUES = [10, 20, 30, 40]
    RESOLUTIONS = [0.2, 0.3, 0.4, 0.5]
else:
    raise ValueError(f"Unsupported GC_CLUSTER_GRID_VARIANT: {GRID_VARIANT}")
EXPECTED_CANDIDATES = len(PCS_VALUES) * len(NEIGHBOR_VALUES) * len(RESOLUTIONS)
EXPECTED_GRAPH_GROUPS = len(PCS_VALUES) * len(NEIGHBOR_VALUES)
EXPECTED_ARI_ROWS = EXPECTED_GRAPH_GROUPS * (len(RESOLUTIONS) - 1)
TABLE_ROOT = WORKFLOW / f"tables/01-celltype_integration_clustering/{GRID_DIR}"
FIGURE_ROOT = WORKFLOW / f"figures/01-celltype_integration_clustering/{GRID_DIR}"
CODE_ROOT = WORKFLOW / f"codes/01-celltype_integration_clustering/{GRID_DIR}"
CHECKPOINT_ROOT = TABLE_ROOT / "graph-checkpoints"
MANIFEST_PATH = TABLE_ROOT / "clustering_grid_manifest.csv"
PREFLIGHT_PATH = TABLE_ROOT / "gpu_preflight.csv"

USE_REP = "X_pca_inte"
NEIGHBOR_ALGORITHM = "cagra"
NEIGHBOR_METHOD = "umap"
UMAP_MIN_DIST = 0.5
UMAP_SPREAD = 1.0
LEIDEN_ITERATIONS = 100
SEED = 0


def token(value: float) -> str:
    return str(value).replace(".", "p")


def label(n_pcs: int, n_neighbors: int, resolution: float) -> str:
    return f"pcs-{n_pcs}_nn-{n_neighbors}_res-{token(resolution)}"


def graph_checkpoint_paths(n_pcs: int, n_neighbors: int) -> tuple[Path, Path]:
    checkpoint_dir = CHECKPOINT_ROOT / f"pcs-{n_pcs}_nn-{n_neighbors}"
    return checkpoint_dir / "adjacent_resolution_ari.csv", checkpoint_dir / "graph_resource.csv"


def label_checkpoint_path(n_pcs: int, n_neighbors: int, resolution: float) -> Path:
    return (
        CHECKPOINT_ROOT
        / f"pcs-{n_pcs}_nn-{n_neighbors}"
        / f"labels_res-{token(resolution)}.npy"
    )


def save_label_checkpoint(path: Path, labels: pd.Series) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    with temporary.open("wb") as handle:
        np.save(handle, labels.astype(int).to_numpy(dtype=np.int32))
    os.replace(temporary, path)


def write_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_text_new(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_frame_recoverable(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        pd.testing.assert_frame_equal(
            existing.reset_index(drop=True),
            frame.reset_index(drop=True),
            check_dtype=False,
            rtol=1e-10,
            atol=1e-12,
        )
        return
    write_new(path, frame)


def write_text_recoverable(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise RuntimeError(f"Existing candidate code differs from regenerated content: {path}")
        return
    write_text_new(path, value)


def write_checkpoint(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        return
    write_new(path, frame)


def save_manifest(manifest: pd.DataFrame) -> None:
    temporary = MANIFEST_PATH.with_suffix(".tmp.csv")
    manifest.to_csv(temporary, index=False)
    os.replace(temporary, MANIFEST_PATH)


def release_gpu() -> tuple[int, int]:
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    return int(free_bytes), int(total_bytes)


def as_numpy(value):
    return cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)


def normalized_entropy(labels: pd.Series, groups: pd.Series) -> float:
    table = pd.crosstab(labels.astype(str), groups.astype(str)).to_numpy(dtype=float)
    totals = table.sum(axis=1)
    probabilities = np.divide(table, totals[:, None], out=np.zeros_like(table), where=totals[:, None] > 0)
    entropy = -(probabilities * np.log(probabilities, where=probabilities > 0, out=np.zeros_like(probabilities))).sum(axis=1)
    denominator = np.log(table.shape[1]) if table.shape[1] > 1 else 1.0
    weights = totals / totals.sum()
    return float(np.sum(weights * entropy / denominator))


def save_candidate_figure(
    coords: np.ndarray,
    clusters: pd.Series,
    series: pd.Series,
    status: pd.Series,
    output_stem: Path,
    title: str,
) -> None:
    cluster_cat = pd.Categorical(clusters.astype(str), categories=sorted(clusters.astype(str).unique(), key=lambda x: int(x)))
    series_cat = pd.Categorical(series.astype(str))
    status_cat = pd.Categorical(status.astype(str))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = [
        (cluster_cat.codes, "Leiden", "tab20"),
        (series_cat.codes, "Series", "tab10"),
        (status_cat.codes, "Status", "Set1"),
    ]
    for ax, (codes, panel_title, cmap) in zip(axes, panels):
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=codes,
            cmap=cmap,
            s=0.12,
            linewidths=0,
            alpha=0.75,
            rasterized=True,
        )
        ax.set_title(panel_title)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
    centroid_frame = pd.DataFrame(
        {"cluster": clusters.astype(str).to_numpy(), "x": coords[:, 0], "y": coords[:, 1]}
    ).groupby("cluster", sort=False)[["x", "y"]].median()
    for cluster, row in centroid_frame.iterrows():
        axes[0].text(
            row["x"],
            row["y"],
            cluster,
            fontsize=6,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.5},
        )
    fig.suptitle(title)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".svg"]:
        path = output_stem.with_suffix(suffix)
        if path.exists():
            continue
        temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
        fig.savefig(temporary, dpi=240, bbox_inches="tight")
        os.replace(temporary, path)
    plt.close(fig)


def candidate_outputs_exist(row: pd.Series) -> bool:
    table_a = WORKFLOW / str(row["expected_cluster_count_table"])
    table_b = WORKFLOW / str(row["expected_parameter_table"])
    fig_dir = WORKFLOW / str(row["expected_figure_dir"])
    code = (
        WORKFLOW / str(row["expected_candidate_code"])
        if "expected_candidate_code" in row.index
        else CODE_ROOT / str(row["candidate_label"]) / "candidate_parameters.py"
    )
    return (
        table_a.exists()
        and table_b.exists()
        and (fig_dir / "umap_leiden_series_status.pdf").exists()
        and (fig_dir / "umap_leiden_series_status.svg").exists()
        and code.exists()
    )


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["reason_if_skipped"] = manifest["reason_if_skipped"].fillna("").astype(str)
    if "reused_from_grid" in manifest.columns:
        manifest["reused_from_grid"] = manifest["reused_from_grid"].fillna("").astype(str)
    if len(manifest) != EXPECTED_CANDIDATES or manifest["candidate_label"].nunique() != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Clustering manifest is not the required {EXPECTED_CANDIDATES}-candidate {GRID_VARIANT} grid"
        )
    observed_grid = set(
        map(tuple, manifest[["n_pcs", "n_neighbors", "resolution"]].itertuples(index=False, name=None))
    )
    expected_grid = {
        (n_pcs, n_neighbors, resolution)
        for n_pcs in PCS_VALUES
        for n_neighbors in NEIGHBOR_VALUES
        for resolution in RESOLUTIONS
    }
    if observed_grid != expected_grid:
        raise RuntimeError("Manifest parameter grid does not match the configured values")
    if not PREFLIGHT_PATH.exists() or not pd.read_csv(PREFLIGHT_PATH)["passed"].all():
        raise RuntimeError("Clustering GPU preflight is missing or failed")
    if "selected_h5ad" in manifest.columns or manifest.astype(str).apply(
        lambda col: col.str.contains("candidate.*h5ad", case=False, regex=True).any()
    ).any():
        raise RuntimeError("Candidate manifest must not contain per-candidate h5ad paths")

    already_completed = manifest["completed"].astype(str).str.lower().eq("true")
    for _, row in manifest[already_completed].iterrows():
        if not candidate_outputs_exist(row):
            raise RuntimeError(f"Manifest says completed but outputs are missing: {row['candidate_label']}")

    print(f"reading {SOURCE}", flush=True)
    adata_harmony = ad.read_h5ad(SOURCE)
    if adata_harmony.shape != (340166, 3000):
        raise RuntimeError(f"Unexpected Harmony source shape: {adata_harmony.shape}")
    if USE_REP not in adata_harmony.obsm or adata_harmony.obsm[USE_REP].shape[1] < max(PCS_VALUES):
        raise RuntimeError("Harmony basis missing or has too few dimensions")
    if "X_umap" in adata_harmony.obsm or adata_harmony.obsp:
        raise RuntimeError("Harmony source is not a clean pre-graph object")

    transfer_started = time.monotonic()
    rsc.get.anndata_to_GPU(adata_harmony)
    free_after_template, total_gpu = cp.cuda.runtime.memGetInfo()
    print(
        f"source template transferred to GPU in {time.monotonic() - transfer_started:.1f}s; "
        f"free={free_after_template / 2**30:.2f} GiB",
        flush=True,
    )

    graph_rows = []
    review_rows = []
    ari_rows = []
    failures = []

    for n_pcs in PCS_VALUES:
        for n_neighbors in NEIGHBOR_VALUES:
            group_mask = (manifest["n_pcs"] == n_pcs) & (manifest["n_neighbors"] == n_neighbors)
            group_indices = manifest.index[group_mask].tolist()
            checkpoint_ari_path, checkpoint_resource_path = graph_checkpoint_paths(
                n_pcs, n_neighbors
            )
            if all(
                (
                    str(manifest.loc[idx, "completed"]).lower() == "true"
                    and candidate_outputs_exist(manifest.loc[idx])
                )
                or str(manifest.loc[idx, "status"]).startswith("skipped")
                for idx in group_indices
            ) and checkpoint_ari_path.exists() and checkpoint_resource_path.exists():
                print(f"reuse completed graph group pcs={n_pcs}, nn={n_neighbors}", flush=True)
                continue

            work_indices = [
                idx
                for idx in group_indices
                if not (
                    (
                        str(manifest.loc[idx, "completed"]).lower() == "true"
                        and candidate_outputs_exist(manifest.loc[idx])
                    )
                    or str(manifest.loc[idx, "status"]).startswith("skipped")
                )
            ]
            manifest.loc[work_indices, "status"] = "running"
            save_manifest(manifest)
            graph_started = time.monotonic()
            free_before, _ = cp.cuda.runtime.memGetInfo()
            print(
                f"graph pcs={n_pcs}, nn={n_neighbors}; free before copy={free_before / 2**30:.2f} GiB",
                flush=True,
            )
            try:
                adata_run = adata_harmony.copy()
                rsc.pp.neighbors(
                    adata_run,
                    n_neighbors=n_neighbors,
                    n_pcs=n_pcs,
                    use_rep=USE_REP,
                    algorithm=NEIGHBOR_ALGORITHM,
                    method=NEIGHBOR_METHOD,
                    random_state=SEED,
                )
                rsc.tl.umap(
                    adata_run,
                    min_dist=UMAP_MIN_DIST,
                    spread=UMAP_SPREAD,
                    random_state=SEED,
                )
                coords = as_numpy(adata_run.obsm["X_umap"])
                if coords.shape != (adata_run.n_obs, 2) or not np.isfinite(coords).all():
                    raise RuntimeError("UMAP coordinates are missing or non-finite")
                labels_by_resolution: dict[float, pd.Series] = {}

                stop_trigger_resolution = None
                stop_trigger_clusters = None
                for resolution in RESOLUTIONS:
                    candidate = label(n_pcs, n_neighbors, resolution)
                    idx = manifest.index[manifest["candidate_label"].eq(candidate)].item()
                    row = manifest.loc[idx]
                    candidate_completed = (
                        str(row["completed"]).lower() == "true" and candidate_outputs_exist(row)
                    )
                    reused_from_grid = str(row.get("reused_from_grid", "")).strip()
                    if str(row["status"]).startswith("skipped"):
                        print(f"  reuse skipped candidate {candidate}", flush=True)
                        continue
                    if candidate_completed and reused_from_grid:
                        print(f"  reuse prior candidate {candidate}", flush=True)
                        prior_n_clusters = int(
                            pd.read_csv(WORKFLOW / row["expected_parameter_table"]).loc[
                                0, "n_clusters"
                            ]
                        )
                        if prior_n_clusters > 20 and stop_trigger_resolution is None:
                            stop_trigger_resolution = resolution
                            stop_trigger_clusters = prior_n_clusters
                        continue

                    if stop_trigger_resolution is not None and not candidate_completed:
                        manifest.loc[idx, "status"] = "skipped_cluster_gt20"
                        manifest.loc[idx, "completed"] = False
                        manifest.loc[idx, "reason_if_skipped"] = (
                            f"lower_resolution_{token(stop_trigger_resolution)}_had_"
                            f"{stop_trigger_clusters}_clusters_gt_20"
                        )
                        save_manifest(manifest)
                        print(
                            f"  skipped {candidate}: prior resolution "
                            f"{stop_trigger_resolution} had {stop_trigger_clusters} clusters",
                            flush=True,
                        )
                        continue

                    cluster_key = f"leiden_res{token(resolution)}"
                    labels_path = label_checkpoint_path(n_pcs, n_neighbors, resolution)
                    if labels_path.exists():
                        label_values = np.load(labels_path)
                        if len(label_values) != adata_run.n_obs:
                            raise RuntimeError(f"Label checkpoint length mismatch for {candidate}")
                        clusters = pd.Series(
                            label_values.astype(str), index=adata_run.obs_names, name=cluster_key
                        )
                    else:
                        rsc.tl.leiden(
                            adata_run,
                            resolution=resolution,
                            key_added=cluster_key,
                            random_state=SEED,
                            n_iterations=LEIDEN_ITERATIONS,
                        )
                        clusters = adata_run.obs[cluster_key].astype(str).copy()
                        save_label_checkpoint(labels_path, clusters)
                    if clusters.isna().any() or len(clusters) != adata_run.n_obs:
                        raise RuntimeError(f"Incomplete Leiden labels for {candidate}")
                    labels_by_resolution[resolution] = clusters
                    if candidate_completed:
                        prior_n_clusters = int(
                            pd.read_csv(WORKFLOW / row["expected_parameter_table"]).loc[
                                0, "n_clusters"
                            ]
                        )
                        print(f"  reuse expanded candidate {candidate}", flush=True)
                        if prior_n_clusters > 20 and stop_trigger_resolution is None:
                            stop_trigger_resolution = resolution
                            stop_trigger_clusters = prior_n_clusters
                        continue

                    table_dir = TABLE_ROOT / candidate
                    figure_dir = FIGURE_ROOT / candidate
                    code_dir = CODE_ROOT / candidate
                    count_path = table_dir / "cluster_counts.csv"
                    parameter_path = table_dir / "clustering_parameters.csv"
                    figure_stem = figure_dir / "umap_leiden_series_status"
                    code_path = code_dir / "candidate_parameters.py"

                    counts = clusters.value_counts().sort_index(key=lambda index: index.astype(int))
                    counts.index.name = "cluster"
                    centroids = pd.DataFrame(
                        {"cluster": clusters.to_numpy(), "umap1": coords[:, 0], "umap2": coords[:, 1]}
                    ).groupby("cluster", sort=False)[["umap1", "umap2"]].median()
                    count_table = counts.rename("n_cells").to_frame().join(centroids).reset_index()
                    count_table["fraction"] = count_table["n_cells"] / adata_run.n_obs
                    write_frame_recoverable(count_path, count_table)

                    save_candidate_figure(
                        coords,
                        clusters,
                        adata_run.obs["series"],
                        adata_run.obs["status"],
                        figure_stem,
                        candidate,
                    )

                    parameter_frame = pd.DataFrame(
                        [
                            {
                                "candidate_label": candidate,
                                "source_h5ad": str(SOURCE),
                                "source_template_transferred_to_gpu_before_loop": True,
                                "candidate_object_created_by_copy": True,
                                "n_cells": adata_run.n_obs,
                                "n_pcs": n_pcs,
                                "n_neighbors": n_neighbors,
                                "resolution": resolution,
                                "use_rep": USE_REP,
                                "neighbor_algorithm": NEIGHBOR_ALGORITHM,
                                "neighbor_method": NEIGHBOR_METHOD,
                                "umap_min_dist": UMAP_MIN_DIST,
                                "umap_spread": UMAP_SPREAD,
                                "clustering_algorithm": "rapids-singlecell cuGraph Leiden",
                                "raw_cluster_column": cluster_key,
                                "n_clusters": clusters.nunique(),
                                "smallest_cluster_cells": int(counts.min()),
                                "largest_cluster_cells": int(counts.max()),
                                "weighted_sample_entropy": normalized_entropy(clusters, adata_run.obs["sample"]),
                                "weighted_series_entropy": normalized_entropy(clusters, adata_run.obs["series"]),
                                "weighted_status_entropy": normalized_entropy(clusters, adata_run.obs["status"]),
                                "umap_all_cells_finite": True,
                                "cluster_count_table": str(count_path),
                                "figure_pdf": str(figure_stem.with_suffix(".pdf")),
                                "figure_svg": str(figure_stem.with_suffix(".svg")),
                                "code_file": str(Path(__file__).resolve()),
                                "candidate_parameter_code": str(code_path),
                                "random_seed": SEED,
                                "backend_package": f"rapids-singlecell {version('rapids-singlecell')}",
                                "candidate_h5ad_saved": False,
                                "candidate_object_deleted_after_outputs": True,
                                "gpu_memory_release_status": "verified_after_shared_graph_resolution_outputs",
                                "review_notes": "Manual review required; not automatically selected.",
                            }
                        ]
                    )
                    write_frame_recoverable(parameter_path, parameter_frame)
                    candidate_code = "\n".join(
                            [
                                "#!/usr/bin/env python3",
                                f"SOURCE_H5AD = {str(SOURCE)!r}",
                                f"N_PCS = {n_pcs}",
                                f"N_NEIGHBORS = {n_neighbors}",
                                f"RESOLUTION = {resolution}",
                                f"USE_REP = {USE_REP!r}",
                                f"NEIGHBOR_ALGORITHM = {NEIGHBOR_ALGORITHM!r}",
                                f"RANDOM_SEED = {SEED}",
                                "if __name__ == '__main__':",
                                "    print({k: v for k, v in globals().items() if k.isupper()})",
                                "",
                            ]
                        )
                    write_text_recoverable(code_path, candidate_code)

                    manifest.loc[idx, "status"] = "completed"
                    manifest.loc[idx, "completed"] = True
                    manifest.loc[idx, "reason_if_skipped"] = ""
                    save_manifest(manifest)
                    review_rows.append(parameter_frame.iloc[0].to_dict())
                    print(
                        f"  completed {candidate}: {clusters.nunique()} clusters, "
                        f"sample_entropy={parameter_frame.iloc[0]['weighted_sample_entropy']:.3f}",
                        flush=True,
                    )
                    if clusters.nunique() > 20 and stop_trigger_resolution is None:
                        stop_trigger_resolution = resolution
                        stop_trigger_clusters = int(clusters.nunique())

                ordered_resolutions = [value for value in RESOLUTIONS if value in labels_by_resolution]
                group_ari_rows = []
                for left, right in zip(ordered_resolutions[:-1], ordered_resolutions[1:]):
                    if not np.isclose(right - left, 0.1):
                        continue
                    group_ari_rows.append(
                        {
                            "n_pcs": n_pcs,
                            "n_neighbors": n_neighbors,
                            "resolution_left": left,
                            "resolution_right": right,
                            "adjusted_rand_index": adjusted_rand_score(
                                labels_by_resolution[left], labels_by_resolution[right]
                            ),
                        }
                    )
                group_ari = pd.DataFrame(
                    group_ari_rows,
                    columns=[
                        "n_pcs",
                        "n_neighbors",
                        "resolution_left",
                        "resolution_right",
                        "adjusted_rand_index",
                    ],
                )
                available_resolutions = [
                    resolution
                    for resolution in RESOLUTIONS
                    if (
                        str(
                            manifest.loc[
                                manifest["candidate_label"].eq(
                                    label(n_pcs, n_neighbors, resolution)
                                ),
                                "completed",
                            ].iloc[0]
                        ).lower()
                        == "true"
                        and not str(
                            manifest.loc[
                                manifest["candidate_label"].eq(
                                    label(n_pcs, n_neighbors, resolution)
                                ),
                                "reused_from_grid",
                            ].iloc[0]
                        ).strip()
                        and label_checkpoint_path(n_pcs, n_neighbors, resolution).exists()
                    )
                ]
                expected_group_ari_rows = sum(
                    np.isclose(right - left, 0.1)
                    for left, right in zip(available_resolutions[:-1], available_resolutions[1:])
                )
                if len(group_ari) != expected_group_ari_rows:
                    raise RuntimeError(
                        f"Incomplete adjacent-resolution ARI rows for pcs={n_pcs}, nn={n_neighbors}"
                    )
                write_checkpoint(checkpoint_ari_path, group_ari)
                ari_rows.extend(group_ari_rows)

                del adata_run, coords, labels_by_resolution
                free_after, total_after = release_gpu()
                graph_resource = {
                        "n_pcs": n_pcs,
                        "n_neighbors": n_neighbors,
                        "elapsed_seconds": time.monotonic() - graph_started,
                        "gpu_free_bytes_before_copy": int(free_before),
                        "gpu_free_bytes_after_delete_release": int(free_after),
                        "gpu_total_bytes": int(total_after),
                        "candidate_object_deleted_after_outputs": True,
                        "gpu_memory_release_verified": free_after >= free_before * 0.95,
                    }
                write_checkpoint(checkpoint_resource_path, pd.DataFrame([graph_resource]))
                graph_rows.append(graph_resource)
                print(
                    f"released graph pcs={n_pcs}, nn={n_neighbors}; free={free_after / 2**30:.2f} GiB",
                    flush=True,
                )
            except Exception as exc:
                if "adata_run" in locals():
                    del adata_run
                release_gpu()
                reason = f"{type(exc).__name__}: {exc}"
                failure_indices = [
                    idx
                    for idx in work_indices
                    if not str(manifest.loc[idx, "status"]).startswith("skipped")
                ]
                manifest.loc[failure_indices, "status"] = "failed"
                manifest.loc[failure_indices, "completed"] = False
                manifest.loc[failure_indices, "reason_if_skipped"] = reason
                save_manifest(manifest)
                failures.append(
                    {"n_pcs": n_pcs, "n_neighbors": n_neighbors, "failure_reason": reason}
                )
                print(f"FAILED graph pcs={n_pcs}, nn={n_neighbors}: {reason}", flush=True)

    del adata_harmony
    release_gpu()

    manifest = pd.read_csv(MANIFEST_PATH)
    if "reused_from_grid" in manifest.columns:
        manifest["reused_from_grid"] = manifest["reused_from_grid"].fillna("").astype(str)
    completed = manifest["completed"].astype(str).str.lower().eq("true")
    skipped = manifest["status"].astype(str).str.startswith("skipped")
    terminal = completed | skipped
    outputs_exist = manifest.apply(candidate_outputs_exist, axis=1)
    completed_outputs_exist = bool(outputs_exist[completed].all())
    if int(terminal.sum()) != EXPECTED_CANDIDATES or not completed_outputs_exist:
        print(
            f"grid incomplete: completed={int(completed.sum())}, skipped={int(skipped.sum())}, "
            f"terminal={int(terminal.sum())}/{EXPECTED_CANDIDATES}, "
            f"completed_outputs_exist={int(outputs_exist[completed].sum())}/{int(completed.sum())}",
            flush=True,
        )
        raise RuntimeError("Clustering grid is incomplete; repair failed groups and resume")

    review_frames = []
    for _, row in manifest[completed & outputs_exist].iterrows():
        review_frames.append(pd.read_csv(WORKFLOW / row["expected_parameter_table"]))
    review = pd.concat(review_frames, ignore_index=True) if review_frames else pd.DataFrame()
    if len(review) != int(completed.sum()):
        raise RuntimeError("Candidate review rows are incomplete")
    review = review.merge(
        manifest[["candidate_label", "reused_from_grid"]].fillna(""),
        on="candidate_label",
        how="left",
        validate="one_to_one",
    )
    review["result_source_grid"] = np.where(
        review["reused_from_grid"].astype(str).str.len().gt(0),
        review["reused_from_grid"],
        GRID_DIR,
    )

    checkpoint_ari_frames = []
    checkpoint_resource_frames = []
    for n_pcs in PCS_VALUES:
        for n_neighbors in NEIGHBOR_VALUES:
            checkpoint_ari_path, checkpoint_resource_path = graph_checkpoint_paths(
                n_pcs, n_neighbors
            )
            if not checkpoint_ari_path.exists() or not checkpoint_resource_path.exists():
                raise RuntimeError(
                    f"Graph checkpoint missing for pcs={n_pcs}, nn={n_neighbors}"
                )
            checkpoint_ari_frames.append(pd.read_csv(checkpoint_ari_path))
            checkpoint_resource_frames.append(pd.read_csv(checkpoint_resource_path))
    final_ari = pd.concat(checkpoint_ari_frames, ignore_index=True)
    final_ari["ari_source_grid"] = GRID_DIR
    unavailable_cross_source_ari_rows = 0
    unavailable_skipped_ari_rows = 0
    if GRID_VARIANT == "expanded-v2":
        legacy_ari_path = (
            WORKFLOW
            / "tables/01-celltype_integration_clustering/05-clustering-parameter-search/adjacent_resolution_ari.csv"
        )
        if not legacy_ari_path.exists():
            raise RuntimeError("Legacy ARI table required for reused candidates is missing")
        legacy_ari = pd.read_csv(legacy_ari_path)
        legacy_ari["ari_source_grid"] = "05-clustering-parameter-search"
        final_ari = pd.concat([legacy_ari, final_ari], ignore_index=True)
    expected_available_ari_rows = 0
    for n_pcs in PCS_VALUES:
        for n_neighbors in NEIGHBOR_VALUES:
            group = manifest[
                (manifest["n_pcs"] == n_pcs)
                & (manifest["n_neighbors"] == n_neighbors)
            ].set_index("resolution")
            for left, right in zip(RESOLUTIONS[:-1], RESOLUTIONS[1:]):
                left_completed = str(group.loc[left, "completed"]).lower() == "true"
                right_completed = str(group.loc[right, "completed"]).lower() == "true"
                if not (left_completed and right_completed):
                    unavailable_skipped_ari_rows += 1
                    continue
                left_reused = bool(str(group.loc[left, "reused_from_grid"]).strip())
                right_reused = bool(str(group.loc[right, "reused_from_grid"]).strip())
                if left_reused != right_reused:
                    unavailable_cross_source_ari_rows += 1
                    continue
                expected_available_ari_rows += 1
    final_ari = final_ari.drop_duplicates(
        subset=["n_pcs", "n_neighbors", "resolution_left", "resolution_right"],
        keep="first",
    ).sort_values(["n_pcs", "n_neighbors", "resolution_left"]).reset_index(drop=True)
    final_resources = pd.concat(checkpoint_resource_frames, ignore_index=True)
    if len(final_ari) != expected_available_ari_rows or len(final_resources) != EXPECTED_GRAPH_GROUPS:
        raise RuntimeError("Graph checkpoint accounting is incomplete")
    if not final_resources["gpu_memory_release_verified"].all():
        raise RuntimeError("One or more graph groups failed GPU memory-release verification")

    write_new(TABLE_ROOT / "candidate_review_manifest.csv", review)
    write_new(TABLE_ROOT / "graph_resource_log.csv", final_resources)
    write_new(TABLE_ROOT / "adjacent_resolution_ari.csv", final_ari)
    write_new(
        TABLE_ROOT / "grid_failures.csv",
        pd.DataFrame(columns=["n_pcs", "n_neighbors", "failure_reason"]),
    )
    completion = pd.DataFrame(
        [
            {
                "expected_candidates": EXPECTED_CANDIDATES,
                "completed_candidates": int(completed.sum()),
                "reused_completed_candidates": int(
                    manifest["reused_from_grid"].astype(str).str.len().gt(0).sum()
                ),
                "new_completed_candidates": int(
                    completed.sum()
                    - manifest["reused_from_grid"].astype(str).str.len().gt(0).sum()
                ),
                "failed_candidates": int(manifest["status"].eq("failed").sum()),
                "skipped_candidates": int(manifest["status"].str.startswith("skipped").sum()),
                "missing_candidates": int((~terminal).sum()),
                "all_completed_lightweight_outputs_exist": completed_outputs_exist,
                "full_adjacent_ari_rows": EXPECTED_ARI_ROWS,
                "available_adjacent_ari_rows": len(final_ari),
                "unavailable_cross_source_ari_rows": unavailable_cross_source_ari_rows,
                "unavailable_skipped_ari_rows": unavailable_skipped_ari_rows,
                "manual_selection_ready": bool(
                    terminal.sum() == EXPECTED_CANDIDATES and completed_outputs_exist
                ),
            }
        ]
    )
    write_new(TABLE_ROOT / "clustering_grid_completion_check.csv", completion)
    versions = {
        package: version(package)
        for package in [
            "rapids-singlecell",
            "cudf-cu12",
            "cugraph-cu12",
            "cuml-cu12",
            "cupy-cuda12x",
            "anndata",
            "numpy",
            "pandas",
            "scikit-learn",
            "matplotlib",
        ]
    }
    versions.update(
        {
            "python": platform.python_version(),
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "random_seed": SEED,
        }
    )
    write_text_new(
        TABLE_ROOT / "package_versions_03_run_clustering_grid.json",
        json.dumps(versions, indent=2),
    )
    write_text_new(
        TABLE_ROOT / "readme.txt",
        "\n".join(
            [
                f"Block 01 / {GRID_DIR}",
                f"Source: {SOURCE}",
                f"Grid variant: {GRID_VARIANT}.",
                f"n_pcs: {PCS_VALUES}.",
                f"n_neighbors: {NEIGHBOR_VALUES}.",
                f"Leiden resolutions: {RESOLUTIONS} ({EXPECTED_CANDIDATES} candidates).",
                "Previously completed candidates are reused without rerunning their Leiden clustering or outputs.",
                "Within each graph group, resolutions after the first candidate with more than 20 clusters are skipped.",
                "Adjacent-resolution ARI spanning one reused and one new candidate is recorded as unavailable rather than rerunning the reused candidate.",
                "All 340166 cells were used; no downsampling or subsampling.",
                "GPU backend: rapids-singlecell CAGRA neighbors, UMAP, and cuGraph Leiden.",
                "Each candidate has PDF/SVG UMAP, cluster counts, parameters, and candidate parameter code; no candidate h5ad was saved.",
                "The user authorized agent selection after the expanded grid is complete.",
            ]
        )
        + "\n",
    )

    print(completion.to_string(index=False), flush=True)
    if not bool(completion.iloc[0]["manual_selection_ready"]):
        raise RuntimeError("Clustering grid is incomplete; inspect manifest and failures")


if __name__ == "__main__":
    main()

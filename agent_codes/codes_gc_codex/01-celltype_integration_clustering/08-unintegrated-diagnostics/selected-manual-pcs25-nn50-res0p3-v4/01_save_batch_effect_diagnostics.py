#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import random
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
RUN = "selected-manual-pcs25-nn50-res0p3-v4"
N_PCS = 25
N_NEIGHBORS = 30
MAX_CELLS_PER_SERIES_PER_CELLTYPE = 500

random.seed(SEED)
np.random.seed(SEED)

WF = Path(__file__).resolve().parents[4]
INPUT = WF / f"h5ad/01-celltype_integration_clustering/06-broad-annotation/{RUN}/adata_anno.h5ad"
TABLE = WF / f"tables/01-celltype_integration_clustering/08-unintegrated-diagnostics/{RUN}"
FIG = WF / f"figures/01-celltype_integration_clustering/08-unintegrated-diagnostics/{RUN}"


def refuse_existing() -> None:
    existing = [str(path) for path in (TABLE, FIG) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing batch-diagnostic overwrite: {existing}")


def local_inverse_simpson(neighbor_labels: np.ndarray, n_categories: int) -> tuple[float, float]:
    values = []
    for row in neighbor_labels:
        _, counts = np.unique(row, return_counts=True)
        proportions = counts / counts.sum()
        values.append(1.0 / float(np.square(proportions).sum()))
    lisi = float(np.mean(values))
    denominator = max(1, min(neighbor_labels.shape[1], n_categories) - 1)
    normalized = float((lisi - 1.0) / denominator)
    return lisi, normalized


def main() -> None:
    refuse_existing()
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    atlas = ad.read_h5ad(INPUT, backed="r")
    required_obs = {"leiden_coarse", "series", "sample"}
    missing = sorted(required_obs - set(atlas.obs.columns))
    if missing:
        raise KeyError(f"Missing batch-diagnostic obs columns: {missing}")
    for basis in ("X_pca", "X_pca_inte"):
        if basis not in atlas.obsm or atlas.obsm[basis].shape[1] < N_PCS:
            raise KeyError(f"Missing or undersized obsm[{basis!r}]")

    obs = atlas.obs[["leiden_coarse", "series", "sample"]].copy()
    rng = np.random.default_rng(SEED)
    selected_positions: list[int] = []
    for _, lineage_frame in obs.groupby("leiden_coarse", observed=True):
        for _, series_frame in lineage_frame.groupby("series", observed=True):
            positions = obs.index.get_indexer(series_frame.index)
            n_take = min(MAX_CELLS_PER_SERIES_PER_CELLTYPE, len(positions))
            selected_positions.extend(rng.choice(positions, size=n_take, replace=False).tolist())
    selected = np.asarray(sorted(set(selected_positions)), dtype=np.int64)
    sampled_obs = obs.iloc[selected].copy()
    sampling_manifest = sampled_obs.reset_index(names="cell_id")
    sampling_manifest["atlas_position"] = selected

    rows: list[dict[str, object]] = []
    for cell_type, lineage_frame in sampling_manifest.groupby("leiden_coarse", observed=True):
        positions = lineage_frame["atlas_position"].to_numpy(dtype=np.int64)
        lineage_meta = obs.iloc[positions]
        if len(positions) < N_NEIGHBORS + 1:
            raise RuntimeError(f"Too few sampled cells for {cell_type}: {len(positions)}")
        for basis in ("X_pca", "X_pca_inte"):
            embedding = np.asarray(atlas.obsm[basis][positions, :N_PCS], dtype=np.float32)
            model = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, metric="euclidean", n_jobs=8)
            indices = model.fit(embedding).kneighbors(embedding, return_distance=False)
            if not np.array_equal(indices[:, 0], np.arange(len(indices))):
                raise RuntimeError(f"Self-neighbor was not first for {cell_type}/{basis}")
            neighbor_indices = indices[:, 1:]
            for batch_key in ("series", "sample"):
                labels = lineage_meta[batch_key].astype(str).to_numpy()
                neighbor_labels = labels[neighbor_indices]
                same_fraction = float((neighbor_labels == labels[:, None]).mean())
                frequencies = pd.Series(labels).value_counts(normalize=True)
                expected_same = float(np.square(frequencies.to_numpy()).sum())
                lisi, lisi_normalized = local_inverse_simpson(neighbor_labels, len(frequencies))
                if 1 < len(frequencies) < len(labels):
                    silhouette = float(
                        silhouette_score(
                            embedding,
                            labels,
                            metric="euclidean",
                            sample_size=min(3000, len(labels)),
                            random_state=SEED,
                        )
                    )
                else:
                    silhouette = np.nan
                rows.append(
                    {
                        "cell_type": str(cell_type),
                        "basis": basis,
                        "integration_state": "before_harmony" if basis == "X_pca" else "after_harmony",
                        "batch_key": batch_key,
                        "n_cells_sampled": len(labels),
                        "n_batches_observed": len(frequencies),
                        "knn_k": N_NEIGHBORS,
                        "same_batch_neighbor_fraction": same_fraction,
                        "expected_same_batch_fraction": expected_same,
                        "same_batch_neighbor_excess": same_fraction - expected_same,
                        "local_inverse_simpson": lisi,
                        "local_inverse_simpson_normalized": lisi_normalized,
                        "batch_silhouette": silhouette,
                    }
                )
    atlas.file.close()

    metrics = pd.DataFrame(rows)
    summary_rows = []
    for (basis, state, batch_key), frame in metrics.groupby(
        ["basis", "integration_state", "batch_key"], observed=True
    ):
        weights = frame["n_cells_sampled"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "basis": basis,
                "integration_state": state,
                "batch_key": batch_key,
                "n_lineages": len(frame),
                "n_lineage_sampled_cells_sum": int(weights.sum()),
                "weighted_same_batch_neighbor_excess": float(
                    np.average(frame["same_batch_neighbor_excess"], weights=weights)
                ),
                "weighted_local_inverse_simpson_normalized": float(
                    np.average(frame["local_inverse_simpson_normalized"], weights=weights)
                ),
                "weighted_batch_silhouette": float(
                    np.average(frame["batch_silhouette"], weights=weights)
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    improvements = []
    for batch_key, frame in summary.groupby("batch_key", observed=True):
        indexed = frame.set_index("integration_state")
        before = indexed.loc["before_harmony"]
        after = indexed.loc["after_harmony"]
        improvements.append(
            {
                "batch_key": batch_key,
                "same_batch_neighbor_excess_reduction": float(
                    before["weighted_same_batch_neighbor_excess"]
                    - after["weighted_same_batch_neighbor_excess"]
                ),
                "normalized_lisi_increase": float(
                    after["weighted_local_inverse_simpson_normalized"]
                    - before["weighted_local_inverse_simpson_normalized"]
                ),
                "batch_silhouette_reduction": float(
                    before["weighted_batch_silhouette"] - after["weighted_batch_silhouette"]
                ),
                "after_harmony_same_batch_neighbor_excess": float(
                    after["weighted_same_batch_neighbor_excess"]
                ),
                "after_harmony_batch_silhouette": float(after["weighted_batch_silhouette"]),
            }
        )
    improvement = pd.DataFrame(improvements)

    TABLE.mkdir(parents=True, exist_ok=False)
    FIG.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(TABLE / "01_batch_effect_metrics_by_cell_type.csv", index=False)
    summary.to_csv(TABLE / "02_batch_effect_weighted_summary.csv", index=False)
    improvement.to_csv(TABLE / "03_harmony_improvement_summary.csv", index=False)
    sampling_manifest.to_csv(TABLE / "04_batch_diagnostic_sampling_manifest.csv", index=False)

    sns.set_theme(style="ticks")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    plot_frame = summary.copy()
    plot_frame["state"] = plot_frame["integration_state"].map(
        {"before_harmony": "Before Harmony", "after_harmony": "After Harmony"}
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    panels = [
        ("weighted_batch_silhouette", "Batch silhouette\n(lower is better)"),
        ("weighted_same_batch_neighbor_excess", "Same-batch neighbor excess\n(lower is better)"),
        ("weighted_local_inverse_simpson_normalized", "Normalized local diversity\n(higher is better)"),
    ]
    for axis, (column, label) in zip(axes, panels):
        sns.barplot(data=plot_frame, x="batch_key", y=column, hue="state", ax=axis)
        axis.set_xlabel("Batch key")
        axis.set_ylabel(label)
        axis.tick_params(axis="both", labelsize=8)
    axes[0].legend(title="", frameon=False)
    for axis in axes[1:]:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    figure.tight_layout()
    for suffix in ("pdf", "svg"):
        figure.savefig(FIG / f"batch_effect_before_after_harmony.{suffix}", bbox_inches="tight")
    plt.close(figure)

    completion = {
        "all_passed": True,
        "input_h5ad": str(INPUT.relative_to(WF)),
        "n_atlas_cells": int(len(obs)),
        "n_sampled_unique_cells": int(len(selected)),
        "n_cell_types": int(metrics["cell_type"].nunique()),
        "batch_keys": ["series", "sample"],
        "bases": ["X_pca", "X_pca_inte"],
        "n_pcs": N_PCS,
        "knn_k": N_NEIGHBORS,
        "sampling_rule": f"up to {MAX_CELLS_PER_SERIES_PER_CELLTYPE} cells per series within each broad cell type",
        "interpretation_guardrail": "descriptive diagnostics have no universal pass/fail cutoff; biological composition differences may contribute to residual local structure",
        "pdf_count": len(list(FIG.glob("*.pdf"))),
        "svg_count": len(list(FIG.glob("*.svg"))),
        "png_count": len(list(FIG.glob("*.png"))),
        "random_seed": SEED,
    }
    if completion["pdf_count"] != 1 or completion["svg_count"] != 1 or completion["png_count"]:
        raise RuntimeError(f"Batch diagnostic figure audit failed: {completion}")
    (TABLE / "batch_effect_diagnostics_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    (TABLE / "package_versions.txt").write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                *(f"{name}={version(name)}" for name in ("anndata", "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn")),
                f"seed={SEED}",
                "environment=uv_envs/rapids/.venv",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (TABLE / "readme.txt").write_text(
        "Saved, reproducible batch-effect diagnostics for the formal PCs25/NN50/res0.3 branch.\n"
        "The same deterministically sampled cells are compared in uncorrected X_pca and Harmony X_pca_inte.\n"
        "Metrics are computed within each broad cell type for both series and sample to reduce cell-composition confounding.\n"
        "Lower silhouette and same-batch-neighbor excess and higher normalized local inverse-Simpson diversity indicate improved mixing.\n"
        "Residual structure is descriptive and may include real dataset-specific subtype composition; there is no universal cutoff.\n",
        encoding="utf-8",
    )
    print(json.dumps(completion, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

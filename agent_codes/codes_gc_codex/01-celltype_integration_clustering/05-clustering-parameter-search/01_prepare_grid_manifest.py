#!/usr/bin/env python3
from __future__ import annotations

from itertools import product
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / "epi-cm-core-workflow"
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
OUTPUT = WORKFLOW / f"tables/01-celltype_integration_clustering/{GRID_DIR}/clustering_grid_manifest.csv"
LEGACY_GRID_DIR = "05-clustering-parameter-search"
if OUTPUT.exists():
    raise FileExistsError(f"Refusing to overwrite existing artifact: {OUTPUT}")


def token(value: float) -> str:
    return str(value).replace(".", "p")


rows = []
for n_pcs, n_neighbors, resolution in product(PCS_VALUES, NEIGHBOR_VALUES, RESOLUTIONS):
    label = f"pcs-{n_pcs}_nn-{n_neighbors}_res-{token(resolution)}"
    table_dir = f"tables/01-celltype_integration_clustering/{GRID_DIR}/{label}"
    figure_dir = f"figures/01-celltype_integration_clustering/{GRID_DIR}/{label}"
    candidate_code = f"codes/01-celltype_integration_clustering/{GRID_DIR}/{label}/candidate_parameters.py"
    reused_from_grid = ""
    completed = False
    status = "planned"
    if GRID_VARIANT == "expanded-v2":
        legacy_table_dir = f"tables/01-celltype_integration_clustering/{LEGACY_GRID_DIR}/{label}"
        legacy_figure_dir = f"figures/01-celltype_integration_clustering/{LEGACY_GRID_DIR}/{label}"
        legacy_code = f"codes/01-celltype_integration_clustering/{LEGACY_GRID_DIR}/{label}/candidate_parameters.py"
        legacy_paths = [
            WORKFLOW / legacy_table_dir / "cluster_counts.csv",
            WORKFLOW / legacy_table_dir / "clustering_parameters.csv",
            WORKFLOW / legacy_figure_dir / "umap_leiden_series_status.pdf",
            WORKFLOW / legacy_figure_dir / "umap_leiden_series_status.svg",
            WORKFLOW / legacy_code,
        ]
        if all(path.exists() and path.stat().st_size > 0 for path in legacy_paths):
            table_dir = legacy_table_dir
            figure_dir = legacy_figure_dir
            candidate_code = legacy_code
            reused_from_grid = LEGACY_GRID_DIR
            completed = True
            status = "completed_reused"
    rows.append(
        {
            "n_pcs": n_pcs,
            "n_neighbors": n_neighbors,
            "resolution": resolution,
            "candidate_label": label,
            "expected_cluster_count_table": f"{table_dir}/cluster_counts.csv",
            "expected_parameter_table": f"{table_dir}/clustering_parameters.csv",
            "expected_figure_dir": figure_dir,
            "expected_candidate_code": candidate_code,
            "reused_from_grid": reused_from_grid,
            "status": status,
            "completed": completed,
            "reason_if_skipped": "",
        }
    )
manifest = pd.DataFrame(rows)
if len(manifest) != EXPECTED_CANDIDATES or manifest["candidate_label"].nunique() != EXPECTED_CANDIDATES:
    raise RuntimeError(f"{GRID_VARIANT} clustering manifest must contain {EXPECTED_CANDIDATES} unique candidates")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(OUTPUT, index=False)
print(
    f"wrote {OUTPUT} with {len(manifest)} candidates; "
    f"reused={int(manifest['completed'].sum())}, planned={int((~manifest['completed']).sum())}"
)

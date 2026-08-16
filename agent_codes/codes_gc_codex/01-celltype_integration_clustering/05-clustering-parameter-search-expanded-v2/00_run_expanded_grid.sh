#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ROOT="$(cd "$WORKFLOW/.." && pwd)"
PYTHON="$ROOT/uv_envs/rapids/.venv/bin/python"
BASE_CODE="$WORKFLOW/codes/01-celltype_integration_clustering/05-clustering-parameter-search"
TABLE_DIR="$WORKFLOW/tables/01-celltype_integration_clustering/05-clustering-parameter-search-expanded-v2"

export GC_CLUSTER_GRID_VARIANT="expanded-v2"

if [[ ! -e "$TABLE_DIR/clustering_grid_manifest.csv" ]]; then
    "$PYTHON" "$BASE_CODE/01_prepare_grid_manifest.py"
fi
if [[ ! -e "$TABLE_DIR/gpu_preflight.csv" ]]; then
    "$PYTHON" "$BASE_CODE/02_clustering_gpu_preflight.py"
fi
exec "$PYTHON" "$BASE_CODE/03_run_clustering_grid.py"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/disk18t/lr_xcy/riku/gc_codex
WF="$ROOT/epi-cm-core-workflow"
PY="$ROOT/uv_envs/rapids/.venv/bin/python"
RUN_VERSION=selected-manual-pcs25-nn50-res0p3-v4
export GC_RUN_VERSION="$RUN_VERSION"

QC="$WF/h5ad/01-celltype_integration_clustering/03-qc/adata_qc.h5ad"
CONSISTENT="$WF/h5ad/01-celltype_integration_clustering/07-score-rank-qc/$RUN_VERSION/adata_anno_score_genes_rank_consistent.h5ad"
MANIFEST="$WF/tables/02-cell_subtype_integration_clustering/01-lineage-selection/$RUN_VERSION/lineage_manifest.csv"

cd "$ROOT"

"$PY" "$WF/codes/02-cell_subtype_integration_clustering/01-lineage-selection/01_extract_lineages.py" \
  --qc-h5ad "$QC" \
  --consistent-h5ad "$CONSISTENT"

mapfile -t LINEAGES < <("$PY" - "$MANIFEST" <<'PY'
import sys
import pandas as pd
for value in pd.read_csv(sys.argv[1])["lineage"].astype(str):
    print(value)
PY
)

for lineage in "${LINEAGES[@]}"; do
  "$PY" "$WF/codes/02-cell_subtype_integration_clustering/02-subtype-harmony/02_run_lineage_harmony_gpu.py" \
    --lineage "$lineage" \
    --n-top-genes 3000 \
    --batch-key sample
  "$PY" "$WF/codes/02-cell_subtype_integration_clustering/03-subtype-clustering-grid/03_run_lineage_grid_gpu.py" \
    --lineage "$lineage"
done

"$PY" - "$MANIFEST" "$WF/tables/02-cell_subtype_integration_clustering/03-subtype-clustering-grid/$RUN_VERSION" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd

manifest = pd.read_csv(sys.argv[1])
grid_root = Path(sys.argv[2])
rows = []
for lineage in manifest["lineage"].astype(str):
    payload = json.loads((grid_root / lineage / "clustering_grid_completion.json").read_text())
    rows.append(payload)
summary = pd.DataFrame(rows)
out = grid_root / "all_lineages_grid_completion.csv"
if out.exists():
    old = pd.read_csv(out)
    if not old.equals(summary):
        raise RuntimeError(f"Existing summary differs: {out}")
else:
    summary.to_csv(out, index=False)
if not (
    len(summary) == len(manifest)
    and summary["terminal_accounting_ok"].astype(bool).all()
    and (summary["failed_candidates"].astype(int) == 0).all()
):
    raise RuntimeError("Not all lineage grids completed successfully")
print(summary.to_string(index=False))
print("STOP_AFTER_GRID_FOR_MANUAL_SELECTION")
PY

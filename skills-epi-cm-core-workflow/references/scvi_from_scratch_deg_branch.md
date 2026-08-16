# From-Scratch scVI Integration and DEG Branch

## Contents

1. Activation and scope
2. Input and matrix contracts
3. Ordered workflow
4. scVI defaults
5. Clustering and selection
6. DEG contract
7. Outputs and validation

## Activation and Scope

Use this branch when the user explicitly requests a workflow that starts from
raw single-cell inputs or raw counts, uses scVI for integration, and continues
through DEG calculation. The explicit request makes scVI the primary
integration method for this run; do not require a failed Harmony run first.

This branch is bounded. Its default endpoint is the complete per-cluster DEG
export from the manually selected scVI clustering. Do not continue into broad
biological annotation, score/rank filtering, lineage subtype analysis, CM
discovery, or spatial validation unless the user explicitly requests those
later steps.

## Input and Matrix Contracts

Accept the earliest valid input supplied by the user:

```text
FASTQ -> Cell Ranger count when no count matrix exists
10x/raw feature-barcode matrices -> read and merge
h5ad with demonstrably raw nonnegative integer counts -> validate and reuse
```

Apply the parent skill's sample naming, hierarchical merge, metadata recovery,
gene-identifier audit, QC, per-sample Scrublet, no-downsampling, GPU, seed, and
output-layout contracts. Preserve `sample`, `series`, `status`,
`original_barcode`, and globally unique `obs_names`.

Maintain three distinct representations:

```text
layers["counts"]
  nonnegative integer raw counts; scVI input only

adata.raw
  full-gene normalize_total(target_sum=1e4) followed by log1p expression;
  Scanpy DEG and marker-expression source

obsm["X_scVI"]
  scVI latent representation; neighbors, UMAP, and clustering source only
```

Do not infer raw counts only from a layer name. Audit that the selected count
matrix is finite, nonnegative, and integer-like. If raw counts cannot be
recovered, stop before scVI.

## Ordered Workflow

1. Generate or read raw counts and save the merged pre-QC object as
   `h5ad/02-merge-metadata/adata_merge.h5ad`.
2. Run the existing Module 01 QC contract and save
   `h5ad/03-qc/adata_qc.h5ad` with `layers["counts"]` intact.
3. From the clean QC object, create a full-gene expression copy, run
   `normalize_total(target_sum=1e4)` and `log1p`, and retain it as the future
   `adata.raw` DEG source.
4. On the raw-count object, select HVGs using `layers["counts"]`. Unless the
   user specifies otherwise, use `flavor="seurat_v3"`, `batch_key="sample"`,
   and `n_top_genes=3000`; if fewer genes are available, use all eligible genes
   and record the reason. Never subset cells.
5. Subset the scVI working `.X`/count layer to selected genes, then attach the
   full-gene normalized/log object as `adata.raw`. Verify exact cell-ID and
   order agreement before assignment.
6. Register and train scVI using the defaults below, store `X_scVI`, save the
   model and `adata_scvi.h5ad`, and close the training process after saving.
7. From a clean read of `adata_scvi.h5ad`, run neighbors, UMAP, and Leiden once
   using only `X_scVI`, with `n_neighbors=20` and `resolution=0.8` unless the
   user specifies other parameters. If more than 20 raw clusters are produced,
   discard that mutable run and rerun once from the clean object at
   `resolution=0.4`.
8. Save the accepted direct clustering as `adata_inte.h5ad` with its UMAP and
   initial/final parameter audit. Do not create a clustering candidate grid.
9. Calculate and export full DEGs for every selected raw cluster using
   `adata.raw`, never `X_scVI`.
10. Write the branch validation report and stop unless later annotation or
    downstream blocks were explicitly requested.

Canonical representation-construction pattern:

```python
adata_counts = sc.read_h5ad("h5ad/03-qc/adata_qc.h5ad")
if "counts" not in adata_counts.layers:
    raise ValueError("Recover a validated raw-count layer before scVI.")

adata_log = adata_counts.copy()
adata_log.X = adata_log.layers["counts"].copy()
sc.pp.normalize_total(adata_log, target_sum=1e4)
sc.pp.log1p(adata_log)

sc.pp.highly_variable_genes(
    adata_counts,
    layer="counts",
    flavor="seurat_v3",
    n_top_genes=3000,
    batch_key="sample",
)
adata_scvi = adata_counts[:, adata_counts.var["highly_variable"]].copy()
if not adata_scvi.obs_names.equals(adata_log.obs_names):
    raise ValueError("Cell IDs/order differ between scVI counts and DEG expression.")
adata_scvi.raw = adata_log
```

## scVI Defaults

Use fixed seed `42` throughout. Unless the user specifies other values, use:

```text
batch_key = sample
layer = counts
n_latent = 30
n_layers = 2
gene_likelihood = nb
max_epochs = None
```

Canonical scVI setup, training, latent-export, and model-save pattern:

```python
from pathlib import Path
import random

import numpy as np
import torch
import scvi

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
scvi.settings.seed = SEED

SCVI_H5AD = Path("h5ad/04-integration-scvi/adata_scvi.h5ad")
SCVI_MODEL_DIR = Path("h5ad/04-integration-scvi/scvi_model")
if SCVI_H5AD.exists() or SCVI_MODEL_DIR.exists():
    raise FileExistsError(
        "The scVI h5ad or model path already exists; use a versioned path "
        "or obtain explicit permission before overwriting it."
    )
SCVI_H5AD.parent.mkdir(parents=True, exist_ok=True)

scvi.model.SCVI.setup_anndata(
    adata_scvi,
    layer="counts",
    batch_key="sample",
)
model = scvi.model.SCVI(
    adata_scvi,
    n_latent=30,
    n_layers=2,
    gene_likelihood="nb",
)
model.train(max_epochs=None)

adata_scvi.obsm["X_scVI"] = model.get_latent_representation()
adata_scvi.write_h5ad(SCVI_H5AD)
model.save(SCVI_MODEL_DIR, overwrite=False, save_anndata=False)
```

The object passed to `setup_anndata` is the HVG-subset `adata_scvi`, but its
`layers["counts"]` remains raw integer counts. The full-gene normalized/log
matrix remains in `adata_scvi.raw`; it is not passed to the scVI likelihood.
After saving, verify that the saved model reloads against `SCVI_H5AD`, then
close the training process or release model/GPU memory before continuing.

Record the scvi-tools, PyTorch, CUDA, Python, AnnData, and Scanpy versions,
actual device, elapsed time, training history/convergence information, actual
epoch count, early-stopping behavior, seed, gene count, and cell count. Save:

```text
h5ad/04-integration-scvi/adata_scvi.h5ad
h5ad/04-integration-scvi/scvi_model/
```

If either path already exists, do not overwrite it silently. Use a versioned
path or stop and ask, following the parent skill's preservation rule.

## Direct Clustering

The scVI clustering grid is disabled. Use the full 30-dimensional `X_scVI`
representation and run one direct clustering with `n_neighbors=20` and Leiden
`resolution=0.8` unless the user explicitly supplies other values. If the
initial run has more than 20 raw clusters, discard it, reload the clean
`adata_scvi.h5ad`, and rerun once with `resolution=0.4`, keeping
`n_neighbors=20` and the same full `X_scVI`. Do not recursively halve the
resolution. Record `integration_source=scVI`, `source_h5ad`,
`use_rep=X_scVI`, `latent_n_dims_used=30`, initial/final parameters, seed,
initial/final cluster counts, and output paths. Do not call latent dimensions
`n_pcs`.

Save the accepted direct graph, UMAP, and clustering as:

```text
h5ad/05-clustering-parameter-search/selected/adata_inte.h5ad
tables/05-clustering-parameter-search/selected_clustering.csv
```

Canonical direct-clustering pattern:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

SEED = 42
sc.settings.set_figure_params(vector_friendly=True)
SELECTED_N_NEIGHBORS = 20
SELECTED_RESOLUTION = 0.8

SCVI_H5AD = Path("h5ad/04-integration-scvi/adata_scvi.h5ad")
SELECTED_H5AD = Path(
    "h5ad/05-clustering-parameter-search/selected/adata_inte.h5ad"
)
SELECTED_TABLE = Path(
    "tables/05-clustering-parameter-search/selected_clustering.csv"
)
SELECTED_FIGURE_DIR = Path(
    "figures/05-clustering-parameter-search/selected"
)

def run_from_clean_scvi(resolution):
    adata = sc.read_h5ad(SCVI_H5AD)
    if "X_scVI" not in adata.obsm:
        raise KeyError("X_scVI is missing from the saved scVI object.")
    if adata.raw is None:
        raise ValueError("Full-gene normalized/log adata.raw is required for DEG.")
    if not adata.obs_names.equals(adata.raw.obs_names):
        raise ValueError("adata.raw cell IDs/order do not match the scVI object.")

    latent = np.asarray(adata.obsm["X_scVI"])
    if latent.shape[0] != adata.n_obs or not np.isfinite(latent).all():
        raise ValueError("X_scVI has invalid dimensions or non-finite values.")

    resolution_text = str(resolution).replace(".", "p")
    cluster_key = f"scvi_leiden_res{resolution_text}"
    sc.pp.neighbors(
        adata,
        use_rep="X_scVI",
        n_neighbors=int(SELECTED_N_NEIGHBORS),
        random_state=SEED,
    )
    sc.tl.umap(adata, random_state=SEED)
    sc.tl.leiden(
        adata,
        resolution=float(resolution),
        key_added=cluster_key,
        random_state=SEED,
    )
    return adata, latent, cluster_key, resolution_text


# FIXED: graph, UMAP, and Leiden use X_scVI; start at resolution 0.8.
initial_resolution = float(SELECTED_RESOLUTION)
adata, latent, cluster_key, resolution_text = run_from_clean_scvi(
    initial_resolution
)
initial_cluster_count = int(adata.obs[cluster_key].nunique())

# If there are more than 20 raw clusters, discard the mutable run and rerun
# once from the clean scVI h5ad at half the default resolution.
if initial_cluster_count > 20:
    final_resolution = initial_resolution / 2
    adata, latent, cluster_key, resolution_text = run_from_clean_scvi(
        final_resolution
    )
    adjustment_reason = "initial raw cluster count exceeded 20"
else:
    final_resolution = initial_resolution
    adjustment_reason = "none"
final_cluster_count = int(adata.obs[cluster_key].nunique())

selected_pdf = SELECTED_FIGURE_DIR / f"umap_{cluster_key}.pdf"
selected_svg = SELECTED_FIGURE_DIR / f"umap_{cluster_key}.svg"
for path in (SELECTED_H5AD, SELECTED_TABLE, selected_pdf, selected_svg):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")

SELECTED_TABLE.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame([{
    "source_h5ad": str(SCVI_H5AD),
    "integration_source": "scVI",
    "use_rep": "X_scVI",
    "latent_n_dims_used": int(latent.shape[1]),
    "n_neighbors": int(SELECTED_N_NEIGHBORS),
    "initial_resolution": initial_resolution,
    "initial_cluster_count": initial_cluster_count,
    "final_resolution": final_resolution,
    "final_cluster_count": final_cluster_count,
    "adjustment_reason": adjustment_reason,
    "cluster_key": cluster_key,
    "seed": SEED,
}]).to_csv(SELECTED_TABLE, index=False)

SELECTED_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(6.0, 5.0))
sc.pl.umap(adata, color=cluster_key, ax=ax, show=False, frameon=False)
fig.savefig(selected_pdf, bbox_inches="tight", dpi=300)
fig.savefig(selected_svg, bbox_inches="tight", dpi=300)
plt.close(fig)
```

## DEG Contract

Run Scanpy t-test against the rest for every selected raw cluster:

```python
sc.tl.rank_genes_groups(
    adata,
    groupby=selected_cluster_key,
    reference="rest",
    method="t-test",
    use_raw=True,
    n_genes=adata.raw.n_vars,
    key_added=f"rank_genes_{selected_cluster_key}",
)
```

The default DEG engine is Scanpy on full-gene normalized/log `adata.raw`.
`X_scVI` is not an expression matrix and must never be used for DEG. Do not use
`model.get_normalized_expression()` or `model.differential_expression()` as a
silent replacement. A user-requested Bayesian scVI-DE branch must be separate,
method-labeled, and must not overwrite the Scanpy DEG tables.

Export one full-length CSV per raw cluster. Required columns and order:

```text
gene, score, logfoldchanges, pvals, pvals_adj
```

Do not export only top-N genes and do not prefilter the full DEG tables by
significance. For later marker selection, filter the saved tables with
`pvals_adj < 0.05` and `logfoldchanges > 0`, then order by `score` descending.
If biological annotation is requested later, use these full raw-cluster DEG
tables for annotation and recompute a second DEG round after final labels are
written, following the parent skill.

Save the raw-cluster tables under:

```text
tables/06-broad-annotation/degs_<selected_cluster_key>_latent30_nn<NN>_res<RES>/
<group>_degs_<selected_cluster_key>_latent30_nn<NN>_res<RES>.csv
```

Also save a manifest with one row per cluster containing the input h5ad,
groupby key, reference, method, expression source, number of tested genes,
output CSV, and row count.

Canonical full-DEG export and final selected-h5ad save pattern, continuing from
the selected-parameter code above:

```python
import re

DEG_DIR = Path(
    "tables/06-broad-annotation/"
    f"degs_{cluster_key}_latent{latent.shape[1]}_"
    f"nn{SELECTED_N_NEIGHBORS}_res{resolution_text}"
)
DEG_MANIFEST = DEG_DIR / "deg_manifest.csv"
if DEG_DIR.exists():
    raise FileExistsError(f"Refusing to reuse existing DEG directory: {DEG_DIR}")
DEG_DIR.mkdir(parents=True, exist_ok=False)

# FIXED: DEG uses full-gene normalized/log adata.raw, never X_scVI.
deg_key = f"rank_genes_{cluster_key}"
sc.tl.rank_genes_groups(
    adata,
    groupby=cluster_key,
    reference="rest",
    method="t-test",
    use_raw=True,
    n_genes=adata.raw.n_vars,
    key_added=deg_key,
)

if hasattr(adata.obs[cluster_key].dtype, "categories"):
    groups = [str(x) for x in adata.obs[cluster_key].cat.categories]
else:
    groups = sorted(adata.obs[cluster_key].astype(str).unique())

manifest_rows = []
for group in groups:
    deg = sc.get.rank_genes_groups_df(adata, group=group, key=deg_key)
    deg = deg.rename(columns={"names": "gene", "scores": "score"})
    required = ["gene", "score", "logfoldchanges", "pvals", "pvals_adj"]
    missing = [column for column in required if column not in deg.columns]
    if missing:
        raise ValueError(f"DEG columns missing for cluster {group}: {missing}")
    deg = deg.loc[:, required]

    safe_group = re.sub(r"[^A-Za-z0-9_.-]+", "_", group)
    output_csv = DEG_DIR / f"{safe_group}_degs_{cluster_key}.csv"
    deg.to_csv(output_csv, index=False)
    manifest_rows.append({
        "input_h5ad": str(SCVI_H5AD),
        "group": group,
        "groupby": cluster_key,
        "reference": "rest",
        "method": "t-test",
        "expression_source": "adata.raw: normalize_total_1e4_log1p",
        "use_raw": True,
        "number_of_tested_genes": int(adata.raw.n_vars),
        "row_count": int(deg.shape[0]),
        "output_csv": str(output_csv),
    })

pd.DataFrame(manifest_rows).to_csv(DEG_MANIFEST, index=False)

# Save once, after the selected clustering and DEG result are both attached.
SELECTED_H5AD.parent.mkdir(parents=True, exist_ok=True)
adata.write_h5ad(SELECTED_H5AD)
```

## Outputs and Validation

Required branch outputs:

```text
adata_merge.h5ad
adata_qc.h5ad
adata_scvi.h5ad
scvi_model/
selected adata_inte.h5ad
selected_clustering.csv
one full DEG CSV per selected raw cluster
DEG manifest
scVI parameter/provenance tables
branch_validation.csv
package_versions.txt
```

The branch is complete only when all checks pass:

- Counts used by scVI are finite, nonnegative, and integer-like.
- `adata.raw` is full-gene normalize-total/log1p expression and shares exact
  cell IDs/order with the scVI object.
- `X_scVI` exists, is finite, has one row per cell, and has 30 columns unless a
  user-specified latent size is recorded.
- The saved model reloads against the saved scVI AnnData or its registered
  schema is otherwise validated.
- The initial and any one-time adjusted clustering run use `X_scVI`; neither
  run uses PCA/Harmony by accident.
- The selected cluster key exists in the selected h5ad and every cluster has a
  non-empty full DEG CSV.
- Every DEG table contains the canonical columns and was computed with
  `use_raw=True` from normalized/log expression.
- No cells were downsampled, and all exclusions are explained by the recorded
  QC contract.
- Existing outputs were not silently overwritten.

# Block 01 raw-count merge

`01_merge_raw_counts.py` reads only rows with `used=true`, `modality=scRNA-seq`
and one of the six supported scRNA GEO series from the canonical v4 manifest:

`tables/01-celltype_integration_clustering/01-sample-manifest/all_datasets_all_samples_inclusion_v4.csv`

Supported source layouts are:

- direct per-GSM 10x `matrix/features/barcodes` members in a GEO RAW tar;
- submitter-provided per-GSM gene-by-cell raw integer count tables in a GEO RAW
  tar (the actual GSE150290, GSE167297 and GSE183904 layouts);
- the shared GSE234129 matrix, split by `GSE234129_meta.tsv.gz::sample` using
  the manifest `sample_title` while preserving the GEO GSM as canonical
  `obs['sample']`.

Run only after canonical v4 manifest validation succeeds:

```bash
/mnt/disk18t/lr_xcy/riku/gc_codex/uv_envs/main/.venv/bin/python \
  01_merge_raw_counts.py
```

The final artifact is
`h5ad/01-celltype_integration_clustering/02-merge-metadata/adata_merge.h5ad`.
Genes are first inner-joined among samples within each series, then the six
series-level matrices are inner-joined. Both levels are recorded in
`gene_join_hierarchy_audit.csv`; flat/outer concatenation is not used. `X`
remains non-negative integer raw counts. Cell IDs are exactly
`series|sample|original_barcode`; required metadata are `sample`, `series`,
`status`, `patient`, `tissue_class`, and `original_barcode`.

The run is resumable per sample. A checkpoint is reused only if the SHA256 of
the complete canonical manifest row matches. Any failed used sample is written
to `failed_samples.csv` and stops the final merge. Existing completed output is
never overwritten.

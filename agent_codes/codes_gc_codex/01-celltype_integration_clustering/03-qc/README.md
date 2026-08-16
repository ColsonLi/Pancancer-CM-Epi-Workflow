# Block 01 full-cell QC

`01_run_qc.py` applies basic QC to the full merged count matrix without
downsampling:

- retain cells with at least 200 detected genes;
- retain cells with at most 20% mitochondrial counts (`MT-`, case-insensitive);
- retain genes detected in at least 3 cells (before Scrublet and again after
  doublet removal);
- run Scrublet independently for every `series + sample`, seed 42, automatic
  per-sample threshold, expected doublet rate 0.05 and simulation ratio 2;
- try RAPIDS-SingleCell on the GPU first; only after a logged failure, retry the
  same complete sample with Scanpy/Scrublet CPU;
- if all allowed Scrublet routes fail, exclude that complete sample and continue;
  stop without a final h5ad only when no sample remains.

Run after `adata_merge.h5ad` exists:

```bash
/mnt/disk18t/lr_xcy/riku/gc_codex/uv_envs/main/.venv/bin/python \
  01_run_qc.py
```

Use `--no-allow-cpu-fallback` to require GPU success for every sample. The GPU
worker is process-isolated in `02_scrublet_worker.py`, allowing CUDA memory to
be fully released before an audited fallback. Results are resumable only when
the complete cell-ID hash and every Scrublet parameter match.

The final artifact is
`h5ad/01-celltype_integration_clustering/03-qc/adata_qc.h5ad`; `X` is still raw
integer counts and has no normalized layer. Required obs QC columns are
`total_counts`, `n_genes_by_counts`, `total_counts_MT`, `pct_counts_MT`,
`total_counts_RIBO`, `pct_counts_RIBO`, `doublet_score`, and
`predicted_doublet`. Audit outputs include seven numbered tables, complete
parameters, `qc_report.txt`, `gene_identifier_audit.csv`, package versions,
per-cell metrics, worker logs, hashes, and validation. Figures are PDF/SVG only.

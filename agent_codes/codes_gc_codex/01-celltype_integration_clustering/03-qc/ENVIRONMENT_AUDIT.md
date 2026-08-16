# Environment/API audit (2026-08-03)

The implementation was checked against the existing local environments before
writing the runner.

Main environment (`uv_envs/main/.venv`): Python 3.12.13, anndata 0.12.19,
Scanpy 1.11.5, NumPy 2.4.6, pandas 2.3.3, SciPy 1.18.0 and h5py 3.16.0.
`scanpy.pp.scrublet` is available and accepts `sim_doublet_ratio`,
`expected_doublet_rate`, `n_prin_comps`, `use_approx_neighbors`, and
`random_state`.

RAPIDS environment (`uv_envs/rapids/.venv`): Python 3.12.13, anndata 0.13.2,
Scanpy 1.12.3, RAPIDS-SingleCell 0.14.1, CuPy 14.1.1, RMM 25.12.00 and cuML
25.12.00. `rapids_singlecell.pp.scrublet` exposes the corresponding API, and
`rapids_singlecell.get.anndata_to_GPU/CPU` are available.

The local CuPy sparse implementation accepts floating-point rather than
integer sparse arrays. The GPU worker therefore converts only its isolated raw
count working copy from int32 to float32 immediately before transfer. It does
not normalize values and cannot mutate `adata_merge.h5ad` or the final integer
count matrix.

The main environment reports a Numba warning that its installed TBB interface
is older than the preferred threading-layer interface. A synthetic CPU
Scrublet check wrote valid complete results but could hang during interpreter
teardown. Because each worker is isolated and writes both outputs atomically,
the worker flushes streams and explicitly exits only after successful output
completion. Errors still propagate normally before that point.

CUDA inspection found one NVIDIA GeForce RTX 2080 Ti with 22,528 MiB reported
memory. This file records only the implementation-time API audit; every formal
run writes its own package/backend/device audit alongside the results.

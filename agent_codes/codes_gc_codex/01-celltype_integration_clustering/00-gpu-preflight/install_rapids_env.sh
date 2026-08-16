#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
python_path="${project_root}/uv_envs/rapids/.venv/bin/python"
cache_dir="${project_root}/uv_cache"

if [[ ! -x "${python_path}" ]]; then
  uv venv --python 3.12 "${project_root}/uv_envs/rapids/.venv"
fi

# rapids-singlecell 0.16.1 is source-only on PyPI and requires a local CUDA
# extension build. The host toolkit is CUDA 11.5, so use the last upstream
# pure-Python wheel and pin its tested-era RAPIDS stack to CUDA 12 wheels.
UV_CACHE_DIR="${cache_dir}" uv pip install \
  --python "${python_path}" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://pypi.nvidia.com \
  --only-binary :all: \
  --no-binary docrep \
  'rapids-singlecell[rapids12]==0.14.1' \
  'cudf-cu12==25.12.0' \
  'cugraph-cu12==25.12.2' \
  'cuml-cu12==25.12.0' \
  'cuvs-cu12==25.12.0' \
  'librmm-cu12==25.12.0'

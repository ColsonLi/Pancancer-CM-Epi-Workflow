#!/usr/bin/env bash
set -euo pipefail

# Resumable, non-destructive receipt of the GEO files that were replaced by
# NCBI "Access Denied" HTML pages in the user-supplied input directory.
project_root="/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow"
receipt_dir="${project_root}/inputs/geo-official-20260806"
log_dir="${receipt_dir}/logs"
mkdir -p "${receipt_dir}" "${log_dir}"

declare -A expected_bytes=(
  [GSE278688_sc_matrix.mtx.gz]=1093755688
  [GSE278688_sc_features.tsv.gz]=221703
  [GSE278688_sc_barcodes.tsv.gz]=911362
  [GSE278688_sc_metadata.csv.gz]=1166884
  [GSE278687_RAW.tar]=1117061120
)

declare -A urls=(
  [GSE278688_sc_matrix.mtx.gz]="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278688/suppl/GSE278688_sc_matrix.mtx.gz"
  [GSE278688_sc_features.tsv.gz]="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278688/suppl/GSE278688_sc_features.tsv.gz"
  [GSE278688_sc_barcodes.tsv.gz]="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278688/suppl/GSE278688_sc_barcodes.tsv.gz"
  [GSE278688_sc_metadata.csv.gz]="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278688/suppl/GSE278688_sc_metadata.csv.gz"
  [GSE278687_RAW.tar]="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278687/suppl/GSE278687_RAW.tar"
)

download_one() {
  local name="$1"
  local final="${receipt_dir}/${name}"
  local part="${final}.part"
  local log="${log_dir}/${name}.curl.log"
  local expected="${expected_bytes[$name]}"

  if [[ -f "${final}" ]] && [[ "$(stat -c '%s' "${final}")" == "${expected}" ]]; then
    printf '%s\tSKIP_COMPLETE\t%s\t%s\n' "$(date --iso-8601=seconds)" "${name}" "${expected}" >>"${log}"
    return 0
  fi

  if [[ -f "${final}" ]]; then
    printf '%s\tERROR_BAD_FINAL_SIZE\t%s\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "${name}" "$(stat -c '%s' "${final}")" "${expected}" >>"${log}"
    return 1
  fi

  printf '%s\tSTART_OR_RESUME\t%s\t%s\n' "$(date --iso-8601=seconds)" "${name}" "${expected}" >>"${log}"
  curl --fail --location --show-error \
    --retry 10 --retry-all-errors --retry-delay 5 --connect-timeout 30 \
    --continue-at - --output "${part}" "${urls[$name]}" >>"${log}" 2>&1

  local observed
  observed="$(stat -c '%s' "${part}")"
  if [[ "${observed}" != "${expected}" ]]; then
    printf '%s\tERROR_SIZE_MISMATCH\t%s\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "${name}" "${observed}" "${expected}" >>"${log}"
    return 1
  fi
  mv "${part}" "${final}"
  printf '%s\tRECEIVED\t%s\t%s\n' "$(date --iso-8601=seconds)" "${name}" "${observed}" >>"${log}"
}

names=(
  GSE278688_sc_matrix.mtx.gz
  GSE278688_sc_features.tsv.gz
  GSE278688_sc_barcodes.tsv.gz
  GSE278688_sc_metadata.csv.gz
  GSE278687_RAW.tar
)

pids=()
for name in "${names[@]}"; do
  download_one "${name}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"

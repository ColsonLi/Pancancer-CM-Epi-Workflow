#!/usr/bin/env python3
"""Run section-owned GC all-pair plot workers with bounded concurrency."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TASK = "01-tangram-pseudobulk-v3-final-v5"
SCOPE = ROOT / f"tables/04-spatial-validation-optional/{TASK}/01-reference-and-manifest/spatial_sample_scope_manifest.csv"
WORKER = ROOT / f"codes/04-spatial-validation-optional/{TASK}/04_plot_all_pairs_for_one_section.py"
LOG_DIR = ROOT / f"tables/04-spatial-validation-optional/{TASK}/03-all-pair-statistics-and-plots/plot_logs_by_sample"
STATUS = ROOT / f"tables/04-spatial-validation-optional/{TASK}/03-all-pair-statistics-and-plots/plot_worker_status.csv"
MANIFEST_DIR = ROOT / f"tables/04-spatial-validation-optional/{TASK}/03-all-pair-statistics-and-plots/plot_manifests_by_sample"
MAX_WORKERS = 16


def is_complete(sample):
    """Avoid rewriting a sample whose full 132-pair figure set is already present."""
    manifest = MANIFEST_DIR / f"{sample}_plot_manifest.csv"
    if not manifest.exists():
        return False
    table = pd.read_csv(manifest)
    path_cols = ["raw_pdf", "raw_svg", "percentile_pdf", "percentile_svg"]
    return (
        len(table) == 132
        and set(path_cols).issubset(table.columns)
        and table[path_cols].map(lambda path: Path(path).is_file()).all().all()
    )


def run(sample):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{sample}.log"
    with log.open("a", encoding="utf-8") as fh:
        proc = subprocess.run([sys.executable, str(WORKER), "--sample", sample], stdout=fh, stderr=subprocess.STDOUT)
    return sample, proc.returncode, str(log)


def main():
    samples = pd.read_csv(SCOPE)["sample"].astype(str).tolist()
    completed = [sample for sample in samples if is_complete(sample)]
    pending = [sample for sample in samples if sample not in completed]
    rows = [
        {"sample": sample, "returncode": 0, "status": "already_completed", "log": ""}
        for sample in completed
    ]
    pd.DataFrame(rows).to_csv(STATUS, index=False)
    print(json.dumps({"already_completed": completed, "pending": len(pending), "max_workers": MAX_WORKERS}), flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run, sample): sample for sample in pending}
        for future in as_completed(futures):
            sample, rc, log = future.result()
            rows.append({"sample": sample, "returncode": rc, "status": "completed" if rc == 0 else "failed", "log": log})
            pd.DataFrame(rows).to_csv(STATUS, index=False)
            print(json.dumps(rows[-1]), flush=True)
            if rc != 0:
                raise RuntimeError(f"Plot worker failed: {sample}; see {log}")


if __name__ == "__main__":
    main()


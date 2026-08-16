#!/usr/bin/env python3
"""Complete the slow GEO matrix transfer with validated parallel byte ranges."""

from __future__ import annotations

import concurrent.futures
import random
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278688/suppl/GSE278688_sc_matrix.mtx.gz"
EXPECTED_BYTES = 1_093_755_688
WORK_DIR = Path(
    "/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow/inputs/geo-official-20260806"
)
FINAL = WORK_DIR / "GSE278688_sc_matrix.mtx.gz"
PREFIX = WORK_DIR / "GSE278688_sc_matrix.mtx.gz.part"
ASSEMBLED = WORK_DIR / "GSE278688_sc_matrix.mtx.gz.multipart.part"
N_RANGES = 4


def split_ranges(start: int, stop: int, n_ranges: int) -> list[tuple[int, int]]:
    total = stop - start + 1
    base, remainder = divmod(total, n_ranges)
    ranges = []
    cursor = start
    for index in range(n_ranges):
        length = base + (1 if index < remainder else 0)
        ranges.append((cursor, cursor + length - 1))
        cursor += length
    if cursor != stop + 1:
        raise AssertionError("Range splitting failed")
    return ranges


def fetch_range(start: int, stop: int) -> Path:
    destination = WORK_DIR / f"GSE278688_sc_matrix.mtx.gz.range-{start}-{stop}.part"
    expected = stop - start + 1
    for attempt in range(1, 11):
        have = destination.stat().st_size if destination.exists() else 0
        if have == expected:
            return destination
        if have > expected:
            raise ValueError(f"Oversized range file: {destination}")
        request_start = start + have
        request = urllib.request.Request(
            URL,
            headers={
                "Range": f"bytes={request_start}-{stop}",
                "User-Agent": "pdac-epi-cm-input-receipt/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206:
                    raise OSError(f"Expected HTTP 206, received {response.status}")
                content_range = response.headers.get("Content-Range", "")
                expected_header = f"bytes {request_start}-{stop}/{EXPECTED_BYTES}"
                if content_range != expected_header:
                    raise OSError(
                        f"Unexpected Content-Range {content_range!r}; expected {expected_header!r}"
                    )
                with destination.open("ab") as handle:
                    while block := response.read(8 * 1024 * 1024):
                        handle.write(block)
        except Exception:
            if attempt == 10:
                raise
            time.sleep(min(5 * attempt, 30))
            continue
    if destination.stat().st_size != expected:
        raise OSError(f"Incomplete range: {destination}")
    return destination


def main() -> None:
    if FINAL.exists():
        if FINAL.stat().st_size == EXPECTED_BYTES:
            print(f"already complete: {FINAL}")
            return
        raise FileExistsError(f"Refusing to overwrite wrong-size final file: {FINAL}")
    if ASSEMBLED.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete assembly: {ASSEMBLED}")
    if not PREFIX.is_file():
        raise FileNotFoundError(PREFIX)
    prefix_size = PREFIX.stat().st_size
    time.sleep(2)
    if PREFIX.stat().st_size != prefix_size:
        raise RuntimeError("Prefix file is still changing; stop the single-connection writer first")
    if not 0 < prefix_size < EXPECTED_BYTES:
        raise ValueError(f"Unexpected prefix size: {prefix_size}")

    ranges = split_ranges(prefix_size, EXPECTED_BYTES - 1, N_RANGES)
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_RANGES) as executor:
        futures = [executor.submit(fetch_range, start, stop) for start, stop in ranges]
        range_paths = [future.result() for future in futures]

    with ASSEMBLED.open("wb") as target, PREFIX.open("rb") as source:
        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
        for path in range_paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    if ASSEMBLED.stat().st_size != EXPECTED_BYTES:
        raise OSError(
            f"Assembled size {ASSEMBLED.stat().st_size} != expected {EXPECTED_BYTES}"
        )
    ASSEMBLED.replace(FINAL)
    PREFIX.unlink()
    for path in range_paths:
        path.unlink()
    print(f"completed multipart receipt: {FINAL} ({FINAL.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fast dependency-light contract tests; does not touch analysis outputs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from block02_common import (
    EXPECTED_CANDIDATES, NN_VALUES, PCS_VALUES, RES_VALUES, assert_grid_contract, candidate_id,
    cell_id_digest, deterministic_unique_tokens, marker_family, partial_h5ad_path,
    prefix_for, save_candidate_once, terminal_metadata, weighted_cluster_mixing,
)


def main() -> None:
    assert_grid_contract(); assert EXPECTED_CANDIDATES == 810
    assert len(PCS_VALUES) == 9 and len(NN_VALUES) == 9 and len(RES_VALUES) == 10
    assert candidate_id(10, 50, 0.1) == "pcs10_nn50_res0p1"
    assert marker_family("Epithelial Cells") == "epithelial"
    assert marker_family("T/NK Cells") == "t_nk"
    assert marker_family("Mast Cells") == "mast"
    assert prefix_for("Mast Cells", "mast_cells") == "Mast"
    assert prefix_for("Myeloid Cells", "myeloid_cells") == "Mye"
    assert cell_id_digest(["a", "b"]) != cell_id_digest(["b", "a"])
    resolved = deterministic_unique_tokens([
        ("lineage_a", "Mye_GENE"), ("lineage_b", "Mye_GENE"), ("lineage_c", "Epi_GENE")
    ])
    assert resolved["lineage_c"] == "Epi_GENE"
    assert resolved["lineage_a"] != resolved["lineage_b"]
    assert len(set(resolved.values())) == 3
    assert partial_h5ad_path(Path("adata.h5ad")).name == ".adata.partial.h5ad"
    labels = np.array([0, 0, 1, 1]); groups = np.array(["a", "b", "a", "b"])
    assert np.isclose(weighted_cluster_mixing(labels, groups), 1.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "terminal.npz"; meta = {"status": "completed", "n_cells": 4}
        save_candidate_once(path, labels, meta); assert terminal_metadata(path) == meta
        try: save_candidate_once(path, labels, meta)
        except FileExistsError: pass
        else: raise AssertionError("immutable terminal was overwritten")
    print("Block02 selftest: PASS")


if __name__ == "__main__": main()

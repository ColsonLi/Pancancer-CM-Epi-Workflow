#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parents[4]
TABLE = ROOT / "epi-cm-core-workflow/tables/01-celltype_integration_clustering/04-integration-harmony"


def write_new(name: str, frame: pd.DataFrame) -> None:
    path = TABLE / name
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False)


def main() -> None:
    attempts = pd.read_csv(TABLE / "02_harmony_attempts.csv")
    params = pd.read_csv(TABLE / "integration_parameters.csv")
    if attempts.empty or params.shape[0] != 1:
        raise RuntimeError("Harmony attempts/parameters are incomplete")
    accepted = attempts.loc[attempts["accepted"].astype(bool)]
    if accepted.shape[0] != 1:
        raise RuntimeError("Expected exactly one accepted Harmony attempt")
    out = attempts.copy()
    out["key"] = str(params.iloc[0]["harmony_key"])
    out["basis"] = str(params.iloc[0]["harmony_basis"])
    out["adjusted_basis"] = str(params.iloc[0]["harmony_adjusted_basis"])
    out["correction_method"] = "original"
    out["use_gemm"] = False
    out["random_seed"] = SEED
    out["input_reset"] = np.where(out["attempt"].eq(1), "initial_clean_X_pca", "restored_clean_X_pca")
    out["final_backend"] = "rapids-singlecell GPU"
    write_new("08_harmony_integrate_attempt_provenance.csv", out)
    write_new("integration_harmony_provenance_addendum.csv", pd.DataFrame([{
        "issue": "08_harmony_integrate_parameters.csv summarizes only the accepted max_iter",
        "authoritative_attempt_table": str(TABLE / "08_harmony_integrate_attempt_provenance.csv"),
        "observed_route": "attempt1 max_iter=10 did_not_converge; attempt2 restored clean X_pca max_iter=30 accepted",
        "method_unchanged": True,
        "cpu_fallback": False,
        "random_seed": SEED,
    }]))
    print("Harmony attempt provenance addendum written")


if __name__ == "__main__":
    main()

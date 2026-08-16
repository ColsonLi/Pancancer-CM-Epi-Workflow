#!/usr/bin/env python3
"""All-pair status-specific Spearman GC Epi-CM associations."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


CODE_DIR = Path(__file__).resolve().parent
WF = CODE_DIR.parents[2]
BLOCK = "03-epi-cm-discovery"
PREFLIGHT_TAG = "agent-led-current-selection-v5-preflight-r2"
NMF_TAG = "agent-led-current-selection-v5-nmf-r1"
TAG = "agent-led-current-selection-v5-cm-analysis-r1"
SEED = 42
PREFLIGHT = WF / "tables" / BLOCK / "01-cm-lineage-analysis" / "01_prepare_inputs_and_frequency_tables" / PREFLIGHT_TAG
NMF = WF / "tables" / BLOCK / "01-cm-lineage-analysis" / "02_balanced_joint_nmf" / NMF_TAG
OUT = WF / "tables" / BLOCK / "01-cm-lineage-analysis" / "04_epi_cm_association_spearman" / TAG


def write_csv_once(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    frame.to_csv(partial, index=index)
    os.replace(partial, path)


def write_text_once(value: str, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(value)
    os.replace(partial, path)


def star(q: float) -> str:
    if not np.isfinite(q):
        return "ns"
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "ns"


def main() -> None:
    if (OUT / "spearman_completion.json").exists():
        raise FileExistsError(OUT / "spearman_completion.json")
    epi = pd.read_csv(PREFLIGHT / "epi_subtype_frequency.csv").set_index("sample")
    activity = pd.read_csv(NMF / "activity_df_sample_by_CM.csv").set_index("sample")
    classification = pd.read_csv(NMF / "joint_module_classification.csv").sort_values("global_order")
    cm_order = classification.CM.astype(str).tolist()
    common = epi.index.intersection(activity.index, sort=False)
    epi = epi.loc[common].astype(float)
    activity = activity.loc[common]
    if epi.shape[1] != 12 or len(cm_order) != 11:
        raise RuntimeError(f"Unexpected association dimensions: epi={epi.shape[1]}, CM={len(cm_order)}")
    if not np.allclose(epi.sum(axis=1), 1.0):
        raise RuntimeError("Epithelial frequency rows do not sum to one")

    all_pairs = []
    scatter_rows = []
    context_records = []
    for context in ("normal-like", "tumor"):
        samples = activity.index[activity.status.astype(str).eq(context)]
        E = epi.loc[samples]
        C = activity.loc[samples, cm_order].astype(float)
        pair_rows = []
        for epithelial_subtype in E.columns:
            for cm in cm_order:
                x = E[epithelial_subtype]
                y = C[cm]
                valid = x.notna() & y.notna()
                if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
                    rho = p_value = np.nan
                else:
                    rho, p_value = spearmanr(x[valid], y[valid])
                pair_rows.append({
                    "context": context, "epithelial_subtype": epithelial_subtype,
                    "CM": cm, "rho": float(rho), "p": float(p_value),
                    "n_samples": int(valid.sum()),
                })
        pairs = pd.DataFrame(pair_rows)
        p = pairs.p.to_numpy(dtype=float)
        valid_p = np.isfinite(p)
        q = np.full(len(p), np.nan)
        q[valid_p] = multipletests(p[valid_p], method="fdr_bh")[1]
        pairs["q"] = q
        pairs["significance"] = pairs.q.map(star)
        pairs["method"] = "spearman"
        all_pairs.append(pairs)

        rho_matrix = pairs.pivot(index="epithelial_subtype", columns="CM", values="rho").reindex(index=epi.columns, columns=cm_order)
        p_matrix = pairs.pivot(index="epithelial_subtype", columns="CM", values="p").reindex(index=epi.columns, columns=cm_order)
        q_matrix = pairs.pivot(index="epithelial_subtype", columns="CM", values="q").reindex(index=epi.columns, columns=cm_order)
        star_matrix = pairs.pivot(index="epithelial_subtype", columns="CM", values="significance").reindex(index=epi.columns, columns=cm_order)
        stem = f"balanced_joint_cm_epi_cm_association_{context}"
        write_csv_once(rho_matrix.reset_index(), OUT / f"{stem}_rho_matrix.csv")
        write_csv_once(p_matrix.reset_index(), OUT / f"{stem}_p_matrix.csv")
        write_csv_once(q_matrix.reset_index(), OUT / f"{stem}_q_matrix.csv")
        write_csv_once(star_matrix.reset_index(), OUT / f"{stem}_q_significance_matrix.csv")
        write_csv_once(pairs, OUT / f"{stem}_all_pairs.csv")

        pair_lookup = pairs.set_index(["epithelial_subtype", "CM"])
        for epithelial_subtype in E.columns:
            for cm in cm_order:
                result = pair_lookup.loc[(epithelial_subtype, cm)]
                for sample in samples:
                    scatter_rows.append({
                        "context": context, "method": "spearman", "sample": sample,
                        "epithelial_subtype": epithelial_subtype, "CM": cm,
                        "epi_frequency": float(E.loc[sample, epithelial_subtype]),
                        "cm_activity": float(C.loc[sample, cm]),
                        "rho": float(result.rho), "p": float(result.p),
                        "q": float(result.q), "significance": result.significance,
                    })
        context_records.append({
            "context": context, "n_samples": len(samples), "n_epi_subtypes": E.shape[1],
            "n_cms": len(cm_order), "n_pairs": len(pairs), "n_valid_pairs": int(valid_p.sum()),
            "n_q_lt_0p05": int((pairs.q < 0.05).sum()),
        })

    combined = pd.concat(all_pairs, ignore_index=True)
    scatter = pd.DataFrame(scatter_rows)
    write_csv_once(combined, OUT / "balanced_joint_cm_epi_cm_association_spearman_all_contexts_all_pairs.csv")
    write_csv_once(scatter, OUT / "balanced_joint_cm_epi_cm_association_spearman_all_pair_scatter_source.csv")
    write_csv_once(pd.DataFrame(context_records), OUT / "spearman_context_pair_inventory.csv")
    write_csv_once(pd.DataFrame([{
        "step": "epi_cm_association_spearman", "epi_input": str(PREFLIGHT / "epi_subtype_frequency.csv"),
        "cm_activity_input": str(NMF / "activity_df_sample_by_CM.csv"),
        "method": "spearman", "min_epi_cells_per_sample": 1,
        "fdr_method": "BH", "fdr_family": "all epithelial_subtype x CM pairs within each status context",
        "q_star_thresholds": "q<0.001:***;q<0.01:**;q<0.05:*;else:ns",
        "n_common_samples": len(common), "n_epi_subtypes": epi.shape[1], "n_cms": len(cm_order),
        "random_seed": SEED, "code_file": str(Path(__file__).resolve()),
    }]), OUT / "spearman_parameters.csv")
    result = {
        "method": "spearman", "n_common_epi_eligible_samples": len(common),
        "n_epi_subtypes": epi.shape[1], "n_cms": len(cm_order),
        "n_pairs_per_context": epi.shape[1] * len(cm_order),
        "contexts": context_records, "scatter_source_rows": len(scatter),
    }
    write_text_once(json.dumps(result, indent=2) + "\n", OUT / "spearman_completion.json")
    write_text_once(
        "GC all-pair status-specific Epi-CM Spearman associations. BH correction is applied across all 12 x 11 pairs within each status.\n",
        OUT / "readme.txt",
    )
    write_text_once(
        f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\n"
        f"scipy spearmanr; statsmodels BH\ncode={Path(__file__).resolve()}\nseed={SEED}\n",
        OUT / "package_versions.txt",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

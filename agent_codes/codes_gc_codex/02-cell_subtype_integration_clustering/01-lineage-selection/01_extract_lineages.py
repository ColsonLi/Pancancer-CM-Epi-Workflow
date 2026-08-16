#!/usr/bin/env python3
"""Extract all actual broad lineages: consistent IDs, QC counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block02_common import (  # noqa: E402
    BROAD_COLUMN_CANDIDATES, CANONICAL_MARKERS, EXPECTED_CANDIDATES, MODULE,
    WORKFLOW_ROOT, atomic_write_h5ad, cell_id_digest, deterministic_unique_tokens,
    marker_family, prefix_for, slugify,
    task_root,
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--qc-h5ad", type=Path)
    p.add_argument("--consistent-h5ad", type=Path)
    return p.parse_args()


def discover(explicit: Path | None, patterns: list[str], name: str) -> Path:
    if explicit:
        if not explicit.exists(): raise FileNotFoundError(explicit)
        return explicit.resolve()
    hits: list[Path] = []
    for pattern in patterns:
        hits.extend(WORKFLOW_ROOT.glob(pattern))
    hits = sorted({p.resolve() for p in hits if p.is_file()})
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one {name}; found {len(hits)}: {hits}. Pass it explicitly.")
    return hits[0]


def count_matrix(qc: ad.AnnData):
    source = "layers[counts]" if "counts" in qc.layers else "X"
    x = qc.layers["counts"] if "counts" in qc.layers else qc.X
    sample = x.data[:100000] if sparse.issparse(x) else np.asarray(x).ravel()[:100000]
    if sample.size and (np.nanmin(sample) < 0 or not np.allclose(sample, np.rint(sample), atol=1e-5)):
        raise RuntimeError(f"QC {source} is not a nonnegative count matrix.")
    return x.copy(), source


def validate_preflight(table_dir: Path, manifest: pd.DataFrame) -> None:
    csv_path = table_dir / "lineage_preflight.csv"
    json_path = table_dir / "lineage_preflight.json"
    if not csv_path.is_file() or not json_path.is_file():
        raise RuntimeError("Lineage manifest exists without completed preflight")
    frame = pd.read_csv(csv_path)
    if set(frame["lineage"].astype(str)) != set(manifest["lineage"].astype(str)):
        raise RuntimeError("Lineage preflight keys differ from lineage manifest")
    payload = json.loads(json_path.read_text())
    if not bool(payload.get("all_lineages_passed", False)) or not frame["preflight_pass"].astype(bool).all():
        failed = frame.loc[~frame["preflight_pass"].astype(bool), ["lineage", "failure_reason"]]
        raise RuntimeError(f"Block02 lineage preflight failed:\n{failed.to_string(index=False)}")


def main() -> None:
    a = args()
    qc_path = discover(a.qc_h5ad, ["h5ad/01-celltype_integration_clustering/**/adata_qc.h5ad"], "QC h5ad")
    consistent_path = discover(
        a.consistent_h5ad,
        ["h5ad/01-celltype_integration_clustering/**/*score*rank*consistent*.h5ad"],
        "score/rank-consistent h5ad",
    )
    table_dir = task_root("tables", "01-lineage-selection")
    manifest_path = table_dir / "lineage_manifest.csv"
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        if all(Path(p).exists() for p in old["lineage_qc_h5ad"]):
            validate_preflight(table_dir, old)
            print(f"validated_existing={manifest_path}"); return
        raise RuntimeError(f"Existing manifest is incomplete; refusing overwrite: {manifest_path}")

    consistent = ad.read_h5ad(consistent_path, backed="r")
    broad_col = next((c for c in BROAD_COLUMN_CANDIDATES if c in consistent.obs), None)
    if broad_col is None: raise KeyError(f"No broad lineage column among {BROAD_COLUMN_CANDIDATES}")
    labels = consistent.obs[broad_col].astype(str)
    if labels.isna().any() or labels.isin(["", "nan", "NA"]).any():
        raise RuntimeError(f"Missing broad labels in obs[{broad_col!r}]")
    if "score_rank_consistent" in consistent.obs and not consistent.obs["score_rank_consistent"].astype(bool).all():
        raise RuntimeError("The supplied consistent atlas contains score_rank_consistent=False cells.")
    ids = consistent.obs_names.copy()
    transfer = consistent.obs.copy()
    consistent.file.close()

    qc = ad.read_h5ad(qc_path)
    if not qc.obs_names.is_unique or not ids.is_unique: raise RuntimeError("obs_names must be unique")
    missing = ids.difference(qc.obs_names)
    if len(missing): raise RuntimeError(f"{len(missing)} consistent IDs are absent from QC")
    rows = []
    used_slugs: set[str] = set()
    for label in sorted(labels.unique()):
        selected = ids[labels.to_numpy() == label]
        slug = slugify(label)
        if slug in used_slugs: slug = f"{slug}_{len(used_slugs) + 1}"
        used_slugs.add(slug)
        out = task_root("h5ad", "01-lineage-selection") / slug / f"adata_{slug}_qc_counts.h5ad"
        ids_csv = table_dir / slug / "selected_cell_ids.csv"
        if out.exists() or ids_csv.exists():
            raise FileExistsError(f"Partial pre-existing lineage output; refusing overwrite: {out} / {ids_csv}")
        sub = qc[selected].copy()
        sub.X, matrix_source = count_matrix(sub)
        sub.layers.clear(); sub.raw = None
        # Annotation fields come from the consistent object; expression counts
        # still come exclusively from the QC object.
        for col in transfer.columns:
            sub.obs[col] = transfer.loc[sub.obs_names, col].values
        sub.uns.clear(); sub.obsm.clear(); sub.obsp.clear(); sub.varm.clear(); sub.varp.clear()
        sub.obs["block02_broad_label"] = label
        sub.obs["block02_count_source"] = matrix_source
        out.parent.mkdir(parents=True, exist_ok=True); ids_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"cell_id": selected}).to_csv(ids_csv, index=False)
        atomic_write_h5ad(sub, out)
        rows.append({
            "lineage": slug, "broad_label": label, "prefix_base": prefix_for(label, slug),
            "marker_family": marker_family(label), "n_cells": sub.n_obs, "n_genes": sub.n_vars,
            "cell_id_sha256": cell_id_digest(sub.obs_names), "broad_column": broad_col,
            "qc_h5ad": str(qc_path), "consistent_h5ad": str(consistent_path),
            "count_source": matrix_source, "lineage_qc_h5ad": str(out), "status": "completed",
            "qc_source_read_only": True,
        })
    del qc
    manifest = pd.DataFrame(rows).sort_values("lineage")
    if int(manifest.n_cells.sum()) != len(ids): raise AssertionError("Lineage partition is not exhaustive")
    resolved_prefix = deterministic_unique_tokens(
        manifest[["lineage", "prefix_base"]].itertuples(index=False, name=None)
    )
    manifest["prefix"] = manifest["lineage"].map(resolved_prefix)
    manifest["cell_abbrev"] = manifest["prefix"].map(slugify)
    if manifest["prefix"].duplicated().any() or manifest["cell_abbrev"].duplicated().any():
        raise AssertionError("Resolved lineage prefixes/cell abbreviations are not globally unique")
    prefix_audit = manifest[["lineage", "broad_label", "prefix_base", "prefix", "cell_abbrev"]].copy()
    prefix_audit["conflict_resolved"] = prefix_audit["prefix_base"] != prefix_audit["prefix"]
    prefix_audit["resolution_rule"] = np.where(
        prefix_audit["conflict_resolved"], "append_double_underscore_lineage_slug", "retain_unique_base"
    )

    max_pc = 50
    preflight = manifest[["lineage", "broad_label", "marker_family", "n_cells", "n_genes"]].copy()
    preflight["max_user_grid_pc"] = max_pc
    preflight["max_pca_components_supported"] = np.minimum(
        np.minimum(preflight["n_cells"].astype(int) - 1, preflight["n_genes"].astype(int) - 1),
        3000,
    )
    preflight["grid_pc50_feasible"] = preflight["max_pca_components_supported"] >= max_pc
    preflight["marker_family_supported"] = preflight["marker_family"].fillna("").isin(CANONICAL_MARKERS)
    preflight["expected_grid_candidates"] = EXPECTED_CANDIDATES
    preflight["preflight_pass"] = preflight["grid_pc50_feasible"] & preflight["marker_family_supported"]
    preflight["failure_reason"] = ""
    preflight.loc[~preflight["grid_pc50_feasible"], "failure_reason"] += "cannot_support_pc50;"
    preflight.loc[~preflight["marker_family_supported"], "failure_reason"] += "unsupported_marker_family;"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    prefix_audit.to_csv(table_dir / "prefix_uniqueness_audit.csv", index=False)
    preflight.to_csv(table_dir / "lineage_preflight.csv", index=False)
    (table_dir / "lineage_preflight.json").write_text(json.dumps({
        "status": "passed" if bool(preflight["preflight_pass"].all()) else "failed",
        "all_lineages_passed": bool(preflight["preflight_pass"].all()),
        "n_lineages": int(len(preflight)),
        "max_user_grid_pc": max_pc,
        "expected_candidates_per_lineage": EXPECTED_CANDIDATES,
        "failure_rule": "stop before any lineage Harmony/grid; never shrink the user grid",
    }, indent=2) + "\n")
    validate_preflight(table_dir, manifest)
    print(manifest[["lineage", "broad_label", "n_cells"]].to_string(index=False))


if __name__ == "__main__": main()

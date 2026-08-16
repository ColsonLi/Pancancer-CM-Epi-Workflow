#!/usr/bin/env python3
"""Audit all direct lineage clustering, full DEG, and candidate annotation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import scanpy as sc


PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
LINEAGES = ["epithelial", "t_nk", "myeloid", "b", "endothelial", "stromal", "mast"]


def main() -> None:
    cluster_root = (
        PROJECT_ROOT / "tables/02-cell_subtype_integration_clustering/03-subtype-clustering"
    )
    deg_root = (
        PROJECT_ROOT / "tables/02-cell_subtype_integration_clustering/05-subtype-deg-annotation"
    )
    output_csv = cluster_root / "module02_direct_clustering_deg_summary.csv"
    output_json = cluster_root / "module02_direct_clustering_deg_validation.json"
    for path in [output_csv, output_json]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite audit output: {path}")

    rows = []
    errors = []
    for slug in LINEAGES:
        parameter_path = cluster_root / slug / "direct_clustering_parameters.csv"
        validation_path = cluster_root / slug / "direct_clustering_validation.json"
        deg_validation_path = deg_root / slug / "raw_cluster_deg_validation.json"
        deg_manifest_path = deg_root / slug / "raw_cluster_deg_file_manifest.csv"
        candidate_path = deg_root / slug / "candidate_subtype_annotation.csv"
        required = [
            parameter_path,
            validation_path,
            deg_validation_path,
            deg_manifest_path,
            candidate_path,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            errors.extend(f"missing:{path}" for path in missing)
            continue

        parameters = pd.read_csv(parameter_path).iloc[0]
        clustering_validation = json.loads(validation_path.read_text(encoding="utf-8"))
        deg_validation = json.loads(deg_validation_path.read_text(encoding="utf-8"))
        deg_manifest = pd.read_csv(deg_manifest_path, dtype={"cluster": str})
        candidate = pd.read_csv(candidate_path, dtype={"cluster": str})
        selected_h5ad = Path(parameters["selected_h5ad"])
        if not selected_h5ad.is_file():
            errors.append(f"missing selected h5ad:{selected_h5ad}")
            continue
        adata = sc.read_h5ad(selected_h5ad, backed="r")
        n_samples = int(adata.obs["sample"].nunique())
        n_series = int(adata.obs["series"].nunique())
        status_values = ";".join(sorted(adata.obs["status"].astype(str).unique()))
        subtype_absent = "cell_subtype" not in adata.obs and "functional_state" not in adata.obs
        adata.file.close()

        n_clusters = int(parameters["final_n_clusters"])
        lineage_errors = []
        if not clustering_validation.get("output_h5ad_readable", False):
            lineage_errors.append("clustering h5ad validation failed")
        if parameters["backend_package"] != "rapids-singlecell_GPU":
            lineage_errors.append("accepted clustering backend was not RAPIDS GPU")
        if bool(parameters["candidate_grid_run"]):
            lineage_errors.append("candidate grid unexpectedly ran")
        if int(deg_validation["observed_csv_count"]) != n_clusters:
            lineage_errors.append("DEG CSV count does not match cluster count")
        if not deg_manifest["n_deg_rows"].eq(20_244).all():
            lineage_errors.append("DEG CSV not full 20,244-gene length")
        if candidate.shape[0] != n_clusters or not candidate["proposed cell_subtype"].is_unique:
            lineage_errors.append("candidate annotation coverage or uniqueness failed")
        if not subtype_absent:
            lineage_errors.append("final subtype labels were written before confirmation")
        errors.extend(f"{slug}:{message}" for message in lineage_errors)
        rows.append(
            {
                "lineage": slug,
                "target_leiden_coarse": parameters["target_leiden_coarse"],
                "n_cells": int(parameters["n_obs"]),
                "n_samples": n_samples,
                "n_series": n_series,
                "status_values": status_values,
                "initial_resolution": float(parameters["initial_resolution"]),
                "initial_n_clusters": int(parameters["initial_n_clusters"]),
                "accepted_resolution": float(parameters["final_resolution"]),
                "accepted_n_clusters": n_clusters,
                "raw_cluster_column": parameters["raw_cluster_column"],
                "backend": parameters["backend_package"],
                "candidate_grid_run": bool(parameters["candidate_grid_run"]),
                "full_deg_csv_count": int(deg_validation["observed_csv_count"]),
                "genes_per_deg_csv": int(deg_manifest["n_deg_rows"].min()),
                "candidate_rows": int(candidate.shape[0]),
                "candidate_subtypes_unique": bool(candidate["proposed cell_subtype"].is_unique),
                "low_confidence_candidates": int(candidate["confidence"].eq("low").sum()),
                "cell_subtype_absent_pending_confirmation": subtype_absent,
                "lineage_validation_errors": " | ".join(lineage_errors),
            }
        )

    summary = pd.DataFrame(rows)
    pending_dirs = sorted(
        str(path)
        for category in ["h5ad", "tables", "figures"]
        for path in (
            PROJECT_ROOT / category / "02-cell_subtype_integration_clustering"
        ).glob("**/*pending*")
        if path.is_dir()
    )
    legacy_output_dirs = sorted(
        str(path)
        for category in ["h5ad", "tables", "figures"]
        for task in ["03-subtype-clustering-grid", "04-subtype-selected-clustering"]
        for path in [
            PROJECT_ROOT / category / "02-cell_subtype_integration_clustering" / task
        ]
        if path.exists()
    )
    if pending_dirs:
        errors.extend(f"pending directory:{path}" for path in pending_dirs)
    if legacy_output_dirs:
        errors.extend(f"legacy grid/selected output directory:{path}" for path in legacy_output_dirs)
    if summary.shape[0] != len(LINEAGES):
        errors.append(f"expected {len(LINEAGES)} lineage summaries, observed {summary.shape[0]}")
    if not summary.empty and int(summary["n_cells"].sum()) != 319_042:
        errors.append(f"lineage cell total mismatch:{int(summary['n_cells'].sum())}")
    if (
        not summary.empty
        and int(summary["accepted_n_clusters"].sum())
        != int(summary["full_deg_csv_count"].sum())
    ):
        errors.append("accepted cluster total differs from full DEG CSV total")

    summary.to_csv(output_csv, index=False)
    validation = {
        "status": "ready_for_subtype_annotation_confirmation" if not errors else "failed",
        "lineages_expected": len(LINEAGES),
        "lineages_observed": int(summary.shape[0]),
        "total_cells": int(summary["n_cells"].sum()) if not summary.empty else 0,
        "total_accepted_raw_clusters": (
            int(summary["accepted_n_clusters"].sum()) if not summary.empty else 0
        ),
        "total_full_deg_csvs": int(summary["full_deg_csv_count"].sum()) if not summary.empty else 0,
        "genes_per_deg_csv": 20_244,
        "all_clustering_backends_rapids_gpu": bool(
            not summary.empty and summary["backend"].eq("rapids-singlecell_GPU").all()
        ),
        "candidate_grid_outputs_absent": not legacy_output_dirs,
        "pending_directories_absent": not pending_dirs,
        "candidate_tables_complete_unique": bool(
            not summary.empty
            and summary["candidate_rows"].eq(summary["accepted_n_clusters"]).all()
            and summary["candidate_subtypes_unique"].all()
        ),
        "final_mapping_written": False,
        "final_subtype_h5ads_written": False,
        "confirmation_gate_pending": True,
        "summary_csv": str(output_csv),
        "errors": errors,
    }
    output_json.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

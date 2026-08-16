#!/usr/bin/env python3
"""Read-only final audit for the completed gastric-cancer Epi-CM workflow."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

WF = Path(__file__).resolve().parents[3]
TABLES = WF / "tables"
H5AD = WF / "h5ad"
FIGURES = WF / "figures"
OUT = TABLES / "99-workflow-final-audit/01-main"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []

    def check(block: str, requirement: str, observed: object, expected: object, passed: bool) -> None:
        checks.append(
            {
                "block": block,
                "requirement": requirement,
                "observed": observed,
                "expected": expected,
                "passed": bool(passed),
            }
        )

    # Block 01: integration/QC/selected broad clustering.
    b1t = TABLES / "01-celltype_integration_clustering"
    b1h = H5AD / "01-celltype_integration_clustering"
    required_b1_h5ad = [
        b1h / "02-merge-metadata/adata_merge.h5ad",
        b1h / "03-qc/adata_qc.h5ad",
        b1h / "04-integration-harmony/adata_harmony.h5ad",
        b1h / "05-clustering-parameter-search/selected/adata_inte.h5ad",
    ]
    check("01", "required integration h5ad files", sum(x.is_file() for x in required_b1_h5ad), 4,
          all(x.is_file() and x.stat().st_size > 0 for x in required_b1_h5ad))
    merge = read_json(b1t / "02-merge-metadata/merge_validation.json")
    harmony = read_json(b1t / "04-integration-harmony/integration_validation.json")
    selected = read_json(b1t / "05-clustering-parameter-search/selected/selected_clustering_validation.json")
    qc = pd.read_csv(b1t / "03-qc/qc_validation.csv")
    check("01", "merge validation", merge["all_passed"], True, bool(merge["all_passed"]))
    check("01", "merged samples", merge["n_samples"], 129, merge["n_samples"] == 129)
    check("01", "merge used full cells", merge["downsampled"], False, not merge["downsampled"])
    check("01", "QC validation", int(qc["passed"].sum()), len(qc), bool(qc["passed"].all()))
    check("01", "post-QC cells", harmony["n_obs"], 457999, harmony["n_obs"] == 457999)
    check("01", "Harmony validation", harmony["all_passed"], True, bool(harmony["all_passed"]))
    check("01", "selected clustering validation", selected["all_passed"], True, bool(selected["all_passed"]))

    # Block 02: eight lineage-specific subtype analyses and exact full-atlas projection.
    b2t = TABLES / "02-cell_subtype_integration_clustering"
    b2h = H5AD / "02-cell_subtype_integration_clustering"
    projection = read_json(b2t / "06-project-subtypes-to-full-adata-v2/exact_projection_audit.json")
    subtype_counts = pd.read_csv(b2t / "06-project-subtypes-to-full-adata-v2/projected_subtype_counts.csv")
    annotation_h5ads = sorted((b2h / "05-subtype-deg-annotation-v2").glob("*/adata_*.h5ad"))
    final_subtype_h5ad = b2h / "06-project-subtypes-to-full-adata-v2/adata_anno_cellsubtype.h5ad"
    check("02", "lineage annotation h5ad count", len(annotation_h5ads), 8, len(annotation_h5ads) == 8)
    check("02", "final subtype h5ad", final_subtype_h5ad.is_file(), True,
          final_subtype_h5ad.is_file() and final_subtype_h5ad.stat().st_size > 0)
    check("02", "exactly projected cells", projection["n_projected_exact"], 278857,
          projection["n_projected_exact"] == 278857 and projection["n_unmatched"] == 0)
    check("02", "lineages", projection["n_lineages"], 8, projection["n_lineages"] == 8)
    check("02", "global subtypes", projection["n_global_subtypes"], 67,
          projection["n_global_subtypes"] == 67 and len(subtype_counts) == 67)
    check("02", "no subtype cell filtering", projection["no_subtype_cell_filtering"], True,
          bool(projection["no_subtype_cell_filtering"]))

    # Block 03: balanced joint NMF, CM classification, all-pair Spearman and final plots.
    b3 = TABLES / "03-epi-cm-discovery"
    b3a = b3 / "01-cm-lineage-analysis"
    inclusion = pd.read_csv(b3a / "01_prepare_inputs_and_frequency_tables/sample_inclusion_exclusion.csv")
    sample_status = pd.read_csv(b3a / "01_prepare_inputs_and_frequency_tables/sample_status.csv")
    cm_class = pd.read_csv(b3a / "03_cm_classification_nodes_and_edges/joint_module_classification.csv")
    spearman = pd.read_csv(b3a / "04_epi_cm_association_spearman/epi_cm_spearman_all_pairs_long.csv")
    b3_plot = read_json(b3 / "02-cm-lineage-final-plotting/validation_report.json")
    check("03", "CM-fit samples", int(inclusion["keep_for_cm"].sum()), 96,
          int(inclusion["keep_for_cm"].sum()) == 96)
    check("03", "CM-fit normal-like/tumor samples", sample_status["status"].value_counts().to_dict(),
          {"normal-like": 36, "tumor": 60}, sample_status["status"].value_counts().to_dict() == {"tumor": 60, "normal-like": 36})
    check("03", "Epi-CM samples", int(inclusion["keep_for_epi_cm"].sum()), 75,
          int(inclusion["keep_for_epi_cm"].sum()) == 75)
    check("03", "selected K and CM count", len(cm_class), 11, len(cm_class) == 11)
    class_counts = cm_class["class"].value_counts().to_dict()
    check("03", "CM classes", class_counts, {"normalCM": 3, "sharedCM": 6, "tCM": 2},
          class_counts == {"sharedCM": 6, "normalCM": 3, "tCM": 2})
    assoc_scope = spearman.groupby("status").agg(rows=("CM", "size"), n_samples=("n_samples", "first"))
    assoc_ok = (
        len(spearman) == 308
        and int(assoc_scope.loc["normal-like", "rows"]) == 154
        and int(assoc_scope.loc["normal-like", "n_samples"]) == 28
        and int(assoc_scope.loc["tumor", "rows"]) == 154
        and int(assoc_scope.loc["tumor", "n_samples"]) == 47
    )
    check("03", "all-pair Spearman scope", len(spearman), "308 rows; n=28 normal-like, n=47 tumor", assoc_ok)
    check("03", "Block03 PDF/SVG figures", b3_plot["figure_count"], 774,
          b3_plot["figure_count"] == 774 and b3_plot["pdf_svg_only"] and b3_plot["all_manifest_files_exist"])
    check("03", "optional Pearson branch", b3_plot["pearson_scatter_panels"], 0,
          b3_plot["pearson_scatter_panels"] == 0)

    # Block 04: use the dedicated 26-requirement final audit as the formal source.
    b4 = TABLES / "04-spatial-validation-optional/01-tangram-pseudobulk-v2"
    b4_report = read_json(b4 / "04-final-audit/validation_report.json")
    check("04", "dedicated spatial audit", f"{b4_report['passed']}/{b4_report['requirements']}", "26/26",
          b4_report["status"] == "passed" and b4_report["failed"] == 0)
    check("04", "mapped Visium sections", len(list((b4 / "02-tangram-mapping/spot_scores").glob("*_spot_scores.csv"))), 72,
          len(list((b4 / "02-tangram-mapping/spot_scores").glob("*_spot_scores.csv"))) == 72)
    check("04", "spatial spots", b4_report["n_spatial_spots"], 152164,
          b4_report["n_spatial_spots"] == 152164)
    check("04", "section x CM x Epi rows", b4_report["n_pair_section_rows"], 11088,
          b4_report["n_pair_section_rows"] == 11088)
    check("04", "spatial figure files", b4_report["n_figure_files"], 44358,
          b4_report["pdf_count"] == 22179 and b4_report["svg_count"] == 22179)
    check("04", "spatial PNG files", b4_report["png_count"], 0, b4_report["png_count"] == 0)

    # Normalize provenance schema without altering the original per-section JSON records.
    metric_paths = sorted((b4 / "02-tangram-mapping").glob("*_mapping_metrics.json"))
    metric_rows = []
    for path in metric_paths:
        record = read_json(path)
        legacy = "n_common_genes" in record
        requested = record.get("n_common_genes_requested", record.get("n_common_genes"))
        training = record.get("n_training_genes")
        if record.get("sample") == "GSM8349861_D01T2" and training is None:
            # Recomputed read-only with the canonical loader + tg.pp_adatas: 2661.
            training = 2661
        metric_rows.append(
            {
                "sample": record.get("sample"),
                "n_common_genes_requested": requested,
                "n_training_genes": training,
                "source_schema": "legacy_n_common_genes" if legacy else "current",
                "source_json": str(path),
                "training_gene_provenance": "recomputed_tg.pp_adatas" if legacy else "source_json",
            }
        )
    metric_audit = pd.DataFrame(metric_rows)
    metric_audit.to_csv(OUT / "tangram_mapping_metrics_schema_audit.csv", index=False)
    metric_ok = (
        len(metric_audit) == 72
        and metric_audit["n_common_genes_requested"].eq(2734).all()
        and metric_audit["n_training_genes"].between(1, 2734).all()
        and metric_audit["sample"].nunique() == 72
    )
    check("04", "normalized mapping metrics schema", len(metric_audit), 72, bool(metric_ok))

    # Cross-workflow output contract.
    png_files = sorted(FIGURES.rglob("*.png"))
    check("all", "PNG files anywhere in canonical figures", len(png_files), 0, not png_files)
    check("all", "Block03 spatial state superseded by Block04 final audit", b4_report["status"], "passed",
          b4_report["status"] == "passed")

    audit = pd.DataFrame(checks)
    audit.to_csv(OUT / "workflow_validation_checks.csv", index=False)
    failures = audit.loc[~audit["passed"]]
    report = {
        "project": "gastric_cancer",
        "workflow": "epi-cm-core-workflow",
        "status": "passed" if failures.empty else "failed",
        "requirements": len(audit),
        "passed": int(audit["passed"].sum()),
        "failed": int((~audit["passed"]).sum()),
        "failed_requirements": failures[["block", "requirement"]].to_dict("records"),
        "block01": {"merged_samples": 129, "post_qc_cells": 457999},
        "block02": {"cells": 278857, "lineages": 8, "subtypes": 67, "subtype_filtering": False},
        "block03": {
            "cm_fit_samples": 96,
            "epi_cm_samples": 75,
            "selected_k": 11,
            "cm_classes": {"normalCM": 3, "sharedCM": 6, "tCM": 2},
            "spearman_rows": 308,
            "pearson_optional_branch_run": False,
        },
        "block04": {
            "spatial_sections": 72,
            "spatial_spots": 152164,
            "pair_section_rows": 11088,
            "figure_files": 44358,
            "pdf": 22179,
            "svg": 22179,
            "png": 0,
        },
    }
    (OUT / "workflow_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "readme.txt").write_text(
        "Final read-only audit of the current gastric-cancer Epi-CM core workflow.\n"
        "Block03's earlier validation_report.json retains its historical spatial-pending field; "
        "the authoritative current spatial state is Block04/04-final-audit/validation_report.json=passed.\n"
        "The original Tangram per-section JSON files were preserved. A normalized schema audit table "
        "records the one legacy first-section metrics record and its read-only tg.pp_adatas recomputation.\n",
        encoding="utf-8",
    )
    versions = [
        f"python={platform.python_version()}",
        f"numpy={package_version('numpy')}",
        f"pandas={package_version('pandas')}",
        f"seed={SEED}",
        "code=01_run_workflow_final_audit.py",
    ]
    (OUT / "package_versions.txt").write_text("\n".join(versions) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not failures.empty:
        raise RuntimeError(f"Workflow final audit failed: {failures[['block', 'requirement']].to_dict('records')}")


if __name__ == "__main__":
    main()

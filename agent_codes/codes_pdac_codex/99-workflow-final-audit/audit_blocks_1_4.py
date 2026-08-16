#!/usr/bin/env python3
"""Independent final contract audit for PDAC Epi-CM Blocks 1--4."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
OUT_DIR = PROJECT_ROOT / "tables/99-workflow-final-audit"
ROWS: list[dict] = []


def check(block: str, requirement: str, passed: bool, evidence: str) -> None:
    ROWS.append(
        {
            "block": block,
            "requirement": requirement,
            "passed": bool(passed),
            "evidence": evidence,
        }
    )


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def h5ad_contract(path: Path, expected_obs: int | None = None) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing {path}"
    a = ad.read_h5ad(path, backed="r")
    shape = tuple(a.shape)
    raw_vars = a.raw.n_vars if a.raw is not None else 0
    valid = expected_obs is None or a.n_obs == expected_obs
    a.file.close()
    return valid, f"{path}; shape={shape}; raw_n_vars={raw_vars}"


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite final audit: {OUT_DIR}")

    # Block 1: canonical Harmony, fixed active-default broad clustering/annotation,
    # full raw-gene DEG exports, and score/rank consistency filter.
    harmony = PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_harmony.h5ad"
    valid, evidence = h5ad_contract(harmony, 392388)
    check("Block1", "Canonical Harmony atlas has 392,388 cells", valid, evidence)
    selected_validation_path = PROJECT_ROOT / "tables/01-celltype_integration_clustering/05-clustering-parameter-search/selected_clustering_validation.json"
    selected_validation = load_json(selected_validation_path)
    check(
        "Block1",
        "Fixed broad clustering used 20 PCs, 20 neighbors, resolution 0.4 on RAPIDS GPU without a grid",
        selected_validation.get("n_pcs") == 20
        and selected_validation.get("n_neighbors") == 20
        and selected_validation.get("resolution") == 0.4
        and selected_validation.get("backend_package") == "rapids-singlecell_GPU"
        and selected_validation.get("candidate_grid_run") is False,
        f"{selected_validation_path}; candidate={selected_validation.get('candidate_label')}",
    )
    selected = PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/05-clustering-parameter-search/selected/adata_inte.h5ad"
    valid, evidence = h5ad_contract(selected, 392388)
    check("Block1", "Clean selected broad clustering exists", valid, evidence)
    broad = PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/06-broad-annotation/adata_anno.h5ad"
    valid, evidence = h5ad_contract(broad, 392388)
    check("Block1", "Seven-lineage broad annotation exists", valid, evidence)
    consistent = PROJECT_ROOT / "h5ad/01-celltype_integration_clustering/07-score-rank-qc/adata_anno_score_genes_rank_consistent.h5ad"
    valid, evidence = h5ad_contract(consistent, 319042)
    check("Block1", "Score/rank-consistent atlas has 319,042 cells", valid, evidence)
    broad_deg_dir = PROJECT_ROOT / "tables/01-celltype_integration_clustering/06-broad-annotation/degs_leiden_coarse_pcs20_nn20_res0p4"
    broad_deg_files = sorted(broad_deg_dir.glob("*.csv"))
    broad_deg_rows = [sum(1 for _ in path.open("rb")) - 1 for path in broad_deg_files]
    check(
        "Block1",
        "Seven post-annotation broad DEG files are full 20,244-gene exports",
        len(broad_deg_files) == 7 and set(broad_deg_rows) == {20244},
        f"directory={broad_deg_dir}; files={len(broad_deg_files)}; row_counts={sorted(set(broad_deg_rows))}",
    )

    # Block 2: seven direct lineage clusterings, full raw-cluster DEGs,
    # reviewed one-to-one subtype annotations, and exact all-cell projection.
    direct_validation_path = PROJECT_ROOT / "tables/02-cell_subtype_integration_clustering/03-subtype-clustering/module02_direct_clustering_deg_validation.json"
    direct = load_json(direct_validation_path)
    check(
        "Block2",
        "Seven direct lineage clusterings produced 74 full 20,244-gene DEG files on RAPIDS GPU without grids",
        direct.get("lineages_observed") == 7
        and direct.get("total_cells") == 319042
        and direct.get("total_accepted_raw_clusters") == 74
        and direct.get("total_full_deg_csvs") == 74
        and direct.get("genes_per_deg_csv") == 20244
        and direct.get("all_clustering_backends_rapids_gpu")
        and direct.get("candidate_grid_outputs_absent"),
        f"{direct_validation_path}",
    )
    direct_summary_path = PROJECT_ROOT / "tables/02-cell_subtype_integration_clustering/03-subtype-clustering/module02_direct_clustering_deg_summary.csv"
    direct_summary = pd.read_csv(direct_summary_path) if direct_summary_path.is_file() else pd.DataFrame()
    expected_resolution = {
        "epithelial": 0.4,
        "t_nk": 0.4,
        "myeloid": 0.4,
        "b": 0.4,
        "endothelial": 0.2,
        "stromal": 0.2,
        "mast": 0.2,
    }
    observed_resolution = (
        direct_summary.set_index("lineage")["accepted_resolution"].to_dict()
        if len(direct_summary)
        else {}
    )
    check(
        "Block2",
        "Mast, Endothelial, and Stromal use resolution 0.2 while the other four lineages retain 0.4",
        observed_resolution == expected_resolution,
        f"{direct_summary_path}; observed={observed_resolution}",
    )
    lineage_slugs = ["epithelial", "t_nk", "myeloid", "b", "endothelial", "stromal", "mast"]
    final_validations = [
        load_json(
            PROJECT_ROOT
            / f"tables/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/{slug}/final_subtype_annotation_validation.json"
        )
        for slug in lineage_slugs
    ]
    check(
        "Block2",
        "Seven finalized lineage H5ADs cover 319,042 cells and 74 unique reviewed subtypes",
        len(final_validations) == 7
        and sum(int(item.get("n_cells", 0)) for item in final_validations) == 319042
        and sum(int(item.get("n_subtypes", 0)) for item in final_validations) == 74
        and all(item.get("output_h5ad_readable") for item in final_validations)
        and all(item.get("all_raw_clusters_mapped_once") for item in final_validations)
        and all(item.get("subtype_names_unique") for item in final_validations),
        "final_subtype_annotation_validation.json for seven lineages",
    )
    projection_validation_path = PROJECT_ROOT / "tables/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata/all_lineages/projection_validation.json"
    projection_validation = load_json(projection_validation_path)
    check(
        "Block2",
        "All seven lineage subtype sets were projected once by exact cell IDs without broad-identity conflicts",
        projection_validation.get("all_lineage_ID_sets_exact")
        and projection_validation.get("all_lineage_ID_sets_disjoint")
        and projection_validation.get("all_cells_projected_once")
        and projection_validation.get("unmatched_cells") == 0
        and projection_validation.get("broad_conflicts_after_prefix_derivation") == 0
        and projection_validation.get("n_cell_subtypes_total") == 74,
        f"{projection_validation_path}",
    )

    # Block 3: balanced joint NMF and complete method-separated associations.
    block3_validation_path = PROJECT_ROOT / "tables/03-epi-cm-discovery/01-cm-lineage-analysis/block03_analysis_validation.json"
    block3 = load_json(block3_validation_path)
    check(
        "Block3",
        "Balanced joint NMF completed with canonical CM names and single-status routes disabled",
        block3.get("validation_passed")
        and block3.get("single_status_routes_disabled")
        and block3.get("all_public_cm_names_canonical")
        and block3.get("spearman_summary", {}).get("n_epi_subtypes") == 14
        and block3.get("H_orientation") == [14, 60],
        f"{block3_validation_path}; selected_k={block3.get('W_orientation', [None, None])[1] if block3 else None}",
    )
    for method in ["spearman", "pearson"]:
        summary = block3.get(f"{method}_summary", {})
        check(
            "Block3",
            f"{method.title()} has a complete Epi x CM Cartesian product in all contexts with BH-FDR",
            summary.get("rows_total") == 3 * summary.get("pairs_per_context", -1)
            and set(summary.get("contexts", [])) == {"overall", "normal-like", "tumor"},
            f"summary={summary}",
        )
    block3_plot_validation_path = PROJECT_ROOT / "tables/03-epi-cm-discovery/02-cm-lineage-final-plotting/plotting_validation.json"
    block3_plot = load_json(block3_plot_validation_path)
    check(
        "Block3",
        "Block 3 PDF/SVG plotting is CSV-driven and includes both association methods",
        block3_plot.get("validation_passed")
        and block3_plot.get("all_pair_scatter_contract")
        and block3_plot.get("association_methods") == ["spearman", "pearson"],
        f"{block3_plot_validation_path}; figure_files={block3_plot.get('figure_files')}",
    )

    # Block 4: real Visium inputs, canonical CUDA Tangram, CSV-driven statistics
    # and all-pair spatial figures.
    staging_path = PROJECT_ROOT / "tables/04-spatial-validation-optional/01-input-staging/staging_validation.json"
    staging = load_json(staging_path)
    check(
        "Block4",
        "Fourteen eligible GSE278687 double-gzip Visium inputs were staged",
        staging.get("eligible_samples") == 14
        and staging.get("all_spatial_archives_readable")
        and staging.get("all_required_spatial_components_present")
        and staging.get("excluded_samples_staged") == 0,
        f"{staging_path}",
    )
    spatial_h5ad_validation_path = PROJECT_ROOT / "tables/04-spatial-validation-optional/02-build-spatial-h5ad/spatial_h5ad_validation.json"
    spatial_h5ad = load_json(spatial_h5ad_validation_path)
    check(
        "Block4",
        "Fourteen clean spatial H5ADs preserve counts and real coordinates after 10k+log",
        spatial_h5ad.get("n_samples") == 14
        and spatial_h5ad.get("total_spots") == 53508
        and spatial_h5ad.get("all_counts_layers_present")
        and spatial_h5ad.get("all_real_spatial_coordinates_present")
        and spatial_h5ad.get("all_normalized_log"),
        f"{spatial_h5ad_validation_path}",
    )
    tangram_reference_validation_path = PROJECT_ROOT / "tables/04-spatial-validation-optional/03-tangram-reference/tangram_reference_validation.json"
    tangram_reference = load_json(tangram_reference_validation_path)
    check(
        "Block4",
        "Tangram reference contains all 74 finalized subtypes with 14 epithelial and 60 non-epithelial nodes",
        tangram_reference.get("validation_passed")
        and tangram_reference.get("n_reference_subtypes") == 74
        and tangram_reference.get("n_epithelial_subtypes") == 14
        and tangram_reference.get("n_non_epithelial_subtypes") == 60
        and tangram_reference.get("all_reference_cells_used"),
        f"{tangram_reference_validation_path}; shape={tangram_reference.get('reference_shape')}",
    )
    tangram_validation_path = PROJECT_ROOT / "tables/04-spatial-validation-optional/04-tangram-mapping/tangram_mapping_validation.json"
    tangram = load_json(tangram_validation_path)
    check(
        "Block4",
        "Canonical Tangram ran 14/14 in cells mode on cuda:0 for 350 epochs at lr=0.05 seed=42",
        tangram.get("validation_passed")
        and tangram.get("samples_completed") == 14
        and tangram.get("mode") == "cells"
        and tangram.get("device") == "cuda:0"
        and tangram.get("num_epochs") == 350
        and tangram.get("learning_rate") == 0.05
        and tangram.get("random_state") == 42
        and tangram.get("canonical_tangram"),
        f"{tangram_validation_path}; cuda={tangram.get('cuda_device')}",
    )
    score_dir = PROJECT_ROOT / "tables/04-spatial-validation-optional/04-tangram-mapping/spot-scores"
    score_paths = sorted(score_dir.glob("*_tangram_pseudobulk_epi_cm_spot_scores.csv"))
    epi_row_sum_valid = True
    total_score_rows = 0
    for path in score_paths:
        score = pd.read_csv(path)
        epi_columns = [column for column in score if column.startswith("EPIfrac__")]
        cm_columns = [column for column in score if column.startswith("CMact__")]
        total = score[epi_columns].sum(axis=1)
        positive = total.gt(0)
        epi_row_sum_valid &= bool(
            epi_columns
            and cm_columns
            and np.allclose(total.loc[positive], 1.0, rtol=1e-6, atol=1e-8)
            and not any("pct" in column.lower() for column in epi_columns + cm_columns)
        )
        total_score_rows += len(score)
    check(
        "Block4",
        "Fourteen canonical spot-score CSVs preserve raw CMact and within-epithelial EPIfrac",
        len(score_paths) == 14 and total_score_rows == 53508 and epi_row_sum_valid,
        f"directory={score_dir}; files={len(score_paths)}; rows={total_score_rows}",
    )
    statistics_validation_path = PROJECT_ROOT / "tables/04-spatial-validation-optional/05-spatial-statistics-and-plots/spatial_statistics_validation.json"
    statistics = load_json(statistics_validation_path)
    check(
        "Block4",
        "CSV-reloaded 14-sample Spearman, Fisher, and signed-Stouffer statistics passed",
        statistics.get("validation_passed")
        and statistics.get("n_samples") == 14
        and statistics.get("all_samples_are_tumor")
        and statistics.get("downstream_source", "").startswith("reloaded"),
        f"{statistics_validation_path}; pairs_per_sample={statistics.get('pairs_per_sample')}",
    )
    manifest_dir = PROJECT_ROOT / "tables/04-spatial-validation-optional/05-spatial-statistics-and-plots/plot-manifests-by-sample"
    manifests = sorted(manifest_dir.glob("*_plot_manifest.csv"))
    plot_rows = pd.concat([pd.read_csv(path) for path in manifests], ignore_index=True) if manifests else pd.DataFrame()
    figure_columns = ["raw_pdf", "raw_svg", "percentile_pdf", "percentile_svg"]
    figures_exist = bool(len(plot_rows)) and all(
        Path(value).is_file() for column in figure_columns for value in plot_rows[column]
    )
    expected_plot_rows = 14 * int(statistics.get("pairs_per_sample", 0))
    check(
        "Block4",
        "Every sample x CM x Epi pair has raw and percentile PDF/SVG spatial maps",
        len(manifests) == 14 and len(plot_rows) == expected_plot_rows and figures_exist,
        f"manifest_files={len(manifests)}; pair_rows={len(plot_rows)}; expected={expected_plot_rows}",
    )

    audit = pd.DataFrame(ROWS)
    passed = int(audit["passed"].sum())
    failed = int((~audit["passed"]).sum())
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    audit.to_csv(OUT_DIR / "blocks1-4-final-audit.csv", index=False)
    summary = {
        "requirements": int(len(audit)),
        "passed": passed,
        "failed": failed,
        "blocks": sorted(audit["block"].unique().tolist()),
        "workflow_complete": failed == 0,
    }
    (OUT_DIR / "blocks1-4-final-audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    failed_table = audit.loc[~audit["passed"]]
    report_lines = [
        "# PDAC Epi-CM Blocks 1-4 final audit",
        "",
        f"Requirements passed: {passed}/{len(audit)}.",
        "",
        "All epithelial and non-epithelial CM nodes are finalized Block 2 direct-clustering "
        "subtypes. Mast, Endothelial, and Stromal use resolution 0.2; the other four "
        "lineages use resolution 0.4.",
    ]
    if failed:
        report_lines.extend(["", "## Failed requirements", ""])
        report_lines.extend(
            f"- {row.block}: {row.requirement} ({row.evidence})"
            for row in failed_table.itertuples(index=False)
        )
    (OUT_DIR / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise RuntimeError(f"Final audit failed {failed}/{len(audit)} requirements")


if __name__ == "__main__":
    main()

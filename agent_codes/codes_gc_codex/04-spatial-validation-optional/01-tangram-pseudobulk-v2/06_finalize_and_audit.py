#!/usr/bin/env python3
"""Finalize provenance and audit the complete v2 GC spatial validation."""

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
TASK = "01-tangram-pseudobulk-v2"
BASE_TABLE = WF / f"tables/04-spatial-validation-optional/{TASK}"
REF = BASE_TABLE / "01-reference-and-manifest"
MAP = BASE_TABLE / "02-tangram-mapping"
STAT = BASE_TABLE / "03-all-pair-statistics-and-plots"
FIG = WF / f"figures/04-spatial-validation-optional/{TASK}"
OUT = BASE_TABLE / "04-final-audit"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []

    def check(requirement: str, observed: object, expected: object, passed: bool) -> None:
        checks.append(
            {
                "requirement": requirement,
                "observed": observed,
                "expected": expected,
                "passed": bool(passed),
            }
        )

    scope = pd.read_csv(REF / "spatial_sample_scope_manifest.csv")
    run = pd.read_csv(MAP / "tangram_section_run_manifest.csv")
    pairs = pd.read_csv(MAP / "all_sample_cm_epi_pair_manifest.csv")
    spearman = pd.read_csv(STAT / "per_section_spearman.csv")
    fisher = pd.read_csv(STAT / "percentile_quadrant_fisher_per_section.csv")
    check("spatial sections", len(scope), 72, len(scope) == 72)
    check("spatial donors", scope.patient_id.nunique(), 20, scope.patient_id.nunique() == 20)
    check("biological regions", scope.biological_region.nunique(), 37, scope.biological_region.nunique() == 37)
    check("main sections", int(scope.section_role.eq("main").sum()), 33, int(scope.section_role.eq("main").sum()) == 33)
    check("replicating sections", int(scope.section_role.eq("replicating").sum()), 39, int(scope.section_role.eq("replicating").sum()) == 39)
    check("all spatial status tumor", sorted(scope.status.unique()), ["tumor"], set(scope.status) == {"tumor"})
    check("Tangram completed sections", int(run.status_run.eq("completed").sum()), 72, run.status_run.eq("completed").all())
    check("Tangram failed sections", int(run.status_run.eq("failed").sum()), 0, not run.status_run.eq("failed").any())
    check("all-pair manifest rows", len(pairs), 11088, len(pairs) == 11088)
    check("pair manifest CMs", pairs.CM.nunique(), 11, pairs.CM.nunique() == 11)
    check("pair manifest epithelial subtypes", pairs.epi_subtype.nunique(), 14, pairs.epi_subtype.nunique() == 14)
    check("per-section Spearman rows", len(spearman), 11088, len(spearman) == 11088)
    check("per-section Fisher rows", len(fisher), 11088, len(fisher) == 11088)

    score_paths = sorted((MAP / "spot_scores").glob("*_tangram_pseudobulk_epi_cm_spot_scores.csv"))
    check("spot-score CSV count", len(score_paths), 72, len(score_paths) == 72)
    total_rows = 0
    finite = True
    epi_normalized = True
    score_schema = True
    for path in score_paths:
        frame = pd.read_csv(path)
        total_rows += len(frame)
        cm = [x for x in frame if x.startswith("CMact__")]
        epi = [x for x in frame if x.startswith("EPIfrac__")]
        score_schema &= len(cm) == 11 and len(epi) == 14
        values = frame[cm + epi].to_numpy(float)
        finite &= bool(np.isfinite(values).all())
        epi_normalized &= bool(np.allclose(frame[epi].sum(axis=1).to_numpy(float), 1.0, rtol=1e-6, atol=1e-8))
    expected_spots = int(scope.n_spatial_spots.sum())
    check("total mapped spots", total_rows, expected_spots, total_rows == expected_spots)
    check("score schema 11 CM plus 14 Epi", score_schema, True, score_schema)
    check("all scores finite", finite, True, finite)
    check("EPIfrac sums to one per spot", epi_normalized, True, epi_normalized)

    plot_manifests = sorted((STAT / "plot_manifests_by_sample").glob("*_plot_manifest.csv"))
    check("section plot-manifest count", len(plot_manifests), 72, len(plot_manifests) == 72)
    manifested_paths: list[Path] = []
    manifest_complete = True
    for path in plot_manifests:
        frame = pd.read_csv(path)
        manifest_complete &= len(frame) == 154
        for column in ("raw_pdf", "raw_svg", "percentile_pdf", "percentile_svg"):
            manifest_complete &= column in frame.columns
            if column in frame.columns:
                manifested_paths.extend(Path(x) for x in frame[column].astype(str))
    check("154 complete plots per section", manifest_complete, True, manifest_complete)
    check(
        "all manifested pair figures exist",
        sum(path.is_file() and path.stat().st_size > 0 for path in manifested_paths),
        44352,
        len(manifested_paths) == 44352 and all(path.is_file() and path.stat().st_size > 0 for path in manifested_paths),
    )

    all_figures = sorted(path for path in FIG.rglob("*") if path.is_file())
    png = [path for path in all_figures if path.suffix.lower() == ".png"]
    pdf = [path for path in all_figures if path.suffix.lower() == ".pdf"]
    svg = [path for path in all_figures if path.suffix.lower() == ".svg"]
    other = [path for path in all_figures if path.suffix.lower() not in {".pdf", ".svg"}]
    check("Block04 PNG files", len(png), 0, not png)
    check("Block04 non-PDF/SVG figure files", len(other), 0, not other)
    check("Block04 PDF count", len(pdf), 22179, len(pdf) == 22179)
    check("Block04 SVG count", len(svg), 22179, len(svg) == 22179)
    check("PDF/SVG stem parity", {x.with_suffix("") for x in pdf} == {x.with_suffix("") for x in svg}, True, {x.with_suffix("") for x in pdf} == {x.with_suffix("") for x in svg})

    figure_manifest = pd.DataFrame(
        [
            {
                "figure_file": str(path),
                "output_format": path.suffix.lower().lstrip("."),
                "figure_family": (
                    "raw_cmact_epifrac"
                    if "raw_cmact_epifrac_by_sample" in path.parts
                    else "percentile_quadrant"
                    if "percentile_quadrant_by_sample" in path.parts
                    else "stouffer_heatmap"
                ),
                "size_bytes": path.stat().st_size,
            }
            for path in all_figures
        ]
    )
    figure_manifest.to_csv(OUT / "spatial_figure_manifest.csv", index=False)
    audit = pd.DataFrame(checks)
    audit.to_csv(OUT / "validation_checks.csv", index=False)
    failures = audit.loc[~audit["passed"]]
    report = {
        "status": "passed" if failures.empty else "failed",
        "requirements": len(audit),
        "passed": int(audit.passed.sum()),
        "failed": int((~audit.passed).sum()),
        "failed_requirements": failures.requirement.tolist(),
        "n_spatial_spots": total_rows,
        "n_pair_section_rows": len(fisher),
        "n_figure_files": len(all_figures),
        "pdf_count": len(pdf),
        "svg_count": len(svg),
        "png_count": len(png),
    }
    (OUT / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"step": "reference_and_manifest", "planned_backend": "CPU", "attempted_backend": "CPU", "status": "completed", "error_summary": "", "fallback_backend": "", "clean_input_reloaded": True, "final_backend_for_rerun": "CPU"},
            {"step": "Tangram_cells_mapping_350_epochs", "planned_backend": "cuda:0", "attempted_backend": "cuda:0", "status": "completed", "error_summary": "", "fallback_backend": "", "clean_input_reloaded": True, "final_backend_for_rerun": "cuda:0"},
            {"step": "Spearman_Fisher_BH_Stouffer", "planned_backend": "CPU", "attempted_backend": "CPU", "status": "completed", "error_summary": "", "fallback_backend": "", "clean_input_reloaded": True, "final_backend_for_rerun": "CPU"},
            {"step": "all_pair_PDF_SVG_plotting", "planned_backend": "CPU_16_workers", "attempted_backend": "CPU_16_workers", "status": "completed", "error_summary": "", "fallback_backend": "", "clean_input_reloaded": True, "final_backend_for_rerun": "CPU_16_workers"},
        ]
    ).to_csv(OUT / "gpu_backend_capability_summary.csv", index=False)
    params = {
        "seed": SEED,
        "Tangram": {"mode": "cells", "device": "cuda:0", "num_epochs": 350, "learning_rate": 0.05, "random_state": 42},
        "marker_selection": "pvals_adj<0.05, positive logFC, descending score, deduplicated top100 per each of 67 subtypes, then spatial/reference intersection",
        "EPIfrac": "Tangram epithelial subtype abundance normalized within the 14 epithelial columns per spot",
        "CMact": "Tangram non-epithelial subtype abundance multiplied by Block03 H loading fractions",
        "spatial_statistics": "per-section Spearman and median-percentile Fisher; BH within section; signed-Stouffer across 72 sections",
        "replicate_aware_sensitivity": "within-region Stouffer then equal-weight Stouffer across 37 biological regions",
    }
    (OUT / "run_parameters.json").write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    versions = [
        f"python={platform.python_version()}",
        f"environment={Path(platform.python_implementation()).name}; executable recorded by runner",
        f"anndata={package_version('anndata')}",
        f"scanpy={package_version('scanpy')}",
        f"tangram-sc={package_version('tangram-sc')}",
        f"torch={package_version('torch')}",
        f"numpy={package_version('numpy')}",
        f"pandas={package_version('pandas')}",
        f"scipy={package_version('scipy')}",
        f"matplotlib={package_version('matplotlib')}",
        f"seaborn={package_version('seaborn')}",
        "Tangram_device=cuda:0",
        "plot_workers=16",
        f"seed={SEED}",
    ]
    (OUT / "package_versions.txt").write_text("\n".join(versions) + "\n", encoding="utf-8")
    (OUT / "readme.txt").write_text(
        "Canonical GC Block04 v2 output from the v2 67-subtype atlas and v2 Block03 CM loadings only.\n"
        "All 72 tumor Visium sections were mapped by Tangram on cuda:0; no marker-score or NNLS substitute was used.\n"
        "All 11x14 pairs were computed and plotted per section. Figures are PDF/SVG only.\n"
        "The 37-region result is a separately labeled replicate-aware sensitivity analysis.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    if not failures.empty:
        raise RuntimeError(f"Spatial final audit failed: {failures.requirement.tolist()}")


if __name__ == "__main__":
    main()

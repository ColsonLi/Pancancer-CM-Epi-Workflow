#!/usr/bin/env python3
"""Build the v5 62-subtype pseudobulk reference and 72-section GC manifest."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import re
import tarfile
import tempfile
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy
from scipy import sparse

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

WF = Path(__file__).resolve().parents[3]
TASK = "01-tangram-pseudobulk-v3-final-v5"
TABLE_DIR = WF / f"tables/04-spatial-validation-optional/{TASK}/01-reference-and-manifest"
H5AD_DIR = WF / f"h5ad/04-spatial-validation-optional/{TASK}/01-reference-and-manifest"
RAW_TAR = Path(
    "/mnt/disk18t/lr_xcy/riku/external_atlas_for_cm-epi_validation/"
    "Gastric_cancer/GSE270678_st/GSE270678_RAW.tar"
)
CURATED_SPATIAL = (
    WF
    / "tables/01-celltype_integration_clustering/01-sample-manifest/"
    "per_dataset/GSE270678_all_samples_inclusion_v4.csv"
)
FINAL_SC = (
    WF
    / "h5ad/02-cell_subtype_integration_clustering/"
    "06-project-subtypes-to-full-adata/agent-led-current-selection-v5/adata_anno_cellsubtype.h5ad"
)
DEG_ROOT = (
    WF
    / "tables/02-cell_subtype_integration_clustering/"
    "05-subtype-deg-annotation/agent-led-current-selection-v5"
)
CM_LOADING = (
    WF
    / "tables/03-epi-cm-discovery/01-cm-lineage-analysis/"
    "02_balanced_joint_nmf/agent-led-current-selection-v5-nmf-r1/loading_df_cell_subtype_by_CM_fraction.csv"
)
REFERENCE_OUT = H5AD_DIR / "adata_cell_subtype_pseudobulk_deg_panel.h5ad"


def decode(values) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def categorical(group: h5py.Group, key: str) -> tuple[list[str], np.ndarray]:
    node = group[key]
    if isinstance(node, h5py.Group) and "categories" in node:
        return decode(node["categories"][()]), node["codes"][()].astype(np.int32)
    values = np.asarray(decode(node[()]), dtype=object)
    cats, codes = np.unique(values, return_inverse=True)
    return cats.tolist(), codes.astype(np.int32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tar_member_prefix(source_matrix_path: str) -> str:
    match = re.search(r"::([^;]+)_filtered_feature_bc_matrix\.h5", str(source_matrix_path))
    if not match:
        raise ValueError(f"Cannot parse 10x member from {source_matrix_path}")
    return match.group(1)


def build_manifest() -> pd.DataFrame:
    curated = pd.read_csv(CURATED_SPATIAL)
    curated = curated.loc[curated["used"].astype(bool)].copy()
    curated["sample"] = curated["source_matrix_path"].map(tar_member_prefix)
    curated["gsm"] = curated["sample"].str.extract(r"^(GSM\d+)", expand=False)
    curated["section_label"] = curated["sample"].str.replace(r"^GSM\d+_", "", regex=True)
    curated["biological_region"] = curated["section_label"].str.replace(r"_[123]$", "", regex=True)
    curated["section_role"] = np.where(
        curated["sample_title"].astype(str).str.contains("Replicate", case=False, na=False),
        "replicating",
        "main",
    )
    curated["patient_id"] = curated["patient"].astype(str).str.extract(r"(\d+)", expand=False).map(
        lambda x: f"D{int(x):02d}"
    )
    curated["status"] = "tumor"
    curated["include_all_samples"] = True
    curated["include_tumor_only"] = True
    curated["h5_member"] = curated["sample"] + "_filtered_feature_bc_matrix.h5"
    curated["positions_member"] = curated["sample"] + "_tissue_positions_list.csv.gz"

    with tarfile.open(RAW_TAR, "r") as archive:
        members = {x.name: x for x in archive.getmembers()}
        n_spots, n_features = [], []
        gene_hashes = []
        for row in curated.itertuples(index=False):
            if row.h5_member not in members or row.positions_member not in members:
                raise FileNotFoundError(f"Missing tar member for {row.sample}")
            with tempfile.TemporaryDirectory() as temp_dir:
                archive.extract(members[row.h5_member], temp_dir, filter="data")
                with h5py.File(Path(temp_dir) / row.h5_member, "r") as handle:
                    matrix = handle["matrix"]
                    shape = tuple(int(x) for x in matrix["shape"][()])
                    genes = decode(matrix["features"]["name"][()])
                    n_features.append(shape[0])
                    n_spots.append(shape[1])
                    gene_hashes.append(
                        hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()
                    )
    curated["n_spatial_spots"] = n_spots
    curated["n_spatial_features"] = n_features
    curated["spatial_gene_order_sha256"] = gene_hashes
    keep = [
        "sample", "gsm", "patient_id", "stage", "section_label", "biological_region",
        "section_role", "sample_title", "status", "treatment_context", "metastasis_evidence",
        "include_all_samples", "include_tumor_only", "n_spatial_spots", "n_spatial_features",
        "spatial_gene_order_sha256", "h5_member", "positions_member", "metadata_evidence_source",
    ]
    result = curated[keep].sort_values("gsm", kind="stable").reset_index(drop=True)
    checks = {
        "sections": len(result),
        "donors": result["patient_id"].nunique(),
        "biological_regions": result["biological_region"].nunique(),
        "main_sections": int(result["section_role"].eq("main").sum()),
        "replicating_sections": int(result["section_role"].eq("replicating").sum()),
        "status_values": sorted(result["status"].unique().tolist()),
        "unique_gene_orders": result["spatial_gene_order_sha256"].nunique(),
    }
    expected = {
        "sections": 72, "donors": 20, "biological_regions": 37,
        "main_sections": 33, "replicating_sections": 39,
    }
    for key, value in expected.items():
        if checks[key] != value:
            raise ValueError(f"Spatial cohort contract failed: {key}={checks[key]}, expected={value}")
    if checks["status_values"] != ["tumor"] or checks["unique_gene_orders"] != 1:
        raise ValueError(f"Spatial status/gene-order contract failed: {checks}")
    (TABLE_DIR / "spatial_manifest_audit.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def spatial_gene_symbols(first_h5_member: str) -> list[str]:
    with tarfile.open(RAW_TAR, "r") as archive, tempfile.TemporaryDirectory() as temp_dir:
        member = archive.getmember(first_h5_member)
        archive.extract(member, temp_dir, filter="data")
        with h5py.File(Path(temp_dir) / first_h5_member, "r") as handle:
            return decode(handle["matrix"]["features"]["name"][()])


def build_marker_table(spatial_genes: set[str], reference_genes: set[str]) -> pd.DataFrame:
    paths = sorted(DEG_ROOT.glob("*/degs_cell_subtype*/*.csv"))
    if len(paths) != 62 or len({path.stem for path in paths}) != 62:
        raise ValueError(f"Require exactly 62 post-annotation DEG CSVs, found {len(paths)}")
    rows = []
    for path in paths:
        subtype = path.stem
        deg = pd.read_csv(path)
        required = {"gene", "score", "logfoldchanges", "pvals_adj"}
        if not required.issubset(deg.columns):
            raise KeyError(f"{path}: missing {sorted(required - set(deg.columns))}")
        for column in ("score", "logfoldchanges", "pvals_adj"):
            deg[column] = pd.to_numeric(deg[column], errors="coerce")
        deg["gene"] = deg["gene"].astype(str)
        deg = deg.loc[
            np.isfinite(deg["score"])
            & np.isfinite(deg["logfoldchanges"])
            & np.isfinite(deg["pvals_adj"])
            & deg["logfoldchanges"].gt(0)
            & deg["pvals_adj"].lt(0.05)
        ].copy()
        deg = deg.sort_values("score", ascending=False, kind="stable").drop_duplicates("gene").head(100)
        if deg.empty:
            raise ValueError(f"{subtype}: no significant positive DEG")
        deg["deg_rank"] = np.arange(1, len(deg) + 1)
        for row in deg.itertuples(index=False):
            rows.append(
                {
                    "cell_subtype": subtype,
                    "gene": row.gene,
                    "deg_rank": int(row.deg_rank),
                    "score": float(row.score),
                    "logfoldchanges": float(row.logfoldchanges),
                    "pvals_adj": float(row.pvals_adj),
                    "in_spatial": row.gene in spatial_genes,
                    "in_reference_raw": row.gene in reference_genes,
                    "source_deg_csv": str(path),
                }
            )
    result = pd.DataFrame(rows)
    result["used_for_tangram"] = result["in_spatial"] & result["in_reference_raw"]
    result.to_csv(TABLE_DIR / "marker_genes_used.csv", index=False)
    per_subtype = result.groupby("cell_subtype").agg(
        n_significant_positive_top100=("gene", "size"),
        n_used_for_tangram=("used_for_tangram", "sum"),
    )
    per_subtype.to_csv(TABLE_DIR / "marker_gene_counts_by_subtype.csv")
    if len(per_subtype) != 62 or per_subtype["n_used_for_tangram"].min() < 1:
        raise ValueError("Every one of the 62 subtypes must retain at least one common marker")
    return result


def build_pseudobulk(marker_genes: list[str]) -> ad.AnnData:
    with h5py.File(FINAL_SC, "r") as handle:
        subtype_categories, subtype_codes = categorical(handle["obs"], "cell_subtype")
        raw_names = decode(handle["raw"]["var"]["_index"][()])
        gene_to_i = {gene: i for i, gene in enumerate(raw_names)}
        selected_i = np.asarray([gene_to_i[gene] for gene in marker_genes], dtype=np.int64)
        node = handle["raw"]["X"]
        shape = tuple(int(x) for x in node.attrs["shape"])
        raw_x = sparse.csr_matrix(
            (node["data"][()], node["indices"][()], node["indptr"][()]), shape=shape
        )
    selected = raw_x[:, selected_i].tocsr()
    rows = np.flatnonzero(subtype_codes >= 0)
    groups = subtype_codes[rows]
    indicator = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (groups, rows)),
        shape=(len(subtype_categories), len(subtype_codes)),
    )
    sums = (indicator @ selected).toarray().astype(np.float32)
    counts = np.bincount(groups, minlength=len(subtype_categories)).astype(np.float32)
    means = sums / counts[:, None]
    obs = pd.DataFrame(index=pd.Index(subtype_categories, name="cell_subtype"))
    obs["cell_subtype"] = obs.index.astype(str)
    obs["cell_class"] = np.where(obs.index.str.startswith("Epi_"), "epithelial", "non_epithelial")
    obs["n_reference_cells"] = counts.astype(np.int64)
    var = pd.DataFrame(index=pd.Index(marker_genes, name="gene"))
    reference = ad.AnnData(X=means, obs=obs, var=var)
    reference.uns["expression_source"] = "v5 atlas adata.raw normalized/log expression; all cells used"
    reference.uns["marker_contract"] = "pvals_adj<0.05, logFC>0, score-desc, deduplicated top100 per subtype, then spatial/raw intersection"
    reference.uns["random_seed"] = SEED
    return reference


def main() -> None:
    started = time.time()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    H5AD_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    manifest.to_csv(TABLE_DIR / "spatial_sample_scope_manifest.csv", index=False)
    spatial_names = spatial_gene_symbols(manifest.loc[0, "h5_member"])
    spatial_set = set(spatial_names)
    with h5py.File(FINAL_SC, "r") as handle:
        reference_names = decode(handle["raw"]["var"]["_index"][()])
    markers = build_marker_table(spatial_set, set(reference_names))
    used = markers.loc[markers["used_for_tangram"]].copy()
    marker_genes = (
        used.groupby("gene", as_index=False)["score"].max()
        .sort_values(["score", "gene"], ascending=[False, True], kind="stable")["gene"]
        .tolist()
    )
    pd.DataFrame({"gene": marker_genes}).to_csv(TABLE_DIR / "tangram_common_deg_genes.csv", index=False)
    pd.DataFrame(
        [
            {"metric": "spatial_features", "value": len(spatial_names)},
            {"metric": "spatial_unique_gene_symbols", "value": len(spatial_set)},
            {"metric": "spatial_duplicate_gene_symbols", "value": len(spatial_names) - len(spatial_set)},
            {"metric": "single_cell_raw_genes", "value": len(reference_names)},
            {"metric": "significant_positive_top100_rows", "value": len(markers)},
            {"metric": "final_tangram_gene_union", "value": len(marker_genes)},
        ]
    ).to_csv(TABLE_DIR / "marker_gene_intersection_summary.csv", index=False)
    reference = build_pseudobulk(marker_genes)
    if reference.n_obs != 62 or reference.obs["cell_class"].eq("epithelial").sum() != 12:
        raise ValueError(f"Reference subtype contract failed: {reference.shape}")
    reference.write_h5ad(REFERENCE_OUT, compression="gzip")
    reference.obs[["n_reference_cells", "cell_class"]].to_csv(TABLE_DIR / "reference_cells_used.csv")
    loadings = pd.read_csv(CM_LOADING, index_col=0)
    non_epi = reference.obs_names[reference.obs["cell_class"].eq("non_epithelial")].tolist()
    epi = reference.obs_names[reference.obs["cell_class"].eq("epithelial")].tolist()
    if set(loadings.index) != set(non_epi) or loadings.shape != (50, 11):
        raise ValueError(f"Block03 loading/reference mismatch: {loadings.shape}")
    loadings.loc[non_epi].to_csv(TABLE_DIR / "cm_loading_fraction_non_epi_subtype_by_CM.csv")
    pd.DataFrame({"cell_subtype": epi}).to_csv(TABLE_DIR / "epithelial_subtypes.csv", index=False)
    report = {
        "status": "completed",
        "input_tar": str(RAW_TAR),
        "input_tar_sha256": sha256(RAW_TAR),
        "input_v5_atlas": str(FINAL_SC),
        "n_sections": len(manifest),
        "n_donors": manifest["patient_id"].nunique(),
        "n_biological_regions": manifest["biological_region"].nunique(),
        "n_reference_subtypes": reference.n_obs,
        "n_epi_subtypes": len(epi),
        "n_non_epi_subtypes": len(non_epi),
        "n_cm": loadings.shape[1],
        "n_common_deg_genes": len(marker_genes),
        "elapsed_seconds": round(time.time() - started, 3),
        "python": platform.python_version(),
        "anndata": ad.__version__,
        "scipy": scipy.__version__,
        "seed": SEED,
    }
    (TABLE_DIR / "reference_build_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (TABLE_DIR / "readme.txt").write_text(
        "Input: GSE270678 RAW tar, curated spatial inclusion CSV, v5 62-subtype atlas/DEGs, and v5 Block03 CM loading fractions.\n"
        "All 72 primary-tumor Visium sections are included; 33 main and 39 replicating sections cover 20 donors and 37 biological regions.\n"
        "Reference: all v5 cells, adata.raw normalized/log expression, significant positive top100 DEG per subtype before spatial intersection.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Stage only the 14 clinically eligible GSE278687 Visium samples."""

from __future__ import annotations

import importlib.metadata
import gzip
import json
import random
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path("/mnt/disk18t/lr_xcy/riku/pdac_codex/epi-cm-core-workflow")
SOURCE_TAR = PROJECT_ROOT / "inputs/geo-official-20260806/GSE278687_RAW.tar"
INCLUDED_MANIFEST = (
    PROJECT_ROOT
    / "tables/01-celltype_integration_clustering/01-input-metadata-audit/included/metadata.csv"
)
STAGING_ROOT = PROJECT_ROOT / "inputs/02-spatial-staging/GSE278687"
TABLE_DIR = PROJECT_ROOT / "tables/04-spatial-validation-optional/01-input-staging"


def require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def main() -> None:
    if not SOURCE_TAR.is_file() or not INCLUDED_MANIFEST.is_file():
        raise FileNotFoundError("Official spatial tar or included sample manifest is missing")
    outputs = [
        TABLE_DIR / "spatial_input_manifest.csv",
        TABLE_DIR / "spatial_archive_contract.csv",
        TABLE_DIR / "package_versions.txt",
        TABLE_DIR / "readme.txt",
        TABLE_DIR / "staging_validation.json",
    ]
    for path in outputs:
        require_new(path)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)

    included = pd.read_csv(INCLUDED_MANIFEST, low_memory=False)
    included = included.loc[included["dataset"].eq("GSE278687")].copy()
    if included["gsm"].nunique() != 14 or not included["status"].eq("tumor").all():
        raise ValueError("Expected exactly 14 eligible tumor spatial samples")
    manifest_rows = []
    archive_rows = []

    with tarfile.open(SOURCE_TAR, "r") as outer:
        members = outer.getmembers()
        for _, row in included.sort_values("gsm").iterrows():
            gsm = str(row["gsm"])
            matches = [
                member for member in members
                if Path(member.name).name.startswith(gsm + "_") and member.isfile()
            ]
            h5_members = [member for member in matches if member.name.endswith("_filtered_feature_bc_matrix.h5")]
            spatial_members = [member for member in matches if member.name.endswith("_spatial.tar.gz")]
            if len(h5_members) != 1 or len(spatial_members) != 1:
                raise ValueError(
                    f"{gsm}: expected one filtered H5 and one spatial tar.gz; "
                    f"found {len(h5_members)} and {len(spatial_members)}"
                )
            sample_dir = STAGING_ROOT / gsm
            sample_dir.mkdir(parents=True, exist_ok=True)
            staged_paths: dict[str, Path] = {}
            for role, member in [("filtered_feature_bc_matrix_h5", h5_members[0]), ("spatial_tar_gz", spatial_members[0])]:
                destination = sample_dir / Path(member.name).name
                if destination.exists():
                    if destination.stat().st_size != member.size:
                        raise ValueError(f"Wrong-size existing staged file: {destination}")
                    action = "reused_exact_size_staged_file"
                else:
                    part = destination.with_suffix(destination.suffix + ".part")
                    if part.exists():
                        raise FileExistsError(f"Incomplete prior extraction requires review: {part}")
                    source = outer.extractfile(member)
                    if source is None:
                        raise OSError(f"Cannot read outer tar member: {member.name}")
                    with source, part.open("wb") as target:
                        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
                    if part.stat().st_size != member.size:
                        raise OSError(f"Incomplete extraction for {member.name}")
                    part.replace(destination)
                    action = "extracted_from_validated_official_geo_tar"
                staged_paths[role] = destination
                manifest_rows.append(
                    {
                        "sample": gsm,
                        "sample_label": row["sample_id"],
                        "patient_id": row["patient_id"],
                        "status": row["status"],
                        "ptnm": row["ptnm"],
                        "clinical_stage": row["clinical_stage"],
                        "matrix_or_spatial_role": role,
                        "outer_tar_path": str(SOURCE_TAR),
                        "outer_tar_member": member.name,
                        "outer_tar_member_size_bytes": member.size,
                        "staged_path": str(destination),
                        "staged_size_bytes": destination.stat().st_size,
                        "action": action,
                    }
                )

            spatial_path = staged_paths["spatial_tar_gz"]
            # GEO names these files *.tar.gz, but they are double-gzipped:
            # gzip -> gzip -> tar.  Preserve the supplied bytes and document the
            # actual container contract instead of renaming or rewriting them.
            with gzip.open(spatial_path, "rb") as first_gzip:
                with tarfile.open(fileobj=first_gzip, mode="r:gz") as inner:
                    inner_members = inner.getmembers()
                    inner_files = [member for member in inner_members if member.isfile()]
            required_patterns = {
                "tissue_positions": any("tissue_positions" in member.name for member in inner_files),
                "scalefactors_json": any("scalefactors_json.json" in member.name for member in inner_files),
                "tissue_image": any(
                    member.name.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
                    for member in inner_files
                ),
            }
            archive_rows.append(
                {
                    "sample": gsm,
                    "spatial_tar_gz": str(spatial_path),
                    "gzip_tar_readable": True,
                    "container_format": "double_gzip_then_tar",
                    "n_inner_members": len(inner_members),
                    "n_inner_files": len(inner_files),
                    **required_patterns,
                    "all_required_spatial_components_present": all(required_patterns.values()),
                    "first_20_inner_members": ";".join(member.name for member in inner_members[:20]),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    archive_contract = pd.DataFrame(archive_rows)
    if manifest["sample"].nunique() != 14 or manifest.shape[0] != 28:
        raise RuntimeError("Staged spatial manifest does not contain 2 files for each of 14 samples")
    if not archive_contract["all_required_spatial_components_present"].all():
        failed = archive_contract.loc[
            ~archive_contract["all_required_spatial_components_present"], "sample"
        ].tolist()
        raise RuntimeError(f"Spatial component validation failed for: {failed}")
    manifest.to_csv(outputs[0], index=False)
    archive_contract.to_csv(outputs[1], index=False)
    package_lines = [
        "environment_category=main",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        "install_manager=uv",
        "index_url=https://pypi.tuna.tsinghua.edu.cn/simple",
        f"numpy={importlib.metadata.version('numpy')}",
        f"pandas={importlib.metadata.version('pandas')}",
        f"random_seed={SEED}",
        f"code_file={Path(__file__).resolve()}",
    ]
    outputs[2].write_text("\n".join(package_lines) + "\n", encoding="utf-8")
    outputs[3].write_text(
        "PDAC spatial input staging\n\n"
        f"Input: {SOURCE_TAR}\n"
        f"Clinical inclusion manifest: {INCLUDED_MANIFEST}\n"
        "Exactly 14 eligible GSE278687 tumor samples were staged.  Each sample has one "
        "filtered 10x H5 count matrix and one readable spatial.tar.gz containing tissue "
        "positions, scale factors, and a tissue image.  Excluded spatial samples were not staged.\n"
        "These are Block 4 inputs only; canonical Tangram mapping is run after the selected "
        "single-cell epithelial subtype and CM outputs exist.\n",
        encoding="utf-8",
    )
    validation = {
        "source_tar": str(SOURCE_TAR),
        "staging_root": str(STAGING_ROOT),
        "eligible_samples": 14,
        "staged_files": manifest.shape[0],
        "all_staged_sizes_match_outer_tar": bool(
            (manifest["outer_tar_member_size_bytes"] == manifest["staged_size_bytes"]).all()
        ),
        "all_spatial_archives_readable": bool(archive_contract["gzip_tar_readable"].all()),
        "all_required_spatial_components_present": bool(
            archive_contract["all_required_spatial_components_present"].all()
        ),
        "excluded_samples_staged": 0,
        "random_seed": SEED,
    }
    outputs[4].write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

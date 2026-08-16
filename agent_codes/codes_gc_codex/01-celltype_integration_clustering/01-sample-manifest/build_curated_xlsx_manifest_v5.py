#!/usr/bin/env python3
"""Build the canonical gastric-cancer sample manifest from five curated XLSX files.

The curated inclusion sheets are the sole authority for inclusion.  Exclusion
worksheets are retained only as an audit trail.  No GEO/paper heuristics are
used to override a curator decision.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path

import pandas as pd


SEED = 42
ROOT = Path(__file__).resolve().parents[4]
WF = ROOT / "epi-cm-core-workflow"
DATA_ROOT = Path(
    "/mnt/disk18t/lr_xcy/riku/external_atlas_for_cm-epi_validation/"
    "Gastric_cancer"
)
TABLE_DIR = WF / "tables/01-celltype_integration_clustering/01-sample-manifest"
MANIFEST_NAME = "all_datasets_curated_xlsx_inclusion_v5.csv"
EXPECTED_INCLUDED = {
    "GSE150290": {"tumor": 23, "AT": 24, "healthy": 0},
    "GSE163558": {"tumor": 1, "AT": 1, "healthy": 0},
    "GSE183904": {"tumor": 23, "AT": 7, "healthy": 1},
    "GSE234129": {"tumor": 1, "AT": 1, "healthy": 0},
    "GSE270680": {"tumor": 25, "AT": 22, "healthy": 0},
}

SPECS = {
    "GSE150290": {
        "workbook": "GSE150290_metadata_with_GEO.xlsx",
        "included_sheet": "Sample_level",
        "excluded_sheet": "未纳入样本",
        "patient": "Patient",
        "sample_label": "GEO sample name",
        "title": "GEO sample name",
        "stage": "Diagnosis",
        "treatment": None,
    },
    "GSE163558": {
        "workbook": "GSE163558_metadata_with_GEO.xlsx",
        "included_sheet": "Non-metastatic samples",
        "excluded_sheet": "未纳入样本",
        "patient": "Patient",
        "sample_label": "Sample",
        "title": "GEO sample name",
        "stage": "Histopathological diagnosis",
        "treatment": None,
    },
    "GSE183904": {
        "workbook": "GSE183904_metadata_with_GEO.xlsx",
        "included_sheet": "Eligible samples",
        "excluded_sheet": "未纳入样本",
        "patient": "Patient ID",
        "sample_label": "GEO sample file",
        "title": "GEO sample name",
        "stage": "Stage",
        "treatment": "Treatment status",
    },
    "GSE234129": {
        "workbook": "GSE234129_metadata_with_GEO.xlsx",
        "included_sheet": "Eligible samples",
        "excluded_sheet": "未纳入样本",
        "patient": "Subject ID",
        "sample_label": "Sample",
        "title": "GEO sample name",
        "stage": "WHO classification",
        "treatment": "GEO treatment",
    },
    "GSE270680": {
        "workbook": "GSE270680_SC_samples_with_GEO.xlsx",
        "included_sheet": "SC_samples",
        "excluded_sheet": "未纳入样本",
        "patient": "Donor id",
        "sample_label": "Sample_id",
        "title": "GEO sample name",
        "stage": None,
        "treatment": None,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(value).lower())


def tissue_class(series: str, row: pd.Series) -> str:
    if series == "GSE150290":
        value = clean(row.get("Tissue label")).casefold()
        return {"tumor": "tumor", "adjacent": "AT", "control": "other"}.get(value, "other")
    if series == "GSE163558":
        value = clean(row.get("Sample")).upper()
        if value.startswith("PT"):
            return "tumor"
        if value.startswith("NT"):
            return "AT"
        return "other"
    if series == "GSE183904":
        category = clean(row.get("Sample category")).casefold()
        if category == "primary tumor":
            return "tumor"
        if category == "primary normal":
            return "healthy" if clean(row.get("Patient ID")) == "NGCII536" else "AT"
        return "other"
    if series == "GSE234129":
        value = clean(row.get("Tissue type")).casefold()
        if value == "primary tumor":
            return "tumor"
        if "normal adjacent" in value:
            return "AT"
        if "healthy" in value:
            return "healthy"
        return "other"
    if series == "GSE270680":
        value = re.sub(r"\*$", "", clean(row.get("Sample_id"))).upper()
        if value.endswith("T"):
            return "tumor"
        if value.endswith("N"):
            return "AT"
        return "other"
    raise KeyError(series)


def selected_tar_members(series: str, gsm: str, members: list[str]) -> list[str]:
    matched = sorted(name for name in members if gsm in Path(name).name)
    if series == "GSE150290":
        return [name for name in matched if name.endswith(".txt.gz")]
    if series == "GSE183904":
        return [name for name in matched if name.endswith(".csv.gz")]
    matrices = [name for name in matched if name.endswith("_matrix.mtx.gz")]
    for matrix in matrices:
        stem = matrix[: -len("_matrix.mtx.gz")]
        triplet = [matrix, stem + "_features.tsv.gz", stem + "_barcodes.tsv.gz"]
        if all(item in members for item in triplet):
            return triplet
    return []


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"Refusing to overwrite a different artifact: {path}")
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def atomic_text(text: str, path: Path) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def main() -> None:
    tar_members: dict[str, list[str]] = {}
    provenance_rows: list[dict[str, object]] = []
    for series, spec in SPECS.items():
        workbook = DATA_ROOT / series / str(spec["workbook"])
        if not workbook.is_file():
            raise FileNotFoundError(workbook)
        provenance_rows.append(
            {
                "series": series,
                "workbook": str(workbook),
                "workbook_sha256": sha256(workbook),
                "included_sheet": spec["included_sheet"],
                "excluded_sheet": spec["excluded_sheet"],
                "authority": "manual_curated_xlsx",
            }
        )
        if series != "GSE234129":
            archive = DATA_ROOT / series / f"{series}_RAW.tar"
            if not archive.is_file():
                raise FileNotFoundError(archive)
            with tarfile.open(archive, "r:*") as handle:
                tar_members[series] = [item.name for item in handle.getmembers() if item.isfile()]

    shared_paths = [
        DATA_ROOT / "GSE234129" / f"GSE234129_{suffix}"
        for suffix in (
            "count_matrix.mtx.gz",
            "barcodes.tsv.gz",
            "features.tsv.gz",
            "meta.tsv.gz",
        )
    ]
    missing_shared = [str(path) for path in shared_paths if not path.is_file()]
    if missing_shared:
        raise FileNotFoundError(f"Missing GSE234129 shared inputs: {missing_shared}")
    shared_meta = pd.read_csv(shared_paths[-1], sep="\t", dtype=str, keep_default_na=False)
    shared_sample_keys = set(shared_meta["sample"].map(normalize_key))

    rows: list[dict[str, object]] = []
    for series, spec in SPECS.items():
        workbook = DATA_ROOT / series / str(spec["workbook"])
        included = pd.read_excel(workbook, sheet_name=str(spec["included_sheet"]), dtype=str).dropna(how="all")
        excluded = pd.read_excel(workbook, sheet_name=str(spec["excluded_sheet"]), dtype=str).dropna(how="all")
        if series == "GSE270680":
            included = included[
                included["Usage"].fillna("").str.strip().str.casefold().eq("yes")
            ].copy()
        for used, sheet, frame in (
            (True, str(spec["included_sheet"]), included),
            (False, str(spec["excluded_sheet"]), excluded),
        ):
            for _, source in frame.iterrows():
                gsm = clean(source.get("GSM"))
                if not gsm and not used:
                    gsm = "NO_GSM__" + re.sub(
                        r"[^A-Za-z0-9_.-]+", "_", clean(source.get(str(spec["sample_label"])))
                    )
                if used and not re.fullmatch(r"GSM\d+", gsm):
                    raise ValueError(f"{series}/{sheet}: invalid GSM {gsm!r}")
                cls = tissue_class(series, source)
                if used and cls not in {"tumor", "AT", "healthy"}:
                    raise ValueError(f"{series}/{gsm}: included row has unsupported tissue class {cls}")
                if series == "GSE234129":
                    sample_label = clean(source.get(str(spec["sample_label"])))
                    local_present = normalize_key(sample_label) in shared_sample_keys
                    source_path = ";".join(str(path) for path in shared_paths)
                else:
                    selected = (
                        selected_tar_members(series, gsm, tar_members[series])
                        if re.fullmatch(r"GSM\d+", gsm)
                        else []
                    )
                    local_present = bool(selected)
                    archive = DATA_ROOT / series / f"{series}_RAW.tar"
                    source_path = str(archive) + "::" + ";".join(selected)
                treatment_value = clean(source.get(spec["treatment"])) if spec["treatment"] else ""
                treatment_context = (
                    treatment_value
                    if treatment_value and treatment_value.casefold() != "not reported"
                    else "treatment-naive_by_user_default_no_treatment_evidence"
                )
                row = {
                    "series": series,
                    "modality": "scRNA-seq",
                    "sample": gsm,
                    "sample_title": clean(source.get(str(spec["title"]))) or clean(source.get(str(spec["sample_label"]))),
                    "sample_label": clean(source.get(str(spec["sample_label"]))),
                    "patient": clean(source.get(str(spec["patient"]))),
                    "stage": clean(source.get(spec["stage"])) if spec["stage"] else "",
                    "tissue_class": cls,
                    "status": "tumor" if cls == "tumor" else ("normal" if cls in {"AT", "healthy"} else "excluded"),
                    "treatment_context": treatment_context,
                    "used": used,
                    "include_decision": "included" if used else "excluded",
                    "exclusion_reason": "" if used else clean(source.get("未满足原因")),
                    "curated_workbook": str(workbook),
                    "curated_workbook_sha256": sha256(workbook),
                    "curated_sheet": sheet,
                    "local_data_present": local_present,
                    "source_matrix_path": source_path,
                    "seed": SEED,
                }
                rows.append(row)

    manifest = pd.DataFrame(rows)
    duplicate_keys = manifest.duplicated(["series", "sample"], keep=False)
    if duplicate_keys.any():
        raise ValueError(
            "Curated workbook rows duplicate series+GSM keys: "
            + json.dumps(
                manifest.loc[duplicate_keys, ["series", "sample", "curated_sheet"]].to_dict("records"),
                ensure_ascii=False,
            )
        )
    included = manifest[manifest["used"]].copy()
    if len(included) != 129 or included["sample"].nunique() != 129:
        raise ValueError(
            f"Expected 129 unique included GSMs, observed rows={len(included)}, "
            f"unique={included['sample'].nunique()}"
        )
    if not included["local_data_present"].all():
        missing = included.loc[~included["local_data_present"], ["series", "sample", "source_matrix_path"]]
        raise FileNotFoundError(f"Included samples without a local count source: {missing.to_dict('records')}")
    if set(included["series"]) != set(EXPECTED_INCLUDED):
        raise ValueError(f"Unexpected included series: {sorted(set(included['series']))}")

    summary = (
        included.groupby(["series", "tissue_class"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=list(EXPECTED_INCLUDED), columns=["tumor", "AT", "healthy"], fill_value=0)
        .reset_index()
    )
    summary["total"] = summary[["tumor", "AT", "healthy"]].sum(axis=1)
    for record in summary.to_dict("records"):
        expected = EXPECTED_INCLUDED[record["series"]]
        observed = {key: int(record[key]) for key in ("tumor", "AT", "healthy")}
        if observed != expected:
            raise ValueError(f"{record['series']} expected {expected}, observed {observed}")
    total = {
        "series": "TOTAL",
        "tumor": int(summary["tumor"].sum()),
        "AT": int(summary["AT"].sum()),
        "healthy": int(summary["healthy"].sum()),
        "total": int(summary["total"].sum()),
    }
    summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)
    if total != {"series": "TOTAL", "tumor": 73, "AT": 55, "healthy": 1, "total": 129}:
        raise ValueError(f"Unexpected overall included summary: {total}")

    manifest = manifest.sort_values(["series", "used", "sample"], ascending=[True, False, True], kind="stable")
    included = included.sort_values(["series", "sample"], kind="stable")
    excluded = manifest[~manifest["used"]].copy()
    validation = pd.DataFrame(
        [
            {"check": "included_rows", "passed": len(included) == 129, "value": len(included)},
            {"check": "included_unique_GSM", "passed": included["sample"].nunique() == 129, "value": included["sample"].nunique()},
            {"check": "included_series", "passed": included["series"].nunique() == 5, "value": included["series"].nunique()},
            {"check": "included_local_matrix_coverage", "passed": bool(included["local_data_present"].all()), "value": int(included["local_data_present"].sum())},
            {"check": "tumor_samples", "passed": total["tumor"] == 73, "value": total["tumor"]},
            {"check": "AT_samples", "passed": total["AT"] == 55, "value": total["AT"]},
            {"check": "healthy_samples", "passed": total["healthy"] == 1, "value": total["healthy"]},
            {"check": "GSE167297_excluded_entirely", "passed": "GSE167297" not in set(manifest["series"]), "value": 0},
        ]
    )
    if not validation["passed"].all():
        raise RuntimeError(validation.to_string(index=False))

    atomic_csv(manifest, TABLE_DIR / MANIFEST_NAME)
    atomic_csv(included, TABLE_DIR / "included_samples_curated_xlsx_v5.csv")
    atomic_csv(excluded, TABLE_DIR / "excluded_samples_audit_curated_xlsx_v5.csv")
    atomic_csv(summary, TABLE_DIR / "sample_count_by_series_tissue_v5.csv")
    atomic_csv(pd.DataFrame(provenance_rows), TABLE_DIR / "workbook_provenance_v5.csv")
    atomic_csv(validation, TABLE_DIR / "sample_manifest_validation_v5.csv")
    for series in EXPECTED_INCLUDED:
        atomic_csv(
            included[included["series"].eq(series)],
            TABLE_DIR / f"{series}_included_samples_v5.csv",
        )
    atomic_text(
        "Canonical cohort: five manually curated XLSX workbooks only.\n"
        "Included: 129 unique GSMs across five GSEs (73 tumor, 55 AT, 1 healthy).\n"
        "AT and healthy map to status=normal; tumor maps to status=tumor.\n"
        "GSE167297 is entirely absent. Peripheral blood, lymph-node, metastatic,\n"
        "peritoneal/ascites, ovary/liver, and other curator-excluded samples are not used.\n"
        "No-treatment-evidence rows follow the user's rule and are treated as naive.\n"
        "All included rows have a validated local raw-count source.\n",
        TABLE_DIR / "readme_curated_xlsx_v5.txt",
    )
    print(summary.to_string(index=False))
    print(f"manifest={TABLE_DIR / MANIFEST_NAME}")
    print(f"excluded_rows={len(excluded)} all_validation_passed={validation['passed'].all()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the evidence-backed GC single-cell/spatial sample manifest.

The inclusion contract is intentionally strict:
  * primary gastric cancer or adjacent gastric tissue must be explicitly
    untreated/treatment-naive and must not come from a patient with documented
    distant metastatic disease;
  * healthy stomach controls are eligible with treatment not applicable;
  * metastatic sites, blood, lymph-node tissue, ascites, organoids, and other
    non-gastric/non-primary materials are excluded;
  * GSE270678 spatial sections inherit treatment context from the linked
    GSE270680 cohort and use the same stage I-III non-distant-metastatic scope.

All GEO samples are retained in the output even when excluded.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import os
import random
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


SEED = 42
random.seed(SEED)

ANALYSIS_ROOT = Path("/mnt/disk18t/lr_xcy/riku/gc_codex")
WORKFLOW_ROOT = ANALYSIS_ROOT / "epi-cm-core-workflow"
DATA_ROOT = Path(
    "/mnt/disk18t/lr_xcy/riku/external_atlas_for_cm-epi_validation/Gastric_cancer"
)
TABLE_DIR = (
    WORKFLOW_ROOT
    / "tables/01-celltype_integration_clustering/01-sample-manifest"
)
SOURCE_DIR = TABLE_DIR / "source_metadata"

SERIES = (
    "GSE150290",
    "GSE163558",
    "GSE167297",
    "GSE183904",
    "GSE234129",
    "GSE270678",
    "GSE270680",
)

LOCAL_DIR = {
    "GSE150290": "GSE150290",
    "GSE163558": "GSE163558",
    "GSE167297": "GSE167297",
    "GSE183904": "GSE183904",
    "GSE234129": "GSE234129",
    "GSE270678": "GSE270678_st",
    "GSE270680": "GSE270680",
}

MODALITY = {
    "GSE150290": "scRNA-seq",
    "GSE163558": "scRNA-seq",
    "GSE167297": "scRNA-seq",
    "GSE183904": "scRNA-seq",
    "GSE234129": "scRNA-seq",
    "GSE270678": "spatial transcriptomics",
    "GSE270680": "scRNA-seq",
}

# GSE163558 Table S1, PMID 35184420 / PMC8858624.
GSE163_SAMPLE_TO_PATIENT = {
    "PT1": "Patient 1",
    "Li1": "Patient 1",
    "PT2": "Patient 2",
    "NT1": "Patient 2",
    "O1": "Patient 3",
    "PT3": "Patient 4",
    "Li2": "Patient 4",
    "LN1": "Patient 4",
    "LN2": "Patient 5",
    "P1": "Patient 6",
}
GSE163_METASTATIC_PATIENTS = {
    "Patient 1",
    "Patient 3",
    "Patient 4",
    "Patient 5",
    "Patient 6",
}

# GSE183904 GEO title mapping plus Supplementary Table 1, PMID 34642171.
GSE183_SAMPLE_TO_PATIENT = {
    7: "NGCII521", 6: "NGCII521",
    22: "NGCII513", 21: "NGCII513",
    26: "NGCII512", 25: "NGCII512",
    5: "NGCII520", 4: "NGCII520",
    24: "NGCII514", 23: "NGCII514",
    2: "NGCII518", 1: "NGCII518",
    10: "NGCII525", 9: "NGCII525",
    12: "NGCII527", 11: "NGCII527",
    36: "NGCII538", 35: "NGCII538", 38: "NGCII538", 37: "NGCII538",
    32: "NGCII539", 13: "NGCII529", 15: "NGCII511",
    18: "NGCII498", 29: "NGCII522", 16: "NGCII499",
    14: "NGCII502", 17: "NGCII509", 33: "NGCII540",
    34: "NGCII541", 39: "NGCII543", 27: "NGCII510",
    3: "NGCII519", 28: "NGCII531", 40: "NGCII545",
    30: "NGCII533", 8: "NGCII524", 31: "NGCII536",
    19: "NGCII011", 20: "NGCII015",
}
GSE183_PATIENT_STAGE = {
    "NGCII521": "II", "NGCII513": "III", "NGCII512": "III",
    "NGCII520": "III", "NGCII514": "III", "NGCII518": "II",
    "NGCII525": "II", "NGCII527": "IV", "NGCII538": "IV",
    "NGCII539": "I", "NGCII529": "I", "NGCII511": "II",
    "NGCII498": "II", "NGCII522": "II", "NGCII499": "III",
    "NGCII502": "III", "NGCII509": "III", "NGCII540": "III",
    "NGCII541": "III", "NGCII543": "III", "NGCII510": "I",
    "NGCII519": "III", "NGCII531": "III", "NGCII545": "III",
    "NGCII533": "III", "NGCII524": "IV", "NGCII536": "Normal",
    "NGCII011": "DM", "NGCII015": "DM",
}


def read_series_matrix(series: str) -> list[dict[str, object]]:
    path = SOURCE_DIR / f"{series}_series_matrix.txt.gz"
    sample_rows: list[list[str]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if row and row[0].startswith("!Sample_"):
                sample_rows.append(row)

    accessions = next(row[1:] for row in sample_rows if row[0] == "!Sample_geo_accession")
    records: list[dict[str, object]] = [
        {"series": series, "sample": accession, "characteristics": {}}
        for accession in accessions
    ]
    repeated: defaultdict[str, int] = defaultdict(int)
    for row in sample_rows:
        key = row[0].removeprefix("!Sample_")
        values = row[1:]
        if len(values) != len(records):
            continue
        repeated[key] += 1
        for record, value in zip(records, values):
            value = value.strip()
            if key == "characteristics_ch1":
                if ":" in value:
                    char_key, char_value = value.split(":", 1)
                    record["characteristics"][char_key.strip().lower()] = char_value.strip()
            elif repeated[key] == 1:
                record[key] = value
            else:
                prior = record.get(key, "")
                record[key] = " | ".join(x for x in (str(prior), value) if x)
    return records


def tar_entries(series: str) -> list[str]:
    local = DATA_ROOT / LOCAL_DIR[series]
    tar_paths = sorted(local.glob("*.tar"))
    if not tar_paths:
        return []
    entries: list[str] = []
    for tar_path in tar_paths:
        with tarfile.open(tar_path, "r:*") as archive:
            entries.extend(member.name for member in archive.getmembers() if member.isfile())
    return entries


def gse234_local_samples() -> set[str]:
    meta = DATA_ROOT / "GSE234129/GSE234129_meta.tsv.gz"
    values: set[str] = set()
    with gzip.open(meta, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            values.add(row["sample"])
    return values


def local_provenance(record: dict[str, object], entries: list[str], gse234_samples: set[str]) -> tuple[bool, str]:
    series = str(record["series"])
    sample = str(record["sample"])
    title = str(record.get("title", ""))
    local_dir = DATA_ROOT / LOCAL_DIR[series]
    if series == "GSE234129":
        files = [
            local_dir / "GSE234129_count_matrix.mtx.gz",
            local_dir / "GSE234129_barcodes.tsv.gz",
            local_dir / "GSE234129_features.tsv.gz",
            local_dir / "GSE234129_meta.tsv.gz",
        ]
        present = title in gse234_samples and all(path.exists() for path in files)
        return present, ";".join(map(str, files))
    matched = [name for name in entries if sample in Path(name).name]
    raw_tar = next(iter(sorted(local_dir.glob("*.tar"))), None)
    source = ";".join(f"{raw_tar}::{name}" for name in matched) if raw_tar else ""
    return bool(matched), source


def normalise_stage(value: str) -> str:
    return re.sub(r"^Stage\s+", "", value, flags=re.I).strip()


def classify_sample(record: dict[str, object]) -> dict[str, object]:
    series = str(record["series"])
    title = str(record.get("title", ""))
    source = str(record.get("source_name_ch1", ""))
    chars = dict(record.get("characteristics", {}))
    patient = ""
    stage = normalise_stage(str(chars.get("stage", "")))
    tissue_class = "out_of_scope"
    status = "excluded"
    treatment_context = "unknown"
    treatment_evidence = ""
    distant_metastasis = False
    metastasis_evidence = ""
    evidence_source = f"{series}_series_matrix.txt.gz"

    if series == "GSE150290":
        patient = title.split("-", 1)[0]
        if source == "Control":
            tissue_class, status = "healthy_stomach", "normal-like"
            treatment_context = "not_applicable_healthy_control"
        elif source == "Cancer cells":
            tissue_class, status = "primary_gastric_tumor", "tumor"
            treatment_context = "unknown_not_explicit_in_geo_or_article"
        elif source == "Adjacent normal cells":
            tissue_class, status = "adjacent_normal_stomach", "normal-like"
            treatment_context = "unknown_not_explicit_in_geo_or_article"
        metastasis_evidence = "No distant-metastasis field was supplied; EGC/AGC is not treated as proof of M0."
        evidence_source += ";PMID35087207;PMC8795238"

    elif series == "GSE163558":
        token_match = re.search(r"\b(PT\d+|NT\d+|LN\d+|Li\d+|O\d+|P\d+)\b", title)
        token = token_match.group(1) if token_match else ""
        patient = GSE163_SAMPLE_TO_PATIENT.get(token, "")
        if source == "primary gastric cancer":
            tissue_class, status = "primary_gastric_tumor", "tumor"
        elif source == "adjacent non-tumoral":
            tissue_class, status = "adjacent_normal_stomach", "normal-like"
        else:
            tissue_class, status = "metastatic_site", "metastatic"
        treatment_context = chars.get("treatment", "unknown").lower()
        treatment_evidence = "GEO characteristic: treatment=untreated"
        distant_metastasis = patient in GSE163_METASTATIC_PATIENTS
        metastasis_evidence = (
            "Supplementary Table S1 links this patient to liver/ovary/lymph-node/peritoneal metastasis."
            if distant_metastasis
            else "Supplementary Table S1 diagnosis states no metastasis."
        )
        evidence_source += ";PMC8858624 Supplementary Table S1"

    elif series == "GSE167297":
        patient = chars.get("patient", "")
        tissue = chars.get("tissue", "").lower()
        if "normal" in tissue:
            tissue_class, status = "adjacent_normal_stomach", "normal-like"
        else:
            tissue_class, status = "primary_gastric_tumor", "tumor"
        treatment_context = "unknown_not_explicit_in_geo"
        metastasis_evidence = "Primary diffuse-type GC layers; no per-sample distant-metastasis field supplied."
        evidence_source += ";PMID34385296"

    elif series == "GSE183904":
        number_match = re.search(r"sample(\d+)", title, flags=re.I)
        number = int(number_match.group(1)) if number_match else -1
        patient = GSE183_SAMPLE_TO_PATIENT.get(number, "")
        stage = GSE183_PATIENT_STAGE.get(patient, "")
        tissue = chars.get("tissue", "").lower()
        if "periton" in tissue:
            tissue_class, status = "metastatic_site", "metastatic"
        elif "normal" in tissue and stage == "Normal":
            tissue_class, status = "healthy_stomach", "normal-like"
        elif "normal" in tissue:
            tissue_class, status = "adjacent_normal_stomach", "normal-like"
        else:
            tissue_class, status = "primary_gastric_tumor", "tumor"
        treatment_context = (
            "not_applicable_healthy_control" if tissue_class == "healthy_stomach"
            else "unknown_not_explicit_in_geo_or_supplementary_clinical_table"
        )
        distant_metastasis = stage in {"IV", "DM"}
        metastasis_evidence = f"Supplementary Table 1 stage={stage or 'unknown'}"
        evidence_source += ";PMC9394383 Supplementary Table 1"

    elif series == "GSE234129":
        patient = title.split("-", 1)[0]
        suffix = title.rsplit("-", 1)[-1]
        if suffix == "Ca":
            tissue_class, status = "primary_gastric_tumor", "tumor"
        elif suffix == "Ad":
            tissue_class, status = "adjacent_normal_stomach", "normal-like"
        elif suffix in {"As", "Ov", "Li"}:
            tissue_class, status = "metastatic_site", "metastatic"
        else:
            tissue_class, status = "blood_or_other", "excluded"
        treatment_context = chars.get("treatment", "unknown").lower().replace(" ", "_")
        treatment_evidence = "GEO characteristic and series design: Treatment Naive"
        metastatic_patients = {"MDA_Pt2", "MDA_Pt3", "MDA_Pt4", "MDA_Pt5", "MDA_Pt9"}
        distant_metastasis = patient in metastatic_patients
        metastasis_evidence = (
            "Same patient has ascites, ovarian metastasis, or liver metastasis in the series."
            if distant_metastasis
            else "No metastatic sample or metastatic diagnosis is reported for Pt1."
        )
        evidence_source += ";PMID37419119;GSE234129_meta.tsv.gz"

    elif series == "GSE270678":
        patient = chars.get("subject id", "")
        tissue_class, status = "primary_gastric_tumor_spatial", "tumor"
        treatment_context = "treatment_naive"
        treatment_evidence = "Linked cohort GSE270680 overall design: 27 treatment-naive donors"
        distant_metastasis = stage.upper() in {"IV", "4"}
        metastasis_evidence = f"GEO stage={stage}; spatial cohort contains stages I-III only."
        evidence_source += ";GSE270680 overall design;PMID41593079"

    elif series == "GSE270680":
        patient = chars.get("cell type", "")
        tissue = chars.get("tissue", "")
        if tissue == "Gastric cancer tumor":
            tissue_class, status = "primary_gastric_tumor", "tumor"
        elif tissue == "Adjacent normal":
            tissue_class, status = "adjacent_normal_stomach", "normal-like"
        elif tissue == "Lymph node":
            tissue_class, status = "lymph_node", "excluded"
        else:
            tissue_class, status = "peripheral_blood", "excluded"
        treatment_context = "treatment_naive"
        treatment_evidence = "GEO overall design: cohort of 27 treatment-naive donors"
        distant_metastasis = stage.upper() in {"IV", "4"}
        metastasis_evidence = f"GEO stage={stage}; cohort contains stages I-III only."
        evidence_source += ";PMID41593079"

    allowed_tissue = tissue_class in {
        "primary_gastric_tumor",
        "primary_gastric_tumor_spatial",
        "adjacent_normal_stomach",
        "healthy_stomach",
    }
    if tissue_class == "healthy_stomach":
        treatment_eligible = True
    else:
        treatment_eligible = treatment_context in {
            "untreated",
            "treatment_naive",
        }
    included = allowed_tissue and treatment_eligible and not distant_metastasis

    reasons: list[str] = []
    if not allowed_tissue:
        reasons.append(f"out_of_scope_tissue:{tissue_class}")
    if allowed_tissue and not treatment_eligible:
        reasons.append("treatment_status_not_explicitly_untreated")
    if distant_metastasis:
        reasons.append("patient_has_documented_distant_metastatic_context")
    if included:
        if tissue_class.startswith("primary_gastric_tumor"):
            decision_reason = "included_primary_untreated_nonmetastatic_gc"
        elif tissue_class == "adjacent_normal_stomach":
            decision_reason = "included_adjacent_normal_from_untreated_nonmetastatic_context"
        else:
            decision_reason = "included_healthy_stomach_control"
    else:
        decision_reason = ";".join(reasons)

    return {
        "patient": patient,
        "stage": stage,
        "tissue_class": tissue_class,
        "status": status,
        "treatment_context": treatment_context,
        "treatment_evidence": treatment_evidence,
        "distant_metastatic_patient_context": distant_metastasis,
        "metastasis_evidence": metastasis_evidence,
        "allowed_tissue_scope": allowed_tissue,
        "treatment_eligible": treatment_eligible,
        "used": included,
        "decision": "used" if included else "not_used",
        "decision_reason": decision_reason,
        "metadata_evidence_source": evidence_source,
    }


OUTPUT_COLUMNS = [
    "series", "modality", "sample", "sample_title", "source_name", "patient",
    "stage", "tissue_class", "status", "treatment_context", "treatment_evidence",
    "distant_metastatic_patient_context", "metastasis_evidence", "allowed_tissue_scope",
    "treatment_eligible", "used", "decision", "decision_reason", "local_data_present",
    "source_matrix_path", "metadata_evidence_source",
]


def write_csv_new(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    gse234_samples = gse234_local_samples()
    manifest: list[dict[str, object]] = []
    for series in SERIES:
        entries = [] if series == "GSE234129" else tar_entries(series)
        for raw in read_series_matrix(series):
            decision = classify_sample(raw)
            present, matrix_path = local_provenance(raw, entries, gse234_samples)
            row = {
                "series": series,
                "modality": MODALITY[series],
                "sample": raw["sample"],
                "sample_title": raw.get("title", ""),
                "source_name": raw.get("source_name_ch1", ""),
                **decision,
                "local_data_present": present,
                "source_matrix_path": matrix_path,
            }
            manifest.append(row)

    manifest.sort(key=lambda row: (SERIES.index(str(row["series"])), str(row["sample"])))
    write_csv_new(TABLE_DIR / "all_datasets_all_samples_inclusion.csv", manifest, OUTPUT_COLUMNS)

    per_dataset_dir = TABLE_DIR / "per_dataset"
    for series in SERIES:
        rows = [row for row in manifest if row["series"] == series]
        write_csv_new(per_dataset_dir / f"{series}_all_samples_inclusion.csv", rows, OUTPUT_COLUMNS)

    summary: list[dict[str, object]] = []
    for series in SERIES:
        rows = [row for row in manifest if row["series"] == series]
        used = [row for row in rows if row["used"]]
        reasons = Counter(str(row["decision_reason"]) for row in rows if not row["used"])
        summary.append({
            "series": series,
            "modality": MODALITY[series],
            "n_all_geo_samples": len(rows),
            "n_used": len(used),
            "n_not_used": len(rows) - len(used),
            "all_local_data_present": all(bool(row["local_data_present"]) for row in rows),
            "used_sample_ids": ";".join(str(row["sample"]) for row in used),
            "not_used_reason_counts": " | ".join(f"{key}={value}" for key, value in sorted(reasons.items())),
        })
    write_csv_new(
        TABLE_DIR / "sample_inclusion_summary_by_dataset.csv",
        summary,
        [
            "series", "modality", "n_all_geo_samples", "n_used", "n_not_used",
            "all_local_data_present", "used_sample_ids", "not_used_reason_counts",
        ],
    )

    checksums = [
        {
            "file": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(SOURCE_DIR.rglob("*"))
        if path.is_file()
    ]
    write_csv_new(
        TABLE_DIR / "source_metadata_sha256.csv",
        checksums,
        ["file", "size_bytes", "sha256"],
    )

    validation = [
        {"check": "all_geo_samples_have_one_manifest_row", "passed": len(manifest) == sum(int(row["n_all_geo_samples"]) for row in summary), "value": len(manifest)},
        {"check": "sample_ids_unique_within_series", "passed": len({(row["series"], row["sample"]) for row in manifest}) == len(manifest), "value": len(manifest)},
        {"check": "all_local_data_present", "passed": all(bool(row["local_data_present"]) for row in manifest), "value": sum(bool(row["local_data_present"]) for row in manifest)},
        {"check": "used_samples_have_allowed_tissue", "passed": all(bool(row["allowed_tissue_scope"]) for row in manifest if row["used"]), "value": sum(bool(row["used"]) for row in manifest)},
        {"check": "used_cancer_contexts_are_explicitly_untreated", "passed": all(bool(row["treatment_eligible"]) for row in manifest if row["used"]), "value": sum(bool(row["used"]) for row in manifest)},
        {"check": "used_samples_have_no_documented_distant_metastatic_context", "passed": all(not bool(row["distant_metastatic_patient_context"]) for row in manifest if row["used"]), "value": sum(bool(row["used"]) for row in manifest)},
    ]
    write_csv_new(TABLE_DIR / "sample_manifest_validation.csv", validation, ["check", "passed", "value"])

    if not all(bool(row["passed"]) for row in validation):
        failed = [str(row["check"]) for row in validation if not row["passed"]]
        raise RuntimeError("Sample-manifest validation failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()

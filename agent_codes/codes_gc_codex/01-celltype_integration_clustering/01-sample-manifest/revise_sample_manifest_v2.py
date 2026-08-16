#!/usr/bin/env python3
"""Create a non-destructive v2 manifest correcting the GSE270680 D27L row.

GSM8350021 is explicitly titled "Lymph Node" and its three local matrix
members use the D27L stem. Its GEO ``source_name``/``tissue`` field says
gastric cancer tumor, which caused v1 to include it. For the tissue-scope
contract, the concordant title and file stem take precedence over that one
contradictory field.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
TABLE_DIR = ROOT / "epi-cm-core-workflow/tables/01-celltype_integration_clustering/01-sample-manifest"
INPUT = TABLE_DIR / "all_datasets_all_samples_inclusion.csv"
OUTPUT = TABLE_DIR / "all_datasets_all_samples_inclusion_v2.csv"


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def write_new(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


with INPUT.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    columns = list(reader.fieldnames or [])
    rows = list(reader)

target = [row for row in rows if row["series"] == "GSE270680" and row["sample"] == "GSM8350021"]
if len(target) != 1:
    raise RuntimeError(f"Expected exactly one GSM8350021 row, found {len(target)}")
row = target[0]
if "Lymph Node" not in row["sample_title"] or not re.search(r"_D27L_", row["source_matrix_path"]):
    raise RuntimeError("GSM8350021 title/file evidence no longer matches the audited D27L correction")
if not as_bool(row["used"]):
    raise RuntimeError("GSM8350021 is no longer used in v1; inspect before applying this revision")

old = dict(row)
row.update(
    {
        "tissue_class": "lymph_node",
        "status": "excluded",
        "allowed_tissue_scope": "False",
        "used": "False",
        "decision": "not_used",
        "decision_reason": "out_of_scope_tissue:lymph_node",
        "metadata_evidence_source": row["metadata_evidence_source"]
        + ";sample_title=Lymph Node;local_matrix_stem=D27L;v2_scope_correction",
    }
)

write_new(OUTPUT, rows, columns)

series_order = list(dict.fromkeys(item["series"] for item in rows))
per_dataset = TABLE_DIR / "per_dataset"
for series in series_order:
    write_new(
        per_dataset / f"{series}_all_samples_inclusion_v2.csv",
        [item for item in rows if item["series"] == series],
        columns,
    )

summary = []
for series in series_order:
    subset = [item for item in rows if item["series"] == series]
    used = [item for item in subset if as_bool(item["used"])]
    reasons = Counter(item["decision_reason"] for item in subset if not as_bool(item["used"]))
    summary.append(
        {
            "series": series,
            "modality": subset[0]["modality"],
            "n_all_geo_samples": len(subset),
            "n_used": len(used),
            "n_not_used": len(subset) - len(used),
            "all_local_data_present": all(as_bool(item["local_data_present"]) for item in subset),
            "used_sample_ids": ";".join(item["sample"] for item in used),
            "not_used_reason_counts": " | ".join(
                f"{key}={value}" for key, value in sorted(reasons.items())
            ),
        }
    )
write_new(
    TABLE_DIR / "sample_inclusion_summary_by_dataset_v2.csv",
    summary,
    [
        "series",
        "modality",
        "n_all_geo_samples",
        "n_used",
        "n_not_used",
        "all_local_data_present",
        "used_sample_ids",
        "not_used_reason_counts",
    ],
)

used_rows = [item for item in rows if as_bool(item["used"])]
gse270680_used = [item for item in used_rows if item["series"] == "GSE270680"]
tn_titles_and_members = all(
    (", Tumor" in item["sample_title"] or ", Normal" in item["sample_title"])
    and re.search(r"_D\d+[TN]_", item["source_matrix_path"])
    for item in gse270680_used
)
validation = [
    {
        "check": "all_geo_samples_have_one_manifest_row",
        "passed": len(rows) == sum(int(item["n_all_geo_samples"]) for item in summary),
        "value": len(rows),
    },
    {
        "check": "sample_ids_unique_within_series",
        "passed": len({(item["series"], item["sample"]) for item in rows}) == len(rows),
        "value": len(rows),
    },
    {
        "check": "all_local_data_present",
        "passed": all(as_bool(item["local_data_present"]) for item in rows),
        "value": sum(as_bool(item["local_data_present"]) for item in rows),
    },
    {
        "check": "used_samples_have_allowed_tissue",
        "passed": all(as_bool(item["allowed_tissue_scope"]) for item in used_rows),
        "value": len(used_rows),
    },
    {
        "check": "used_cancer_contexts_are_explicitly_untreated",
        "passed": all(as_bool(item["treatment_eligible"]) for item in used_rows),
        "value": len(used_rows),
    },
    {
        "check": "used_samples_have_no_documented_distant_metastatic_context",
        "passed": all(not as_bool(item["distant_metastatic_patient_context"]) for item in used_rows),
        "value": len(used_rows),
    },
    {
        "check": "gse270680_used_samples_are_explicit_T_or_N_by_title_and_matrix_stem",
        "passed": tn_titles_and_members,
        "value": len(gse270680_used),
    },
]
write_new(
    TABLE_DIR / "sample_manifest_validation_v2.csv",
    validation,
    ["check", "passed", "value"],
)

revision = [
    {
        "series": row["series"],
        "sample": row["sample"],
        "sample_title": row["sample_title"],
        "local_matrix_token": "D27L",
        "old_tissue_class": old["tissue_class"],
        "new_tissue_class": row["tissue_class"],
        "old_used": old["used"],
        "new_used": row["used"],
        "revision_reason": "Concordant sample title and local matrix stem identify lymph-node tissue; out of allowed stomach tissue scope.",
    }
]
write_new(
    TABLE_DIR / "manifest_revision_v1_to_v2.csv",
    revision,
    list(revision[0]),
)

if not all(as_bool(item["passed"]) for item in validation):
    raise RuntimeError("One or more v2 manifest checks failed")

used_sc = sum(as_bool(item["used"]) and item["modality"] == "scRNA-seq" for item in rows)
used_spatial = sum(
    as_bool(item["used"]) and item["modality"].lower().startswith("spatial") for item in rows
)
print(f"v2 manifest passed: {len(rows)} total, {used_sc} used scRNA-seq, {used_spatial} used spatial")

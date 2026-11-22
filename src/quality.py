"""Data Quality Index (DQI): deterministic 0-100 completeness/validity score per record.

Applied to records *after* they are extracted from FHIR (or HL7 v2), so a missing value
means the corresponding FHIR element / HL7 field was absent. Shared by the training
pipeline, the REST API and the HL7 listener so every path uses identical rules.
"""
import math

from src.config import load_config

ISSUE_LABELS = {
    "missing_race": "Race/ethnicity not recorded",
    "missing_bmi": "BMI observation missing",
    "out_of_bounds_bmi": "BMI implausible",
    "missing_age": "Birth date missing",
    "out_of_bounds_hba1c": "HbA1c missing or implausible",
}


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def score_record(race, bmi, hba1c, age) -> dict:
    """Return {"score": int, "issues": [str]} for one record."""
    config = load_config()
    w = config["quality_weights"]
    b = config["quality_bounds"]

    score = 100
    issues = []

    if _is_missing(race):
        score -= w["missing_race_penalty"]
        issues.append("missing_race")

    if _is_missing(bmi):
        score -= w["missing_bmi_penalty"]
        issues.append("missing_bmi")
    elif not (b["bmi_min"] <= bmi <= b["bmi_max"]):
        score -= w["out_of_bounds_bmi_penalty"]
        issues.append("out_of_bounds_bmi")

    if _is_missing(age):
        score -= w["missing_age_penalty"]
        issues.append("missing_age")

    if _is_missing(hba1c) or not (b["hba1c_min"] <= hba1c <= b["hba1c_max"]):
        score -= w["out_of_bounds_hba1c_penalty"]
        issues.append("out_of_bounds_hba1c")

    return {"score": max(score, 0), "issues": issues}


def score_dataframe(df):
    """Add `data_quality_index` and `data_quality_issues` columns (returns a copy)."""
    out = df.copy()
    results = [
        score_record(r, b, h, a)
        for r, b, h, a in zip(out["race_demographic"], out["bmi"], out["baseline_hba1c"], out["age"])
    ]
    out["data_quality_index"] = [r["score"] for r in results]
    out["data_quality_issues"] = [", ".join(r["issues"]) for r in results]
    return out

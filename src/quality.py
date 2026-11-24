"""Record-level Data Quality Index (0-100) for a single flat record.

Used where one record arrives on its own (REST /predict, FHIR $predict, the HL7 listener).
Whole test patients are scored by the validation script (src/stdc/validation.py) instead.
"""
import math

from src import terminology as t
from src.config import load_config

ISSUE_LABELS = {"missing_race_ethnicity": "Race/ethnicity not recorded", "missing_birthdate": "Birth date missing",
                "missing_sex": "Sex not recorded",
                **{f"missing_{k}": f"{k.replace('_', ' ')} missing" for k in t.LABS},
                **{f"implausible_{k}": f"{k.replace('_', ' ')} implausible" for k in t.LABS}}


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def score_record(rec: dict) -> dict:
    cfg = load_config()
    w, bounds = cfg["quality_weights"], cfg["quality_bounds"]
    score, issues = 100, []
    if _missing(rec.get("race_ethnicity")):
        score -= w["missing_race_ethnicity"]
        issues.append("missing_race_ethnicity")
    if _missing(rec.get("birth_date")) and _missing(rec.get("age")):
        score -= w["missing_birthdate"]
        issues.append("missing_birthdate")
    if _missing(rec.get("sex") or rec.get("gender")):
        score -= w["missing_sex"]
        issues.append("missing_sex")
    for key in t.LABS:
        v = rec.get(key)
        if _missing(v):
            score -= w["missing_lab"]
            issues.append(f"missing_{key}")
        elif not bounds[key][0] <= v <= bounds[key][1]:
            score -= w["implausible_lab"]
            issues.append(f"implausible_{key}")
    return {"score": max(score, 0), "issues": issues, "labels": [ISSUE_LABELS[i] for i in issues]}

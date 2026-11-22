"""Data Quality Index (DQI): deterministic 0-100 completeness/validity score per record.

Shared by the training pipeline (batch scoring) and the API (per-request scoring)
so both use identical rules.
"""
import math
from typing import Optional

from src.config import load_config


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def score_record(race: Optional[str], bmi: Optional[float], hba1c: Optional[float]) -> dict:
    """Return the DQI score and the list of issues found for one record."""
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

    if _is_missing(hba1c) or not (b["hba1c_min"] <= hba1c <= b["hba1c_max"]):
        score -= w["out_of_bounds_hba1c_penalty"]
        issues.append("out_of_bounds_hba1c")

    return {"score": max(score, 0), "issues": issues}


def score_dataframe(df):
    """Add a `data_quality_index` column to a DataFrame (returns a copy)."""
    out = df.copy()
    out["data_quality_index"] = [
        score_record(r, b, h)["score"]
        for r, b, h in zip(out["race_demographic"], out["bmi"], out["baseline_hba1c"])
    ]
    return out

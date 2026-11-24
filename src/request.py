"""Step 01 · The test-data request ("Test Patient Data: 100 Records Requested").

A request names the cohorts, their share of the delivery, who is eligible (age, chronic
condition groups, history depth) and the resource targets per patient. Optionally it carries
the demographic mix and data-quality fault rates of the requester's real population, taken
from an aggregate profile produced by tools/data_quality_profile.py.

Defaults reproduce the reference request: 75 typical/acute (Cohort A) + 25 chronic/complex
(Cohort B) per 100 records, US Core, strict in-bundle referential integrity, 3+ years of
continuous history for Cohort B.
"""
import copy
from datetime import date

from src.stdc.config import CohortSpec, _cohort

NOT_RECORDED = "Not recorded"

COHORT_A = {
    "name": "A", "label": "Typical / acute care", "pool": "A", "share": 0.75,
    "min_age": 18, "max_age": 64, "min_condition_groups": 0, "exclude_condition_groups": True,
    "min_history_years": 0, "max_gap_days": None,
    "purpose": "Baseline response times for standard encounters.",
    "targets": {
        "Patient": {"min": 1, "max": 1}, "Encounter": {"min": 2, "max": 5}, "Condition": {"min": 1, "max": 3},
        "AllergyIntolerance": {"min": 0, "max": 2}, "MedicationRequest|MedicationStatement": {"min": 1, "max": 4},
        "Observation": {"min": 20, "max": 50}, "Procedure": {"min": 0, "max": 2},
        "Immunization": {"min": 5, "max": 10}, "DiagnosticReport": {"min": 1, "max": 3},
        "CarePlan": {"min": 0, "max": 1}, "CareTeam": {"min": 0, "max": 1},
    },
}
COHORT_B = {
    "name": "B", "label": "Chronic / complex care", "pool": "B", "share": 0.25,
    "min_age": 50, "max_age": 90, "min_condition_groups": 2, "exclude_condition_groups": False,
    "min_history_years": 3, "max_gap_days": 365,
    "purpose": "Stress-test large $everything payloads and pagination.",
    "targets": {
        "Patient": {"min": 1, "max": 1}, "Encounter": {"min": 30, "max": None}, "Condition": {"min": 15, "max": None},
        "AllergyIntolerance": {"min": 5, "max": 10, "severity": "warning"},
        "MedicationRequest|MedicationStatement": {"min": 20, "max": None},
        "Observation": {"min": 500, "max": None}, "Procedure": {"min": 10, "max": None},
        "Immunization": {"min": 10, "max": 15}, "DiagnosticReport": {"min": 20, "max": None},
        "CarePlan": {"min": 2, "max": 5}, "CareTeam": {"min": 2, "max": 5},
    },
}
RESOURCE_ROWS = ["Patient", "Encounter", "Condition", "AllergyIntolerance", "MedicationRequest|MedicationStatement",
                 "Observation", "Procedure", "Immunization", "DiagnosticReport", "CarePlan", "CareTeam"]
STRESS_FACTOR = {
    "Patient": "Baseline record retrieval", "Encounter": "Relational query depth",
    "Condition": "Problem list aggregation", "AllergyIntolerance": "Secondary resource loading",
    "MedicationRequest|MedicationStatement": "High-volume history retrieval",
    "Observation": "Primary pagination trigger", "Procedure": "Linked resource resolution",
    "Immunization": "Standard history retrieval", "DiagnosticReport": "Complex relational linking",
    "CarePlan": "Multi-resource aggregation", "CareTeam": "Multi-resource aggregation",
}

# Faults that can be injected to mirror the requester's data quality (profile key -> fault)
FAULTS = {
    "missing_birthdate": ("Patient.birthDate missing", "V01 fails; not repairable"),
    "missing_race_ethnicity": ("Race/ethnicity extensions missing", "reported in fidelity; no check fails"),
    "dangling_reference": ("A reference points outside the bundle", "V03 fails; repaired"),
    "missing_obs_category": ("An Observation has no category", "V04 fails; not repairable"),
    "reversed_period": ("An Encounter ends before it starts", "V07 fails; repaired"),
}


def default_request() -> dict:
    return {
        "label": "VAL test patients",
        "delivery_size": 100,
        "max_per_cohort": 363,          # evaluation sample per cohort (all eligible, up to this)
        "seed": 20261102,
        "as_of": date.today().isoformat(),
        "cohorts": [copy.deepcopy(COHORT_A), copy.deepcopy(COHORT_B)],
        "demographics": None,           # {"sex": {...%}, "race_ethnicity": {...%}} → selection quotas
        "faults": {k: 0 for k in FAULTS},  # % of patients
        "acceptance": {"h0_rate": 0.85, "target_rate": 0.90, "alpha": 0.05},
        "split": {"train": 0.6, "test": 0.2, "validation": 0.2},
        "source_profile": None,
    }


def cohort_specs(request: dict) -> list[CohortSpec]:
    return [_cohort(c) for c in request["cohorts"]]


def problems(request: dict) -> list[str]:
    out = []
    total = sum(c["share"] for c in request["cohorts"])
    if abs(total - 1) > 1e-6:
        out.append(f"Cohort shares add up to {total:.0%}, not 100%.")
    for c in request["cohorts"]:
        if c["min_age"] > c["max_age"]:
            out.append(f"Cohort {c['name']}: minimum age is above maximum age.")
        for rtype, t in c["targets"].items():
            if t.get("max") is not None and t["max"] < t["min"]:
                out.append(f"Cohort {c['name']} {rtype}: max is below min.")
    s = request["split"]
    if abs(sum(s.values()) - 1) > 1e-6:
        out.append("Train/test/validation split must add up to 100%.")
    return out


def _shares(counts: dict, k_min: int, notes: list, name: str, drop=()) -> dict:
    est = {}
    for key, c in counts.items():
        if key in drop:
            continue
        if c is None:
            est[key] = k_min / 2
            notes.append(f"{name} '{key}' was suppressed (<{k_min}); estimated at {k_min / 2:g} patients.")
        else:
            est[key] = c
    total = sum(est.values())
    return {k: round(100 * v / total, 2) for k, v in est.items()} if total else {}


def apply_profile(request: dict, profile: dict) -> tuple[dict, list[str]]:
    """Copy the demographic mix and data-quality fault rates of a real-data profile into a request."""
    req = copy.deepcopy(request)
    notes = []
    k = profile.get("k_min", 11)
    n = profile["n_patients"]
    c = profile["counts"]
    req["demographics"] = {
        "sex": _shares({s: v for s, v in c["sex"].items() if s in ("female", "male")}, k, notes, "Sex"),
        "race_ethnicity": _shares(c["race_ethnicity"], k, notes, "Race/ethnicity"),
    }
    q = profile["quality"]
    for fault in FAULTS:
        count = q.get(fault, 0)
        if count is None:
            notes.append(f"Quality '{fault}' was suppressed (<{k}); estimated at {k / 2:g} patients.")
            count = k / 2
        req["faults"][fault] = round(100 * count / n, 2)
    req["source_profile"] = profile.get("source", "uploaded profile")
    req["label"] = f"Matched to {req['source_profile']}"
    return req, notes

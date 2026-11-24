"""Aggregate-only reference profile, for comparing Synthea data with real data.

A profile only ever holds aggregates, never patient rows. Categories seen
fewer than k_min_cell_size times are suppressed before anything is stored,
and minimum/maximum values are not kept.

Profile layout:
    {
      "source": "cdm" | "records",
      "n_patients": int,
      "k_min": int,
      "quantile_points": [0.05, ...],
      "categorical": {"gender": {"female": 0.53, ...}, "condition_group": {"t2d": 0.12, ...}},
      "numeric": {"age": [q05, ..., q95], "n_encounter": [...], ...},
      "correlation": {"features": [...], "matrix": [[...], ...]}
    }

condition_group proportions are the share of patients in each group, so they
compare across terminologies (SNOMED in Synthea, ICD-10 in many CDMs).
"""

import re
import statistics

from . import fhir
from .synthea import age_on, condition_groups

QUANTILE_POINTS = [0.05, 0.25, 0.50, 0.75, 0.95]
COUNTED_TYPES = ["Encounter", "Condition", "Observation", "MedicationRequest", "Procedure",
                 "Immunization", "DiagnosticReport", "AllergyIntolerance"]
FEATURES = ["age"] + [f"n_{t.lower()}" for t in COUNTED_TYPES]
CORRELATED = ["age", "n_encounter", "n_condition", "n_observation", "n_medicationrequest"]

IDENTIFIER_PATTERN = re.compile(r"(name|mrn|ssn|birth|dob|address|phone|email|patient_id|identifier)", re.I)


def quantiles(values, points=QUANTILE_POINTS):
    """Linear-interpolated quantiles (numpy's default method)."""
    xs = sorted(values)
    if not xs:
        raise ValueError("cannot take quantiles of an empty list")
    out = []
    for p in points:
        pos = p * (len(xs) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        out.append(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))
    return out


def suppress_counts(counts, k_min, total=None):
    """Drop categories under k_min. Proportions are of total (default: sum of kept)."""
    kept = {k: n for k, n in counts.items() if n >= k_min}
    denom = total if total is not None else sum(kept.values())
    if not denom:
        return {}
    return {k: n / denom for k, n in sorted(kept.items(), key=lambda kv: -kv[1])}


def assert_aggregate_only(profile):
    """Raise if a profile looks like it carries row-level or identifying data."""
    for section in ("categorical", "numeric"):
        for key in profile.get(section, {}):
            if IDENTIFIER_PATTERN.search(str(key)):
                raise ValueError(f"profile field '{key}' looks like an identifier")
    for name, dist in profile.get("categorical", {}).items():
        if not isinstance(dist, dict) or not all(0 <= p <= 1 for p in dist.values()):
            raise ValueError(f"categorical '{name}' must map values to proportions")
    for name, qs in profile.get("numeric", {}).items():
        if len(qs) != len(profile["quantile_points"]):
            raise ValueError(f"numeric '{name}' must hold exactly the profile quantiles")
    if profile.get("n_patients", 0) < profile["k_min"]:
        raise ValueError("profile built from fewer patients than the minimum cell size")


def extract_features(bundle, as_of, groups):
    p = fhir.patient(bundle) or {}
    counts = {t: 0 for t in COUNTED_TYPES}
    for r in fhir.resources(bundle):
        if r.get("resourceType") in counts:
            counts[r["resourceType"]] += 1
    return {
        "age": age_on(p.get("birthDate"), as_of),
        "gender": p.get("gender"),
        "condition_groups": condition_groups(bundle, groups),
        **{f"n_{t.lower()}": n for t, n in counts.items()},
    }


def correlation_matrix(rows, features=CORRELATED):
    complete = [r for r in rows if all(r.get(f) is not None for f in features)]
    cols = [[float(r[f]) for r in complete] for f in features]
    k = len(features)
    m = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            try:
                m[i][j] = m[j][i] = statistics.correlation(cols[i], cols[j])
            except statistics.StatisticsError:
                pass
    return m


def build_profile_from_features(feats, k_min, source="records"):
    n = len(feats)
    gender, groups = {}, {}
    for f in feats:
        gender[f["gender"]] = gender.get(f["gender"], 0) + 1
        for g in f["condition_groups"]:
            groups[g] = groups.get(g, 0) + 1
    profile = {
        "source": source,
        "n_patients": n,
        "k_min": k_min,
        "quantile_points": QUANTILE_POINTS,
        "categorical": {
            "gender": suppress_counts(gender, k_min),
            "condition_group": suppress_counts(groups, k_min, total=n),
        },
        "numeric": {name: quantiles([f[name] for f in feats if f[name] is not None])
                    for name in FEATURES},
        "correlation": {"features": CORRELATED, "matrix": correlation_matrix(feats)},
    }
    assert_aggregate_only(profile)
    return profile

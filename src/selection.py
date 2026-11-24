"""Select each cohort's patients from its Synthea pool, and optionally inject data-quality faults.

Eligibility uses only the clinical cohort definition (alive, age, chronic condition groups), as
in the reference project: resource volumes, history depth and conformance are what validation
measures, so they are deliberately not used to pick patients.

When the request carries a demographic mix (from a real-data profile), patients are drawn by
race/ethnicity quota. If a pool has too few patients of a group, the shortfall is filled from the
other groups and reported — Synthea follows the census of the simulated state, so small groups
can run short.
"""
import random

from src.request import NOT_RECORDED


def eligible(row: dict, cohort: dict) -> bool:
    if not row["alive"] or row["age"] is None:
        return False
    if not cohort["min_age"] <= row["age"] <= cohort["max_age"]:
        return False
    n = len(row["condition_groups"])
    if cohort["exclude_condition_groups"] and n:
        return False
    return n >= cohort["min_condition_groups"]


def _quota(n: int, shares: dict) -> dict:
    total = sum(shares.values()) or 1
    raw = {k: n * v / total for k, v in shares.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def select_cohort(rows: list[dict], cohort: dict, n_max: int, demographics: dict | None,
                  seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(f"{seed}:{cohort['name']}")
    pool = sorted((r for r in rows if eligible(r, cohort)), key=lambda r: r["patient_id"])
    n = min(n_max, len(pool))
    report = {"eligible": len(pool), "selected": n, "shortfall": {}}
    mix = (demographics or {}).get("race_ethnicity")
    if not mix:
        return rng.sample(pool, n), report

    by_group = {}
    for r in pool:
        by_group.setdefault(r["race_ethnicity"], []).append(r)
    chosen = []
    for group, want in _quota(n, mix).items():
        have = by_group.get(group, [])
        take = rng.sample(have, min(want, len(have)))
        chosen += take
        if len(take) < want:
            report["shortfall"][group] = want - len(take)
    rest = [r for r in pool if r not in chosen]
    chosen += rng.sample(rest, min(n - len(chosen), len(rest)))
    return chosen, report


# ----------------------------------------------------------------------------- faults

def _of_type(bundle, rtype):
    return [e["resource"] for e in bundle.get("entry", []) if e["resource"].get("resourceType") == rtype]


def inject(bundle: dict, fault: str, rng: random.Random) -> bool:
    """Apply one fault in place. Returns False when the record has nothing to break."""
    patient = _of_type(bundle, "Patient")[0]
    if fault == "missing_birthdate":
        return patient.pop("birthDate", None) is not None
    if fault == "missing_race_ethnicity":
        before = len(patient.get("extension", []))
        patient["extension"] = [e for e in patient.get("extension", [])
                                if not e.get("url", "").endswith(("-race", "-ethnicity"))]
        return len(patient["extension"]) < before
    observations = [o for o in _of_type(bundle, "Observation") if o.get("encounter")]
    if fault == "dangling_reference" and observations:
        rng.choice(observations)["encounter"] = {"reference": "urn:uuid:00000000-0000-0000-0000-000000000000"}
        return True
    if fault == "missing_obs_category" and observations:
        rng.choice(observations).pop("category", None)
        return True
    encounters = [e for e in _of_type(bundle, "Encounter") if e.get("period", {}).get("end")]
    if fault == "reversed_period" and encounters:
        enc = rng.choice(encounters)
        if enc["period"]["start"] != enc["period"]["end"]:
            enc["period"]["start"], enc["period"]["end"] = enc["period"]["end"], enc["period"]["start"]
            return True
    return False


def plan_faults(n: int, rates: dict, seed: int) -> list[list[str]]:
    """Which faults each of n patients receives (exact counts per fault)."""
    rng = random.Random(f"{seed}:faults")
    plan = [[] for _ in range(n)]
    for fault, pct in (rates or {}).items():
        for i in rng.sample(range(n), min(n, round(n * pct / 100))):
            plan[i].append(fault)
    return plan

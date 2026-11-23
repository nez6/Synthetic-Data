"""Independent audit of the test-data spec.

Written separately from validation.py and sharing no code with it or with the generator
(tests/test_testdata.py fails if this module imports either), so it can score the validation
script's output without trusting the script's own checks. Together with official FHIR R4
schema validation it gives the independent pass/fail, which is the result to report: the
script's own pass rate is partly guaranteed, because the script repairs what it checks.
"""
from datetime import date

LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"
A1C, BMI = "4548-4", "39156-5"


def _all_refs(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "reference" and isinstance(v, str):
                out.append(v)
            else:
                _all_refs(v, out)
    elif isinstance(node, list):
        for v in node:
            _all_refs(v, out)
    return out


def _code(obs):
    for c in (obs.get("code") or {}).get("coding") or []:
        if c.get("system") == LOINC:
            return c.get("code")
    return None


def audit_bundle(bundle: dict, rules: dict, as_of: date) -> list[str]:
    """Return failure messages (empty = pass).

    rules: {"min_age", "max_age", "min_hba1c_obs", "min_bmi_obs", "hba1c_bounds", "bmi_bounds",
            "synthetic_system"}
    """
    entries = bundle.get("entry") or []
    res = [e.get("resource") or {} for e in entries]
    failures = []

    patients = [r for r in res if r.get("resourceType") == "Patient"]
    if len(patients) != 1:
        return [f"{len(patients)} Patient resources"]
    p = patients[0]
    if not any(i.get("system") == rules["synthetic_system"] for i in p.get("identifier") or []):
        failures.append("no synthetic identifier")
    if "HTEST" not in {s.get("code") for s in (p.get("meta") or {}).get("security") or []}:
        failures.append("no HTEST label")
    try:
        b = date.fromisoformat(p["birthDate"])
        age = as_of.year - b.year - ((as_of.month, as_of.day) < (b.month, b.day))
        if age < rules["min_age"] or age > rules["max_age"]:
            failures.append(f"age {age} outside {rules['min_age']}-{rules['max_age']}")
    except (KeyError, ValueError):
        failures.append("no valid birthDate")

    ids = {f"{r.get('resourceType')}/{r.get('id')}" for r in res} | {e.get("fullUrl") for e in entries}
    bad_refs = [x for x in _all_refs(res, []) if x not in ids]
    if bad_refs:
        failures.append(f"{len(bad_refs)} unresolved references")

    observations = [r for r in res if r.get("resourceType") == "Observation"]
    a1c = [o for o in observations if _code(o) == A1C]
    bmi = [o for o in observations if _code(o) == BMI]
    if len(a1c) < rules["min_hba1c_obs"]:
        failures.append(f"only {len(a1c)} HbA1c results")
    if len(bmi) < rules["min_bmi_obs"]:
        failures.append(f"only {len(bmi)} BMI results")
    for o, (lo, hi), label in [(o, rules["hba1c_bounds"], "HbA1c") for o in a1c] + \
                              [(o, rules["bmi_bounds"], "BMI") for o in bmi]:
        q = o.get("valueQuantity") or {}
        if q.get("system") != UCUM:
            failures.append(f"{label} unit not UCUM")
        v = q.get("value")
        if not isinstance(v, (int, float)) or not lo <= v <= hi:
            failures.append(f"{label} value {v} implausible")
    if any(not o.get("category") for o in observations):
        failures.append("Observation without category")

    for r in res:
        per = r.get("period") or {}
        if per.get("start") and per.get("end") and per["end"] < per["start"]:
            failures.append(f"{r.get('resourceType')} period reversed")
    return sorted(set(failures))

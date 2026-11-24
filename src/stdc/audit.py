"""Independent audit of the VAL test data specification.

Written separately from validation.py and fhir.py and sharing no code with
them (tests/test_audit.py fails if it imports either), so it can score the
validation script's output without trusting the script's own checks. It reads
the raw JSON config and bundles and re-derives everything itself.

Together with the HL7 FHIR Validator (US Core conformance, run by
scripts/independent_check.py) it provides the independent pass/fail.

Rules re-checked:
  - exactly one Patient, carrying the Synthea identifier; SSNs are 999- test numbers
  - every reference resolves inside the bundle (urn:uuid, Type/id or #contained);
    conditional references only to the provider directory
  - resources per patient within the cohort's error-level min/max
  - cohort history depth and continuity
"""

from datetime import date, timedelta

DIRECTORY = {"Practitioner", "PractitionerRole", "Organization", "Location"}
SYNTHEA_SYSTEM = "https://github.com/synthetichealth/synthea"
SSN_SYSTEM = "http://hl7.org/fhir/sid/us-ssn"


def _references(node, found):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "reference" and isinstance(value, str):
                found.append(value)
            elif key != "contained":
                _references(value, found)
    elif isinstance(node, list):
        for item in node:
            _references(item, found)
    return found


def _encounter_dates(resources):
    out = []
    for r in resources:
        if r.get("resourceType") == "Encounter":
            start = (r.get("period") or {}).get("start", "")
            try:
                out.append(date.fromisoformat(start[:10]))
            except ValueError:
                pass
    return sorted(out)


def audit_bundle(bundle, cohort, as_of):
    """Return a list of failure messages (empty = pass).

    cohort is the cohort's dict from config/emrs.json; as_of is a date.
    """
    failures = []
    entries = bundle.get("entry", [])
    resources = [e.get("resource", {}) for e in entries]

    patients = [r for r in resources if r.get("resourceType") == "Patient"]
    if len(patients) != 1:
        return [f"{len(patients)} Patient resources"]
    ids = patients[0].get("identifier", [])
    if not any(i.get("system") == SYNTHEA_SYSTEM for i in ids):
        failures.append("Patient lacks the Synthea identifier")
    if any(i.get("system") == SSN_SYSTEM and not str(i.get("value", "")).startswith("999-") for i in ids):
        failures.append("SSN is not a 999- test number")

    known = set()
    for e in entries:
        r = e.get("resource", {})
        if e.get("fullUrl"):
            known.add(e["fullUrl"])
        known.add(f"{r.get('resourceType')}/{r.get('id')}")
    unresolved = 0
    for r in resources:
        contained = {"#" + str(c.get("id")) for c in r.get("contained", [])}
        for ref in _references(r, []):
            if ref.startswith("#"):
                ok = ref in contained
            elif "?" in ref:
                ok = ref.split("?", 1)[0] in DIRECTORY
            else:
                ok = ref in known
            unresolved += not ok
    if unresolved:
        failures.append(f"{unresolved} references do not resolve")

    for key, target in cohort["targets"].items():
        if target.get("severity", "error") != "error":
            continue
        types = set(key.split("|"))
        n = sum(1 for r in resources if r.get("resourceType") in types)
        if n < target["min"] or (target.get("max") is not None and n > target["max"]):
            failures.append(f"{key}: {n} outside {target['min']}-{target.get('max') or 'any'}")

    years = cohort.get("min_history_years") or 0
    if years:
        start = as_of - timedelta(days=round(years * 365.25))
        dates = _encounter_dates(resources)
        if not dates or dates[0] > start:
            failures.append(f"history shorter than {years} years")
        elif cohort.get("max_gap_days"):
            points = [start] + [d for d in dates if d > start]
            gap = max((b - a).days for a, b in zip(points, points[1:])) if len(points) > 1 else (as_of - start).days
            if gap > cohort["max_gap_days"]:
                failures.append(f"{gap}-day gap in recent history")
    return failures

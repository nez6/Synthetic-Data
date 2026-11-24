"""Flatten a FHIR Bundle (any supported version) into one flat, model-ready record.

Values are taken as recorded (the latest value per vital/lab), *including* implausible ones:
the raw-data model is meant to see the data's problems. Missing elements come back as None.
"""
from datetime import date, datetime

from src import terminology as t


def _resources(bundle: dict, resource_type: str) -> list[dict]:
    return [e["resource"] for e in bundle.get("entry", [])
            if e.get("resource", {}).get("resourceType") == resource_type]


def _omb_code(ext: dict):
    for sub in ext.get("extension", []):
        if sub.get("url") == "ombCategory":
            return sub.get("valueCoding", {}).get("code")
    return None


def race_ethnicity(patient: dict):
    race = ethnicity = None
    for ext in patient.get("extension", []):
        url = ext.get("url", "")
        if url.endswith("-race"):
            race = _omb_code(ext)
        elif url.endswith("-ethnicity"):
            ethnicity = _omb_code(ext)
    if ethnicity == t.HISPANIC[0]:
        return t.HISPANIC_GROUP
    return t.RACE_BY_CODE.get(race)


def age_on(birth_date, on: date):
    if not birth_date:
        return None
    try:
        b = date.fromisoformat(str(birth_date)[:10])
    except ValueError:
        return None
    return on.year - b.year - ((on.month, on.day) < (b.month, b.day))


def _loincs(cc):
    return {c.get("code") for c in (cc or {}).get("coding", []) if c.get("system") == t.LOINC}


def _language(patient):
    for comm in patient.get("communication", []):
        for c in comm.get("language", {}).get("coding", []):
            code = (c.get("code") or "").lower()
            if code in ("en", "en-us"):
                return "en-US"
            if code in t.LANGUAGES:
                return code
            if code:
                return "other"
    return None


def bundle_to_record(bundle: dict, as_of: date | None = None) -> dict:
    patients = _resources(bundle, "Patient")
    patient = patients[0] if patients else {}

    latest = {}       # lab key -> (when, value)
    smoking = (None, None)
    for obs in _resources(bundle, "Observation"):
        if obs.get("status") in ("entered-in-error", "cancelled"):
            continue
        when = obs.get("effectiveDateTime") or ""
        codes = _loincs(obs.get("code"))
        if t.SMOKING_LOINC[0] in codes:
            code = next((c.get("code") for c in obs.get("valueCodeableConcept", {}).get("coding", [])), None)
            if when >= (smoking[0] or ""):
                smoking = (when, t.SMOKING_BY_CODE.get(code))
        parts = [(codes, obs.get("valueQuantity"))] + [(_loincs(c.get("code")), c.get("valueQuantity"))
                                                       for c in obs.get("component", [])]
        for part_codes, q in parts:
            for code in part_codes:
                key = t.LAB_BY_CODE.get(code)
                if key and q and isinstance(q.get("value"), (int, float)) and when >= latest.get(key, ("",))[0]:
                    latest[key] = (when, q["value"])

    conditions = set()
    for cond in _resources(bundle, "Condition"):
        for c in cond.get("code", {}).get("coding", []):
            if c.get("code") in t.CONDITION_BY_CODE:
                conditions.add(t.CONDITION_BY_CODE[c["code"]])

    effective = max((w for w, _ in latest.values()), default=None) or None
    if as_of is None:
        as_of = datetime.fromisoformat(effective.replace("Z", "+00:00")).date() if effective else date.today()
    address = (patient.get("address") or [{}])[0]

    record = {
        "patient_id": patient.get("id"),
        "gender": patient.get("gender"),
        "birth_date": patient.get("birthDate"),
        "effective": effective,
        "age": age_on(patient.get("birthDate"), as_of),
        "sex": patient.get("gender"),
        "race_ethnicity": race_ethnicity(patient),
        "language": _language(patient),
        "state": address.get("state"),
        "smoking": smoking[1],
        **{k: latest[k][1] if k in latest else None for k in t.LABS},
        "conditions": sorted(conditions),
        "n_encounters": len(_resources(bundle, "Encounter")),
    }
    record["complex_care"] = t.complex_care(conditions)
    return record


def detect_version(bundle: dict) -> tuple[str, list[str]]:
    """Guess the FHIR version from structural fingerprints. Returns (version, evidence)."""
    evidence = []
    for enc in _resources(bundle, "Encounter"):
        cls = enc.get("class")
        if isinstance(cls, list) or "actualPeriod" in enc or enc.get("status") == "completed":
            evidence.append("Encounter.class is a list / actualPeriod / status 'completed' → R5")
            return "R5", evidence
        if isinstance(cls, str) or "patient" in enc:
            evidence.append("Encounter.class is a plain code / Encounter.patient → DSTU2")
            return "DSTU2", evidence
        if isinstance(cls, dict) and cls.get("system") == "http://hl7.org/fhir/v3/ActCode":
            evidence.append("Encounter.class uses the pre-R4 ActCode URL → STU3")
            return "STU3", evidence
    for obs in _resources(bundle, "Observation"):
        if isinstance(obs.get("category"), dict) or "comments" in obs:
            evidence.append("Observation.category is a single object → DSTU2")
            return "DSTU2", evidence
        if "context" in obs or "comment" in obs:
            evidence.append("Observation.context / Observation.comment → STU3")
            return "STU3", evidence
    for cond in _resources(bundle, "Condition"):
        if isinstance(cond.get("clinicalStatus"), str):
            evidence.append("Condition.clinicalStatus is a plain code → DSTU2/STU3")
            return ("DSTU2" if "patient" in cond else "STU3"), evidence
    evidence.append("No pre-R4 or R5 fingerprints found; R4 and R4B are identical for these resources")
    return "R4/R4B", evidence

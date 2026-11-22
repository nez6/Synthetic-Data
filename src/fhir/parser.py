"""Flatten a FHIR Bundle (any supported version) back into a model-ready record."""
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


def _demographic_group(patient: dict):
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


def age_on(birth_date: str, on: date):
    if not birth_date:
        return None
    b = date.fromisoformat(birth_date[:10])
    return on.year - b.year - ((on.month, on.day) < (b.month, b.day))


def _observation_value(observations: list[dict], loinc_code: str):
    for obs in observations:
        codes = {c.get("code") for c in obs.get("code", {}).get("coding", []) if c.get("system") == t.LOINC}
        if loinc_code in codes and obs.get("status") not in ("entered-in-error", "cancelled"):
            value = obs.get("valueQuantity", {}).get("value")
            return value, obs.get("effectiveDateTime")
    return None, None


def bundle_to_record(bundle: dict) -> dict:
    """Return a flat record. Missing elements come back as None (the DQI layer scores them)."""
    patients = _resources(bundle, "Patient")
    patient = patients[0] if patients else {}
    observations = _resources(bundle, "Observation")

    hba1c, effective = _observation_value(observations, t.HBA1C["code"])
    bmi, bmi_effective = _observation_value(observations, t.BMI["code"])
    effective = effective or bmi_effective
    as_of = datetime.fromisoformat(effective.replace("Z", "+00:00")).date() if effective else date.today()

    record = {
        "patient_id": patient.get("id"),
        "gender": patient.get("gender"),
        "birth_date": patient.get("birthDate"),
        "effective": effective,
        "race_demographic": _demographic_group(patient),
        "age": age_on(patient.get("birthDate"), as_of),
        "bmi": bmi,
        "baseline_hba1c": hba1c,
        "recommended_medication_plan": None,
        "recommended_meal_plan": None,
    }
    for cp in _resources(bundle, "CarePlan"):
        kinds = {c.get("text") for c in cp.get("category", [])}
        if "medication-plan" in kinds:
            record["recommended_medication_plan"] = cp.get("description")
        if "meal-plan" in kinds:
            record["recommended_meal_plan"] = cp.get("description")
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
    evidence.append("No pre-R4 or R5 fingerprints found; R4 and R4B are identical for these resources")
    return "R4/R4B", evidence

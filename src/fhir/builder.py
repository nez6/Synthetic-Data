"""Turn a flat patient record into a FHIR Bundle for a chosen FHIR version.

Record keys:
    patient_id, gender, birth_date (YYYY-MM-DD), effective (ISO datetime),
    race_demographic (str | None), bmi (float | None), baseline_hba1c (float | None),
    recommended_medication_plan / recommended_meal_plan (optional labels)

Fields that are missing in the record are *omitted* from the FHIR output (no race
extension, no BMI Observation), the same way a real source system would send them.
"""
from datetime import datetime, timedelta

from src import terminology as t
from src.fhir.versions import VERSIONS

BASE_URL = "https://example.org/fhir"

_OBS_CATEGORY_OLD = "http://hl7.org/fhir/observation-category"
_OBS_CATEGORY_NEW = "http://terminology.hl7.org/CodeSystem/observation-category"
_ACTCODE_OLD = "http://hl7.org/fhir/v3/ActCode"
_ACTCODE_NEW = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
_CATEGORY_DISPLAY = {"laboratory": "Laboratory", "vital-signs": "Vital Signs"}


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and not v.strip())


def _omb_extension(url: str, code: str, display: str) -> dict:
    return {
        "url": url,
        "extension": [
            {"url": "ombCategory",
             "valueCoding": {"system": t.CDC_RACE_ETHNICITY, "code": code, "display": display}},
            {"url": "text", "valueString": display},
        ],
    }


def _demographic_extensions(group, version: str) -> list[dict]:
    if _missing(group):
        return []
    if version == "DSTU2":
        race_url = "http://fhir.org/guides/argonaut/StructureDefinition/argo-race"
        eth_url = "http://fhir.org/guides/argonaut/StructureDefinition/argo-ethnicity"
    else:
        race_url = "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race"
        eth_url = "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity"

    if group == t.HISPANIC_GROUP:
        return [_omb_extension(eth_url, *t.HISPANIC)]
    code, display = t.RACE_CODES[group]
    return [_omb_extension(race_url, code, display), _omb_extension(eth_url, *t.NOT_HISPANIC)]


def _patient(rec: dict, version: str) -> dict:
    profile = ("http://fhir.org/guides/argonaut/StructureDefinition/argo-patient" if version == "DSTU2"
               else "http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient")
    patient = {
        "resourceType": "Patient",
        "id": rec["patient_id"],
        "meta": {"profile": [profile]},
    }
    ext = _demographic_extensions(rec.get("race_demographic"), version)
    if ext:
        patient["extension"] = ext
    patient["identifier"] = [{"system": "https://example.org/mrn", "value": rec["patient_id"]}]
    patient["name"] = [{"family": "Synthetic", "given": [f"Patient {rec['patient_id']}"]}]
    if rec.get("gender"):
        patient["gender"] = rec["gender"]
    if not _missing(rec.get("birth_date")):
        patient["birthDate"] = rec["birth_date"]
    return patient


def _encounter(rec: dict, version: str) -> dict:
    start = rec["effective"]
    end = (datetime.fromisoformat(start) + timedelta(minutes=30)).isoformat()
    enc = {"resourceType": "Encounter", "id": f"{rec['patient_id']}-enc"}
    patient_ref = {"reference": f"Patient/{rec['patient_id']}"}

    if version == "DSTU2":
        enc.update(status="finished", **{"class": "ambulatory"}, patient=patient_ref,
                   period={"start": start, "end": end})
    elif version == "STU3":
        enc.update(status="finished",
                   **{"class": {"system": _ACTCODE_OLD, "code": "AMB", "display": "ambulatory"}},
                   subject=patient_ref, period={"start": start, "end": end})
    elif version in ("R4", "R4B"):
        enc.update(status="finished",
                   **{"class": {"system": _ACTCODE_NEW, "code": "AMB", "display": "ambulatory"}},
                   subject=patient_ref, period={"start": start, "end": end})
    else:  # R5
        enc.update(status="completed",
                   **{"class": [{"coding": [{"system": _ACTCODE_NEW, "code": "AMB", "display": "ambulatory"}]}]},
                   subject=patient_ref, actualPeriod={"start": start, "end": end})
    return enc


def _observation(rec: dict, version: str, spec: dict, value: float, suffix: str, note: str = None) -> dict:
    cat_system = _OBS_CATEGORY_OLD if version in ("DSTU2", "STU3") else _OBS_CATEGORY_NEW
    category = {"coding": [{"system": cat_system, "code": spec["category"],
                            "display": _CATEGORY_DISPLAY[spec["category"]]}]}
    obs = {
        "resourceType": "Observation",
        "id": f"{rec['patient_id']}-{suffix}",
        "status": "final",
        "category": category if version == "DSTU2" else [category],
        "code": {"coding": [{"system": t.LOINC, "code": spec["code"], "display": spec["display"]}],
                 "text": spec["display"]},
        "subject": {"reference": f"Patient/{rec['patient_id']}"},
    }
    enc_ref = {"reference": f"Encounter/{rec['patient_id']}-enc"}
    obs["context" if version == "STU3" else "encounter"] = enc_ref
    obs["effectiveDateTime"] = rec["effective"]
    obs["valueQuantity"] = {"value": value, "unit": spec["unit"], "system": t.UCUM, "code": spec["unit"]}
    if note:
        if version == "DSTU2":
            obs["comments"] = note
        elif version == "STU3":
            obs["comment"] = note
        else:
            obs["note"] = [{"text": note}]
    return obs


def _careplan(rec: dict, version: str, kind: str, description: str) -> dict:
    cp = {"resourceType": "CarePlan", "id": f"{rec['patient_id']}-cp-{kind}", "status": "active"}
    if version != "DSTU2":
        cp["intent"] = "plan"
    cp["category"] = [{"text": f"{kind}-plan"}]
    cp["description"] = description
    cp["subject"] = {"reference": f"Patient/{rec['patient_id']}"}
    return cp


def build_bundle(rec: dict, version: str = "R4") -> dict:
    if version not in VERSIONS:
        raise ValueError(f"Unsupported FHIR version {version!r}; choose one of {list(VERSIONS)}")

    resources = [_patient(rec, version), _encounter(rec, version)]
    if not _missing(rec.get("baseline_hba1c")):
        resources.append(_observation(rec, version, t.HBA1C, float(rec["baseline_hba1c"]), "hba1c",
                                      note="Synthetic record generated for demonstration."))
    if not _missing(rec.get("bmi")):
        resources.append(_observation(rec, version, t.BMI, float(rec["bmi"]), "bmi"))
    if not _missing(rec.get("recommended_medication_plan")):
        resources.append(_careplan(rec, version, "medication", rec["recommended_medication_plan"]))
    if not _missing(rec.get("recommended_meal_plan")):
        resources.append(_careplan(rec, version, "meal", rec["recommended_meal_plan"]))

    bundle = {"resourceType": "Bundle", "id": f"{rec['patient_id']}-bundle", "type": "collection"}
    if version in ("R4", "R4B", "R5"):
        bundle["timestamp"] = rec["effective"]  # FHIR requires a timezone; generator always includes one
    bundle["entry"] = [
        {"fullUrl": f"{BASE_URL}/{r['resourceType']}/{r['id']}", "resource": r} for r in resources
    ]
    return bundle

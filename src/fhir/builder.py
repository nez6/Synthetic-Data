"""Turn a flat patient record into a one-visit FHIR Bundle for a chosen FHIR version.

Used for the FHIR-version comparison, `$convert`, HL7 → FHIR, and exporting a dataset in a
version other than R4 (the full multi-visit test records are R4; see src/testdata/generator.py).

Record keys (as produced by src/fhir/parser.py): patient_id, gender, birth_date, effective,
race_ethnicity, language, state, smoking, bmi, systolic_bp, hba1c, egfr, conditions.
Missing values are *omitted* from the output, the way a real source system would send them.
"""
from datetime import datetime, timedelta, timezone

from src import terminology as t
from src.fhir.versions import VERSIONS

BASE_URL = "https://example.org/fhir"

_OBS_CATEGORY_OLD = "http://hl7.org/fhir/observation-category"
_OBS_CATEGORY_NEW = "http://terminology.hl7.org/CodeSystem/observation-category"
_COND_CATEGORY_OLD = "http://hl7.org/fhir/condition-category"
_COND_CATEGORY_NEW = "http://terminology.hl7.org/CodeSystem/condition-category"
_ACTCODE_OLD = "http://hl7.org/fhir/v3/ActCode"
_ACTCODE_NEW = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
_CATEGORY_DISPLAY = {"laboratory": "Laboratory", "vital-signs": "Vital Signs", "social-history": "Social History"}


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


def demographic_extensions(group, version: str = "R4") -> list[dict]:
    if _missing(group) or (group != t.HISPANIC_GROUP and group not in t.RACE_CODES):
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


def language_element(code) -> dict | None:
    if _missing(code):
        return None
    display = t.LANGUAGES.get(code, "Other")
    real_code = "fr" if code == "other" else code
    return {"language": {"coding": [{"system": t.BCP47, "code": real_code, "display": display}], "text": display}}


def _patient(rec: dict, version: str) -> dict:
    profile = ("http://fhir.org/guides/argonaut/StructureDefinition/argo-patient" if version == "DSTU2"
               else "http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient")
    patient = {"resourceType": "Patient", "id": rec["patient_id"], "meta": {"profile": [profile]}}
    ext = demographic_extensions(rec.get("race_ethnicity"), version)
    if ext:
        patient["extension"] = ext
    patient["identifier"] = [{"system": "https://example.org/mrn", "value": rec["patient_id"]}]
    patient["name"] = [{"family": "Synthetic", "given": [f"Patient {rec['patient_id']}"]}]
    if not _missing(rec.get("gender")):
        patient["gender"] = rec["gender"]
    if not _missing(rec.get("birth_date")):
        patient["birthDate"] = rec["birth_date"]
    if not _missing(rec.get("state")):
        patient["address"] = [{"state": rec["state"], "country": "US"}]
    lang = language_element(rec.get("language"))
    if lang:
        patient["communication"] = [lang]
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


def _observation(rec, version, oid, category, code, display, value: dict, note=None) -> dict:
    cat_system = _OBS_CATEGORY_OLD if version in ("DSTU2", "STU3") else _OBS_CATEGORY_NEW
    cat = {"coding": [{"system": cat_system, "code": category, "display": _CATEGORY_DISPLAY[category]}]}
    obs = {
        "resourceType": "Observation", "id": oid, "status": "final",
        "category": cat if version == "DSTU2" else [cat],
        "code": {"coding": [{"system": t.LOINC, "code": code, "display": display}], "text": display},
        "subject": {"reference": f"Patient/{rec['patient_id']}"},
    }
    obs["context" if version == "STU3" else "encounter"] = {"reference": f"Encounter/{rec['patient_id']}-enc"}
    obs["effectiveDateTime"] = rec["effective"]
    obs.update(value)
    if note:
        if version == "DSTU2":
            obs["comments"] = note
        elif version == "STU3":
            obs["comment"] = note
        else:
            obs["note"] = [{"text": note}]
    return obs


def _condition(rec, version, key) -> dict:
    c = t.CONDITIONS[key]
    cond = {"resourceType": "Condition", "id": f"{rec['patient_id']}-{key}"}
    code = {"coding": [{"system": t.SNOMED, "code": c["snomed"][0], "display": c["snomed"][1]},
                       {"system": t.ICD10, "code": c["icd10"][0], "display": c["icd10"][1]}], "text": c["label"]}
    ref = {"reference": f"Patient/{rec['patient_id']}"}
    if version == "DSTU2":
        cond.update(patient=ref, code=code, clinicalStatus="active", verificationStatus="confirmed",
                    category={"coding": [{"system": _COND_CATEGORY_OLD, "code": "diagnosis"}]})
    elif version == "STU3":
        cond.update(clinicalStatus="active", verificationStatus="confirmed",
                    category=[{"coding": [{"system": _COND_CATEGORY_OLD, "code": "problem-list-item"}]}],
                    code=code, subject=ref)
    else:
        cond.update(
            clinicalStatus={"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                        "code": "active"}]},
            verificationStatus={"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                                            "code": "confirmed"}]},
            category=[{"coding": [{"system": _COND_CATEGORY_NEW, "code": "problem-list-item"}]}],
            code=code, subject=ref)
    return cond


def build_bundle(rec: dict, version: str = "R4") -> dict:
    if version not in VERSIONS:
        raise ValueError(f"Unsupported FHIR version {version!r}; choose one of {list(VERSIONS)}")
    rec = dict(rec)
    rec["effective"] = rec.get("effective") or datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    resources = [_patient(rec, version), _encounter(rec, version)]
    first = True
    for key, lab in t.LABS.items():
        if _missing(rec.get(key)):
            continue
        value = {"valueQuantity": {"value": float(rec[key]), "unit": lab["unit"], "system": t.UCUM,
                                   "code": lab["unit"]}}
        resources.append(_observation(rec, version, f"{rec['patient_id']}-{key.replace('_', '-')}",
                                      lab["category"], lab["code"],
                                      lab["display"], value,
                                      note="Synthetic record generated for testing." if first else None))
        first = False
    if not _missing(rec.get("smoking")) and rec["smoking"] in t.SMOKING:
        code, display = t.SMOKING[rec["smoking"]]
        resources.append(_observation(
            rec, version, f"{rec['patient_id']}-smoking", "social-history", *t.SMOKING_LOINC,
            {"valueCodeableConcept": {"coding": [{"system": t.SNOMED, "code": code, "display": display}]}}))
    for key in rec.get("conditions") or []:
        if key in t.CONDITIONS:
            resources.append(_condition(rec, version, key))

    bundle = {"resourceType": "Bundle", "id": f"{rec['patient_id']}-bundle", "type": "collection"}
    if version in ("R4", "R4B", "R5"):
        bundle["timestamp"] = rec["effective"]  # FHIR requires a timezone; inputs always carry one
    bundle["entry"] = [{"fullUrl": f"{BASE_URL}/{r['resourceType']}/{r['id']}", "resource": r} for r in resources]
    return bundle

"""Generate US Core R4 test patients that match a CohortSpec exactly.

Categorical demographics (race/ethnicity, sex, language, state, diabetes control) are
allocated by quota (largest remainder), so a cohort of 500 with 18% Black/African American
contains exactly 90 such patients. Faults are also injected by quota, after a clean
record has been built, and recorded per patient so the validation can be checked.
"""
import random
import uuid
from datetime import date, datetime, time, timedelta, timezone

from src import terminology as t
from src.data_ingestion import MEAL_PLANS, MED_PLANS, care_tier
from src.fhir.builder import _demographic_extensions
from src.testdata.spec import CONTROL_TIERS, FAULTS, LANGUAGES, NOT_RECORDED, STATES, CohortSpec

BASE = "https://example.org/fhir"
SYNTHETIC_ID_SYSTEM = "https://github.com/nez6/Diabetes-Project-/synthetic-test-data"
MRN_SYSTEM = "https://example.org/mrn"
HTEST = {"system": "http://terminology.hl7.org/CodeSystem/v3-ActReason", "code": "HTEST",
         "display": "test health data"}
SNOMED = "http://snomed.info/sct"
ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
BCP47 = "urn:ietf:bcp:47"
OBS_CAT = "http://terminology.hl7.org/CodeSystem/observation-category"

# RxNorm ingredient-level codes
METFORMIN = ("6809", "metformin", "Metformin 500 mg by mouth twice daily with meals")
EMPAGLIFLOZIN = ("1545653", "empagliflozin", "Empagliflozin 10 mg by mouth once daily")
INSULIN_GLARGINE = ("274783", "insulin glargine", "Insulin glargine 10 units subcutaneously at bedtime")
MEDS_BY_TIER = {"mono": [METFORMIN], "combination": [METFORMIN, EMPAGLIFLOZIN],
                "intensive": [METFORMIN, INSULIN_GLARGINE]}

GIVEN = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Jamie", "Quinn",
         "Rowan", "Skyler", "Dakota", "Reese", "Emerson", "Finley", "Harper", "Kai", "Logan", "Parker"]
FAMILY = ["Testpatient", "Synthetic", "Example", "Sample", "Demo", "Mock", "Trial", "Fixture"]


def allocate(n: int, weights: dict, rng: random.Random) -> list:
    """Exactly-n list of keys whose counts follow `weights` (largest remainder), shuffled."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must not all be zero")
    raw = {k: n * w / total for k, w in weights.items()}
    counts = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[: n - sum(counts.values())]:
        counts[k] += 1
    out = [k for k, c in counts.items() for _ in range(c)]
    rng.shuffle(out)
    return out


def _ref(rtype: str, rid: str) -> dict:
    return {"reference": f"{rtype}/{rid}"}


def _quantity(value: float, unit: str) -> dict:
    return {"value": value, "unit": unit, "system": t.UCUM, "code": unit}


def _obs(pid, oid, spec, value, when, enc_id):
    return {
        "resourceType": "Observation", "id": oid, "status": "final",
        "category": [{"coding": [{"system": OBS_CAT, "code": spec["category"],
                                  "display": "Laboratory" if spec["category"] == "laboratory" else "Vital Signs"}]}],
        "code": {"coding": [{"system": t.LOINC, "code": spec["code"], "display": spec["display"]}],
                 "text": spec["display"]},
        "subject": _ref("Patient", pid), "encounter": _ref("Encounter", enc_id),
        "effectiveDateTime": when, "valueQuantity": _quantity(value, spec["unit"]),
    }


def build_patient(i: int, spec: CohortSpec, attrs: dict, rng: random.Random) -> dict:
    as_of = date.fromisoformat(spec.as_of)
    pid = f"td-{i + 1:05d}"
    pract_id = f"{pid}-pract"

    # Birth date chosen so the patient is exactly `age` on the as-of date.
    age = rng.randint(spec.min_age, spec.max_age)
    anchor = as_of if (as_of.month, as_of.day) != (2, 29) else as_of - timedelta(days=1)
    birth = anchor.replace(year=anchor.year - age) - timedelta(days=rng.randint(0, 363))
    city, zip3 = rng.choice(STATES[attrs["state"]])

    patient = {
        "resourceType": "Patient", "id": pid,
        "meta": {"profile": ["http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient"],
                 "security": [dict(HTEST)]},
    }
    ext = _demographic_extensions(None if attrs["race"] == NOT_RECORDED else attrs["race"], "R4")
    if ext:
        patient["extension"] = ext
    patient.update({
        "identifier": [{"system": SYNTHETIC_ID_SYSTEM, "value": str(uuid.UUID(int=rng.getrandbits(128)))},
                       {"system": MRN_SYSTEM, "value": pid}],
        "name": [{"use": "official", "family": rng.choice(FAMILY) + str(rng.randint(100, 999)),
                  "given": [rng.choice(GIVEN) + str(rng.randint(10, 99))]}],
        "telecom": [{"system": "phone", "value": f"555-01{rng.randint(0, 99):02d}", "use": "home"}],
        "gender": attrs["sex"],
        "birthDate": birth.isoformat(),
        "address": [{"line": [f"{rng.randint(1, 999)} Test Street"], "city": city, "state": attrs["state"],
                     "postalCode": f"{zip3}{rng.randint(0, 99):02d}", "country": "US"}],
    })
    if attrs["language"] != NOT_RECORDED:
        patient["communication"] = [{"language": {"coding": [{"system": BCP47, "code": attrs["language"],
                                                              "display": LANGUAGES[attrs["language"]]}],
                                                  "text": LANGUAGES[attrs["language"]]}}]

    practitioner = {"resourceType": "Practitioner", "id": pract_id,
                    "identifier": [{"system": "https://example.org/practitioner-id", "value": f"P{i + 1:05d}"}],
                    "name": [{"family": "Clinician", "given": ["Test"], "prefix": ["Dr."]}]}

    # Visits: evenly spaced back from as_of, with a little jitter
    n_visits = max(1, spec.history_years * spec.visits_per_year)
    gap = 365 / spec.visits_per_year
    last = as_of - timedelta(days=rng.randint(7, 45))
    visit_days = sorted(min(last, last - timedelta(days=int(k * gap + (rng.randint(-15, 15) if k else 0))))
                        for k in range(n_visits))
    visit_days = [max(d, birth + timedelta(days=1)) for d in visit_days]

    _, lo, hi = CONTROL_TIERS[attrs["control"]]
    latest_a1c = round(rng.uniform(lo, hi), 1)
    bmi0 = min(60.0, max(17.0, rng.gauss(spec.bmi_mean, spec.bmi_sd)))

    resources = [patient, practitioner]
    first_visit = visit_days[0]
    resources.append({
        "resourceType": "Condition", "id": f"{pid}-t2d",
        "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                       "code": "active"}]},
        "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                                           "code": "confirmed"}]},
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                  "code": "problem-list-item", "display": "Problem List Item"}]}],
        "code": {"coding": [{"system": SNOMED, "code": "44054006", "display": "Diabetes mellitus type 2 (disorder)"},
                            {"system": ICD10, "code": "E11.9",
                             "display": "Type 2 diabetes mellitus without complications"}],
                 "text": "Type 2 diabetes mellitus"},
        "subject": _ref("Patient", pid),
        "onsetDateTime": (first_visit - timedelta(days=rng.randint(30, 900))).isoformat(),
    })

    last_when = None
    for k, day in enumerate(visit_days):
        start = datetime.combine(day, time(rng.randint(8, 16), rng.choice([0, 15, 30, 45])), tzinfo=timezone.utc)
        when = start.isoformat()
        last_when = when
        enc_id = f"{pid}-enc-{k + 1}"
        resources.append({
            "resourceType": "Encounter", "id": enc_id, "status": "finished",
            "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB",
                      "display": "ambulatory"},
            "type": [{"coding": [{"system": SNOMED, "code": "185349003",
                                  "display": "Encounter for check up (procedure)"}]}],
            "subject": _ref("Patient", pid),
            "participant": [{"individual": _ref("Practitioner", pract_id)}],
            "period": {"start": when, "end": (start + timedelta(minutes=30)).isoformat()},
        })
        steps_back = n_visits - 1 - k
        a1c = latest_a1c if steps_back == 0 else round(min(14.0, max(5.0, latest_a1c + 0.15 * steps_back
                                                                      + rng.gauss(0, 0.4))), 1)
        resources.append(_obs(pid, f"{pid}-hba1c-{k + 1}", t.HBA1C, a1c, when, enc_id))
        bmi = round(min(60.0, max(16.0, bmi0 + rng.gauss(0, 0.4))), 1)
        resources.append(_obs(pid, f"{pid}-bmi-{k + 1}", t.BMI, bmi, when, enc_id))

    tier = care_tier(latest_a1c)
    for code, display, dosage in MEDS_BY_TIER[tier]:
        resources.append({
            "resourceType": "MedicationRequest", "id": f"{pid}-med-{code}", "status": "active", "intent": "order",
            "medicationCodeableConcept": {"coding": [{"system": RXNORM, "code": code, "display": display}],
                                          "text": display},
            "subject": _ref("Patient", pid), "authoredOn": last_when, "requester": _ref("Practitioner", pract_id),
            "dosageInstruction": [{"text": dosage}],
        })
    for kind, text in (("medication", MED_PLANS[tier]), ("meal", MEAL_PLANS[tier])):
        resources.append({
            "resourceType": "CarePlan", "id": f"{pid}-cp-{kind}", "status": "active", "intent": "plan",
            "category": [{"coding": [{"system": "http://hl7.org/fhir/us/core/CodeSystem/careplan-category",
                                      "code": "assess-plan"}], "text": f"{kind}-plan"}],
            "description": text, "subject": _ref("Patient", pid),
        })

    return {"resourceType": "Bundle", "id": f"{pid}-bundle", "type": "collection",
            "timestamp": datetime.combine(as_of, time(0), tzinfo=timezone.utc).isoformat(),
            "entry": [{"fullUrl": f"{BASE}/{r['resourceType']}/{r['id']}", "resource": r} for r in resources]}


# ----------------------------------------------------------------------------- faults

def _of_type(bundle, rtype):
    return [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == rtype]


def _hba1c_obs(bundle):
    return [o for o in _of_type(bundle, "Observation") if o["code"]["coding"][0]["code"] == t.HBA1C["code"]]


def inject(bundle: dict, fault: str, rng: random.Random):
    patient = _of_type(bundle, "Patient")[0]
    if fault == "missing_bmi":
        bundle["entry"] = [e for e in bundle["entry"] if not (
            e["resource"]["resourceType"] == "Observation"
            and e["resource"]["code"]["coding"][0]["code"] == t.BMI["code"])]
    elif fault == "implausible_hba1c":
        _hba1c_obs(bundle)[-1]["valueQuantity"]["value"] = 99.0
    elif fault == "missing_birthdate":
        patient.pop("birthDate", None)
    elif fault == "non_ucum_unit":
        obs = _hba1c_obs(bundle)[-1]
        obs["valueQuantity"] = {"value": obs["valueQuantity"]["value"], "unit": "percent"}
    elif fault == "missing_obs_category":
        rng.choice(_of_type(bundle, "Observation")).pop("category", None)
    elif fault == "reversed_period":
        enc = rng.choice(_of_type(bundle, "Encounter"))
        enc["period"]["start"], enc["period"]["end"] = enc["period"]["end"], enc["period"]["start"]
    elif fault == "dangling_reference":
        rng.choice(_of_type(bundle, "Observation"))["encounter"] = {"reference": "Encounter/not-in-bundle"}
    elif fault == "missing_htest_label":
        patient["meta"].pop("security", None)
    else:
        raise ValueError(f"unknown fault {fault}")


def generate(spec: CohortSpec) -> tuple[list[dict], list[dict]]:
    """Return (bundles, truth) where truth[i] holds the requested attributes and injected faults."""
    rng = random.Random(spec.seed)
    n = spec.n
    races = allocate(n, spec.race_mix, rng)
    sexes = allocate(n, {"female": spec.pct_female, "male": 100 - spec.pct_female}, rng)
    langs = allocate(n, spec.language_mix, rng)
    states = allocate(n, {s: 1 for s in spec.states}, rng)
    controls = allocate(n, spec.control_mix, rng)

    bundles, truth = [], []
    for i in range(n):
        attrs = {"race": races[i], "sex": sexes[i], "language": langs[i], "state": states[i],
                 "control": controls[i]}
        bundles.append(build_patient(i, spec, attrs, rng))
        truth.append({**attrs, "patient_id": f"td-{i + 1:05d}", "faults": []})

    for fault in FAULTS:
        k = round(n * spec.faults.get(fault, 0) / 100)
        for i in rng.sample(range(n), k):
            inject(bundles[i], fault, rng)
            truth[i]["faults"].append(fault)
    return bundles, truth

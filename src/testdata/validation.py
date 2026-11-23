"""Validation script: checks test-patient bundles against the cohort spec and US Core basics.

    V01  Patient: exactly one, with id, valid gender, ISO birthDate, age inside the cohort range
    V02  Test-data guard: synthetic identifier present (never repaired) and HTEST label (repairable)
    V03  Referential integrity: every reference resolves inside the bundle
    V04  US Core required elements per resource type
    V05  Terminology and units: LOINC, SNOMED/ICD-10-CM, RxNorm, UCUM
    V06  Clinical plausibility: HbA1c 3–16%, BMI 10–80
    V07  Dates: periods ordered; nothing after the as-of date or before birth
    V08  Volumes: enough HbA1c and BMI measurements per patient
    V09  Demographic completeness: race/ethnicity and language recorded (warning only)

Repairs are deterministic and only made where the right answer is certain (same rule as the
reference project): add the HTEST label to a record that carries the synthetic identifier,
drop references to resources that are not in the bundle (when the reference is optional), set
an Observation's category from its LOINC code, rewrite 'percent' as UCUM '%', and swap reversed
periods. Nothing is invented: a missing birth date, missing BMI or implausible lab stays failed.
"""
import copy
from dataclasses import asdict, dataclass
from datetime import date, datetime

from src import terminology as t
from src.config import load_config
from src.testdata import VALIDATION_SCRIPT_VERSION
from src.testdata.generator import HTEST, ICD10, OBS_CAT, RXNORM, SNOMED, SYNTHETIC_ID_SYSTEM
from src.testdata.spec import CohortSpec

CHECKS = {
    "V01": "Patient present with id, gender, birthDate, age in cohort range",
    "V02": "Synthetic-data guard (identifier + HTEST label)",
    "V03": "References resolve within the bundle",
    "V04": "US Core required elements present",
    "V05": "Expected code systems and UCUM units",
    "V06": "Clinically plausible HbA1c and BMI",
    "V07": "Dates ordered and in range",
    "V08": "Enough HbA1c / BMI measurements",
    "V09": "Race/ethnicity and language recorded (warning)",
}
ERROR_CHECKS = [c for c in CHECKS if c != "V09"]

REQUIRED = {
    "Patient": ("identifier", "name", "gender"),
    "Practitioner": ("identifier", "name"),
    "Encounter": ("status", "class", "type", "subject"),
    "Condition": ("clinicalStatus", "category", "code", "subject"),
    "Observation": ("status", "category", "code", "subject", "effective[x]"),
    "MedicationRequest": ("status", "intent", "medication[x]", "subject", "authoredOn", "requester"),
    "CarePlan": ("status", "intent", "category", "subject"),
}
TERMINOLOGY = {
    "Observation": ("code", {t.LOINC}),
    "Condition": ("code", {SNOMED, ICD10}),
    "MedicationRequest": ("medicationCodeableConcept", {RXNORM}),
}
CATEGORY_FROM_LOINC = {t.HBA1C["code"]: t.HBA1C["category"], t.BMI["code"]: t.BMI["category"]}
GENDERS = {"male", "female", "other", "unknown"}
REQUIRED_REFS = {"subject", "patient"}


@dataclass(frozen=True)
class Issue:
    check_id: str
    resource: str
    message: str
    repairable: bool
    severity: str = "error"


@dataclass(frozen=True)
class Repair:
    check_id: str
    resource: str
    action: str


def errors(issues):
    return [i for i in issues if i.severity == "error"]


def conformance_score(issues) -> float:
    failed = {i.check_id for i in errors(issues)}
    return 1 - len(failed) / len(ERROR_CHECKS)


# ----------------------------------------------------------------------------- helpers

def _resources(bundle, rtype=None):
    for e in bundle.get("entry", []):
        r = e.get("resource", {})
        if rtype is None or r.get("resourceType") == rtype:
            yield r


def _label(r):
    return f"{r.get('resourceType', '?')}/{r.get('id', '?')}"


def _date(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _has(r, element):
    if element.endswith("[x]"):
        prefix = element[:-3]
        return any(k.startswith(prefix) and k != prefix and r[k] not in (None, "", [], {}) for k in r)
    return r.get(element) not in (None, "", [], {})


def _known_refs(bundle):
    known = set()
    for e in bundle.get("entry", []):
        r = e.get("resource", {})
        known.add(f"{r.get('resourceType')}/{r.get('id')}")
        if e.get("fullUrl"):
            known.add(e["fullUrl"])
    return known


def _walk_refs(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "reference" and isinstance(v, str):
                yield path, v
            else:
                yield from _walk_refs(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_refs(v, path + (i,))


def _loinc(obs):
    return next((c.get("code") for c in obs.get("code", {}).get("coding", []) if c.get("system") == t.LOINC), None)


def _patient(bundle):
    ps = list(_resources(bundle, "Patient"))
    return ps[0] if len(ps) == 1 else None


def _age(birth, as_of):
    return as_of.year - birth.year - ((as_of.month, as_of.day) < (birth.month, birth.day))


# ----------------------------------------------------------------------------- checks

def _v01(bundle, spec, as_of):
    ps = list(_resources(bundle, "Patient"))
    if len(ps) != 1:
        return [Issue("V01", "Bundle", f"expected 1 Patient, found {len(ps)}", False)]
    p, out = ps[0], []
    if not p.get("id"):
        out.append(Issue("V01", "Patient/?", "Patient has no id", False))
    if p.get("gender") not in GENDERS:
        out.append(Issue("V01", _label(p), f"invalid gender {p.get('gender')!r}", False))
    birth = _date(p.get("birthDate"))
    if birth is None or birth > as_of:
        out.append(Issue("V01", _label(p), f"missing, invalid or future birthDate {p.get('birthDate')!r}", False))
    elif not spec.min_age <= _age(birth, as_of) <= spec.max_age:
        out.append(Issue("V01", _label(p), f"age {_age(birth, as_of)} outside cohort {spec.min_age}–{spec.max_age}",
                         False))
    return out


def _v02(bundle, spec, as_of):
    p = _patient(bundle)
    if p is None:
        return [Issue("V02", "Bundle", "no single Patient to check", False)]
    out = []
    if not any(i.get("system") == SYNTHETIC_ID_SYSTEM for i in p.get("identifier", [])):
        out.append(Issue("V02", _label(p), "no synthetic-data identifier: may be real patient data", False))
    if not any(s.get("code") == HTEST["code"] for s in p.get("meta", {}).get("security", [])):
        out.append(Issue("V02", _label(p), "missing HTEST test-data security label", True))
    return out


def _v03(bundle, spec, as_of):
    known = _known_refs(bundle)
    out = []
    for r in _resources(bundle):
        for path, ref in _walk_refs(r):
            if ref not in known:
                required = path[:1] and path[0] in REQUIRED_REFS
                out.append(Issue("V03", _label(r), f"{'.'.join(map(str, path))} → {ref!r} not in bundle",
                                 not required))
    return out


def _v04(bundle, spec, as_of):
    out = []
    for r in _resources(bundle):
        for element in REQUIRED.get(r.get("resourceType"), ()):
            if not _has(r, element):
                fixable = r["resourceType"] == "Observation" and element == "category" \
                    and _loinc(r) in CATEGORY_FROM_LOINC
                out.append(Issue("V04", _label(r), f"missing required {element}", fixable))
    return out


def _v05(bundle, spec, as_of):
    out = []
    for r in _resources(bundle):
        rule = TERMINOLOGY.get(r.get("resourceType"))
        if rule and rule[0] in r:
            systems = {c.get("system") for c in r[rule[0]].get("coding", [])}
            if not systems & rule[1]:
                out.append(Issue("V05", _label(r), f"{rule[0]} not coded in {', '.join(sorted(rule[1]))}", False))
        vq = r.get("valueQuantity")
        if r.get("resourceType") == "Observation" and vq is not None and vq.get("system") != t.UCUM:
            fixable = _loinc(r) == t.HBA1C["code"] and vq.get("unit") in ("%", "percent")
            out.append(Issue("V05", _label(r), f"unit {vq.get('unit')!r} is not UCUM-coded", fixable))
    return out


def _v06(bundle, spec, as_of):
    b = load_config()["quality_bounds"]
    bounds = {t.HBA1C["code"]: ("HbA1c", b["hba1c_min"], b["hba1c_max"]),
              t.BMI["code"]: ("BMI", b["bmi_min"], b["bmi_max"])}
    out = []
    for obs in _resources(bundle, "Observation"):
        rule = bounds.get(_loinc(obs))
        value = obs.get("valueQuantity", {}).get("value")
        if rule and isinstance(value, (int, float)) and not rule[1] <= value <= rule[2]:
            out.append(Issue("V06", _label(obs), f"{rule[0]} {value} outside {rule[1]}–{rule[2]}", False))
    return out


def _v07(bundle, spec, as_of):
    p = _patient(bundle)
    birth = _date(p.get("birthDate")) if p else None
    out = []
    for r in _resources(bundle):
        per = r.get("period")
        if isinstance(per, dict) and per.get("start") and per.get("end"):
            if datetime.fromisoformat(per["end"]) < datetime.fromisoformat(per["start"]):
                out.append(Issue("V07", _label(r), "period ends before it starts", True))
        when = r.get("effectiveDateTime") or (per or {}).get("start") or r.get("authoredOn")
        d = _date(when) if when else None
        if d and (d > as_of or (birth and d < birth)):
            out.append(Issue("V07", _label(r), f"date {d} outside birth..{as_of}", False))
    return out


def _v08(bundle, spec, as_of):
    codes = [_loinc(o) for o in _resources(bundle, "Observation")]
    out = []
    n_a1c, n_bmi = codes.count(t.HBA1C["code"]), codes.count(t.BMI["code"])
    if n_a1c < spec.min_hba1c_obs:
        out.append(Issue("V08", "Observation", f"{n_a1c} HbA1c results, need at least {spec.min_hba1c_obs}", False))
    if n_bmi < spec.min_bmi_obs:
        out.append(Issue("V08", "Observation", f"{n_bmi} BMI results, need at least {spec.min_bmi_obs}", False))
    return out


def _v09(bundle, spec, as_of):
    p = _patient(bundle)
    if p is None:
        return []
    out = []
    urls = {e.get("url", "") for e in p.get("extension", [])}
    if not any(u.endswith("-race") or u.endswith("-ethnicity") for u in urls):
        out.append(Issue("V09", _label(p), "race/ethnicity not recorded", False, "warning"))
    if not p.get("communication"):
        out.append(Issue("V09", _label(p), "preferred language not recorded", False, "warning"))
    return out


_CHECKS = [_v01, _v02, _v03, _v04, _v05, _v06, _v07, _v08, _v09]


def validate(bundle, spec: CohortSpec):
    as_of = date.fromisoformat(spec.as_of)
    return [i for check in _CHECKS for i in check(bundle, spec, as_of)]


# ----------------------------------------------------------------------------- repairs

def _repair(bundle):
    repairs = []
    known = _known_refs(bundle)
    for r in _resources(bundle):
        if r.get("resourceType") == "Patient" \
                and any(i.get("system") == SYNTHETIC_ID_SYSTEM for i in r.get("identifier", [])) \
                and not any(s.get("code") == HTEST["code"] for s in r.get("meta", {}).get("security", [])):
            r.setdefault("meta", {}).setdefault("security", []).append(dict(HTEST))
            repairs.append(Repair("V02", _label(r), "added HTEST security label"))

        for key in [k for k, v in r.items() if isinstance(v, dict) and isinstance(v.get("reference"), str)
                    and v["reference"] not in known and k not in REQUIRED_REFS]:
            repairs.append(Repair("V03", _label(r), f"dropped {key} → {r[key]['reference']!r} (not in bundle)"))
            del r[key]

        if r.get("resourceType") == "Observation" and not r.get("category") and _loinc(r) in CATEGORY_FROM_LOINC:
            cat = CATEGORY_FROM_LOINC[_loinc(r)]
            r["category"] = [{"coding": [{"system": OBS_CAT, "code": cat}]}]
            repairs.append(Repair("V04", _label(r), f"category set to {cat} from LOINC {_loinc(r)}"))

        vq = r.get("valueQuantity")
        if r.get("resourceType") == "Observation" and vq and vq.get("system") != t.UCUM \
                and _loinc(r) == t.HBA1C["code"] and vq.get("unit") in ("%", "percent"):
            r["valueQuantity"] = {"value": vq.get("value"), "unit": "%", "system": t.UCUM, "code": "%"}
            repairs.append(Repair("V05", _label(r), f"unit {vq.get('unit')!r} → UCUM '%'"))

        per = r.get("period")
        if isinstance(per, dict) and per.get("start") and per.get("end") \
                and datetime.fromisoformat(per["end"]) < datetime.fromisoformat(per["start"]):
            per["start"], per["end"] = per["end"], per["start"]
            repairs.append(Repair("V07", _label(r), "swapped reversed period"))
    return repairs


@dataclass
class ValidationResult:
    bundle: dict
    issues_before: list
    repairs: list
    issues_after: list
    script_version: str = VALIDATION_SCRIPT_VERSION

    @property
    def passed_before(self):
        return not errors(self.issues_before)

    @property
    def passed(self):
        return not errors(self.issues_after)

    def as_dict(self):
        return {"issues_before": [asdict(i) for i in self.issues_before],
                "repairs": [asdict(r) for r in self.repairs],
                "issues_after": [asdict(i) for i in self.issues_after],
                "passed_before": self.passed_before, "passed": self.passed,
                "conformance_before": conformance_score(self.issues_before),
                "conformance_after": conformance_score(self.issues_after)}


def run_validation(bundle, spec: CohortSpec) -> ValidationResult:
    """Validate, repair what is safely repairable, validate again. The input is never modified.

    A record without the synthetic identifier may be real data and is returned untouched.
    """
    before = validate(bundle, spec)
    repaired = copy.deepcopy(bundle)
    if any(i.check_id == "V02" and not i.repairable for i in before):
        return ValidationResult(repaired, before, [], before)
    repairs = _repair(repaired) if any(i.repairable for i in before) else []
    return ValidationResult(repaired, before, repairs, validate(repaired, spec))

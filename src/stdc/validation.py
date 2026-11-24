"""Validation script: checks Synthea patient bundles against the VAL test data spec.

Checks (each can report several issues per bundle):

    V01  Patient: exactly one, with id, valid gender and an ISO birthDate not in the future
    V02  PHI guard: Synthea identifier present and every SSN is a 999- test number
    V03  Referential integrity: every in-bundle and contained reference resolves;
         conditional references only point at the provider directory
    V04  US Core required elements present for each resource type
    V05  Terminology: expected code systems (SNOMED/ICD-10-CM, LOINC, RxNorm, CVX)
    V06  C-CDA readiness: allergy reaction severities present; medication dosage present (warning)
    V07  Dates: periods ordered, nothing after the reference date or before birth;
         Cohort B has enough continuous history
    V08  Resource volumes per patient within the cohort's targets

Each issue is an error (fails the record) or a warning (reported only). Repairs
are deterministic and only made where the right result is certain: lower-casing
severities, setting a DiagnosticReport category when all its results share one
(laboratory -> LAB, imaging -> RAD, otherwise OTH), swapping reversed periods,
dropping references to resources that are not in the bundle, and trimming
resource volumes down to the cohort maximum (keeping active, then most recent,
resources). Nothing is invented: a record
under a minimum, or missing a required element, stays failed.

V02 is never repaired. A bundle without Synthea markers may be real patient
data and is returned untouched.
"""

import copy
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from . import VALIDATION_SCRIPT_VERSION
from . import fhir

CHECKS = {
    "V01": "Patient present with id, gender and valid birthDate",
    "V02": "Synthea markers present (PHI guard)",
    "V03": "References resolve within the bundle",
    "V04": "US Core required elements present",
    "V05": "Expected code systems",
    "V06": "C-CDA readiness (allergy severity, medication dosage)",
    "V07": "Dates ordered, in range; Cohort B history depth",
    "V08": "Resource volumes within cohort targets",
}

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SEVERITIES = ("mild", "moderate", "severe")

# US Core required elements. "name[x]" means any choice type (nameDateTime, namePeriod...).
REQUIRED = {
    "Encounter": ("status", "class", "type", "subject"),
    "Condition": ("clinicalStatus", "category", "code", "subject"),
    "Observation": ("status", "category", "code", "subject", "effective[x]"),
    "MedicationRequest": ("status", "intent", "medication[x]", "subject", "authoredOn", "requester"),
    "AllergyIntolerance": ("clinicalStatus", "verificationStatus", "code", "patient"),
    "Immunization": ("status", "vaccineCode", "patient", "occurrence[x]"),
    "Procedure": ("status", "code", "subject", "performed[x]"),
    "DiagnosticReport": ("status", "category", "code", "subject", "effective[x]", "issued"),
    "CarePlan": ("status", "intent", "category", "subject"),
    "CareTeam": ("status", "subject", "participant"),
}

# resource type -> (field holding the code, allowed systems)
TERMINOLOGY = {
    "Condition": ("code", (fhir.SNOMED, fhir.ICD10_CM)),
    "Observation": ("code", (fhir.LOINC,)),
    "DiagnosticReport": ("code", (fhir.LOINC,)),
    "Immunization": ("vaccineCode", (fhir.CVX,)),
    # US Core Procedure Codes: SNOMED, CPT, CDT (dental), HCPCS, ICD-10-PCS, LOINC.
    "Procedure": ("code", (fhir.SNOMED, fhir.CPT, fhir.CDT, fhir.HCPCS, fhir.ICD10_PCS, fhir.LOINC)),
    "AllergyIntolerance": ("code", (fhir.SNOMED, fhir.RXNORM)),
    "MedicationRequest": ("medicationCodeableConcept", (fhir.RXNORM,)),
    "Medication": ("code", (fhir.RXNORM,)),
}

ACTIVE = {"active", "in-progress", "recurrence", "relapse"}

V2_0074 = "http://terminology.hl7.org/CodeSystem/v2-0074"
V2_0074_DISPLAY = {"LAB": "Laboratory", "RAD": "Radiology", "OTH": "Other"}
DR_CATEGORY_FROM_OBSERVATION = {"laboratory": "LAB", "imaging": "RAD"}


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


@dataclass(frozen=True)
class Context:
    as_of: date
    cohort: object  # config.CohortSpec


@dataclass
class ValidationResult:
    bundle: dict
    issues_before: list
    repairs: list
    issues_after: list
    script_version: str = VALIDATION_SCRIPT_VERSION

    @property
    def passed(self):
        return passed(self.issues_after)

    @property
    def phi_guard_failed(self):
        return any(i.check_id == "V02" for i in self.issues_after)


def errors(issues):
    return [i for i in issues if i.severity == "error"]


def passed(issues):
    return not errors(issues)


def conformance_score(issues):
    """Share of the checks with no error (0-1), for the paired comparison."""
    failed = {i.check_id for i in errors(issues)}
    return 1 - len(failed) / len(CHECKS)


# ------------------------------------------------------------------ helpers

def _date(value):
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _has(resource, element):
    if element.endswith("[x]"):
        prefix = element[:-3]
        return any(k.startswith(prefix) and k != prefix and resource[k] not in (None, "", [], {})
                   for k in resource)
    return resource.get(element) not in (None, "", [], {})


def _status(r):
    s = r.get("status")
    if s is None:
        s = fhir.first_coding(r.get("clinicalStatus")).get("code")
    return s


def _count(bundle, key):
    types = key.split("|")
    return sum(1 for r in fhir.resources(bundle) if r.get("resourceType") in types)


# ------------------------------------------------------------------- checks

def _v01(bundle, ctx):
    ps = list(fhir.resources(bundle, "Patient"))
    if len(ps) != 1:
        return [Issue("V01", "Bundle", f"expected 1 Patient, found {len(ps)}", False)]
    p, out = ps[0], []
    if not p.get("id"):
        out.append(Issue("V01", "Patient/?", "Patient has no id", False))
    if p.get("gender") not in fhir.GENDERS:
        out.append(Issue("V01", fhir.label(p), f"invalid gender {p.get('gender')!r}", False))
    b = p.get("birthDate")
    if not (isinstance(b, str) and ISO_DATE.match(b)) or _date(b) is None or _date(b) > ctx.as_of:
        out.append(Issue("V01", fhir.label(p), f"invalid or future birthDate {b!r}", False))
    return out


def _v02(bundle, ctx):
    p = fhir.patient(bundle)
    if p is None:
        return [Issue("V02", "Bundle", "no single Patient to check for Synthea markers", False)]
    ids = p.get("identifier") or []
    out = []
    if not any(i.get("system") == fhir.SYNTHEA_ID_SYSTEM for i in ids):
        out.append(Issue("V02", fhir.label(p), "no Synthea identifier; may be real data", False))
    for i in ids:
        if i.get("system") == fhir.SSN_SYSTEM and not str(i.get("value", "")).startswith(fhir.SYNTHETIC_SSN_PREFIX):
            out.append(Issue("V02", fhir.label(p), "SSN is not a 999- test number; may be real data", False))
    return out


def _v03(bundle, ctx):
    index = fhir.Index(bundle)
    p = fhir.patient(bundle)
    patient_key = index.resolve(f"Patient/{p.get('id')}") if p else None
    out = []
    for r in fhir.resources(bundle):
        contained = {f"#{c.get('id')}" for c in r.get("contained", [])}
        for path, ref in fhir.walk_references(r):
            kind = fhir.ref_kind(ref)
            if kind == "conditional":
                if ref.split("?")[0] not in fhir.DIRECTORY_TYPES:
                    out.append(Issue("V03", fhir.label(r), f"conditional reference {ref[:60]!r} outside the directory", False))
            elif kind == "contained":
                if ref not in contained:
                    out.append(Issue("V03", fhir.label(r), f"contained reference {ref!r} not found", False))
            elif kind == "local":
                key = index.resolve(ref)
                if key is None:
                    out.append(Issue("V03", fhir.label(r), f"{'.'.join(map(str, path))} -> {ref!r} not in bundle", True))
                elif path[:1] in (("subject",), ("patient",)) and key != patient_key:
                    out.append(Issue("V03", fhir.label(r), "subject is not this bundle's patient", False))
            else:
                out.append(Issue("V03", fhir.label(r), "empty or invalid reference", False))
    return out


def _report_category(report, index, by_key):
    """DiagnosticReport category implied by its results, or None if not certain."""
    cats = set()
    for item in report.get("result", []):
        obs = by_key.get(index.resolve(item.get("reference")))
        if obs is None:
            return None
        codes = {fhir.first_coding(c).get("code") for c in obs.get("category", [])}
        if len(codes) != 1:
            return None
        cats |= codes
    if len(cats) != 1:
        return None
    code = DR_CATEGORY_FROM_OBSERVATION.get(cats.pop(), "OTH")
    return [{"coding": [{"system": V2_0074, "code": code, "display": V2_0074_DISPLAY[code]}]}]


def _v04(bundle, ctx):
    out, index, by_key = [], None, None
    for r in fhir.resources(bundle):
        for element in REQUIRED.get(r.get("resourceType"), ()):
            if not _has(r, element):
                fixable = False
                if r["resourceType"] == "DiagnosticReport" and element == "category":
                    if index is None:
                        index, by_key = fhir.Index(bundle), {fhir.entry_key(e): e["resource"] for e in fhir.entries(bundle)}
                    fixable = _report_category(r, index, by_key) is not None
                out.append(Issue("V04", fhir.label(r), f"missing required {element}", fixable))
    return out


def _v05(bundle, ctx):
    out = []
    for r in fhir.resources(bundle):
        rule = TERMINOLOGY.get(r.get("resourceType"))
        if not rule or rule[0] not in r:
            continue  # absent fields are V04's concern; medicationReference is checked via Medication
        systems = {c.get("system") for c in (r[rule[0]].get("coding") or [])}
        if not systems & set(rule[1]):
            out.append(Issue("V05", fhir.label(r), f"{rule[0]} not coded in {', '.join(rule[1])}", False))
    return out


def _v06(bundle, ctx):
    out = []
    for a in fhir.resources(bundle, "AllergyIntolerance"):
        for rx in a.get("reaction", []):
            sev = rx.get("severity")
            if sev not in SEVERITIES:
                fixable = isinstance(sev, str) and sev.lower() in SEVERITIES
                out.append(Issue("V06", fhir.label(a), f"reaction severity {sev!r} missing or invalid", fixable))
    for m in fhir.resources(bundle, "MedicationRequest"):
        if not m.get("dosageInstruction"):
            out.append(Issue("V06", fhir.label(m), "no dosageInstruction", False, "warning"))
    return out


def _v07(bundle, ctx):
    p = fhir.patient(bundle)
    birth = _date(p.get("birthDate")) if p else None
    out, starts = [], []
    for r in fhir.resources(bundle):
        if r.get("resourceType") == "Patient":
            continue
        for k in ("period", "effectivePeriod", "performedPeriod", "billablePeriod"):
            per = r.get(k)
            if isinstance(per, dict) and per.get("start") and per.get("end"):
                try:
                    if datetime.fromisoformat(per["end"]) < datetime.fromisoformat(per["start"]):
                        out.append(Issue("V07", fhir.label(r), f"{k} ends before it starts", True))
                except ValueError:
                    out.append(Issue("V07", fhir.label(r), f"{k} has a malformed date", False))
        d = _date(fhir.resource_date(r))
        if d and (d > ctx.as_of or (birth and d < birth)):
            out.append(Issue("V07", fhir.label(r), f"date {d} outside birth..{ctx.as_of}", False))
        if r.get("resourceType") == "Encounter" and d:
            starts.append(d)
    c = ctx.cohort
    if c.min_history_years:
        # Continuous history: encounters reach back min_history_years from the reference
        # date, with no gap longer than max_gap_days inside that window.
        window_start = ctx.as_of - timedelta(days=round(c.min_history_years * 365.25))
        starts.sort()
        if not starts or starts[0] > window_start:
            have = (ctx.as_of - starts[0]).days / 365.25 if starts else 0
            out.append(Issue("V07", "Bundle", f"{have:.1f} years of history, need {c.min_history_years}", False))
        elif c.max_gap_days:
            recent = [window_start] + [d for d in starts if d > window_start]
            gap = max((b - a).days for a, b in zip(recent, recent[1:])) if len(recent) > 1 else (ctx.as_of - window_start).days
            if gap > c.max_gap_days:
                out.append(Issue("V07", "Bundle", f"{gap}-day gap between encounters in the last "
                                                  f"{c.min_history_years} years, max {c.max_gap_days}", False))
    return out


def _v08(bundle, ctx):
    out = []
    for key, t in ctx.cohort.targets.items():
        n = _count(bundle, key)
        if n < t.min:
            out.append(Issue("V08", key, f"{n} {key}, cohort {ctx.cohort.name} needs at least {t.min}",
                             False, t.severity))
        elif t.max is not None and n > t.max:
            out.append(Issue("V08", key, f"{n} {key}, cohort {ctx.cohort.name} allows at most {t.max}",
                             True, t.severity))
    return out


_CHECK_FUNCS = [_v01, _v02, _v03, _v04, _v05, _v06, _v07, _v08]


def validate(bundle, ctx):
    """Run every check. Never modifies the bundle."""
    issues = []
    for check in _CHECK_FUNCS:
        issues.extend(check(bundle, ctx))
    return issues


# ------------------------------------------------------------------ repairs

def _repair_severities(bundle):
    out = []
    for a in fhir.resources(bundle, "AllergyIntolerance"):
        for rx in a.get("reaction", []):
            sev = rx.get("severity")
            if isinstance(sev, str) and sev not in SEVERITIES and sev.lower() in SEVERITIES:
                rx["severity"] = sev.lower()
                out.append(Repair("V06", fhir.label(a), f"severity {sev!r} -> {sev.lower()!r}"))
    return out


def _repair_report_categories(bundle):
    index = fhir.Index(bundle)
    by_key = {fhir.entry_key(e): e["resource"] for e in fhir.entries(bundle)}
    out = []
    for r in fhir.resources(bundle, "DiagnosticReport"):
        if not r.get("category"):
            cat = _report_category(r, index, by_key)
            if cat:
                r["category"] = cat
                out.append(Repair("V04", fhir.label(r), f"category set to {cat[0]['coding'][0]['code']} from its results"))
    return out


def _repair_periods(bundle):
    out = []
    for r in fhir.resources(bundle):
        for k in ("period", "effectivePeriod", "performedPeriod", "billablePeriod"):
            per = r.get(k)
            try:
                if (isinstance(per, dict) and per.get("start") and per.get("end")
                        and datetime.fromisoformat(per["end"]) < datetime.fromisoformat(per["start"])):
                    per["start"], per["end"] = per["end"], per["start"]
                    out.append(Repair("V07", fhir.label(r), f"swapped reversed {k}"))
            except ValueError:
                pass
    return out


def _repair_dangling(bundle):
    index = fhir.Index(bundle)
    dangling = lambda ref: fhir.ref_kind(ref) == "local" and index.resolve(ref) is None  # noqa: E731
    out, doomed = [], set()
    for e in fhir.entries(bundle):
        r = e["resource"]
        if not any(dangling(ref) for _, ref in fhir.walk_references(r)):
            continue
        if fhir._prune(r, dangling):
            doomed.add(fhir.entry_key(e))
            out.append(Repair("V03", fhir.label(r), "removed: required reference not in bundle"))
        else:
            out.append(Repair("V03", fhir.label(r), "dropped references to resources not in bundle"))
    if doomed:
        fhir.remove_entries(bundle, doomed)
    return out


def _priority(r):
    """Sort key: active resources first, then most recent."""
    return (_status(r) in ACTIVE, fhir.resource_date(r))


def _trim(bundle, key, keep):
    types = key.split("|")
    candidates = [e for e in fhir.entries(bundle) if e["resource"].get("resourceType") in types]
    candidates.sort(key=lambda e: _priority(e["resource"]), reverse=True)
    drop = {fhir.entry_key(e) for e in candidates[keep:]}
    if not drop:
        return []
    removed = fhir.remove_entries(bundle, drop)
    return [Repair("V08", key, f"kept {keep} of {len(candidates)} {key} (active, then most recent); "
                               f"{len(removed) - len(drop)} dependent resources removed")]


def _repair_volumes(bundle, cohort):
    out = []
    # Encounters first: trimming them cascades to the events recorded at those visits.
    order = sorted(cohort.targets, key=lambda k: k != "Encounter")
    for key in order:
        t = cohort.targets[key]
        if t.max is not None and _count(bundle, key) > t.max:
            out.extend(_trim(bundle, key, t.max))
    return out


def run_validation(bundle, ctx):
    """Validate, repair what is safely repairable, and validate again.

    The input bundle is left untouched; the repaired copy is returned.
    """
    issues_before = validate(bundle, ctx)
    if any(i.check_id == "V02" for i in issues_before):
        return ValidationResult(copy.deepcopy(bundle), issues_before, [], issues_before)
    repaired = copy.deepcopy(bundle)
    repairs = []
    if any(i.repairable for i in issues_before):
        repairs += _repair_severities(repaired)
        repairs += _repair_report_categories(repaired)
        repairs += _repair_periods(repaired)
        repairs += _repair_dangling(repaired)
        repairs += _repair_volumes(repaired, ctx.cohort)
    return ValidationResult(repaired, issues_before, repairs, validate(repaired, ctx))

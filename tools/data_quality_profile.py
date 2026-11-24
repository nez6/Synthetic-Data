#!/usr/bin/env python3
"""Data-quality profile for a FHIR R4 patient population — run it on YOUR system.

Reads FHIR R4 data exported from your application and writes an aggregate-only profile
(demographic mix, condition prevalence, vital/lab distributions and data-quality problem
rates). Upload the profile to the Synthetic Test Data Creator to build a synthetic test
dataset with the same population AND the same data-quality problems.

Privacy: no patient-level data is written. Every count between 1 and k-1 (default k = 11)
is replaced by null ("suppressed"), numeric summaries need at least k patients, and the
script refuses to profile fewer than k patients.

Input (any mix, files or folders, searched recursively):
  * Bundles:      *.json, *.json.gz  (one Bundle per patient, e.g. Synthea output)
  * Bulk export:  *.ndjson, *.ndjson.gz  (one resource or one Bundle per line, e.g. FHIR $export)

Usage:
  python data_quality_profile.py --input export/ --out profile.json
  python data_quality_profile.py --input Patient.ndjson Observation.ndjson Condition.ndjson --as-of 2026-11-01

Requires only the Python 3.9+ standard library.
"""
import argparse
import gzip
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone

PROFILE_VERSION = "1.0"
LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"

AGE_BANDS = [("18-34", 18, 34), ("35-49", 35, 49), ("50-64", 50, 64), ("65-79", 65, 79), ("80+", 80, 200)]
LABS = {  # key: (LOINC, plausible low, plausible high)
    "bmi": ("39156-5", 10, 80),
    "systolic_bp": ("8480-6", 60, 260),
    "hba1c": ("4548-4", 3, 16),
    "egfr": ("33914-3", 1, 200),
}
SMOKING_LOINC = "72166-2"
SMOKING = {"266919005": "never", "8517006": "former", "449868002": "current",
           "428041000124106": "current", "77176002": "current", "65568007": "current"}
# SNOMED codes and ICD-10-CM prefixes per condition group (CHF/CKD/T2D/COPD as in the
# Synthetic-Test-Data-Creator reference config, plus hypertension)
CONDITIONS = {
    "hypertension": ({"59621000", "38341003"}, ("I10",)),
    "t2d": ({"44054006"}, ("E11",)),
    "ckd": ({"46177005", "431855005", "431856006", "433144002", "431857002", "709044004", "127013003"},
            ("N18", "E11.2")),
    "chf": ({"88805009", "84114007", "42343007", "703272007"}, ("I50",)),
    "copd": ({"185086009", "87433001", "13645005"}, ("J43", "J44")),
}
COMPLEX_GROUPS = ("t2d", "ckd", "chf", "copd")
RACE = {"1002-5": "American Indian/Alaska Native", "2028-9": "Asian", "2054-5": "Black/African American",
        "2076-8": "Native Hawaiian/Pacific Islander", "2106-3": "White"}
HISPANIC = "2135-2"
LANGUAGE = {"en": "en-US", "en-us": "en-US", "es": "es", "zh": "zh", "vi": "vi"}
DIRECTORY_TYPES = {"Practitioner", "PractitionerRole", "Organization", "Location"}
# Resources whose references are checked (billing resources such as Claim/ExplanationOfBenefit are skipped)
CLINICAL_TYPES = {"Encounter", "Observation", "Condition", "MedicationRequest", "Procedure", "Immunization",
                  "AllergyIntolerance", "DiagnosticReport", "CarePlan", "CareTeam"}
NOT_RECORDED = "Not recorded"


# ----------------------------------------------------------------------------- reading

def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def _files(inputs):
    for p in inputs:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for n in sorted(names):
                    if n.endswith((".json", ".json.gz", ".ndjson", ".ndjson.gz")):
                        yield os.path.join(root, n)
        else:
            yield p


def _bundle_resources(bundle):
    out = []
    for e in bundle.get("entry", []):
        if e.get("resource"):
            r = dict(e["resource"])
            r["_fullUrl"] = e.get("fullUrl")
            out.append(r)
    return out


def _known(resources):
    known = set()
    for r in resources:
        rid = r.get("id")
        if rid:
            known.add(f"{r.get('resourceType')}/{rid}")
            known.add(f"urn:uuid:{rid}")
        if r.get("_fullUrl"):
            known.add(r["_fullUrl"])
    return known


def _subject_id(r):
    for key in ("subject", "patient"):
        ref = (r.get(key) or {}).get("reference", "")
        if ref:
            return ref.split("/")[-1].replace("urn:uuid:", "")
    return None


def group_by_patient(resources):
    """Group loose resources (bulk NDJSON) into [Patient, ...] lists, plus the set of known references."""
    resources = list(resources)
    patients, others = {}, defaultdict(list)
    for r in resources:
        if r.get("resourceType") == "Patient":
            patients[r.get("id")] = r
        else:
            pid = _subject_id(r)
            if pid:
                others[pid].append(r)
    return [[p] + others.get(pid, []) for pid, p in patients.items()], _known(resources)


def iter_patients(inputs):
    """Yield (resources, known_refs) per patient. Bundles holding one Patient are streamed one at a time;
    loose resources and multi-patient Bundles are grouped at the end."""
    loose = []
    for path in _files(inputs):
        with _open(path) as f:
            items = (json.loads(line) for line in f if line.strip()) if ".ndjson" in path else [json.load(f)]
            for item in items:
                if not isinstance(item, dict) or "resourceType" not in item:
                    continue  # not FHIR (e.g. a manifest or index file in the export folder)
                if item.get("resourceType") != "Bundle":
                    loose.append(item)
                    continue
                res = _bundle_resources(item)
                if sum(r.get("resourceType") == "Patient" for r in res) == 1:
                    res.sort(key=lambda r: r.get("resourceType") != "Patient")
                    yield res, _known(res)
                else:
                    loose.extend(res)
    if loose:
        groups, known = group_by_patient(loose)
        for g in groups:
            yield g, known


# ----------------------------------------------------------------------------- per patient

def _date(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def _codes(cc):
    return [(c.get("system"), c.get("code")) for c in (cc or {}).get("coding", [])]


def _refs(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "reference" and isinstance(v, str):
                out.append(v)
            elif k != "contained":
                _refs(v, out)
    elif isinstance(node, list):
        for v in node:
            _refs(v, out)
    return out


def summarize_patient(resources, known, as_of):
    p = resources[0]
    s = {"problems": set()}

    birth = _date(p.get("birthDate"))
    if birth is None:
        s["problems"].add("missing_birthdate")
        s["age_band"] = NOT_RECORDED
    else:
        age = as_of.year - birth.year - ((as_of.month, as_of.day) < (birth.month, birth.day))
        s["age"] = age
        s["age_band"] = next((b for b, lo, hi in AGE_BANDS if lo <= age <= hi), "under 18")
    s["sex"] = p.get("gender") or NOT_RECORDED

    race = eth = None
    for ext in p.get("extension", []):
        omb = next((x.get("valueCoding", {}).get("code") for x in ext.get("extension", [])
                    if x.get("url") == "ombCategory"), None)
        if ext.get("url", "").endswith("-race"):
            race = omb
        elif ext.get("url", "").endswith("-ethnicity"):
            eth = omb
    s["race_ethnicity"] = "Hispanic/Latino" if eth == HISPANIC else RACE.get(race, NOT_RECORDED)
    if s["race_ethnicity"] == NOT_RECORDED:
        s["problems"].add("missing_race_ethnicity")

    lang = None
    for comm in p.get("communication", []):
        for _, code in _codes(comm.get("language")):
            if code:
                lang = LANGUAGE.get(code.lower(), LANGUAGE.get(code.lower().split("-")[0], "other"))
                break
    s["language"] = lang or NOT_RECORDED
    if not lang:
        s["problems"].add("missing_language")
    s["state"] = ((p.get("address") or [{}])[0].get("state") or NOT_RECORDED).upper()[:2] \
        if p.get("address") else NOT_RECORDED

    latest, smoking = {}, (None, NOT_RECORDED)
    conditions, encounter_dates = set(), []
    for r in resources[1:]:
        rtype = r.get("resourceType")
        per = r.get("period") or {}
        if per.get("start") and per.get("end"):
            try:
                if datetime.fromisoformat(per["end"].replace("Z", "+00:00")) < \
                        datetime.fromisoformat(per["start"].replace("Z", "+00:00")):
                    s["problems"].add("reversed_period")
            except ValueError:
                pass
        for ref in (_refs({k: v for k, v in r.items() if k != "_fullUrl"}, []) if rtype in CLINICAL_TYPES else []):
            rt = ref.split("/")[0] if "/" in ref and not ref.startswith(("http", "urn")) else None
            if rt in DIRECTORY_TYPES or "?" in ref or ref.startswith("#"):
                continue
            if ref not in known:
                s["problems"].add("dangling_reference")

        if rtype == "Encounter" and _date(per.get("start")):
            encounter_dates.append(_date(per["start"]))
        elif rtype == "Condition":
            for system, code in _codes(r.get("code")):
                for group, (snomed, icd) in CONDITIONS.items():
                    if (code in snomed) or (code and system and "icd-10" in system and code.startswith(icd)):
                        conditions.add(group)
        elif rtype == "Observation":
            if not r.get("category"):
                s["problems"].add("missing_obs_category")
            loincs = {c for sys_, c in _codes(r.get("code")) if sys_ == LOINC}
            when = r.get("effectiveDateTime") or ""
            if SMOKING_LOINC in loincs:
                code = next((c for _, c in _codes(r.get("valueCodeableConcept"))), None)
                if when >= (smoking[0] or ""):
                    smoking = (when, SMOKING.get(code, NOT_RECORDED))
            # values can sit on the Observation itself or in a component (e.g. the US Core BP panel)
            parts = [(loincs, r.get("valueQuantity"))] + [
                ({c for sys_, c in _codes(comp.get("code")) if sys_ == LOINC}, comp.get("valueQuantity"))
                for comp in r.get("component", [])]
            for key, (code, lo, hi) in LABS.items():
                for part_codes, quantity in parts:
                    if code not in part_codes:
                        continue
                    q = quantity or {}
                    v = q.get("value")
                    if q.get("system") != UCUM:
                        s["problems"].add("non_ucum_unit")
                    if isinstance(v, (int, float)):
                        if not lo <= v <= hi:
                            s["problems"].add("implausible_lab")
                        elif when >= latest.get(key, ("", None))[0]:
                            latest[key] = (when, v)
                    if key == "bmi":
                        s["has_bmi"] = True
    if not s.get("has_bmi"):
        s["problems"].add("missing_bmi")
    s["labs"] = {k: v for k, (_, v) in latest.items()}
    s["smoking"] = smoking[1]
    s["conditions"] = conditions
    s["complex_care"] = sum(g in conditions for g in COMPLEX_GROUPS) >= 2
    s["n_encounters"] = len(encounter_dates)
    s["history_years"] = ((max(encounter_dates) - min(encounter_dates)).days / 365.25) if encounter_dates else 0.0
    return s


# ----------------------------------------------------------------------------- profile

QUALITY_KEYS = ["missing_birthdate", "missing_race_ethnicity", "missing_language", "missing_bmi",
                "implausible_lab", "non_ucum_unit", "missing_obs_category", "reversed_period",
                "dangling_reference"]


def _suppress(counter, k_min, path, suppressed):
    out = {}
    for key, n in sorted(counter.items(), key=lambda kv: str(kv[0])):
        if 0 < n < k_min:
            out[key] = None
            suppressed.append(f"{path}.{key}")
        else:
            out[key] = n
    return out


def _numeric(values, k_min):
    if len(values) < max(k_min, 2):
        return None
    return {"n": len(values), "mean": round(statistics.fmean(values), 2), "sd": round(statistics.stdev(values), 2),
            "p10": round(sorted(values)[len(values) // 10], 2), "p90": round(sorted(values)[9 * len(values) // 10], 2)}


def build_profile(summaries, k_min=11, as_of=None, source="unspecified"):
    n = len(summaries)
    if k_min and n < k_min:
        raise SystemExit(f"Only {n} patients found; at least {k_min} are needed so the profile stays anonymous.")
    suppressed = []
    counts = {
        "age_band": Counter(s["age_band"] for s in summaries),
        "sex": Counter(s["sex"] for s in summaries),
        "race_ethnicity": Counter(s["race_ethnicity"] for s in summaries),
        "language": Counter(s["language"] for s in summaries),
        "state": Counter(s["state"] for s in summaries),
        "smoking": Counter(s["smoking"] for s in summaries),
        "conditions": Counter(c for s in summaries for c in s["conditions"]),
    }
    for g in CONDITIONS:
        counts["conditions"].setdefault(g, 0)
    quality = Counter(p for s in summaries for p in s["problems"])
    for q in QUALITY_KEYS:
        quality.setdefault(q, 0)
    visits = [s["n_encounters"] / max(s["history_years"], 1.0) for s in summaries if s["n_encounters"]]
    return {
        "profile_version": PROFILE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "as_of": str(as_of),
        "k_min": k_min,
        "n_patients": n,
        "counts": {name: _suppress(c, k_min, f"counts.{name}", suppressed) for name, c in counts.items()},
        "complex_care": _suppress(Counter({"patients": sum(s["complex_care"] for s in summaries)}), k_min,
                                  "complex_care", suppressed)["patients"],
        "quality": _suppress(quality, k_min, "quality", suppressed),
        "numeric": {
            **{k: _numeric([s["labs"][k] for s in summaries if k in s["labs"]], k_min) for k in LABS},
            "age": _numeric([s["age"] for s in summaries if "age" in s], k_min),
            "encounters_per_patient": _numeric([s["n_encounters"] for s in summaries], k_min),
            "history_years": _numeric([s["history_years"] for s in summaries], k_min),
            "visits_per_year": _numeric(visits, k_min),
        },
        "suppressed": suppressed,
    }


def profile_paths(inputs, as_of=None, k_min=11, source="unspecified"):
    as_of = as_of or date.today()
    summaries = [summarize_patient(rs, known, as_of) for rs, known in iter_patients(inputs)]
    return build_profile(summaries, k_min=k_min, as_of=as_of, source=source)


def profile_bundles(bundles, as_of=None, k_min=11, source="unspecified"):
    """Profile in-memory patient Bundles (used by the app on its own synthetic data)."""
    as_of = as_of or date.today()
    summaries = []
    for b in bundles:
        res = _bundle_resources(b)
        res.sort(key=lambda r: r.get("resourceType") != "Patient")
        summaries.append(summarize_patient(res, _known(res), as_of))
    return build_profile(summaries, k_min=k_min, as_of=as_of, source=source)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="FHIR files or folders")
    ap.add_argument("--out", default="profile.json")
    ap.add_argument("--as-of", help="date ages are computed on (default: today)")
    ap.add_argument("--k-min", type=int, default=11, help="smallest count reported (default 11)")
    ap.add_argument("--source", default="my application", help="free-text label stored in the profile")
    a = ap.parse_args(argv)
    as_of = date.fromisoformat(a.as_of) if a.as_of else date.today()
    profile = profile_paths(a.input, as_of=as_of, k_min=a.k_min, source=a.source)
    with open(a.out, "w") as f:
        json.dump(profile, f, indent=2)
    n = profile["n_patients"]
    print(f"Profiled {n} patients → {a.out}  ({len(profile['suppressed'])} small cells suppressed)")
    for q, c in profile["quality"].items():
        print(f"  {q:24} {'<' + str(a.k_min) if c is None else f'{c} ({100 * c / n:.1f}%)'}")


if __name__ == "__main__":
    sys.exit(main())

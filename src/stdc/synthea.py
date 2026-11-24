"""Load Synthea FHIR R4 output and select the cohorts.

Synthea writes one ``<Given>_<Family>_<uuid>.json`` bundle per patient, plus
``hospitalInformation*.json`` and ``practitionerInformation*.json`` directory
bundles, which are skipped here.

Selection uses only the clinical cohort definition (age, alive, chronic
condition groups). Resource volumes, history depth and conformance are what
the validation script measures, so they are deliberately not used to pick
patients.
"""

import gzip
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import date

from . import fhir

# Plain or gzipped (.json.gz, as committed under data/) Synthea patient files.
PATIENT_FILE = re.compile(r"_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.json(\.gz)?$")


@dataclass(frozen=True)
class PatientSummary:
    patient_id: str
    path: str
    age: int | None
    alive: bool
    condition_groups: tuple
    n_resources: int


def patient_files(directory):
    """Patient bundle paths under directory (recursive), sorted for reproducibility."""
    out = []
    for root, _, files in os.walk(directory):
        out.extend(os.path.join(root, f) for f in files if PATIENT_FILE.search(f))
    return sorted(out)


def load_bundle(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def age_on(birth_date, as_of):
    try:
        b = date.fromisoformat(birth_date)
    except (TypeError, ValueError):
        return None
    return as_of.year - b.year - ((as_of.month, as_of.day) < (b.month, b.day))


def condition_groups(bundle, groups):
    """Names of the configured condition groups present anywhere in the record."""
    found = set()
    for c in fhir.resources(bundle, "Condition"):
        for coding in (c.get("code") or {}).get("coding") or []:
            code, system = coding.get("code"), coding.get("system")
            for name, vs in groups.items():
                if system == fhir.SNOMED and code in vs.get("snomed", []):
                    found.add(name)
                elif system == fhir.ICD10_CM and any(str(code).startswith(p) for p in vs.get("icd10_prefixes", [])):
                    found.add(name)
    return tuple(sorted(found))


def summarize(path, bundle, as_of, groups):
    p = fhir.patient(bundle) or {}
    return PatientSummary(
        patient_id=p.get("id", os.path.basename(path)),
        path=path,
        age=age_on(p.get("birthDate"), as_of),
        alive=not (p.get("deceasedDateTime") or p.get("deceasedBoolean")),
        condition_groups=condition_groups(bundle, groups),
        n_resources=len(fhir.entries(bundle)),
    )


def eligible(summary, cohort):
    if not summary.alive or summary.age is None:
        return False
    if not cohort.min_age <= summary.age <= cohort.max_age:
        return False
    n = len(summary.condition_groups)
    if cohort.exclude_condition_groups and n:
        return False
    return n >= cohort.min_condition_groups


def cohort_counts(n, cohorts):
    counts, assigned = {}, 0
    for i, c in enumerate(cohorts):
        counts[c.name] = n - assigned if i == len(cohorts) - 1 else round(n * c.share)
        assigned += counts[c.name]
    return counts


def select(summaries_by_pool, cohorts, n, seed, counts=None):
    """Pick each cohort's patients at random (seeded) from its eligible pool.

    summaries_by_pool maps cohort name -> summaries from that cohort's Synthea run.
    Returns a list of (summary, cohort name). Raises when a pool is too small.
    """
    selected = []
    for cohort in cohorts:
        need = (counts or cohort_counts(n, cohorts))[cohort.name]
        pool = sorted((s for s in summaries_by_pool.get(cohort.name, []) if eligible(s, cohort)),
                      key=lambda s: s.patient_id)
        if len(pool) < need:
            raise ValueError(
                f"Cohort {cohort.name} needs {need} eligible patients but its Synthea pool has "
                f"{len(pool)}. Increase synthea.pools.{cohort.name}.population.")
        rng = random.Random(f"{seed}:{cohort.name}")
        selected.extend((s, cohort.name) for s in rng.sample(pool, need))
    return selected


RUN_INFO = "run_info.json"


def read_run_info(directory):
    """Synthea version, seed and command recorded by scripts/run_synthea.py, or {} if absent."""
    for root, _, files in os.walk(directory):
        if RUN_INFO in files:
            with open(os.path.join(root, RUN_INFO), encoding="utf-8") as f:
                return json.load(f)
    return {}


def scan_pool(directory, as_of, groups):
    return [summarize(p, load_bundle(p), as_of, groups) for p in patient_files(directory)]


def manifest(selected, emr_key, seed, synthea_version):
    return {
        "emr": emr_key,
        "seed": seed,
        "synthea_version": synthea_version,
        "patients": [{**asdict(s), "cohort": c, "file": os.path.basename(s.path)} for s, c in selected],
    }

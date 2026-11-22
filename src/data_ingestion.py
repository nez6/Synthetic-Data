"""Step 1-2 of the pipeline: create a synthetic, deliberately messy EHR and export it as FHIR.

Faults are injected on purpose to simulate disparate source systems, so the Data
Quality layer has real problems to find:
  * ~5% of patients have no race/ethnicity recorded
  * 40 patients have no BMI measurement
  * 20 patients have an implausible HbA1c (99.0 — a typical unit/entry error)
"""
import json
from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pandas as pd

from src import terminology as t
from src.config import load_config, resolve
from src.fhir.builder import build_bundle

RACE_PROBS = {
    "American Indian/Alaska Native": 0.04,
    "Black/African American": 0.24,
    "Hispanic/Latino": 0.30,
    "Asian": 0.15,
    "White": 0.27,
}
MISSING_RACE_RATE = 0.05

MED_PLANS = {
    "intensive": "Insulin Therapy Intensive + Metformin",
    "combination": "Combination Oral (Metformin + SGLT2 Inhibitor)",
    "mono": "Monotherapy (Metformin Base) or Lifestyle Modification",
}
MEAL_PLANS = {
    "intensive": "Strict Carb Counting / Low Glycemic Index",
    "combination": "Moderate Carb / Plant-Forward Mediterranean Diet",
    "mono": "Caloric Restriction / Standard DASH Diet",
}


def care_tier(hba1c: float) -> str:
    """Ground-truth label rule. Depends only on clinical values, never on race."""
    if hba1c > 9.0:
        return "intensive"
    if hba1c > 7.5:
        return "combination"
    return "mono"


def generate_records(n: int, seed: int, id_prefix: str = "pt") -> list[dict]:
    rng = np.random.default_rng(seed)
    # Same fault rates as the reference 1,200-patient dataset (40 missing BMI, 20 bad HbA1c).
    n_missing_bmi = round(n * 40 / 1200)
    n_bad_hba1c = round(n * 20 / 1200)

    groups = list(RACE_PROBS)
    race = rng.choice(groups, n, p=list(RACE_PROBS.values())).astype(object)
    race[rng.random(n) < MISSING_RACE_RATE] = None

    age = rng.integers(18, 85, n)
    gender = rng.choice(["female", "male"], n)

    bmi = rng.uniform(18.5, 45.0, n).round(1).astype(object)
    bmi[rng.choice(n, n_missing_bmi, replace=False)] = None

    hba1c = rng.uniform(5.5, 12.0, n).round(1)
    hba1c[rng.choice(n, n_bad_hba1c, replace=False)] = 99.0

    records = []
    for i in range(n):
        visit_day = date(2026, 1, 5) + timedelta(days=int(rng.integers(0, 300)))
        visit = datetime.combine(visit_day, time(int(rng.integers(8, 17)), int(rng.choice([0, 15, 30, 45]))),
                                 tzinfo=timezone.utc)
        birth = visit_day.replace(year=visit_day.year - int(age[i])) - timedelta(days=int(rng.integers(0, 365)))
        tier = care_tier(hba1c[i])
        records.append({
            "patient_id": f"pt-{i + 1:04d}" if id_prefix == "pt" else f"{id_prefix}-{seed % 1000000:06d}-{i + 1}",
            "gender": gender[i],
            "birth_date": birth.isoformat(),
            "effective": visit.isoformat(),
            "race_demographic": race[i],
            "age": int(age[i]),
            "bmi": None if bmi[i] is None else float(bmi[i]),
            "baseline_hba1c": float(hba1c[i]),
            "recommended_medication_plan": MED_PLANS[tier],
            "recommended_meal_plan": MEAL_PLANS[tier],
        })
    return records


def fetch_data(n_samples: int = None, seed: int = None, fhir_version: str = None) -> dict:
    """Generate source records, export them as FHIR Bundles (NDJSON). Returns file paths."""
    config = load_config()
    n = n_samples or config["pipeline"]["n_samples"]
    seed = config["pipeline"]["random_state"] if seed is None else seed
    version = fhir_version or config["data"]["fhir_version"]

    records = generate_records(n, seed)

    source_path = resolve(config["data"]["source_path"])
    source_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(source_path, index=False)

    bundles_path = resolve(config["data"]["bundles_path"])
    bundles_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bundles_path, "w") as f:
        for rec in records:
            f.write(json.dumps(build_bundle(rec, version)) + "\n")

    print(f"Wrote {n} source records → {source_path}")
    print(f"Exported {n} FHIR {version} Bundles → {bundles_path}")
    return {"source_path": source_path, "bundles_path": bundles_path, "fhir_version": version, "n": n}


def load_bundles(path=None) -> list[dict]:
    path = path or resolve(load_config()["data"]["bundles_path"])
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    fetch_data()

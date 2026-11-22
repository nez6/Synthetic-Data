"""Generate a synthetic, deliberately messy EHR extract.

Gaps (missing race, missing BMI) and implausible lab values (HbA1c = 99.0) are
injected on purpose to simulate disparate source systems, so the Data Quality
layer has real faults to detect.
"""
import numpy as np
import pandas as pd

from src.config import load_config, resolve

RACES = [
    "American Indian/Alaska Native",
    "Black/African American",
    "Hispanic/Latino",
    "Asian",
    "White",
]
RACE_PROBS = [0.04, 0.24, 0.30, 0.15, 0.27]
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


def generate_data(n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    race = rng.choice(RACES, n_samples, p=RACE_PROBS).astype(object)
    race[rng.random(n_samples) < MISSING_RACE_RATE] = None

    age = rng.integers(18, 85, n_samples)

    bmi = rng.uniform(18.5, 45.0, n_samples)
    bmi[rng.choice(n_samples, 40, replace=False)] = np.nan

    hba1c = rng.uniform(5.5, 12.0, n_samples)
    hba1c[rng.choice(n_samples, 20, replace=False)] = 99.0  # implausible lab entries

    tiers = [care_tier(h) for h in hba1c]

    return pd.DataFrame({
        "race_demographic": race,
        "age": age,
        "bmi": bmi.round(1),
        "baseline_hba1c": hba1c.round(1),
        "recommended_medication_plan": [MED_PLANS[t] for t in tiers],
        "recommended_meal_plan": [MEAL_PLANS[t] for t in tiers],
    })


def fetch_data() -> pd.DataFrame:
    config = load_config()
    df = generate_data(config["pipeline"]["n_samples"], config["pipeline"]["random_state"])
    path = resolve(config["data"]["local_raw_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} synthetic records to {path}")
    return df


if __name__ == "__main__":
    fetch_data()

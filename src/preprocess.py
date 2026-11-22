"""Step 3: flatten FHIR Bundles → table, score Data Quality, clean, encode, split."""
import json
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import load_config, resolve
from src.fhir.parser import bundle_to_record
from src.quality import score_dataframe

FEATURES = ["race_demographic", "age", "bmi", "baseline_hba1c"]


def flatten_bundles(bundles: list[dict]) -> pd.DataFrame:
    """Extract model features (and CarePlan labels) from each FHIR Bundle."""
    df = pd.DataFrame([bundle_to_record(b) for b in bundles])
    for col in ("age", "bmi", "baseline_hba1c"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def evaluate_and_clean_quality(df: pd.DataFrame, threshold: int = None):
    config = load_config()
    unknown = config["data"]["unknown_race_label"]
    threshold = config["min_quality_threshold"] if threshold is None else threshold

    scored = score_dataframe(df)
    scored.to_csv(resolve(config["data"]["flattened_path"]), index=False)

    group = scored["race_demographic"].fillna(unknown)
    report = (
        scored.assign(group=group)
        .groupby("group")
        .agg(
            records=("data_quality_index", "size"),
            mean_dqi=("data_quality_index", "mean"),
            missing_bmi_rate=("bmi", lambda s: float(s.isna().mean())),
            below_threshold=("data_quality_index", lambda s: int((s < threshold).sum())),
        )
        .round(3)
    )
    print("\n--- BASELINE DATA QUALITY REPORT PER DEMOGRAPHIC ---")
    print(report.to_string())

    cleaned = scored[scored["data_quality_index"] >= threshold].copy()
    print(f"\nKept {len(cleaned)}/{len(scored)} rows with DQI >= {threshold}")

    # Missing race becomes an explicit 'Unknown' category (never a guessed race);
    # missing BMI takes the median of observed values.
    bmi_median = float(cleaned["bmi"].median())
    cleaned["race_demographic"] = cleaned["race_demographic"].fillna(unknown)
    cleaned["bmi"] = cleaned["bmi"].fillna(bmi_median)

    summary = {
        "overall_mean_dqi": round(float(scored["data_quality_index"].mean()), 3),
        "rows_total": int(len(scored)),
        "rows_kept": int(len(cleaned)),
        "threshold": threshold,
        "bmi_median": bmi_median,
        "per_demographic": report.to_dict(orient="index"),
    }
    return cleaned, summary


def preprocess_pipeline(df: pd.DataFrame, threshold: int = None):
    config = load_config()
    df_cleaned, quality_summary = evaluate_and_clean_quality(df, threshold)
    resolve(config["artifacts"]["model_dir"]).mkdir(parents=True, exist_ok=True)

    encoders = {}
    for col in ["race_demographic", config["data"]["target_medication"], config["data"]["target_meal"]]:
        le = LabelEncoder()
        df_cleaned[col] = le.fit_transform(df_cleaned[col])
        encoders[col] = le

    with open(resolve(config["artifacts"]["encoder_path"]), "wb") as f:
        pickle.dump(encoders, f)
    with open(resolve(config["artifacts"]["quality_report_path"]), "w") as f:
        json.dump(quality_summary, f, indent=2)

    X = df_cleaned[FEATURES]
    y_med = df_cleaned[config["data"]["target_medication"]]
    y_meal = df_cleaned[config["data"]["target_meal"]]

    splits = train_test_split(
        X, y_med, y_meal,
        test_size=config["pipeline"]["test_size"],
        random_state=config["pipeline"]["random_state"],
    )
    return splits, quality_summary, encoders

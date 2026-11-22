import json
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import load_config, resolve
from src.quality import score_dataframe

FEATURES = ["race_demographic", "age", "bmi", "baseline_hba1c"]


def evaluate_and_clean_quality(df: pd.DataFrame):
    config = load_config()
    unknown = config["data"]["unknown_race_label"]

    scored = score_dataframe(df)
    group_key = scored["race_demographic"].fillna(unknown)

    # Per-demographic quality profile: surfaces systemic documentation gaps.
    report = (
        scored.assign(group=group_key)
        .groupby("group")
        .agg(
            records=("data_quality_index", "size"),
            mean_dqi=("data_quality_index", "mean"),
            missing_bmi_rate=("bmi", lambda s: float(s.isna().mean())),
        )
        .round(3)
    )
    print("\n--- BASELINE DATA QUALITY REPORT PER DEMOGRAPHIC ---")
    print(report.to_string())

    threshold = config["min_quality_threshold"]
    cleaned = scored[scored["data_quality_index"] >= threshold].copy()
    print(f"\nKept {len(cleaned)}/{len(scored)} rows with DQI >= {threshold}")

    # Impute: missing race becomes an explicit 'Unknown' category (never a guessed
    # race); missing BMI takes the median of observed values.
    cleaned["race_demographic"] = cleaned["race_demographic"].fillna(unknown)
    cleaned["bmi"] = cleaned["bmi"].fillna(cleaned["bmi"].median())

    summary = {
        "overall_mean_dqi": round(float(scored["data_quality_index"].mean()), 3),
        "rows_total": int(len(scored)),
        "rows_kept": int(len(cleaned)),
        "threshold": threshold,
        "per_demographic": report.to_dict(orient="index"),
    }
    return cleaned, summary


def preprocess_pipeline(df: pd.DataFrame):
    config = load_config()
    df_cleaned, quality_summary = evaluate_and_clean_quality(df)
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
    return splits, quality_summary

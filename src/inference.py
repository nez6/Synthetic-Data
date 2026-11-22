"""Shared prediction logic used by the REST API, the FHIR endpoint and the HL7 listener."""
import json
import pickle

import pandas as pd

from src.config import load_config, resolve
from src.preprocess import FEATURES
from src.quality import ISSUE_LABELS, score_record
from src.routing import recommend_provider


class ModelNotReady(RuntimeError):
    pass


_cache = {"key": None, "artifacts": None}


def load_artifacts():
    """Load models, reloading automatically whenever they are retrained."""
    a = load_config()["artifacts"]
    paths = [resolve(a[k]) for k in ("med_model_path", "meal_model_path", "encoder_path", "quality_report_path")]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise ModelNotReady(f"Model not trained yet (missing {', '.join(missing)}). Train it first.")
    key = tuple(p.stat().st_mtime for p in paths)
    if _cache["key"] != key:
        loaded = []
        for p in paths[:3]:
            with open(p, "rb") as f:
                loaded.append(pickle.load(f))
        loaded.append(json.loads(paths[3].read_text()))
        _cache.update(key=key, artifacts=tuple(loaded))
    return _cache["artifacts"]


def predict(record: dict) -> dict:
    config = load_config()
    med_model, meal_model, encoders, quality_report = load_artifacts()

    race, bmi, hba1c, age = (record.get(k) for k in ("race_demographic", "bmi", "baseline_hba1c", "age"))
    quality = score_record(race, bmi, hba1c, age)

    unknown = config["data"]["unknown_race_label"]
    race_value = race if race in encoders["race_demographic"].classes_ else unknown
    bmi_value = bmi if bmi is not None else quality_report["bmi_median"]
    hba1c_value = hba1c if hba1c is not None else 0.0
    age_value = age if age is not None else 50

    features = pd.DataFrame([{
        "race_demographic": encoders["race_demographic"].transform([race_value])[0],
        "age": age_value,
        "bmi": bmi_value,
        "baseline_hba1c": hba1c_value,
    }])[FEATURES]

    med = encoders[config["data"]["target_medication"]].inverse_transform(med_model.predict(features))[0]
    meal = encoders[config["data"]["target_meal"]].inverse_transform(meal_model.predict(features))[0]

    return {
        "data_quality_index": quality["score"],
        "data_quality_issues": quality["issues"],
        "data_quality_issue_labels": [ISSUE_LABELS[i] for i in quality["issues"]],
        "low_quality_warning": quality["score"] < config["min_quality_threshold"],
        "features_used": {"race_demographic": race_value, "age": age_value, "bmi": bmi_value,
                          "baseline_hba1c": hba1c_value},
        "predicted_medication_strategy": med,
        "predicted_dietary_strategy": meal,
        "recommended_provider_routing": recommend_provider(hba1c_value, bmi_value, med),
    }

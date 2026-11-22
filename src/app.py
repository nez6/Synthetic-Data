import pickle
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config, resolve
from src.preprocess import FEATURES
from src.quality import score_record
from src.routing import recommend_provider

app = FastAPI(title="Demographic Clinical Router & Quality Engine API")


@lru_cache(maxsize=1)
def load_artifacts():
    artifacts = load_config()["artifacts"]
    try:
        with open(resolve(artifacts["med_model_path"]), "rb") as f:
            med_model = pickle.load(f)
        with open(resolve(artifacts["meal_model_path"]), "rb") as f:
            meal_model = pickle.load(f)
        with open(resolve(artifacts["encoder_path"]), "rb") as f:
            encoders = pickle.load(f)
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model artifacts missing ({e.filename}). Run `python -m src.train` first.")
    return med_model, meal_model, encoders


class ClinicalInput(BaseModel):
    race_demographic: Optional[str] = None
    age: int = Field(ge=0, le=120)
    bmi: Optional[float] = None
    baseline_hba1c: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_care_and_provider_routing(patient: ClinicalInput):
    config = load_config()
    med_model, meal_model, encoders = load_artifacts()

    quality = score_record(patient.race_demographic, patient.bmi, patient.baseline_hba1c)

    race = patient.race_demographic or config["data"]["unknown_race_label"]
    race_encoder = encoders["race_demographic"]
    if race not in race_encoder.classes_:
        race = config["data"]["unknown_race_label"]
    race_encoded = race_encoder.transform([race])[0]

    # Missing BMI: fall back to a typical value so a prediction can still be made;
    # the DQI already reflects the gap.
    bmi = patient.bmi if patient.bmi is not None else 30.0

    features = pd.DataFrame([{
        "race_demographic": race_encoded,
        "age": patient.age,
        "bmi": bmi,
        "baseline_hba1c": patient.baseline_hba1c,
    }])[FEATURES]

    decoded_med = encoders[config["data"]["target_medication"]].inverse_transform(med_model.predict(features))[0]
    decoded_meal = encoders[config["data"]["target_meal"]].inverse_transform(meal_model.predict(features))[0]

    return {
        "input_record_data_quality_index": f"{quality['score']}%",
        "data_quality_issues": quality["issues"],
        "low_quality_warning": quality["score"] < config["min_quality_threshold"],
        "predicted_medication_strategy": decoded_med,
        "predicted_dietary_strategy": decoded_meal,
        "recommended_provider_routing": recommend_provider(patient.baseline_hba1c, bmi, decoded_med),
    }

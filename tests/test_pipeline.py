"""End-to-end: train the models, then exercise the API."""
import pytest
from fastapi.testclient import TestClient

from src.data_ingestion import care_tier, generate_data
from src.train import run_training


@pytest.fixture(scope="module")
def client():
    metrics = run_training()
    assert metrics["medication_accuracy"] > 0.9
    assert metrics["meal_accuracy"] > 0.9
    from src.app import app, load_artifacts
    load_artifacts.cache_clear()
    return TestClient(app)


def test_labels_do_not_depend_on_race():
    df = generate_data(500, seed=0)
    assert all(
        care_tier(h) in ("intensive", "combination", "mono") for h in df["baseline_hba1c"]
    )
    assert df["race_demographic"].isna().any()
    assert df["bmi"].isna().sum() == 40
    assert (df["baseline_hba1c"] == 99.0).sum() == 20


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_severe_patient(client):
    res = client.post("/predict", json={
        "race_demographic": "Hispanic/Latino", "age": 52, "bmi": 31.2, "baseline_hba1c": 10.5})
    assert res.status_code == 200
    body = res.json()
    assert body["input_record_data_quality_index"] == "100%"
    assert body["predicted_medication_strategy"].startswith("Insulin")
    assert "Endocrinology" in body["recommended_provider_routing"]["specialty"]


def test_predict_missing_race_flags_quality(client):
    res = client.post("/predict", json={"age": 40, "bmi": 24.0, "baseline_hba1c": 6.2})
    body = res.json()
    assert body["input_record_data_quality_index"] == "75%"
    assert body["data_quality_issues"] == ["missing_race"]
    assert body["predicted_medication_strategy"].startswith("Monotherapy")


def test_predict_implausible_lab_warns(client):
    res = client.post("/predict", json={
        "race_demographic": "Asian", "age": 40, "bmi": 24.0, "baseline_hba1c": 99.0})
    body = res.json()
    assert body["input_record_data_quality_index"] == "50%"
    assert body["low_quality_warning"] is True

"""End to end on 12 real Synthea patients (tests/fixtures/pools): request → validate → ML → deliver → API."""
import gzip
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import mlpipe, pools, runner
from src.request import default_request
from tools.data_quality_profile import profile_paths

FIXTURES = Path(__file__).parent / "fixtures" / "pools"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runs")
    mp = pytest.MonkeyPatch()
    mp.setattr(pools, "pools_dir", lambda: FIXTURES)
    mp.setattr(runner, "runs_dir", lambda: tmp)
    mp.setattr(mlpipe, "TRACKING_URI", f"sqlite:///{tmp / 'mlflow.db'}")
    yield tmp
    mp.undo()


@pytest.fixture(scope="module")
def report(env):
    req = default_request()
    req.update(max_per_cohort=6, delivery_size=8, label="fixture run")
    req["faults"]["missing_birthdate"] = 20
    return runner.run(req)


def test_run_tests_every_patient_and_delivers_passing_ones(report):
    assert report["n_tested"] == 12 and set(report["selection"]) == {"A", "B"}
    assert report["faults_applied"]["missing_birthdate"] == 2
    assert 0 < report["n_delivered"] <= 8
    o = report["cohort_scores"]["Overall"]
    assert o["validated_pass_rate"] > o["baseline_pass_rate"]
    for b in runner.load_delivery(report["run_id"]):
        patient = b["entry"][0]["resource"] if b["entry"][0]["resource"]["resourceType"] == "Patient" else \
            next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Patient")
        assert any(s["code"] == "HTEST" for s in patient["meta"]["security"])


def test_missing_birthdate_is_never_repaired(report):
    recs = runner.load_run(report["run_id"])["records"]
    hit = [r for r in recs if "missing_birthdate" in r["faults"]]
    assert hit and all(not r["pass_after"] for r in hit)


def test_trimmed_cohort_a_records_fit_the_targets_and_keep_history(report):
    recs = runner.load_run(report["run_id"])["records"]
    for r in recs:
        if r["cohort"] == "A" and r["pass_after"]:
            raw, validated = runner.record_before_after(report["run_id"], r["patient_id"])
            counts = Counter(e["resource"]["resourceType"] for e in validated["entry"])
            assert 2 <= counts["Encounter"] <= 5
            assert 20 <= counts["Observation"] <= 50       # lab history kept when old visits are trimmed
            assert 5 <= counts["Immunization"] <= 10


def test_ml_split_is_per_cohort(report):
    ml = runner.load_run(report["run_id"])["ml"]
    for c in ("A", "B"):
        n = ml["split_counts"][c]
        assert sum(n.values()) == 6 and n["train"] >= n["test"]
    assert set(ml["models"]) == {"raw", "validated"}
    assert ml["features"] == ["age", "sex", "race_ethnicity", "cohort"]


def test_ml_learns_a_demographic_pattern():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"patient_id": [f"p{i}" for i in range(n)], "age": rng.integers(18, 90, n),
                       "sex": rng.choice(["female", "male"], n), "race_ethnicity": rng.choice(["A", "B"], n),
                       "cohort": rng.choice(["A", "B"], n)})
    df["passed_before"] = df["age"] < 50                       # older patients fail before validation
    df["passed_after"] = rng.random(n) < 0.92                  # after: failures unrelated to demographics
    df["split"] = mlpipe.split(df, {"train": .6, "test": .2, "validation": .2}, 1)
    assert (df.groupby("cohort")["split"].value_counts(normalize=True).xs("train", level=1) > 0.55).all()
    raw, _ = mlpipe.train_one(df, "passed_before", 1)
    val, _ = mlpipe.train_one(df, "passed_after", 1)
    assert raw["validation"]["roc_auc"] > 0.9
    assert max(raw["importance"], key=raw["importance"].get) == "age"
    assert val["validation"]["roc_auc"] < 0.7


def test_profile_script_on_synthea_fixtures(tmp_path):
    prof = profile_paths([str(FIXTURES)], k_min=0, source="fixtures")
    assert prof["n_patients"] == 12 and prof["counts"]["state"] == {"MA": 12}
    assert prof["numeric"]["systolic_bp"]["n"] == 12          # read from the BP panel's component
    with pytest.raises(SystemExit):
        profile_paths([str(FIXTURES)], k_min=13)      # fewer patients than k → refused


def test_api(env, report):
    from src.app import app
    client = TestClient(app)
    assert client.get("/requests/default").json()["delivery_size"] == 100
    assert client.get("/tools/data_quality_profile.py").text.startswith("#!/usr/bin/env python3")
    run_id = report["run_id"]
    assert client.get(f"/runs/{run_id}").json()["n_tested"] == 12
    csv = client.get(f"/runs/{run_id}/dataset", params={"format": "csv"})
    assert csv.status_code == 200 and csv.text.splitlines()[0].startswith("patient_id")
    nd = client.get(f"/runs/{run_id}/dataset", params={"format": "ndjson"})
    first = json.loads(gzip.decompress(nd.content).splitlines()[0])
    assert first["resourceType"] == "Bundle"
    pid = runner.dataset_table(run_id)["patient_id"].iloc[0]
    r5 = client.get(f"/runs/{run_id}/patients/{pid}", params={"fhir_version": "R5"}).json()
    assert r5["resourceType"] == "Bundle" and "timestamp" in r5
    pred = client.post("/predict", json={"age": 70, "sex": "female", "race_ethnicity": "White", "cohort": "B"})
    assert pred.status_code == 200 and 0 <= pred.json()["probability_pass"] <= 1

    job = client.post("/runs", json={"request": {**default_request(), "max_per_cohort": 3, "delivery_size": 4}})
    assert job.status_code == 202
    for _ in range(120):
        status = client.get(f"/jobs/{job.json()['job_id']}").json()
        if status["status"] != "running":
            break
        time.sleep(0.5)
    assert status["status"] == "done", status

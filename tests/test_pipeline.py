"""End-to-end: generate FHIR data, train, then exercise the REST API, FHIR API and HL7 listener."""
import socket

import pytest
from fastapi.testclient import TestClient

from src.data_ingestion import generate_records
from src.fhir.builder import build_bundle
from src.hl7v2 import inbox, message as m
from src.hl7v2.listener import handle_message
from src.hl7v2.mllp import send, start_in_thread
from src.train import run_training


@pytest.fixture(scope="module")
def summary():
    return run_training(regenerate=True)


@pytest.fixture(scope="module")
def client(summary):
    from src.app import app
    return TestClient(app)  # no context manager → lifespan (MLLP on 2575) is not started


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_training_reads_fhir_and_filters_bad_data(summary):
    assert summary["n_bundles"] == 1200
    q = summary["quality"]
    assert q["rows_kept"] < q["rows_total"]
    assert summary["medication_accuracy"] > 0.9
    assert "Unknown" in summary["per_group_accuracy"]


def test_generated_faults():
    recs = generate_records(1200, seed=0)
    assert sum(r["bmi"] is None for r in recs) == 40
    assert sum(r["baseline_hba1c"] == 99.0 for r in recs) == 20
    assert any(r["race_demographic"] is None for r in recs)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_severe_patient(client):
    body = client.post("/predict", json={"race_demographic": "Hispanic/Latino", "age": 52,
                                         "bmi": 31.2, "baseline_hba1c": 10.5}).json()
    assert body["input_record_data_quality_index"] == "100%"
    assert body["predicted_medication_strategy"].startswith("Insulin")
    assert "Endocrinology" in body["recommended_provider_routing"]["specialty"]


def test_predict_missing_race_flags_quality(client):
    body = client.post("/predict", json={"age": 40, "bmi": 24.0, "baseline_hba1c": 6.2}).json()
    assert body["data_quality_index"] == 75
    assert body["data_quality_issues"] == ["missing_race"]


@pytest.mark.parametrize("version", ["DSTU2", "STU3", "R4", "R4B", "R5"])
def test_fhir_predict_every_version(client, version):
    rec = generate_records(1, seed=3)[0]
    res = client.post(f"/fhir/{version}/Bundle/$predict", json=build_bundle(rec, version))
    assert res.status_code == 200
    body = res.json()
    assert body["extracted_record"]["baseline_hba1c"] == rec["baseline_hba1c"]
    assert body["schema_validation"]["valid"] in (True, None)


def test_fhir_metadata(client):
    assert client.get("/fhir/R5/metadata").json()["fhirVersion"] == "5.0.0"
    assert client.get("/fhir/R9/metadata").status_code == 404


def test_hl7_listener_end_to_end(summary):
    port = _free_port()
    server = start_in_thread("127.0.0.1", port, handle_message)
    try:
        rec = {**generate_records(1, seed=11, id_prefix="hl7")[0], "baseline_hba1c": 10.2}
        ack = m.parse(send("127.0.0.1", port, m.build_oru(rec, "E2E-1")))
        msa = m.first(ack, "MSA")
        assert m.field(msa, 1) == "AA" and m.field(msa, 2) == "E2E-1"

        stored = next(r for r in inbox.recent(20) if r["control_id"] == "E2E-1")
        assert stored["fhir_bundle"]["resourceType"] == "Bundle"
        assert "Endocrinology" in stored["prediction"]["recommended_provider_routing"]["specialty"]

        adt = m.build_oru(rec, "E2E-2").replace("ORU^R01^ORU_R01", "ADT^A08^ADT_A01")
        assert m.field(m.first(m.parse(send("127.0.0.1", port, adt)), "MSA"), 1) == "AR"
    finally:
        server.shutdown()

"""Cloud data ingestion (package → serve over HTTP → download + verify) and the endpoints the dashboard uses."""
import functools
import http.server
import json
import shutil
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import cloud_data, mlpipe, pools, runner
from src.request import default_request

FIXTURES = Path(__file__).parent / "fixtures" / "pools"


@pytest.fixture()
def fake_bucket(tmp_path, monkeypatch):
    """Package the fixture pools and serve them from a local HTTP server standing in for GCS."""
    monkeypatch.setattr(pools, "pools_dir", lambda: FIXTURES)
    monkeypatch.setattr(cloud_data, "DIST", tmp_path / "dist")
    manifest = cloud_data.package()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "dist"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(cloud_data, "base_url", lambda: f"http://127.0.0.1:{server.server_port}")
    yield tmp_path, manifest
    server.shutdown()


def test_download_fetches_verifies_and_unpacks_every_pool(fake_bucket, monkeypatch):
    tmp, manifest = fake_bucket
    dest = tmp / "pools"
    monkeypatch.setattr(pools, "pools_dir", lambda: dest)
    assert cloud_data.download(progress=lambda m: None) == {name: "downloaded" for name in manifest["pools"]}
    for name in manifest["pools"]:
        assert sorted(p.name for p in (dest / name / "patients").iterdir()) == \
            sorted(p.name for p in (FIXTURES / name / "patients").iterdir())
    # second call is a no-op: the checksum marker matches
    assert set(cloud_data.download(progress=lambda m: None).values()) == {"present"}


def test_download_rejects_a_corrupted_archive(fake_bucket, monkeypatch):
    tmp, manifest = fake_bucket
    name = next(iter(manifest["pools"]))
    (tmp / "dist" / manifest["pools"][name]["file"]).write_bytes(b"not the archive")
    monkeypatch.setattr(pools, "pools_dir", lambda: tmp / "pools")
    with pytest.raises(RuntimeError, match="checksum"):
        cloud_data.download(progress=lambda m: None)
    assert not (tmp / "pools" / name).exists()


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    mp = pytest.MonkeyPatch()
    mp.setattr(pools, "pools_dir", lambda: FIXTURES)
    mp.setattr(runner, "runs_dir", lambda: tmp)
    mp.setattr(mlpipe, "TRACKING_URI", f"sqlite:///{tmp / 'mlflow.db'}")
    req = default_request()
    req.update(max_per_cohort=6, delivery_size=8, label="dashboard api")
    report = runner.run(req)
    from src.app import app
    with TestClient(app) as client:
        yield client, report
    mp.undo()


def test_full_run_payload_has_everything_the_dashboard_reads(api):
    client, report = api
    body = client.get(f"/runs/{report['run_id']}/full").json()
    assert set(body) == {"report", "request", "records", "ml"}
    assert len(body["records"]) == report["n_tested"]


def test_delivered_records_plot_and_before_after(api):
    client, report = api
    rid = report["run_id"]
    recs = client.get(f"/runs/{rid}/delivered-records").json()
    assert len(recs) == report["n_delivered"] and all("race_ethnicity" in r for r in recs)
    plot = client.get(f"/runs/{rid}/plot.png")
    assert plot.status_code == 200 and plot.content[:4] == b"\x89PNG"
    pid = client.get(f"/runs/{rid}/full").json()["records"][0]["patient_id"]
    both = client.get(f"/runs/{rid}/patients/{pid}/before-after").json()
    assert both["raw"]["resourceType"] == both["validated"]["resourceType"] == "Bundle"
    assert client.get(f"/runs/{rid}/patients/nope/before-after").status_code == 404


def test_pools_eligible_counts_per_cohort(api):
    client, _ = api
    rows = client.post("/pools/eligible", json=default_request()).json()
    assert [r["cohort"] for r in rows] == ["A", "B"] and all(r["eligible"] >= 0 for r in rows)


def test_health_reports_bootstrap_state(api):
    client, _ = api
    assert set(client.get("/health").json()["bootstrap"]) == {"stage", "message", "error"}


def test_predict_returns_a_probability(api):
    client, _ = api
    res = client.post("/predict", json={"age": 70, "sex": "female", "race_ethnicity": "White", "cohort": "B"})
    assert res.status_code == 200 and 0 <= res.json()["probability_pass"] <= 1

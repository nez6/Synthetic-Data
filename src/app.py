"""REST API for the Synthetic Test Data Creator.

    01  GET  /tools/data_quality_profile.py      download the profiling script (run it on your data)
        GET  /requests/default                    the default request (75 A / 25 B, resource targets)
        POST /requests/from-profile               request pre-filled from an aggregate profile
    02-04 POST /runs                              submit a request → job id (runs in the background)
        GET  /jobs/{job_id}                       progress and, when done, the run id
        GET  /runs, /runs/{run_id}                results: cohort scores, checks, ML report
        GET  /runs/{run_id}/dataset?format=…      pull the custom dataset (ndjson | csv | zip)
        GET  /runs/{run_id}/patients/{id}?fhir_version=…   one delivered patient, any FHIR version
    ML  POST /predict                             P(record passes validation) from demographics only
    Also: /pools, /fhir/{version}/metadata, /fhir/{version}/$convert, /hl7/inbox, /health
"""
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from src import pools, runner
from src.config import load_config, resolve
from src.fhir.builder import build_bundle
from src.fhir.parser import bundle_to_record
from src.fhir.versions import VERSIONS
from src.hl7v2 import inbox
from src.hl7v2.mllp import is_listening, start_in_thread
from src.inference import ModelNotReady, predict_pass
from src.request import apply_profile, default_request, problems

_listener = {"server": None, "error": None}
_jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(_app):
    cfg = load_config()["hl7"]
    if cfg.get("start_with_api"):
        from src.hl7v2.listener import handle_message
        try:
            _listener["server"] = start_in_thread(cfg["host"], cfg["port"], handle_message)
        except OSError as e:
            _listener["error"] = f"Could not bind {cfg['host']}:{cfg['port']}: {e}"
    yield
    if _listener["server"]:
        _listener["server"].shutdown()


app = FastAPI(title="Synthetic Test Data Creator API", lifespan=lifespan,
              description="Request a custom synthetic FHIR test dataset, validate it, and pull it back.")


def _run_or_404(run_id: str):
    if not (runner.runs_dir() / run_id / "report.json").exists():
        raise HTTPException(404, f"No run {run_id}")


# ----------------------------------------------------------------------------- basics

@app.get("/health")
def health():
    cfg = load_config()["hl7"]
    return {"status": "ok", "pools": pools.list_pools(), "runs": len(runner.list_runs()),
            "hl7_listener": {"port": cfg["port"], "listening": is_listening(cfg["host"], cfg["port"]),
                             "error": _listener["error"]}}


@app.get("/pools")
def list_pools():
    return pools.list_pools()


# ----------------------------------------------------------------------------- 01 request

@app.get("/tools/data_quality_profile.py")
def download_profile_script():
    return FileResponse(resolve(load_config()["paths"]["profile_script"]), media_type="text/x-python",
                        filename="data_quality_profile.py")


@app.get("/requests/default")
def get_default_request():
    return default_request()


@app.post("/requests/from-profile")
def request_from_profile(profile: dict = Body(...)):
    if "counts" not in profile or "n_patients" not in profile:
        raise HTTPException(400, "Body must be a profile produced by data_quality_profile.py")
    req, notes = apply_profile(default_request(), profile)
    return {"request": req, "notes": notes}


# ----------------------------------------------------------------------------- 02-04 runs

class RunBody(BaseModel):
    request: dict | None = Field(default=None, description="A request; omit to use the default")
    real_profile: dict | None = Field(default=None, description="Optional aggregate profile to compare against")


def _work(job_id: str, request: dict, profile: dict | None):
    job = _jobs[job_id]
    try:
        report = runner.run(request, real_profile=profile,
                            progress=lambda f, msg: job.update(progress=round(f, 3), message=msg))
        job.update(status="done", run_id=report["run_id"], progress=1.0)
    except Exception as e:  # noqa: BLE001
        job.update(status="failed", error=str(e), trace=traceback.format_exc()[-2000:])


@app.post("/runs", status_code=202)
def submit_run(body: RunBody = Body(default=RunBody())):
    request = body.request or default_request()
    if body.real_profile and not body.request:
        request, _ = apply_profile(request, body.real_profile)
    issues = problems(request)
    if issues:
        raise HTTPException(422, issues)
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"job_id": job_id, "status": "running", "progress": 0.0, "message": "queued",
                     "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    threading.Thread(target=_work, args=(job_id, request, body.real_profile), daemon=True).start()
    return _jobs[job_id]


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Unknown job")
    return _jobs[job_id]


@app.get("/runs")
def list_runs():
    return runner.list_runs()


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    _run_or_404(run_id)
    data = runner.load_run(run_id)
    report = data["report"]
    return {**{k: v for k, v in report.items() if k != "profiles"}, "ml_report": data["ml"]}


@app.get("/runs/{run_id}/dataset")
def get_dataset(run_id: str, format: str = "ndjson"):
    """Pull the delivered dataset: ndjson (FHIR R4 Bundles), csv (one row per patient) or zip (everything)."""
    _run_or_404(run_id)
    if format == "ndjson":
        path = runner.runs_dir() / run_id / "delivery.ndjson.gz"
        return FileResponse(path, media_type="application/gzip", filename=f"{run_id}.ndjson.gz")
    if format == "csv":
        return Response(runner.dataset_table(run_id).to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={run_id}.csv"})
    if format == "zip":
        return Response(runner.zip_run(run_id), media_type="application/zip",
                        headers={"Content-Disposition": f"attachment; filename={run_id}.zip"})
    raise HTTPException(400, "format must be ndjson, csv or zip")


@app.get("/runs/{run_id}/patients/{patient_id}")
def get_patient(run_id: str, patient_id: str, fhir_version: str = "R4"):
    """One delivered patient. R4 is the full validated record; other versions are a one-visit snapshot."""
    _run_or_404(run_id)
    for b in runner.load_delivery(run_id):
        rec = bundle_to_record(b)
        if rec["patient_id"] == patient_id:
            if fhir_version == "R4":
                return b
            if fhir_version not in VERSIONS:
                raise HTTPException(400, f"fhir_version must be one of {list(VERSIONS)}")
            return build_bundle(rec, fhir_version)
    raise HTTPException(404, f"{patient_id} is not in the delivery set of {run_id}")


# ----------------------------------------------------------------------------- ML

class PredictInput(BaseModel):
    age: int | None = Field(default=None, ge=0, le=120)
    sex: str | None = Field(default=None, examples=["female"])
    race_ethnicity: str | None = Field(default=None, examples=["Black/African American"])
    cohort: str = Field(default="A", examples=["A", "B"])
    run_id: str | None = None
    model: str = Field(default="validated", pattern="^(validated|raw)$")


@app.post("/predict")
def predict(body: PredictInput):
    """Probability that a record with these demographics passes validation (model sees demographics only)."""
    try:
        out = predict_pass([body.model_dump()], run_id=body.run_id, which=body.model)
    except ModelNotReady as e:
        raise HTTPException(503, str(e))
    return {"run_id": out["run_id"], "model": out["model"], **out["predictions"][0]}


# ----------------------------------------------------------------------------- FHIR + HL7

@app.get("/fhir/{version}/metadata")
def capability_statement(version: str):
    if version not in VERSIONS:
        raise HTTPException(404, f"Unknown FHIR version {version!r}")
    return {"resourceType": "CapabilityStatement", "status": "active", "kind": "instance",
            "date": datetime.now(timezone.utc).date().isoformat(),
            "fhirVersion": VERSIONS[version]["fhirVersion"], "format": ["json"]}


@app.post("/fhir/{version}/$convert")
def fhir_convert(version: str, bundle: dict = Body(...)):
    if version not in VERSIONS:
        raise HTTPException(404, f"Unknown FHIR version {version!r}")
    return build_bundle(bundle_to_record(bundle), version)


@app.get("/hl7/inbox")
def hl7_inbox(limit: int = 50):
    return inbox.recent(limit)

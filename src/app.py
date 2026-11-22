"""REST API: plain JSON prediction, FHIR endpoints (all versions) and HL7 inbox.

The same process also runs the HL7 v2 MLLP listener (config: hl7.start_with_api).
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config
from src.fhir.builder import build_bundle
from src.fhir.parser import bundle_to_record, detect_version
from src.fhir.validate import validate_bundle
from src.fhir.versions import VERSIONS
from src.hl7v2 import inbox
from src.hl7v2.mllp import is_listening, start_in_thread
from src.inference import ModelNotReady, predict

_listener = {"server": None, "error": None}


@asynccontextmanager
async def lifespan(_app):
    cfg = load_config()["hl7"]
    if cfg.get("start_with_api"):
        from src.hl7v2.listener import handle_message
        try:
            _listener["server"] = start_in_thread(cfg["host"], cfg["port"], handle_message)
            print(f"HL7 v2 MLLP listener started on {cfg['host']}:{cfg['port']}")
        except OSError as e:
            _listener["error"] = f"Could not bind {cfg['host']}:{cfg['port']}: {e}"
            print(_listener["error"])
    yield
    if _listener["server"]:
        _listener["server"].shutdown()


app = FastAPI(title="Demographic Clinical Router & Quality Engine API", lifespan=lifespan)


def _run(record: dict) -> dict:
    try:
        return predict(record)
    except ModelNotReady as e:
        raise HTTPException(503, str(e))


def _check_version(version: str):
    if version not in VERSIONS:
        raise HTTPException(404, f"Unknown FHIR version {version!r}. Use one of {list(VERSIONS)}")


# ----------------------------------------------------------------------------- basics

class ClinicalInput(BaseModel):
    race_demographic: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=0, le=120)
    bmi: Optional[float] = None
    baseline_hba1c: Optional[float] = None


@app.get("/health")
def health():
    cfg = load_config()["hl7"]
    return {"status": "ok",
            "hl7_listener": {"host": cfg["host"], "port": cfg["port"],
                             "listening": is_listening(cfg["host"], cfg["port"]),
                             "error": _listener["error"]}}


@app.post("/predict")
def predict_care_and_provider_routing(patient: ClinicalInput):
    result = _run(patient.model_dump())
    return {"input_record_data_quality_index": f"{result['data_quality_index']}%", **result}


# ----------------------------------------------------------------------------- FHIR

@app.get("/fhir/{version}/metadata")
def capability_statement(version: str):
    _check_version(version)
    return {
        "resourceType": "CapabilityStatement",
        "status": "active",
        "date": datetime.now(timezone.utc).date().isoformat(),
        "kind": "instance",
        "fhirVersion": VERSIONS[version]["fhirVersion"],
        "format": ["json"],
        "rest": [{"mode": "server", "resource": [
            {"type": "Bundle", "operation": [{"name": "predict",
                                              "definition": "POST /fhir/{version}/Bundle/$predict"}]}]}],
    }


@app.post("/fhir/{version}/Bundle/$predict")
def fhir_predict(version: str, bundle: dict = Body(...)):
    """Accept a FHIR Bundle (Patient + Observations), validate it, score DQI and predict."""
    _check_version(version)
    if bundle.get("resourceType") != "Bundle":
        raise HTTPException(400, "Body must be a FHIR Bundle")
    validation = validate_bundle(bundle, version)
    detected, evidence = detect_version(bundle)
    record = bundle_to_record(bundle)
    return {
        "fhir_version_declared": version,
        "fhir_version_detected": detected,
        "detection_evidence": evidence,
        "schema_validation": validation,
        "extracted_record": record,
        **_run(record),
    }


@app.post("/fhir/{version}/$convert")
def fhir_convert(version: str, bundle: dict = Body(...)):
    """Re-express a Bundle (any supported version) in the requested FHIR version."""
    _check_version(version)
    return build_bundle(bundle_to_record(bundle), version)


# ----------------------------------------------------------------------------- HL7

@app.get("/hl7/inbox")
def hl7_inbox(limit: int = 50):
    return inbox.recent(limit)

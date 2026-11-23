"""Run a test-data job end to end and store it:  generate → validate + repair → audit → score.

Each run is saved to data/testdata/<run_id>/:
    spec.json        the cohort spec (key demographics, faults, acceptance criteria)
    baseline.ndjson  raw generated Bundles (with injected faults)
    validated.ndjson every Bundle after the validation script's repairs
    delivery.ndjson  validated Bundles that pass the independent audit (the deliverable)
    records.json     per-record issues, repairs and scores
    report.json      dataset scores, hypothesis tests, demographic fidelity
"""
import io
import json
import re
import zipfile
from datetime import date, datetime, timezone

import pandas as pd

from src.config import load_config, resolve
from src.fhir.parser import bundle_to_record
from src.fhir.validate import validate_bundle
from src.hl7v2.message import build_oru
from src.testdata import VALIDATION_SCRIPT_VERSION
from src.testdata.audit import audit_bundle
from src.testdata.fidelity import age_summary, compare
from src.testdata.generator import SYNTHETIC_ID_SYSTEM, generate
from src.testdata.spec import CohortSpec
from src.testdata.stats import evaluate
from src.testdata.validation import CHECKS, run_validation


def runs_dir():
    return resolve("data/testdata")


def _audit_rules(spec: CohortSpec) -> dict:
    b = load_config()["quality_bounds"]
    return {"min_age": spec.min_age, "max_age": spec.max_age, "min_hba1c_obs": spec.min_hba1c_obs,
            "min_bmi_obs": spec.min_bmi_obs, "hba1c_bounds": (b["hba1c_min"], b["hba1c_max"]),
            "bmi_bounds": (b["bmi_min"], b["bmi_max"]), "synthetic_system": SYNTHETIC_ID_SYSTEM}


def _independent(bundle, rules, as_of, schema: bool):
    failures = audit_bundle(bundle, rules, as_of)
    if schema:
        v = validate_bundle(bundle, "R4")
        failures += [f"schema: {e}" for e in v["errors"][:3]]
    return failures


def run(spec: CohortSpec, schema_validation: bool = True, progress=None) -> dict:
    as_of = date.fromisoformat(spec.as_of)
    rules = _audit_rules(spec)
    bundles, truth = generate(spec)

    records, validated = [], []
    for i, (bundle, tr) in enumerate(zip(bundles, truth)):
        res = run_validation(bundle, spec)
        ind_before = _independent(bundle, rules, as_of, schema_validation)
        ind_after = _independent(res.bundle, rules, as_of, schema_validation)
        records.append({**tr, **res.as_dict(),
                        "independent_failures_before": ind_before, "independent_failures_after": ind_after,
                        "independent_pass_before": not ind_before, "independent_pass_after": not ind_after})
        validated.append(res.bundle)
        if progress:
            progress((i + 1) / len(bundles))

    col = lambda k: [r[k] for r in records]  # noqa: E731
    scores = evaluate(col("passed_before"), col("passed"), col("independent_pass_before"),
                      col("independent_pass_after"), col("conformance_before"), col("conformance_after"),
                      spec.h0_rate, spec.target_rate, spec.alpha)
    delivered_flags = col("independent_pass_after")
    delivery = [b for b, ok in zip(validated, delivered_flags) if ok]

    checks = []
    for cid, desc in CHECKS.items():
        checks.append({
            "check": cid, "description": desc,
            "records_failing_before": sum(any(i["check_id"] == cid and i["severity"] == "error"
                                              for i in r["issues_before"]) for r in records),
            "records_failing_after": sum(any(i["check_id"] == cid and i["severity"] == "error"
                                             for i in r["issues_after"]) for r in records),
            "warnings": sum(any(i["check_id"] == cid and i["severity"] == "warning"
                                for i in r["issues_after"]) for r in records),
            "repairs": sum(sum(1 for x in r["repairs"] if x["check_id"] == cid) for r in records),
        })

    ages_gen, ages_dlv = age_summary(bundles, as_of), age_summary(delivery, as_of)
    report = {
        "run_id": None,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation_script_version": VALIDATION_SCRIPT_VERSION,
        "label": spec.label, "n": spec.n,
        "scores": scores, "checks": checks,
        "fidelity": compare(spec, truth, delivered_flags),
        "age": {"requested": [spec.min_age, spec.max_age],
                "generated": [min(ages_gen), max(ages_gen), round(sum(ages_gen) / len(ages_gen), 1)] if ages_gen else None,
                "delivered": [min(ages_dlv), max(ages_dlv), round(sum(ages_dlv) / len(ages_dlv), 1)] if ages_dlv else None},
        "n_delivered": len(delivery),
        "schema_validation": schema_validation,
    }

    slug = re.sub(r"[^a-z0-9]+", "-", spec.label.lower()).strip("-")[:30]
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slug}"
    report["run_id"] = run_id
    out = runs_dir() / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "records.json").write_text(json.dumps(records))
    _ndjson(out / "baseline.ndjson", bundles)
    _ndjson(out / "validated.ndjson", validated)
    _ndjson(out / "delivery.ndjson", delivery)
    return report


def _ndjson(path, bundles):
    with open(path, "w") as f:
        for b in bundles:
            f.write(json.dumps(b) + "\n")


def _read_ndjson(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


# ----------------------------------------------------------------------------- reading runs

def list_runs() -> list[dict]:
    d = runs_dir()
    if not d.exists():
        return []
    out = []
    for p in sorted(d.iterdir(), reverse=True):
        if (p / "report.json").exists():
            r = json.loads((p / "report.json").read_text())
            out.append({"run_id": r["run_id"], "label": r["label"], "created": r["created"], "n": r["n"],
                        "delivered": r["n_delivered"],
                        "independent_pass_rate": r["scores"]["independent_after"]["rate"],
                        "reject_h0": r["scores"]["reject_h0"]})
    return out


def load_run(run_id: str) -> dict:
    p = runs_dir() / run_id
    return {"report": json.loads((p / "report.json").read_text()),
            "spec": json.loads((p / "spec.json").read_text()),
            "records": json.loads((p / "records.json").read_text())}


def load_bundles(run_id: str, which: str = "delivery") -> list[dict]:
    return _read_ndjson(runs_dir() / run_id / f"{which}.ndjson")


# ----------------------------------------------------------------------------- exports

def hl7_messages(run_id: str) -> list[tuple[str, str]]:
    """One ORU^R01 per delivered patient (latest HbA1c and BMI)."""
    out = []
    for i, b in enumerate(load_bundles(run_id)):
        rec = bundle_to_record(b)
        out.append((rec["patient_id"], build_oru(rec, control_id=f"TD{i + 1:06d}")))
    return out


def zip_run(run_id: str) -> bytes:
    p = runs_dir() / run_id
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("delivery.ndjson", "baseline.ndjson", "spec.json", "report.json", "records.json"):
            z.write(p / name, f"{run_id}/{name}")
        for pid, msg in hl7_messages(run_id):
            z.writestr(f"{run_id}/hl7/{pid}.hl7", msg.replace("\r", "\n") + "\n")
    return buf.getvalue()


def use_as_training_data(run_id: str) -> int:
    """Replace the project's training dataset (tabs 1-2) with this run's delivered Bundles."""
    cfg = load_config()["data"]
    bundles = load_bundles(run_id)
    bundles_path, source_path = resolve(cfg["bundles_path"]), resolve(cfg["source_path"])
    bundles_path.parent.mkdir(parents=True, exist_ok=True)
    _ndjson(bundles_path, bundles)
    pd.DataFrame([bundle_to_record(b) for b in bundles]).to_csv(source_path, index=False)
    return len(bundles)

"""Run one test-data request end to end ("Four steps produced 120 VAL test patients").

    01 request (+ optional real-data profile)   src/request.py
    02 select Synthea patients from the pools   src/pools.py, src/selection.py
    03 validate and repair, audit, score        src/stdc (validation, audit, stats)
       train / test / validate the ML model     src/mlpipe.py
    04 select and deliver (HTEST-labelled)

Each run is saved to data/runs/<run_id>/:
    request.json, real_profile.json   what was asked for
    records.json                      per-patient selection, faults, issues, repairs, scores
    report.json                       cohort statistics, checks, profiles, comparison
    delivery.ndjson.gz                the delivered FHIR R4 Bundles (one per line)
    ml/                               ML dataset, models, plots, ml_report.json
"""
import copy
import gzip
import io
import json
import math
import random
import re
import zipfile
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime

import pandas as pd

from src import mlpipe, pools
from src.config import load_config, resolve
from src.fhir.parser import bundle_to_record
from src.hl7v2.message import build_oru
from src.request import FAULTS, cohort_specs, problems
from src.selection import inject, plan_faults, select_cohort
from src.stdc import audit as std_audit
from src.stdc import stats as std_stats
from src.stdc.validation import CHECKS, Context, conformance_score, errors, run_validation
from tools.data_quality_profile import profile_bundles

HTEST = {"system": "http://terminology.hl7.org/CodeSystem/v3-ActReason", "code": "HTEST",
         "display": "test health data"}


def runs_dir():
    return resolve(load_config()["paths"]["runs_dir"])


def _label_htest(bundle):
    for e in bundle.get("entry", []):
        r = e["resource"]
        if r.get("resourceType") == "Patient":
            sec = r.setdefault("meta", {}).setdefault("security", [])
            if HTEST not in sec:
                sec.append(dict(HTEST))


def _finite(obj):
    """Replace inf/NaN (e.g. a paired t of +inf when every record improves equally) with None: valid JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def _issue_rows(issues):
    return [asdict(i) for i in issues]


def run(request: dict, real_profile: dict | None = None, schema_check: bool = False, progress=None) -> dict:
    blocking = problems(request)
    if blocking:
        raise ValueError("; ".join(blocking))
    as_of = date.fromisoformat(request["as_of"])
    seed = request["seed"]
    specs = {s.name: s for s in cohort_specs(request)}
    cohorts = {c["name"]: c for c in request["cohorts"]}
    say = progress or (lambda f, m: None)

    # ---- 02 select patients from the Synthea pools
    selected, selection_report = [], {}
    for name, c in cohorts.items():
        rows = pools.load_index(c["pool"])
        if not rows:
            raise RuntimeError(f"Synthea pool '{c['pool']}' is empty. Build it first (python -m src.pools build "
                               f"{c['pool']}).")
        chosen, rep = select_cohort(rows, c, request["max_per_cohort"], request.get("demographics"), seed)
        selection_report[name] = rep
        selected += [(name, r) for r in chosen]

    # ---- optional: inject the requester's data-quality problems
    plan = plan_faults(len(selected), request.get("faults"), seed)

    records, delivered_candidates = [], {n: [] for n in cohorts}
    for i, (cohort_name, row) in enumerate(selected):
        bundle = pools.load_patient(cohorts[cohort_name]["pool"], row)
        rng = random.Random(f"{seed}:inject:{row['patient_id']}")  # per patient, so a record can be rebuilt
        applied = [f for f in plan[i] if inject(bundle, f, rng)]
        ctx = Context(as_of=as_of, cohort=specs[cohort_name])
        res = run_validation(bundle, ctx)
        ind_before = std_audit.audit_bundle(bundle, cohorts[cohort_name], as_of)
        ind_after = std_audit.audit_bundle(res.bundle, cohorts[cohort_name], as_of)
        if schema_check:
            from src.fhir.validate import validate_bundle
            ind_after += [f"schema: {e}" for e in validate_bundle(res.bundle, "R4")["errors"][:3]]
        n_before = len(bundle.get("entry", []))
        n_after = len(res.bundle.get("entry", []))
        records.append({
            "patient_id": row["patient_id"], "cohort": cohort_name, "pool": cohorts[cohort_name]["pool"],
            "file": row["file"], "age": row["age"], "sex": row["sex"], "race_ethnicity": row["race_ethnicity"],
            "condition_groups": row["condition_groups"], "faults": applied,
            "issues_before": _issue_rows(res.issues_before), "repairs": [asdict(r) for r in res.repairs],
            "issues_after": _issue_rows(res.issues_after),
            "passed_before": not errors(res.issues_before), "passed_after": res.passed,
            "conformance_before": conformance_score(res.issues_before),
            "conformance_after": conformance_score(res.issues_after),
            "independent_failures_before": ind_before, "independent_failures_after": ind_after,
            "independent_before": not ind_before, "independent_after": not ind_after,
            # A record passes only if BOTH the validation script and the independent audit pass it.
            "pass_before": not errors(res.issues_before) and not ind_before,
            "pass_after": res.passed and not ind_after,
            "resources_before": n_before, "resources_after": n_after,
            "removed_by_type": dict(Counter(e["resource"]["resourceType"] for e in bundle["entry"])
                                    - Counter(e["resource"]["resourceType"] for e in res.bundle["entry"])),
        })
        if res.passed and not ind_after:
            delivered_candidates[cohort_name].append((row["patient_id"], res.bundle))
        say((i + 1) / len(selected) * 0.8, f"Validated {i + 1}/{len(selected)} patients")

    # ---- statistics per cohort (Bonferroni across cohorts, as in the reference test plan)
    acc = request["acceptance"]
    alpha = std_stats.bonferroni(acc["alpha"], len(cohorts))
    by_cohort = {}
    for name in list(cohorts) + ["Overall"]:
        rs = [r for r in records if name == "Overall" or r["cohort"] == name]
        if not rs:
            continue
        ev = std_stats.evaluate([r["pass_before"] for r in rs], [r["pass_after"] for r in rs],
                                [r["conformance_before"] for r in rs], [r["conformance_after"] for r in rs],
                                acc["h0_rate"], acc["target_rate"], alpha)
        ev["script_baseline_pass_rate"] = sum(r["passed_before"] for r in rs) / len(rs)
        ev["script_pass_rate"] = sum(r["passed_after"] for r in rs) / len(rs)
        ev["audit_baseline_pass_rate"] = sum(r["independent_before"] for r in rs) / len(rs)
        ev["audit_pass_rate"] = sum(r["independent_after"] for r in rs) / len(rs)
        ev["patients_with_removals"] = sum(1 for r in rs if r["resources_after"] < r["resources_before"])
        ev["mean_share_removed"] = sum(1 - r["resources_after"] / r["resources_before"] for r in rs) / len(rs)
        by_cohort[name] = ev

    checks = [{"check": cid, "description": desc,
               "failing_before": sum(any(i["check_id"] == cid and i["severity"] == "error" for i in r["issues_before"])
                                     for r in records),
               "failing_after": sum(any(i["check_id"] == cid and i["severity"] == "error" for i in r["issues_after"])
                                    for r in records),
               "warnings_after": sum(any(i["check_id"] == cid and i["severity"] == "warning"
                                         for i in r["issues_after"]) for r in records),
               "repairs": sum(sum(1 for x in r["repairs"] if x["check_id"] == cid) for r in records)}
              for cid, desc in CHECKS.items()]

    # ---- 04 select and deliver
    sel_rng = random.Random(f"{seed}:deliver")
    delivery, delivery_counts, need = [], {}, {}
    remaining = request["delivery_size"]
    for i, (name, c) in enumerate(cohorts.items()):
        need[name] = remaining if i == len(cohorts) - 1 else round(request["delivery_size"] * c["share"])
        remaining -= need[name]
    for name, cands in delivered_candidates.items():
        picks = sel_rng.sample(cands, min(need[name], len(cands)))
        for _, b in picks:
            _label_htest(b)
        delivery += [b for _, b in picks]
        delivery_counts[name] = {"requested": need[name], "delivered": len(picks), "passing_available": len(cands)}

    # ---- synthetic vs real: profile the generated and the delivered data with the same script
    say(0.85, "Profiling the synthetic data")
    raw_bundles = [pools.load_patient(cohorts[c]["pool"], {"file": r["file"]}) for c, r in selected]
    synth_raw = profile_bundles(raw_bundles, as_of=as_of, k_min=0, source="synthetic (as selected)")
    synth_delivered = profile_bundles(delivery, as_of=as_of, k_min=0, source="synthetic (delivered)") \
        if delivery else None

    # ---- save, then the ML step
    slug = re.sub(r"[^a-z0-9]+", "-", request["label"].lower()).strip("-")[:30]
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slug}"
    out = runs_dir() / run_id
    out.mkdir(parents=True, exist_ok=True)

    say(0.9, "Training and validating the model")
    ml_frame = pd.DataFrame(records).drop(columns=["passed_before", "passed_after"]).rename(
        columns={"pass_before": "passed_before", "pass_after": "passed_after"})
    ml = mlpipe.run(ml_frame, out / "ml", request, run_id)

    report = {
        "run_id": run_id, "created": datetime.now().isoformat(timespec="seconds"),
        "label": request["label"], "as_of": request["as_of"],
        "synthea": {name: {k: (pools.load_info(c["pool"]) or {}).get(k)
                           for k in ("synthea_version", "jar_sha256", "state", "years_of_history")}
                    for name, c in cohorts.items()},
        "selection": selection_report, "n_tested": len(records),
        "cohort_scores": by_cohort, "alpha_per_cohort": alpha, "checks": checks,
        "faults_requested": request.get("faults"),
        "faults_applied": dict(Counter(f for r in records for f in r["faults"])),
        "delivery": delivery_counts, "n_delivered": len(delivery),
        "profiles": {"real": real_profile, "synthetic_raw": synth_raw, "synthetic_delivered": synth_delivered},
        "ml": {k: ml[k] for k in ("final_scores", "split_counts", "mlflow_run_id")},
    }
    (out / "request.json").write_text(json.dumps(request, indent=2))
    if real_profile:
        (out / "real_profile.json").write_text(json.dumps(real_profile, indent=2))
    (out / "records.json").write_text(json.dumps(records))
    report = _finite(report)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    with gzip.open(out / "delivery.ndjson.gz", "wt", encoding="utf-8") as f:
        for b in delivery:
            f.write(json.dumps(b) + "\n")
    say(1.0, "Done")
    return report


# ----------------------------------------------------------------------------- reading runs

def list_runs() -> list[dict]:
    d = runs_dir()
    out = []
    if d.exists():
        for p in sorted(d.iterdir(), reverse=True):
            if (p / "report.json").exists():
                r = json.loads((p / "report.json").read_text())
                o = r["cohort_scores"].get("Overall", {})
                out.append({"run_id": r["run_id"], "label": r["label"], "created": r["created"],
                            "tested": r["n_tested"], "delivered": r["n_delivered"],
                            "pass_rate_after": o.get("validated_pass_rate"), "decision": o.get("decision")})
    return out


def load_run(run_id: str) -> dict:
    p = runs_dir() / run_id
    return {"report": json.loads((p / "report.json").read_text()),
            "request": json.loads((p / "request.json").read_text()),
            "records": json.loads((p / "records.json").read_text()),
            "ml": json.loads((p / "ml" / "ml_report.json").read_text())}


def load_delivery(run_id: str) -> list[dict]:
    with gzip.open(runs_dir() / run_id / "delivery.ndjson.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record_before_after(run_id: str, patient_id: str) -> tuple[dict, dict]:
    """Rebuild one patient's raw and validated bundle (raw bundles are not stored per run)."""
    data = load_run(run_id)
    rec = next(r for r in data["records"] if r["patient_id"] == patient_id)
    request = data["request"]
    bundle = pools.load_patient(rec["pool"], {"file": rec["file"]})
    raw = copy.deepcopy(bundle)
    rng = random.Random(f"{request['seed']}:inject:{patient_id}")
    for f in rec["faults"]:
        inject(raw, f, rng)
    spec = next(s for s in cohort_specs(request) if s.name == rec["cohort"])
    res = run_validation(raw, Context(as_of=date.fromisoformat(request["as_of"]), cohort=spec))
    return raw, res.bundle


def dataset_table(run_id: str) -> pd.DataFrame:
    """The delivered patients as one flat row each (for CSV export / quick inspection)."""
    rows = []
    for b in load_delivery(run_id):
        rec = bundle_to_record(b)
        rec["conditions"] = ",".join(rec["conditions"])
        rec["resources"] = len(b.get("entry", []))
        rows.append(rec)
    return pd.DataFrame(rows)


def zip_run(run_id: str) -> bytes:
    p = runs_dir() / run_id
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("request.json", "report.json", "records.json", "delivery.ndjson.gz", "real_profile.json"):
            if (p / name).exists():
                z.write(p / name, f"{run_id}/{name}")
        for f in (p / "ml").iterdir():
            z.write(f, f"{run_id}/ml/{f.name}")
        z.writestr(f"{run_id}/delivery.csv", dataset_table(run_id).to_csv(index=False))
        for i, b in enumerate(load_delivery(run_id)):
            rec = bundle_to_record(b)
            z.writestr(f"{run_id}/hl7/{rec['patient_id']}.hl7",
                       build_oru(rec, control_id=f"TD{i + 1:06d}").replace("\r", "\n") + "\n")
    return buf.getvalue()


FAULT_LABELS = {k: v[0] for k, v in FAULTS.items()}

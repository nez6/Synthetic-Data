"""Serve the ML model: does a patient with these demographics pass validation?

Loads model_validated.pkl (or model_raw.pkl) from a run folder; defaults to the latest run.
"""
import pickle

from src import mlpipe, runner


class ModelNotReady(RuntimeError):
    pass


_cache = {}


def _model_path(run_id: str | None, which: str):
    runs = runner.list_runs()
    if not runs:
        raise ModelNotReady("No run yet: submit a test-data request first (it trains the model).")
    run_id = run_id or runs[0]["run_id"]
    path = runner.runs_dir() / run_id / "ml" / f"model_{which}.pkl"
    if not path.exists():
        raise ModelNotReady(f"Run {run_id} has no {which} model.")
    return run_id, path


def load_model(run_id: str | None = None, which: str = "validated"):
    run_id, path = _model_path(run_id, which)
    key = (str(path), path.stat().st_mtime)
    if key not in _cache:
        with open(path, "rb") as f:
            _cache[key] = pickle.load(f)
    return run_id, _cache[key]


def predict_pass(rows: list[dict], run_id: str | None = None, which: str = "validated") -> dict:
    """rows: [{age, sex, race_ethnicity, cohort}] → probability each record passes validation."""
    run_id, model = load_model(run_id, which)
    clean = [{"age": r.get("age"), "sex": r.get("sex") or "unknown",
              "race_ethnicity": r.get("race_ethnicity") or "Not recorded", "cohort": r.get("cohort") or "A"}
             for r in rows]
    probs = mlpipe.predict(model, clean)
    return {"run_id": run_id, "model": which,
            "predictions": [{"input": c, "probability_pass": round(p, 4), "predicted_pass": p >= 0.5}
                            for c, p in zip(clean, probs)]}

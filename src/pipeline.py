"""Entry points used by the Docker image (and handy locally).

    python -m src.pipeline train      # 00 download data from GCS → 01-04 default request → model + MLflow run
    python -m src.pipeline serve      # start the API; on first start it downloads the data and trains a model
                                      # in the background so /predict works without any manual step
    python -m src.pipeline ui         # start the Streamlit front end (set API_URL to the API it should call)

`serve` listens on $PORT (Cloud Run sets it; default 8000).
"""
import argparse
import os
import threading
import traceback

STATE = {"stage": "idle", "message": "", "error": None}


def _set(stage, message=""):
    STATE.update(stage=stage, message=message)
    print(f"[bootstrap] {stage}: {message}", flush=True)


def ensure_data():
    from src import cloud_data, pools
    if pools.list_pools():
        return
    _set("downloading", "fetching Synthea pools from Google Cloud Storage")
    cloud_data.download(progress=lambda m: _set("downloading", m))


def train(request: dict | None = None) -> dict:
    """Download the data if needed, then run one request end to end (selection, validation, ML, MLflow)."""
    from src import runner
    from src.request import default_request
    ensure_data()
    _set("training", "running the default request (validate + train/test/validate + MLflow)")
    report = runner.run(request or default_request(), progress=lambda f, m: _set("training", f"{f:.0%} {m}"))
    _set("ready", f"model from run {report['run_id']}")
    return report


def bootstrap():
    """Make a fresh container useful: data present and at least one trained model."""
    from src import runner
    try:
        ensure_data()
        if not runner.list_runs():
            train()
        _set("ready", "data and model available")
    except Exception as e:  # noqa: BLE001 — surfaced on /health instead of crashing the server
        STATE["error"] = f"{e}\n{traceback.format_exc()[-1500:]}"
        _set("failed", str(e))


def bootstrap_in_background():
    threading.Thread(target=bootstrap, daemon=True, name="bootstrap").start()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "serve", "ui"])
    a = ap.parse_args()
    if a.command == "train":
        r = train()
        print(f"run {r['run_id']}: {r['n_tested']} tested, {r['n_delivered']} delivered")
    elif a.command == "ui":
        import subprocess
        import sys
        port = os.environ.get("UI_PORT", "8501")
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", "interface/frontend.py",
                                  "--server.port", port, "--server.address", "0.0.0.0"]))
    else:
        import uvicorn
        os.environ.setdefault("BOOTSTRAP", "1")
        uvicorn.run("src.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()

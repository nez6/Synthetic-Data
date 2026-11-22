"""Step 4: train two Random Forests from FHIR-derived features and log everything to MLflow.

    python -m src.train                 # uses existing FHIR Bundles, generating them if absent
    python -m src.train --regenerate    # regenerate synthetic data first
"""
import argparse
import json
import os
import pickle
import re

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix  # noqa: E402

from src.config import ROOT, load_config, resolve  # noqa: E402
from src.data_ingestion import fetch_data, load_bundles  # noqa: E402
from src.preprocess import FEATURES, flatten_bundles, preprocess_pipeline  # noqa: E402

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"


def _metric_name(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def run_training(regenerate: bool = False, threshold: int = None, n_estimators: int = None) -> dict:
    config = load_config()
    n_estimators = n_estimators or config["model"]["n_estimators"]

    bundles_path = resolve(config["data"]["bundles_path"])
    if regenerate or not bundles_path.exists():
        fetch_data()
    bundles = load_bundles(bundles_path)
    print(f"Loaded {len(bundles)} FHIR Bundles from {bundles_path}")

    df = flatten_bundles(bundles)
    splits, quality, encoders = preprocess_pipeline(df, threshold)
    X_train, X_test, y_med_train, y_med_test, y_meal_train, y_meal_test = splits

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(config["pipeline"]["project_name"])

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("min_quality_threshold", quality["threshold"])
        mlflow.log_param("training_input", f"FHIR {config['data']['fhir_version']} Bundles")
        mlflow.log_param("n_bundles", len(bundles))

        mlflow.log_metric("dqi_overall_mean", quality["overall_mean_dqi"])
        mlflow.log_metric("rows_kept", quality["rows_kept"])
        mlflow.log_metric("rows_dropped", quality["rows_total"] - quality["rows_kept"])
        for group, stats in quality["per_demographic"].items():
            mlflow.log_metric(f"dqi_score_{_metric_name(group)}", stats["mean_dqi"])

        rf = dict(n_estimators=n_estimators, random_state=config["pipeline"]["random_state"])
        med_model = RandomForestClassifier(**rf).fit(X_train, y_med_train)
        meal_model = RandomForestClassifier(**rf).fit(X_train, y_meal_train)

        med_pred = med_model.predict(X_test)
        meal_pred = meal_model.predict(X_test)
        med_acc = accuracy_score(y_med_test, med_pred)
        meal_acc = accuracy_score(y_meal_test, meal_pred)
        mlflow.log_metric("medication_accuracy", med_acc)
        mlflow.log_metric("meal_accuracy", meal_acc)

        # Fairness check: is the model equally accurate for every demographic group?
        race_names = encoders["race_demographic"].inverse_transform(X_test["race_demographic"])
        per_group = {}
        for g in sorted(set(race_names)):
            mask = race_names == g
            per_group[g] = {"test_records": int(mask.sum()),
                            "medication_accuracy": round(float((med_pred[mask] == y_med_test.values[mask]).mean()), 3)}
            mlflow.log_metric(f"med_accuracy_{_metric_name(g)}", per_group[g]["medication_accuracy"])

        med_labels = list(encoders[config["data"]["target_medication"]].classes_)
        cm = confusion_matrix(y_med_test, med_pred, labels=range(len(med_labels))).tolist()

        with open(resolve(config["artifacts"]["med_model_path"]), "wb") as f:
            pickle.dump(med_model, f)
        with open(resolve(config["artifacts"]["meal_model_path"]), "wb") as f:
            pickle.dump(meal_model, f)

        summary = {
            "run_id": run.info.run_id,
            "n_bundles": len(bundles),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_estimators": n_estimators,
            "medication_accuracy": med_acc,
            "meal_accuracy": meal_acc,
            "quality": quality,
            "per_group_accuracy": per_group,
            "feature_importance": dict(zip(FEATURES, med_model.feature_importances_.round(4).tolist())),
            "confusion_matrix": {"labels": med_labels, "matrix": cm},
        }
        summary_path = resolve(config["artifacts"]["training_summary_path"])
        summary_path.write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(resolve(config["artifacts"]["quality_report_path"])))
        mlflow.log_artifact(str(summary_path))

    print(f"\nmedication_accuracy={med_acc:.3f}  meal_accuracy={meal_acc:.3f}")
    print("Pipeline metrics logged to MLflow (mlflow.db).")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--regenerate", action="store_true", help="regenerate synthetic FHIR data first")
    run_training(regenerate=p.parse_args().regenerate)

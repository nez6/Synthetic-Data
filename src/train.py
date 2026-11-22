import pickle
import re

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.config import ROOT, load_config, resolve
from src.data_ingestion import fetch_data
from src.preprocess import preprocess_pipeline


def _metric_name(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def run_training():
    config = load_config()

    df = fetch_data()
    splits, quality = preprocess_pipeline(df)
    X_train, X_test, y_med_train, y_med_test, y_meal_train, y_meal_test = splits

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(config["pipeline"]["project_name"])

    with mlflow.start_run():
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("n_estimators", config["model"]["n_estimators"])
        mlflow.log_param("min_quality_threshold", quality["threshold"])

        mlflow.log_metric("dqi_overall_mean", quality["overall_mean_dqi"])
        mlflow.log_metric("rows_kept", quality["rows_kept"])
        mlflow.log_metric("rows_dropped", quality["rows_total"] - quality["rows_kept"])
        for group, stats in quality["per_demographic"].items():
            mlflow.log_metric(f"dqi_score_{_metric_name(group)}", stats["mean_dqi"])

        rf_kwargs = dict(n_estimators=config["model"]["n_estimators"],
                         random_state=config["pipeline"]["random_state"])
        med_model = RandomForestClassifier(**rf_kwargs).fit(X_train, y_med_train)
        meal_model = RandomForestClassifier(**rf_kwargs).fit(X_train, y_meal_train)

        med_acc = accuracy_score(y_med_test, med_model.predict(X_test))
        meal_acc = accuracy_score(y_meal_test, meal_model.predict(X_test))
        mlflow.log_metric("medication_accuracy", med_acc)
        mlflow.log_metric("meal_accuracy", meal_acc)

        with open(resolve(config["artifacts"]["med_model_path"]), "wb") as f:
            pickle.dump(med_model, f)
        with open(resolve(config["artifacts"]["meal_model_path"]), "wb") as f:
            pickle.dump(meal_model, f)
        mlflow.log_artifact(str(resolve(config["artifacts"]["quality_report_path"])))

    print(f"\nmedication_accuracy={med_acc:.3f}  meal_accuracy={meal_acc:.3f}")
    print("Pipeline metrics logged to MLflow (mlflow.db).")
    return {"medication_accuracy": med_acc, "meal_accuracy": meal_acc}


if __name__ == "__main__":
    run_training()

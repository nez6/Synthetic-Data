"""Step 03 · Train, test, validate ("Each cohort is split 60/20/20 to train the model").

The model never sees clinical data, only demographics (age, sex, race/ethnicity) and the cohort.
It predicts whether a patient record passes validation. Two models are trained on identical
splits so the effect of the validation script is visible:

    raw        target = record passed BEFORE validation (the data as generated)
    validated  target = record passed AFTER validation  (the data that would be delivered)

Each cohort is split on its own: train (60%) fits the model, test (20%) picks hyperparameters
("checks results while tuning"), validation (20%) is held back for the final score.

If a demographic group fails more often, the raw model can learn it (ROC AUC above 0.5). After
validation, an AUC near 0.5 means data quality no longer depends on who the patient is.
"""
import io
import json
import os
import pickle

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from src.config import ROOT, load_config  # noqa: E402

FEATURES = ["age", "sex", "race_ethnicity", "cohort"]
NUMERIC, CATEGORICAL = ["age"], ["sex", "race_ethnicity", "cohort"]
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
CANDIDATES = [("logistic_regression", {"C": c}) for c in (0.1, 1.0, 10.0)] + \
             [("random_forest", {"max_depth": d, "n_estimators": 200}) for d in (3, 6)]
SERIES = "#2a78d6"


def split(df: pd.DataFrame, ratios: dict, seed: int) -> pd.Series:
    """Assign train / test / validation within each cohort (stratified by the target when possible)."""
    out = pd.Series(index=df.index, dtype=object)
    for _, g in df.groupby("cohort"):
        strat = g["passed_after"] if g["passed_after"].value_counts().min() >= 3 else None
        rest_share = ratios["test"] + ratios["validation"]
        train, rest = train_test_split(g.index, test_size=rest_share, random_state=seed,
                                       stratify=strat)
        strat_rest = g.loc[rest, "passed_after"] if strat is not None and \
            g.loc[rest, "passed_after"].value_counts().min() >= 2 else None
        test, val = train_test_split(rest, test_size=ratios["validation"] / rest_share, random_state=seed,
                                     stratify=strat_rest)
        out[train], out[test], out[val] = "train", "test", "validation"
    return out


def _pipeline(kind: str, params: dict, seed: int) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    if kind == "logistic_regression":
        model = LogisticRegression(C=params["C"], class_weight="balanced", max_iter=2000)
    else:
        model = RandomForestClassifier(**params, class_weight="balanced", random_state=seed)
    return Pipeline([("features", pre), ("model", model)])


def _scores(y, proba, pred) -> dict:
    y = np.asarray(y)
    two = len(set(y)) == 2
    return {"n": int(len(y)), "actual_pass_rate": float(y.mean()) if len(y) else None,
            "predicted_pass_rate": float(np.mean(pred)) if len(y) else None,
            "accuracy": float(accuracy_score(y, pred)) if len(y) else None,
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if two else None,
            "roc_auc": float(roc_auc_score(y, proba)) if two else None}


def train_one(df: pd.DataFrame, target: str, seed: int) -> dict:
    tr, te, va = (df[df["split"] == s] for s in ("train", "test", "validation"))
    X = lambda d: d[FEATURES]  # noqa: E731
    if tr[target].nunique() < 2:
        model = DummyClassifier(strategy="constant", constant=bool(tr[target].iloc[0])).fit(X(tr), tr[target])
        chosen, tuning = ("constant", {"value": bool(tr[target].iloc[0])}), []
        note = "Every training record has the same outcome, so the model predicts that constant."
    else:
        tuning = []
        for kind, params in CANDIDATES:
            m = _pipeline(kind, params, seed).fit(X(tr), tr[target])
            p = m.predict_proba(X(te))[:, 1]
            score = roc_auc_score(te[target], p) if te[target].nunique() == 2 else accuracy_score(te[target], p >= .5)
            tuning.append({"model": kind, "params": params, "test_score": round(float(score), 4)})
        best = max(tuning, key=lambda r: r["test_score"])
        chosen = (best["model"], best["params"])
        model = _pipeline(*chosen, seed).fit(X(tr), tr[target])
        note = None

    def evaluate(d):
        if not len(d):
            return {"n": 0}
        proba = model.predict_proba(X(d))[:, -1] if hasattr(model, "predict_proba") else model.predict(X(d))
        return _scores(d[target].astype(int), proba, model.predict(X(d)).astype(int))

    result = {"target": target, "model": chosen[0], "params": chosen[1], "tuning": tuning, "note": note,
              "train": evaluate(tr), "test": evaluate(te), "validation": evaluate(va),
              "validation_by_cohort": {c: evaluate(va[va["cohort"] == c]) for c in sorted(df["cohort"].unique())}}
    if note is None and va[target].nunique() == 2:
        imp = permutation_importance(model, X(va), va[target], scoring="roc_auc", n_repeats=20, random_state=seed)
        result["importance"] = dict(zip(FEATURES, np.round(imp.importances_mean, 4).tolist()))
    else:
        result["importance"] = {f: 0.0 for f in FEATURES}
    return result, model


def _plot_pass_rates(df) -> bytes:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    rows = []
    for c in sorted(df["cohort"].unique()):
        for s in ("train", "test", "validation"):
            d = df[(df["cohort"] == c) & (df["split"] == s)]
            rows.append((f"{c} · {s}", d["passed_before"].mean() * 100, d["passed_after"].mean() * 100))
    y = np.arange(len(rows))
    ax.barh(y + 0.2, [r[2] for r in rows], 0.38, color=SERIES, label="after validation")
    ax.barh(y - 0.2, [r[1] for r in rows], 0.38, color="#b7d3f6", label="before validation")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("pass rate (%)")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def run(records: pd.DataFrame, out_dir, request: dict, run_id: str) -> dict:
    """records: one row per patient with FEATURES, passed_before, passed_after. Writes models + report."""
    cfg = load_config()
    seed = cfg["project"]["random_state"]
    df = records.copy()
    df["split"] = split(df, request["split"], seed)

    raw, raw_model = train_one(df, "passed_before", seed)
    val, val_model = train_one(df, "passed_after", seed)

    counts = {c: {s: int(((df["cohort"] == c) & (df["split"] == s)).sum()) for s in ("train", "test", "validation")}
              for c in sorted(df["cohort"].unique())}
    final = {}
    for c in sorted(df["cohort"].unique()) + ["Overall"]:
        d = df if c == "Overall" else df[df["cohort"] == c]
        v = d[d["split"] == "validation"]
        final[c] = {"validation_split_pass_rate": float(v["passed_after"].mean()) if len(v) else None,
                    "validation_split_n": int(len(v)),
                    "all_records_pass_rate": float(d["passed_after"].mean()),
                    "all_records_passed": int(d["passed_after"].sum()), "all_records_n": int(len(d)),
                    "baseline_pass_rate": float(d["passed_before"].mean())}

    report = {"features": FEATURES, "split_ratios": request["split"], "split_counts": counts,
              "final_scores": final, "models": {"raw": raw, "validated": val}}

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model_raw.pkl"), "wb") as f:
        pickle.dump(raw_model, f)
    with open(os.path.join(out_dir, "model_validated.pkl"), "wb") as f:
        pickle.dump(val_model, f)
    png = _plot_pass_rates(df)
    with open(os.path.join(out_dir, "pass_rates.png"), "wb") as f:
        f.write(png)
    df[["patient_id", "cohort", "split", *FEATURES[:-1], "passed_before", "passed_after"]].to_csv(
        os.path.join(out_dir, "ml_dataset.csv"), index=False)
    with open(os.path.join(out_dir, "ml_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(cfg["project"]["mlflow_experiment"])
    with mlflow.start_run(run_name=run_id) as parent:
        mlflow.log_params({"request": request["label"], "delivery_size": request["delivery_size"],
                           "features": ",".join(FEATURES), "split": json.dumps(request["split"]),
                           **{f"n_{c}": sum(v.values()) for c, v in counts.items()}})
        for c, s in final.items():
            mlflow.log_metric(f"final_pass_rate_{c}", s["all_records_pass_rate"])
            mlflow.log_metric(f"baseline_pass_rate_{c}", s["baseline_pass_rate"])
        for name, res, path in (("raw", raw, "model_raw.pkl"), ("validated", val, "model_validated.pkl")):
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({"target": res["target"], "model": res["model"],
                                   **{f"param_{k}": v for k, v in res["params"].items()}})
                for split_name in ("train", "test", "validation"):
                    for k, v in res[split_name].items():
                        if isinstance(v, float):
                            mlflow.log_metric(f"{split_name}_{k}", v)
                mlflow.log_artifact(os.path.join(out_dir, path))
        mlflow.log_artifact(os.path.join(out_dir, "pass_rates.png"))
        mlflow.log_artifact(os.path.join(out_dir, "ml_report.json"))
        mlflow.log_artifact(os.path.join(out_dir, "ml_dataset.csv"))
        report["mlflow_run_id"] = parent.info.run_id
    with open(os.path.join(out_dir, "ml_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def predict(model, rows: list[dict]) -> list[float]:
    X = pd.DataFrame(rows)[FEATURES]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, -1].tolist()
    return model.predict(X).astype(float).tolist()

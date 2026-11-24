"""🧠 ML · "Each cohort is split 60/20/20 to train the model" + raw vs validated + synthetic vs real."""
import pandas as pd
import requests
import streamlit as st

from interface.common import api_url, explain, grouped_hbar, load_run, pct, pick_run
from src import runner
from src.request import NOT_RECORDED

MODEL_NAMES = {"raw": "Raw data (before validation)", "validated": "Validated data (after validation)"}


def _split_boxes(ml, rep):
    counts = ml["split_counts"]
    ratios = ml["split_ratios"]
    st.markdown(f"**Training {ratios['train']:.0%} · Test {ratios['test']:.0%} (tuning) · "
                f"Validation {ratios['validation']:.0%} (held back for the final score)**")
    for c, n in counts.items():
        cols = st.columns([1, 1, 1, 1])
        cols[0].markdown(f"**Cohort {c}**  \n{sum(n.values())} patients")
        for col, s, caption in zip(cols[1:], ("train", "test", "validation"),
                                   ("Model learns the patterns", "Checks results while tuning",
                                    "Held back for the final score")):
            with col.container(border=True):
                st.markdown(f"### {n[s]}")
                st.caption(f"{s} · {caption}")


def _final(ml):
    st.markdown("#### Final validation score")
    st.caption("Pass rate after validation. 'Held-back split' = the validation 20% only; 'all records' = every tested "
               "patient in the cohort.")
    cols = st.columns(len(ml["final_scores"]))
    for col, (c, s) in zip(cols, ml["final_scores"].items()):
        col.metric(f"{'Cohort ' + c if c != 'Overall' else 'Overall'}",
                   pct(s["all_records_pass_rate"]),
                   f"from {pct(s['baseline_pass_rate'])} baseline", delta_color="off")
        col.caption(f"{s['all_records_passed']} of {s['all_records_n']} pass · held-back split: "
                    f"{pct(s['validation_split_pass_rate'])} of {s['validation_split_n']}")


def _models(ml):
    st.markdown("#### Raw vs validated data: what the model learns")
    explain("Both models see **only demographics** (age, sex, race/ethnicity, cohort) and predict whether a record "
            "passes. On the **raw** data a high ROC AUC means some demographic groups fail more often: data quality "
            "depends on who the patient is. After **validation**, an AUC near 0.5 (or a constant prediction) means the "
            "remaining failures are not tied to demographics, which is what you want from test data.")
    rows = []
    for key, m in ml["models"].items():
        v = m["validation"]
        rows.append({"data": MODEL_NAMES[key], "chosen model": m["model"], "params": str(m["params"]),
                     "actual pass rate (validation split)": pct(v.get("actual_pass_rate")),
                     "predicted pass rate": pct(v.get("predicted_pass_rate")),
                     "accuracy": pct(v.get("accuracy")),
                     "balanced accuracy": pct(v.get("balanced_accuracy")),
                     "ROC AUC": "–" if v.get("roc_auc") is None else f"{v['roc_auc']:.3f}",
                     "note": m.get("note") or ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    imp = pd.DataFrame([{"feature": f, "data": MODEL_NAMES[k], "importance": v}
                        for k, m in ml["models"].items() for f, v in m["importance"].items()])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Which demographics drive the prediction?** (permutation importance, drop in ROC AUC)")
        grouped_hbar(imp, "feature", "data", "importance", "importance", fmt=".3f")
    with c2:
        st.markdown("**Per cohort, validation split**")
        st.dataframe(pd.DataFrame([{"data": MODEL_NAMES[k], "cohort": c, "n": s.get("n"),
                                    "actual pass": pct(s.get("actual_pass_rate")),
                                    "predicted pass": pct(s.get("predicted_pass_rate")),
                                    "ROC AUC": "–" if s.get("roc_auc") is None else f"{s['roc_auc']:.3f}"}
                                   for k, m in ml["models"].items() for c, s in m["validation_by_cohort"].items()]),
                     hide_index=True, width="stretch")
    with st.expander("Hyperparameter tuning on the test split"):
        for k, m in ml["models"].items():
            st.markdown(f"**{MODEL_NAMES[k]}**")
            st.dataframe(pd.DataFrame(m["tuning"] or [{"note": m.get("note")}]), hide_index=True)


def _compare_profiles(rep):
    st.markdown("#### Synthetic vs your real data")
    p = rep.get("profiles") or {}
    real, raw, dlv = p.get("real"), p.get("synthetic_raw"), p.get("synthetic_delivered")
    if not real:
        st.info("No real-data profile was attached to this run. Load one in **01 · Profile your data** and run a "
                "request to compare. Below: the synthetic data as selected vs as delivered.")
    explain("The same script profiles all three datasets, so they are measured identically. 'As selected' should "
            "mirror your real data (same mix, same injected problems); 'delivered' shows what validation removed.")

    def share(prof, section, key):
        if not prof:
            return None
        v = prof["counts"][section].get(key) if section != "quality" else prof["quality"].get(key)
        return None if v is None else 100 * v / prof["n_patients"]

    rows = []
    for section, title in (("race_ethnicity", "Race / ethnicity"), ("sex", "Sex"), ("age_band", "Age band"),
                           ("conditions", "Chronic condition"), ("quality", "Data-quality problem")):
        keys = set()
        for prof in (real, raw, dlv):
            if prof:
                keys |= set(prof["quality"] if section == "quality" else prof["counts"][section])
        for key in sorted(keys):
            r = {"dimension": title, "category": key,
                 "real %": share(real, section, key), "synthetic as selected %": share(raw, section, key),
                 "synthetic delivered %": share(dlv, section, key)}
            if r["real %"] is not None and r["synthetic as selected %"] is not None:
                r["difference (selected − real)"] = r["synthetic as selected %"] - r["real %"]
            rows.append(r)
    df = pd.DataFrame(rows)
    dim = st.radio("Dimension", list(dict.fromkeys(df["dimension"])), horizontal=True, key="cmpdim")
    view = df[df["dimension"] == dim].drop(columns=["dimension"])
    st.dataframe(view.round(1), hide_index=True, width="stretch")
    long = view.melt(id_vars="category", value_vars=[c for c in ("real %", "synthetic as selected %",
                                                                 "synthetic delivered %") if view[c].notna().any()],
                     var_name="dataset", value_name="share").dropna()
    if not long.empty:
        grouped_hbar(long, "category", "dataset", "share", "% of patients",
                     colors=("#55554f", "#2a78d6", "#eb6834"))
    if real and dim == "Race / ethnicity" and NOT_RECORDED in view["category"].values:
        st.caption("Synthea always records race/ethnicity; 'Not recorded' in the synthetic data comes from the "
                   "missing_race_ethnicity fault, injected at your real rate.")


def _predict():
    st.markdown("#### Try the model (calls the deployed API)")
    c1, c2, c3, c4, c5 = st.columns(5)
    age = c1.number_input("Age", 0, 110, 45)
    sex = c2.selectbox("Sex", ["female", "male"])
    race = c3.selectbox("Race / ethnicity", ["White", "Black/African American", "Hispanic/Latino", "Asian",
                                             "American Indian/Alaska Native", "Native Hawaiian/Pacific Islander",
                                             NOT_RECORDED])
    cohort = c4.selectbox("Cohort", ["A", "B"])
    which = c5.selectbox("Model", ["validated", "raw"])
    if st.button("Predict"):
        try:
            res = requests.post(f"{api_url()}/predict", json={"age": age, "sex": sex, "race_ethnicity": race,
                                                              "cohort": cohort, "model": which}, timeout=10)
        except requests.RequestException as e:
            st.error(f"API not reachable at {api_url()}: {e}")
            return
        if res.status_code != 200:
            st.error(f"{res.status_code}: {res.json().get('detail')}")
            return
        body = res.json()
        st.metric("Probability this record passes validation", pct(body["probability_pass"]),
                  "predicted to pass" if body["predicted_pass"] else "predicted to fail", delta_color="off")
        st.caption(f"Run {body['run_id']} · {body['model']} model · POST {api_url()}/predict")


def render():
    st.subheader("🧠 Train · test · validate")
    st.caption("Built with a mini ML data-validation pipeline: the model never sees clinical data, only demographics.")
    run_id = pick_run("ml_run")
    if not run_id:
        return
    data = load_run(run_id)
    ml, rep = data["ml"], data["report"]
    _split_boxes(ml, rep)
    _final(ml)
    _models(ml)
    st.image(str(runner.runs_dir() / run_id / "ml" / "pass_rates.png"),
             caption="Pass rate before and after validation, per cohort and split (also logged to MLflow)")
    st.caption(f"MLflow run `{ml.get('mlflow_run_id')}` in experiment 'synthetic_test_data_ml' "
               "(`make mlflow`, then http://localhost:5000): parameters, metrics, both models, this plot and the "
               "ML dataset.")
    _compare_profiles(rep)
    _predict()

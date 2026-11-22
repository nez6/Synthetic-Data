"""Streamlit dashboard: walk through Data → FHIR → Quality → Training → Prediction → HL7.

Run with:  streamlit run interface/frontend.py   (or `make run` to start API + UI together)
"""
import contextlib
import difflib
import io
import json
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from src import terminology as t  # noqa: E402
from src.config import load_config, resolve  # noqa: E402
from src.data_ingestion import fetch_data, generate_records, load_bundles  # noqa: E402
from src.fhir.builder import build_bundle  # noqa: E402
from src.fhir.parser import bundle_to_record, detect_version  # noqa: E402
from src.fhir.validate import validate_bundle  # noqa: E402
from src.fhir.versions import DIFFERENCES, VERSION_ORDER, VERSIONS, differences_between  # noqa: E402
from src.hl7v2 import inbox  # noqa: E402
from src.hl7v2 import message as hl7  # noqa: E402
from src.hl7v2.mllp import is_listening, send  # noqa: E402
from src.preprocess import flatten_bundles  # noqa: E402
from src.quality import ISSUE_LABELS, score_dataframe  # noqa: E402

CONFIG = load_config()
SERIES = "#2a78d6"
BUNDLES_PATH = resolve(CONFIG["data"]["bundles_path"])
SOURCE_PATH = resolve(CONFIG["data"]["source_path"])
SUMMARY_PATH = resolve(CONFIG["artifacts"]["training_summary_path"])
HL7_HOST, HL7_PORT = CONFIG["hl7"]["host"], CONFIG["hl7"]["port"]

st.set_page_config(page_title="EHR Decision Dashboard", page_icon="📋", layout="wide")


# ============================================================================ helpers

def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_dataset(_mtime: float):
    bundles = load_bundles(BUNDLES_PATH)
    source = pd.read_csv(SOURCE_PATH)
    flat = score_dataframe(flatten_bundles(bundles))
    return bundles, source, flat


def dataset():
    if not BUNDLES_PATH.exists():
        return None, None, None
    return load_dataset(mtime(BUNDLES_PATH))


def hbar(df: pd.DataFrame, cat: str, val: str, title: str, fmt: str = ".1f", domain=None):
    x = alt.X(f"{val}:Q", title=title, scale=alt.Scale(domain=domain) if domain else alt.Undefined)
    chart = (
        alt.Chart(df)
        .mark_bar(color=SERIES, cornerRadiusEnd=4, height=16)
        .encode(y=alt.Y(f"{cat}:N", sort="-x", title=None), x=x,
                tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{val}:Q", format=fmt, title=title)])
        .properties(height=34 * len(df) + 20)
    )
    st.altair_chart(chart, width="stretch")


def api_url() -> str:
    return st.session_state.get("api_url", "http://localhost:8000")


def api_up() -> bool:
    try:
        return requests.get(f"{api_url()}/health", timeout=1).status_code == 200
    except requests.RequestException:
        return False


def explain(text: str):
    st.info(text, icon="💡")


def show_prediction(result: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("Data Quality Index", f"{result['data_quality_index']}%")
    c2.metric("Issues found", len(result["data_quality_issues"]))
    c3.metric("Below training threshold?", "Yes ⚠️" if result["low_quality_warning"] else "No ✅")
    if result["data_quality_issue_labels"]:
        st.warning("Data quality issues: " + " · ".join(result["data_quality_issue_labels"]))

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**💊 Predicted medication plan**")
        st.info(result["predicted_medication_strategy"])
    with m2:
        st.markdown("**🥗 Predicted meal plan**")
        st.success(result["predicted_dietary_strategy"])

    doc = result["recommended_provider_routing"]
    st.markdown("**🩺 Clinician routing**")
    d1, d2 = st.columns([1, 2])
    d1.warning(f"**{doc['specialty']}**\n\n{doc['urgency']}")
    d2.info(doc["clinical_rationale"])


# ============================================================================ sidebar

with st.sidebar:
    st.header("System status")
    bundles, source, flat = dataset()
    trained = SUMMARY_PATH.exists()
    st.session_state["api_url"] = st.text_input("API address", value=api_url())
    up = api_up()
    listening = is_listening(HL7_HOST, HL7_PORT)
    st.markdown(
        f"{'✅' if bundles else '⬜'} **Data generated** {f'({len(bundles):,} FHIR Bundles)' if bundles else ''}\n\n"
        f"{'✅' if trained else '⬜'} **Model trained**\n\n"
        f"{'✅' if up else '❌'} **API** {api_url()}\n\n"
        f"{'✅' if listening else '❌'} **HL7 listener** {HL7_HOST}:{HL7_PORT}"
    )
    if not up:
        st.caption("Start the API with `make api` (or `make run` for everything).")
    st.button("🔄 Refresh status")
    st.divider()
    st.caption("All patients are synthetic. Demonstration only — not for clinical use.")

st.title("📋 EHR Decision Support: Data Quality + Clinician Routing")

tabs = st.tabs(["🧭 How it works", "1️⃣ Data", "2️⃣ Train", "3️⃣ Predict",
                "4️⃣ FHIR versions", "5️⃣ HL7 v2 listener"])


# ============================================================================ 0. How it works

with tabs[0]:
    st.subheader("What this application does")
    st.markdown(
        """
Hospitals store patient data in different systems that exchange it in two standard formats:
**HL7 v2** (pipe-delimited messages, the 1980s workhorse still used for most lab results) and
**FHIR** (modern JSON resources such as `Patient` and `Observation`).

This app takes diabetes patients in those formats and:

1. **Checks data quality**: every record gets a **Data Quality Index (DQI)** from 0 to 100,
   penalised for missing race/ethnicity, missing BMI, missing birth date, or an implausible HbA1c lab value.
   Low-quality records are excluded from training, and the score is broken down **by demographic group**
   so you can see whether documentation gaps affect some groups more than others.
2. **Predicts a care pathway**: two machine-learning models (Random Forests) recommend a
   **medication plan** and a **meal plan** from demographic group, age, BMI and HbA1c.
3. **Routes to a clinician**: simple rules map the result to a specialty and urgency.
"""
    )
    st.graphviz_chart(
        """
digraph {
  rankdir=LR; node [shape=box style="rounded,filled" fillcolor="#eef4fc" color="#2a78d6" fontname="Helvetica" fontsize=11];
  edge [color="#888888" fontname="Helvetica" fontsize=9];
  ehr  [label="Synthetic EHR\\n(source records)"];
  fhir [label="FHIR Bundles\\nPatient · Encounter ·\\nObservation · CarePlan"];
  hl7  [label="HL7 v2 ORU^R01\\n(lab system)"];
  mllp [label="MLLP listener\\nport 2575"];
  flat [label="Flatten to\\nfeatures"];
  dqi  [label="Data Quality\\nIndex"];
  train[label="Train models\\n(+ MLflow)"];
  model[label="Saved models"];
  api  [label="FastAPI\\n/predict  /fhir/{v}/Bundle/$predict"];
  route[label="Medication + meal plan\\n+ clinician routing"];
  ehr -> fhir [label="export"];
  fhir -> flat -> dqi -> train -> model;
  hl7 -> mllp [label="TCP"];
  mllp -> api [label="convert to FHIR R4"];
  model -> api -> route;
}
"""
    )
    st.subheader("Try it in order")
    st.markdown(
        """
| Tab | What you do | What you learn |
|---|---|---|
| **1 · Data** | Generate synthetic patients and browse them as a table, as FHIR, and as the scored training table | How EHR data becomes FHIR, and where it is incomplete |
| **2 · Train** | Click **Train** and see accuracy, fairness by group and the quality report | How data quality affects what the model learns |
| **3 · Predict** | Enter a patient; the app builds a FHIR Bundle and sends it to the API | What the model and routing rules return |
| **4 · FHIR versions** | Compare one patient across DSTU2, STU3, R4, R4B and R5 | How FHIR changed, and why integrations break between versions |
| **5 · HL7 v2 listener** | Send lab messages over MLLP and watch them arrive in the inbox | How legacy HL7 feeds are received, acknowledged and converted to FHIR |
"""
    )
    with st.expander("Data Quality Index rules"):
        w, b = CONFIG["quality_weights"], CONFIG["quality_bounds"]
        st.table(pd.DataFrame([
            ["Race/ethnicity not recorded", "no us-core-race/ethnicity extension · empty PID-10/PID-22", -w["missing_race_penalty"]],
            ["BMI missing", "no Observation with LOINC 39156-5 · no BMI OBX", -w["missing_bmi_penalty"]],
            [f"BMI outside {b['bmi_min']}–{b['bmi_max']}", "Observation.valueQuantity", -w["out_of_bounds_bmi_penalty"]],
            ["Birth date missing", "Patient.birthDate · PID-7", -w["missing_age_penalty"]],
            [f"HbA1c missing or outside {b['hba1c_min']}–{b['hba1c_max']}%", "Observation LOINC 4548-4 · OBX", -w["out_of_bounds_hba1c_penalty"]],
        ], columns=["Check", "Where it is read from (FHIR · HL7 v2)", "Penalty"]))
        st.caption(f"Records scoring below {CONFIG['min_quality_threshold']} are excluded from training.")
    with st.expander("Clinician routing rules"):
        st.table(pd.DataFrame([
            ["HbA1c ≥ 9.5%", "Endocrinology & Metabolic Specialist", "High: immediate consult"],
            ["Combination therapy predicted, or BMI > 32", "Internal Medicine + Registered Dietitian", "Routine"],
            ["Everyone else", "Primary Care + Behavioral Health Coach", "Preventative"],
        ], columns=["Condition", "Specialty", "Urgency"]))
    with st.expander("Glossary"):
        st.markdown(
            """
- **FHIR** (Fast Healthcare Interoperability Resources): HL7's modern standard. Data is split into *resources* (Patient, Observation…) exchanged as JSON over REST. A **Bundle** groups several resources.
- **US Core**: the US rules ("profile") for FHIR, including how to record race and ethnicity.
- **HL7 v2**: the older, pipe-delimited message standard. Each line is a *segment* (MSH header, PID patient, OBX result).
- **ORU^R01**: the HL7 v2 message type for "here are lab results".
- **MLLP**: how HL7 v2 messages are sent over a TCP connection; the receiver replies with an **ACK** (AA = accepted, AE = error, AR = rejected).
- **LOINC**: the code system for lab tests (4548-4 = HbA1c, 39156-5 = BMI).
- **HbA1c**: average blood sugar over ~3 months; above 6.5% indicates diabetes.
"""
        )


# ============================================================================ 1. Data

with tabs[1]:
    st.subheader("Step 1 · Generate and explore the data")
    explain("The app creates synthetic patients, the way a hospital EHR would hold them, and exports "
            "each one as a **FHIR R4 Bundle**. Problems are injected on purpose: about 5% have no race/ethnicity, "
            "~3% have no BMI, and ~2% have an impossible HbA1c of 99%. "
            "The FHIR Bundles are what the model trains on.")
    g1, g2, g3 = st.columns([1, 1, 2])
    n = g1.number_input("Number of patients", 100, 10000, CONFIG["pipeline"]["n_samples"], step=100)
    seed = g2.number_input("Random seed", 0, 99999, CONFIG["pipeline"]["random_state"])
    g3.write("")
    g3.write("")
    if g3.button("🧪 Generate synthetic data", type="primary"):
        with st.spinner("Generating patients and exporting FHIR Bundles…"):
            fetch_data(int(n), int(seed))
        st.cache_data.clear()
        st.rerun()

    bundles, source, flat = dataset()
    if bundles is None:
        st.warning("No data yet. Click **Generate synthetic data**.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Patients / FHIR Bundles", f"{len(bundles):,}")
        k2.metric("Mean Data Quality Index", f"{flat['data_quality_index'].mean():.1f}")
        k3.metric("Records with any issue", f"{(flat['data_quality_issues'] != '').sum():,}")
        k4.metric("Below training threshold",
                  f"{(flat['data_quality_index'] < CONFIG['min_quality_threshold']).sum():,}")

        view = st.radio("View the data as…", ["A · Source EHR table", "B · FHIR Bundles (training input)",
                                              "C · Flattened from FHIR + Data Quality Index"], horizontal=True)

        if view.startswith("A"):
            st.caption(f"What the hospital system knows, before any standards are applied. File: `{SOURCE_PATH.relative_to(resolve('.'))}`")
            st.dataframe(source, width="stretch", height=420)

        elif view.startswith("B"):
            st.caption(f"Each line of `{BUNDLES_PATH.relative_to(resolve('.'))}` is one FHIR Bundle. "
                       "Missing data is simply absent: no race extension, no BMI Observation.")
            ids = [b["entry"][0]["resource"]["id"] for b in bundles]
            problem_ids = flat.loc[flat["data_quality_issues"] != "", "patient_id"].tolist()
            only_problems = st.checkbox("Only show patients with data quality issues")
            pid = st.selectbox("Patient", problem_ids if only_problems else ids)
            bundle = bundles[ids.index(pid)]
            row = flat[flat["patient_id"] == pid].iloc[0]
            left, right = st.columns([2, 1])
            with right:
                st.markdown("**Resources in this Bundle**")
                st.table(pd.DataFrame([{"resourceType": e["resource"]["resourceType"], "id": e["resource"]["id"]}
                                       for e in bundle["entry"]]))
                st.metric("Data Quality Index", f"{row['data_quality_index']}")
                if row["data_quality_issues"]:
                    st.warning(" · ".join(ISSUE_LABELS[i] for i in row["data_quality_issues"].split(", ")))
            with left:
                st.code(json.dumps(bundle, indent=2), language="json", height=520)

        else:
            st.caption("Features extracted back out of each FHIR Bundle, plus the quality score. "
                       f"Saved to `{CONFIG['data']['flattened_path']}`.")
            f1, f2 = st.columns(2)
            issue_filter = f1.multiselect("Filter by issue", list(ISSUE_LABELS),
                                          format_func=lambda i: ISSUE_LABELS[i])
            group_filter = f2.multiselect("Filter by demographic group",
                                          t.DEMOGRAPHIC_GROUPS + ["(not recorded)"])
            shown = flat.copy()
            shown["group"] = shown["race_demographic"].fillna("(not recorded)")
            for issue in issue_filter:
                shown = shown[shown["data_quality_issues"].str.contains(issue)]
            if group_filter:
                shown = shown[shown["group"].isin(group_filter)]
            cols = ["patient_id", "race_demographic", "age", "bmi", "baseline_hba1c",
                    "data_quality_index", "data_quality_issues",
                    "recommended_medication_plan", "recommended_meal_plan"]
            st.dataframe(shown[cols], width="stretch", height=360)
            st.caption(f"{len(shown):,} rows")

        st.markdown("#### Data quality by demographic group")
        st.caption("Are documentation gaps spread evenly? A lower score for one group means its records are less complete.")
        by_group = (flat.assign(group=flat["race_demographic"].fillna("(not recorded)"))
                    .groupby("group")
                    .agg(records=("patient_id", "size"),
                         mean_dqi=("data_quality_index", "mean"),
                         pct_with_issue=("data_quality_issues", lambda s: 100 * (s != "").mean()))
                    .reset_index())
        c1, c2 = st.columns(2)
        with c1:
            hbar(by_group, "group", "mean_dqi", "Mean Data Quality Index", domain=[0, 100])
        with c2:
            hbar(by_group, "group", "pct_with_issue", "% of records with at least one issue", fmt=".1f")
        st.dataframe(by_group.round(1), width="stretch", hide_index=True)

        st.markdown("#### HbA1c distribution")
        plausible = flat[flat["baseline_hba1c"].between(3, 16)]
        n_bad = int((~flat["baseline_hba1c"].between(3, 16)).sum())
        hist = (alt.Chart(plausible).mark_bar(color=SERIES, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                .encode(x=alt.X("baseline_hba1c:Q", bin=alt.Bin(step=0.25), title="HbA1c (%)"),
                        y=alt.Y("count():Q", title="Patients"),
                        tooltip=[alt.Tooltip("baseline_hba1c:Q", bin=alt.Bin(step=0.25), title="HbA1c"),
                                 alt.Tooltip("count():Q", title="Patients")])
                .properties(height=220))
        rules = alt.Chart(pd.DataFrame({"x": [7.5, 9.0], "label": ["> 7.5 combination", "> 9.0 intensive"]})) \
            .mark_rule(strokeDash=[4, 3], color="#888").encode(x="x:Q", tooltip=["label"])
        st.altair_chart(hist + rules, width="stretch")
        st.caption(f"Dashed lines are the care-tier thresholds. {n_bad} records with an implausible value "
                   "(99%) are not plotted; the DQI flags them and training excludes them.")


# ============================================================================ 2. Train

with tabs[2]:
    st.subheader("Step 2 · Train the models")
    explain("Training reads the FHIR Bundles, extracts the features, scores each record's quality, **drops records "
            "below the quality threshold**, fills small gaps (missing race → an explicit 'Unknown' group, never "
            "a guessed race; missing BMI → the median), then trains two Random Forests. Every run is logged to MLflow.")
    s1, s2, s3 = st.columns([1, 1, 1])
    threshold = s1.select_slider("Minimum Data Quality Index to train on", [0, 25, 50, 75, 100],
                                 value=CONFIG["min_quality_threshold"],
                                 help="Try 0 to include the implausible HbA1c records and see what changes.")
    n_est = s2.slider("Trees per forest", 10, 300, CONFIG["model"]["n_estimators"], step=10)
    s3.write("")
    s3.write("")
    if s3.button("🚀 Train model", type="primary", disabled=bundles is None):
        from src.train import run_training
        log = io.StringIO()
        with st.status("Training…", expanded=True) as status:
            with contextlib.redirect_stdout(log):
                run_training(threshold=threshold, n_estimators=n_est)
            st.code(log.getvalue())
            status.update(label="Training complete", state="complete", expanded=False)
    if bundles is None:
        st.warning("Generate data in tab 1 first.")

    if SUMMARY_PATH.exists():
        s = json.loads(SUMMARY_PATH.read_text())
        q = s["quality"]
        st.markdown(f"#### Latest run · `{s['run_id'][:8]}`")
        m = st.columns(5)
        m[0].metric("FHIR Bundles read", f"{s['n_bundles']:,}")
        m[1].metric("Kept for training", f"{q['rows_kept']:,}", f"-{q['rows_total'] - q['rows_kept']} dropped",
                    delta_color="off")
        m[2].metric("Quality threshold", q["threshold"])
        m[3].metric("Medication accuracy", f"{s['medication_accuracy']:.1%}")
        m[4].metric("Meal plan accuracy", f"{s['meal_accuracy']:.1%}")
        st.caption("Accuracy is near 100% because the synthetic labels follow a simple HbA1c rule. The point is the "
                   "pipeline, not model difficulty. Try a threshold of 0 to see bad data enter training.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Is the model equally accurate for each group?**")
            pg = pd.DataFrame([{"group": g, **v} for g, v in s["per_group_accuracy"].items()])
            pg["accuracy_pct"] = 100 * pg["medication_accuracy"]
            hbar(pg, "group", "accuracy_pct", "Medication accuracy on test set (%)", domain=[0, 100])
            st.dataframe(pg[["group", "test_records", "medication_accuracy"]], hide_index=True,
                         width="stretch")
        with c2:
            st.markdown("**What does the model rely on?**")
            fi = pd.DataFrame(s["feature_importance"].items(), columns=["feature", "importance"])
            hbar(fi, "feature", "importance", "Feature importance (medication model)", fmt=".3f")
            st.caption("HbA1c should dominate. If demographic group mattered a lot, that would be a fairness red flag.")

        st.markdown("**Quality report by demographic group (from this run)**")
        st.dataframe(pd.DataFrame(q["per_demographic"]).T, width="stretch")

        st.markdown("**Confusion matrix (medication plan): rows = actual, columns = predicted**")
        labels = s["confusion_matrix"]["labels"]
        st.dataframe(pd.DataFrame(s["confusion_matrix"]["matrix"], index=labels, columns=labels),
                     width="stretch")

        with st.expander("MLflow run history"):
            try:
                import mlflow
                from src.train import TRACKING_URI
                mlflow.set_tracking_uri(TRACKING_URI)
                runs = mlflow.search_runs(experiment_names=[CONFIG["pipeline"]["project_name"]])
                keep = [c for c in ["start_time", "run_id", "params.min_quality_threshold", "params.n_estimators",
                                    "metrics.rows_kept", "metrics.dqi_overall_mean",
                                    "metrics.medication_accuracy", "metrics.meal_accuracy"] if c in runs.columns]
                st.dataframe(runs[keep], width="stretch", hide_index=True)
                st.caption("Full MLflow UI: run `make mlflow`, then open http://localhost:5000")
            except Exception as e:  # noqa: BLE001
                st.caption(f"MLflow history unavailable: {e}")


# ============================================================================ 3. Predict

with tabs[3]:
    st.subheader("Step 3 · Predict for one patient")
    explain("Enter a patient. The app builds a **FHIR Bundle** in the version you choose and POSTs it to the API at "
            "`/fhir/{version}/Bundle/$predict`. The API validates it, scores its quality, runs both models and "
            "applies the routing rules.")
    left, right = st.columns([1, 2])
    with left:
        race = st.selectbox("Demographic group", t.DEMOGRAPHIC_GROUPS + ["(not recorded)"], index=1)
        age = st.number_input("Age", 18, 100, 52)
        no_bmi = st.checkbox("BMI not measured")
        bmi = None if no_bmi else st.number_input("BMI", 10.0, 80.0, 31.2)
        hba1c = st.number_input("HbA1c (%)", 0.0, 100.0, 9.8, help="Try 99 to see an implausible lab value flagged.")
        version = st.selectbox("Send as FHIR version", VERSION_ORDER, index=VERSION_ORDER.index("R4"))
        go = st.button("Run prediction", type="primary")
    with right:
        if go:
            today = date.today()
            rec = {"patient_id": "manual-entry", "gender": "unknown",
                   "birth_date": today.replace(year=today.year - int(age)).isoformat(),
                   "effective": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                   "race_demographic": None if race == "(not recorded)" else race,
                   "bmi": bmi, "baseline_hba1c": hba1c}
            bundle = build_bundle(rec, version)
            try:
                res = requests.post(f"{api_url()}/fhir/{version}/Bundle/$predict", json=bundle, timeout=10)
            except requests.RequestException as e:
                st.error(f"Could not reach the API at {api_url()} ({e}). Start it with `make api`.")
                st.stop()
            if res.status_code != 200:
                st.error(f"API returned {res.status_code}: {res.json().get('detail', res.text)}")
                st.stop()
            body = res.json()
            show_prediction(body)
            v = body["schema_validation"]
            st.caption(("✅ Valid FHIR. " if v["valid"] else "⚠️ Not schema-checked. " if v["valid"] is None
                        else "❌ Schema errors. ") + v["note"] +
                       f" API detected version: **{body['fhir_version_detected']}**.")
            with st.expander(f"The FHIR {version} Bundle that was sent"):
                st.code(json.dumps(bundle, indent=2), language="json")
            with st.expander("Full API response"):
                st.json(body)
        else:
            st.caption("Results appear here.")


# ============================================================================ 4. FHIR versions

with tabs[4]:
    st.subheader("Step 4 · The same patient in five FHIR versions")
    explain("FHIR has had several major releases. Each one renamed or restructured some elements, which is why an "
            "integration built for one version can break on another. Pick a patient and two versions to see "
            "exactly what changes. Every version except DSTU2 is checked against the official schema.")
    st.dataframe(pd.DataFrame([{"version": k, **v} for k, v in VERSIONS.items()]),
                 hide_index=True, width="stretch")

    if bundles is None:
        sample_rec = generate_records(1, seed=1)[0]
        st.caption("No dataset yet, so a sample patient is shown.")
        recs = {sample_rec["patient_id"]: sample_rec}
    else:
        recs = {b["entry"][0]["resource"]["id"]: b for b in bundles}
    c1, c2, c3 = st.columns(3)
    pid = c1.selectbox("Patient", list(recs), key="fhir_pid")
    va = c2.selectbox("Version A", VERSION_ORDER, index=VERSION_ORDER.index("R4"))
    vb = c3.selectbox("Version B", VERSION_ORDER, index=VERSION_ORDER.index("R5"))
    src = recs[pid]
    record = bundle_to_record(src) if src.get("resourceType") == "Bundle" else src
    ba, bb = build_bundle(record, va), build_bundle(record, vb)

    diffs = differences_between(va, vb)
    if va == vb:
        st.caption("Pick two different versions to compare.")
    elif not diffs:
        st.success(f"{va} and {vb} are identical for these resources.")
    else:
        st.markdown(f"**{len(diffs)} differences between {va} and {vb}**")
        st.dataframe(pd.DataFrame([{"Element": d["element"], va: d[va], vb: d[vb], "Why it changed": d["why"]}
                                   for d in diffs]), hide_index=True, width="stretch")

    ja, jb = json.dumps(ba, indent=2), json.dumps(bb, indent=2)
    j1, j2 = st.columns(2)
    for col, v, b, j in ((j1, va, ba, ja), (j2, vb, bb, jb)):
        with col:
            res = validate_bundle(b, v)
            badge = "✅ valid" if res["valid"] else "⚠️ not schema-checked" if res["valid"] is None else "❌ invalid"
            st.markdown(f"**FHIR {v}** ({VERSIONS[v]['fhirVersion']}) · {badge}")
            if res["errors"]:
                st.error("\n".join(res["errors"]))
            st.code(j, language="json", height=480)
    if va != vb:
        with st.expander("Line-by-line diff", expanded=False):
            st.code("\n".join(difflib.unified_diff(ja.splitlines(), jb.splitlines(), f"FHIR {va}", f"FHIR {vb}",
                                                   lineterm="", n=2)), language="diff")

    st.markdown("#### All differences across versions")
    st.dataframe(pd.DataFrame(DIFFERENCES)[["element"] + VERSION_ORDER + ["why"]],
                 hide_index=True, width="stretch")

    with st.expander("🔍 Paste any FHIR Bundle to detect its version and validate it"):
        pasted = st.text_area("FHIR Bundle JSON", height=200, value=jb)
        if st.button("Detect & validate"):
            try:
                pb = json.loads(pasted)
                det, why = detect_version(pb)
                st.markdown(f"Detected version: **{det}**  \n" + "  \n".join(f"- {w}" for w in why))
                for v in VERSION_ORDER:
                    r = validate_bundle(pb, v)
                    st.write(f"{v}: " + ("✅ valid" if r["valid"] else "— not checked" if r["valid"] is None
                                         else f"❌ {len(r['errors'])} errors"))
            except json.JSONDecodeError as e:
                st.error(f"Not valid JSON: {e}")


# ============================================================================ 5. HL7 v2

with tabs[5]:
    st.subheader("Step 5 · Receive lab results over HL7 v2")
    explain(f"Most lab systems still send results as **HL7 v2 ORU^R01** messages over **MLLP** (TCP). This app runs "
            f"an MLLP **listener on {HL7_HOST}:{HL7_PORT}** inside the API. For each message it: parses the segments → "
            "converts them to a FHIR R4 Bundle → scores data quality → predicts → stores it in the inbox → replies "
            "with an **ACK** (AA = accepted, AE = error, AR = rejected).")
    if listening:
        st.success(f"Listener is running on {HL7_HOST}:{HL7_PORT}")
    else:
        st.error(f"No listener on {HL7_HOST}:{HL7_PORT}. Start the API (`make api` or `make run`).")

    st.markdown("#### Compose a message")
    mode = st.radio("Patient", ["Random new patient", "Patient from the dataset"], horizontal=True)
    f1, f2, f3 = st.columns(3)
    drop_race = f1.checkbox("Leave race/ethnicity empty")
    drop_bmi = f2.checkbox("Leave out the BMI result")
    bad_lab = f3.checkbox("Send an implausible HbA1c (99)")

    if mode == "Patient from the dataset" and bundles is not None:
        pid = st.selectbox("Dataset patient", [b["entry"][0]["resource"]["id"] for b in bundles], key="hl7_pid")
        rec = bundle_to_record(next(b for b in bundles if b["entry"][0]["resource"]["id"] == pid))
    else:
        if "hl7_seed" not in st.session_state or st.button("🎲 New random patient"):
            st.session_state["hl7_seed"] = uuid.uuid4().int % 10**8
        rec = generate_records(1, seed=st.session_state["hl7_seed"], id_prefix="hl7")[0]
    rec = dict(rec)
    if drop_race:
        rec["race_demographic"] = None
    if drop_bmi:
        rec["bmi"] = None
    if bad_lab:
        rec["baseline_hba1c"] = 99.0
    control_id = st.session_state.setdefault("hl7_ctrl", uuid.uuid4().hex[:10].upper())
    composed = hl7.build_oru(rec, control_id).replace("\r", "\n")

    edited = st.text_area("HL7 v2 message (editable; one segment per line)", composed, height=170,
                          key=f"hl7_text_{control_id}_{rec['patient_id']}_{drop_race}{drop_bmi}{bad_lab}")
    with st.expander("What do these fields mean?"):
        st.dataframe(pd.DataFrame(hl7.describe(hl7.parse(edited))), hide_index=True, width="stretch")

    b1, b2, _ = st.columns([1, 1, 2])
    if b1.button("📤 Send over MLLP", type="primary", disabled=not listening):
        ack = send(HL7_HOST, HL7_PORT, hl7.normalize(edited))
        msa = hl7.first(hl7.parse(ack), "MSA")
        code = hl7.field(msa, 1)
        (st.success if code == "AA" else st.error)(
            f"ACK received: **{code}** ({hl7.ACK_CODES.get(code, '?')}): {hl7.field(msa, 3)}")
        st.code(ack.replace("\r", "\n"), language="text")
        st.session_state["hl7_ctrl"] = uuid.uuid4().hex[:10].upper()
    if b2.button("📤 Send 10 random patients", disabled=not listening):
        codes = []
        for _ in range(10):
            r = generate_records(1, seed=uuid.uuid4().int % 10**8, id_prefix="hl7")[0]
            a = send(HL7_HOST, HL7_PORT, hl7.build_oru(r, uuid.uuid4().hex[:10].upper()))
            codes.append(hl7.field(hl7.first(hl7.parse(a), "MSA"), 1))
        st.success(f"Sent 10 messages. ACKs: {', '.join(codes)}")

    st.markdown("#### Inbox: messages received by the listener")
    rows = inbox.recent(200)
    if not rows:
        st.caption("No messages yet.")
    else:
        table = pd.DataFrame([{
            "id": r["id"], "received (UTC)": r["received_at"], "control id": r["control_id"],
            "type": r["message_type"], "patient": r["patient_id"], "ACK": r["ack_code"], "DQI": r["dqi"],
            "issues": ", ".join(r["issues"] or []),
            "routed to": (r["prediction"] or {}).get("recommended_provider_routing", {}).get("specialty"),
        } for r in rows])
        st.dataframe(table, hide_index=True, width="stretch", height=260)
        pick = st.selectbox("Inspect message", [r["id"] for r in rows],
                            format_func=lambda i: next(f"#{r['id']} · {r['patient_id']} · {r['ack_code']}"
                                                       for r in rows if r["id"] == i))
        r = next(r for r in rows if r["id"] == pick)
        i1, i2, i3 = st.tabs(["Raw HL7", "→ FHIR R4 Bundle", "→ Prediction"])
        with i1:
            st.code(r["raw"].replace("\r", "\n"), language="text")
            st.caption(f"ACK sent back: {r['ack_code']}: {r['ack_text']}")
        with i2:
            if r["fhir_bundle"]:
                st.code(json.dumps(r["fhir_bundle"], indent=2), language="json", height=420)
            else:
                st.caption("Not converted (message was rejected).")
        with i3:
            if r["prediction"]:
                show_prediction(r["prediction"])
            else:
                st.caption("No prediction: " + (r["ack_text"] or ""))
        if st.button("🗑️ Clear inbox"):
            inbox.clear()
            st.rerun()

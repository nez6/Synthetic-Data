import requests
import streamlit as st

st.set_page_config(page_title="EHR Decision Dashboard", layout="wide")

st.title("📋 EHR Strategic Decision Support Dashboard")
st.caption("Synthetic data, for demonstration only. Not for clinical use.")
st.markdown("---")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("Patient Core Parameters")
    race = st.selectbox(
        "Demographic Group",
        ["White", "Black/African American", "Hispanic/Latino", "Asian",
         "American Indian/Alaska Native", "(not recorded)"],
    )
    age = st.number_input("Age Profile", 18, 100, 52)
    bmi = st.number_input("Calculated BMI", 10.0, 80.0, 31.2)
    hba1c = st.number_input("Lab Verified HbA1c (%)", 0.0, 100.0, 9.8,
                            help="Try a value outside 3–16 to see the DQI penalty.")

    api_endpoint = st.text_input("Model Target Endpoint", value="http://localhost:8000/predict")
    submit = st.button("Execute Pipeline Audit", type="primary")

with right_col:
    st.header("Pipeline Evaluation Results")

    if submit:
        payload = {
            "race_demographic": None if race == "(not recorded)" else race,
            "age": int(age),
            "bmi": float(bmi),
            "baseline_hba1c": float(hba1c),
        }

        with st.spinner("Processing structural records across analytical models..."):
            try:
                res = requests.post(api_endpoint, json=payload, timeout=5)
            except requests.RequestException as ex:
                st.error(f"Network Pipeline Communication Fault: {ex}")
                st.stop()

        if res.status_code != 200:
            st.error(f"API returned {res.status_code}: {res.text}")
            st.stop()

        data = res.json()

        st.metric(label="Incoming Record Data Quality Score",
                  value=data["input_record_data_quality_index"])
        if data["low_quality_warning"]:
            st.warning("Low data quality: issues found → " + ", ".join(data["data_quality_issues"]))

        med_tile, meal_tile = st.columns(2)
        with med_tile:
            st.subheader("💊 Predicted Pharmacotherapy")
            st.info(data["predicted_medication_strategy"])
        with meal_tile:
            st.subheader("🥗 Dietary Care Pathway")
            st.success(data["predicted_dietary_strategy"])

        st.markdown("---")
        st.header("🩺 Targeted Clinician Matching Recommendation")
        doc = data["recommended_provider_routing"]
        doc_col1, doc_col2 = st.columns([1, 2])
        with doc_col1:
            st.warning(f"**Specialty Assigned:**\n\n{doc['specialty']}")
            st.error(f"**Routing Priority Status:**\n\n{doc['urgency']}")
        with doc_col2:
            st.info(f"**Clinical Routing Rationale:**\n\n{doc['clinical_rationale']}")

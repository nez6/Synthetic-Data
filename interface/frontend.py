"""Streamlit dashboard for the Synthetic Test Data Creator.

    streamlit run interface/frontend.py      (or `make run` to start API + listener + dashboard)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from interface import client  # noqa: E402
from interface.common import api_url  # noqa: E402
from interface.views import api_view, fhir_view, hl7_view, ml_view, profile_view, request_view, results_view  # noqa: E402

st.set_page_config(page_title="Synthetic Test Data Creator", page_icon="🧬", layout="wide")

with st.sidebar:
    st.header("System status")
    st.session_state["api_url"] = st.text_input("API address", value=api_url(),
                                                help="Every tab reads from and writes to this API.").rstrip("/")
    h = client.health()
    if h is None:
        st.error(f"❌ API not reachable at {api_url()}. Cloud Run may be waking up; wait ~30 s and press Refresh.")
    else:
        boot = h.get("bootstrap") or {}
        hl7 = h["hl7_listener"]
        st.markdown(
            f"✅ **API** {api_url()}\n\n"
            + "\n\n".join([f"{'✅' if p['alive'] else '⬜'} **Pool {p['pool']}**: {p['alive']:,} people, "
                           f"ages {p['ages']}" for p in h["pools"]] or ["⬜ **No Synthea pools yet**"])
            + f"\n\n{'✅' if h['runs'] else '⬜'} **Runs**: {h['runs']}"
            + f"\n\n{'✅' if 'profile' in st.session_state else '⬜'} **Real-data profile loaded**"
            + f"\n\n{'✅' if hl7['listening'] else '❌'} **HL7 listener** :{hl7['port']} (inside the API)")
        if boot.get("stage") in ("downloading", "training"):
            st.info(f"⏳ Server warming up: {boot['message']}")
        elif boot.get("stage") == "failed":
            st.error(f"Start-up failed: {boot.get('error')}")
    st.button("🔄 Refresh")
    st.markdown(f"[API docs]({api_url()}/docs)")
    st.divider()
    st.caption("All patients are synthetic (Synthea) and carry the HL7 HTEST label. Not for clinical use.")

st.title("🧬 Synthetic Test Data Creator")
st.caption("Request a custom synthetic patient dataset that matches your population and data quality, "
           "validate it, train a model on it, and pull it from the API.")

tabs = st.tabs(["🧭 How it works", "01 · Profile your data", "📝 Request", "✅ Results", "🧠 ML (train / test / validate)",
                "🔌 API", "🔁 FHIR versions", "📨 HL7 v2"])

with tabs[0]:
    cols = st.columns(4)
    steps = [
        ("01 PROFILE REAL PATIENTS", "Run the downloadable script on your own FHIR data. It returns **aggregates "
         "only** (demographic mix, conditions, data-quality problem rates); small cells are suppressed, so no "
         "patient row leaves your system."),
        ("02 GENERATE WITH SYNTHEA", "Synthea v4.0.0 simulates patient pools (fixed seeds, reproducible). Your request "
         "picks eligible patients per cohort, matching your demographic mix where the pool allows."),
        ("03 VALIDATE, REPAIR, LEARN", "Checks V01–V08 (US Core, references, codes, C-CDA readiness, dates, "
         "resource volumes) with safe repairs, an independent audit, and a model trained 60/20/20 per cohort."),
        ("04 SELECT AND DELIVER", "Patients that pass are labelled HTEST and delivered. Pull the dataset from the "
         "API as FHIR NDJSON, CSV or a zip with HL7 v2 messages."),
    ]
    for col, (title, text) in zip(cols, steps):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.write(text)
    st.markdown(
        """
#### How to use it
1. **01 · Profile your data**: download the script, run it on your FHIR export, upload the `profile.json`
   (or try the demo that profiles the Synthea pools themselves).
2. **📝 Request**: set the cohorts and resource targets ("100 records: 75 typical, 25 complex"). The profile
   fills in your demographic mix and data-quality problems. Click **Generate & validate**.
3. **✅ Results**: pass rates before and after validation, every check, the statistical acceptance test, each
   patient's issues and repairs, and the delivery set.
4. **🧠 ML**: each cohort split 60/20/20; a model that sees **demographics only** learns which records pass
   validation. Compare the raw and validated data, and your real data with the synthetic data.
5. **🔌 API**: pull the custom dataset into your own application.
"""
    )
    st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box style="rounded,filled" fillcolor="#eef4fc" color="#2a78d6" fontname="Helvetica" fontsize=11];
  edge [color="#888888" fontname="Helvetica" fontsize=9];
  real [label="Your FHIR data\\n(stays on your system)"]; prof [label="profile.json\\n(aggregates only)"];
  req [label="Request\\ncohorts · targets · mix"]; syn [label="Synthea pools\\nA · B"];
  sel [label="Select per cohort\\n(eligibility + mix)"]; val [label="Validate & repair\\nV01–V08 + audit"];
  ml [label="ML 60/20/20\\ndemographics → pass?"]; del [label="Delivery set\\nHTEST · FHIR · CSV · HL7"];
  api [label="API\\n/runs · /dataset · /predict"];
  real -> prof [label="data_quality_profile.py"]; prof -> req; req -> sel; syn -> sel -> val -> del -> api; val -> ml -> api;
}""")

with tabs[1]:
    profile_view.render()
with tabs[2]:
    request_view.render()
with tabs[3]:
    results_view.render()
with tabs[4]:
    ml_view.render()
with tabs[5]:
    api_view.render()
with tabs[6]:
    fhir_view.render()
with tabs[7]:
    hl7_view.render()

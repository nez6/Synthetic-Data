"""🔌 API · pull a custom dataset into your own application."""
import json

import pandas as pd
import requests
import streamlit as st

from interface.common import api_url, explain, pick_run


def render():
    st.subheader("🔌 Pull your dataset from the API")
    explain("Everything in this dashboard is also available over HTTP, so your own application (or a test harness) "
            "can request a dataset, wait for it, and download it. Interactive docs: "
            f"{api_url()}/docs")
    base = api_url()
    run_id = pick_run("api_run") or "<run_id>"
    st.markdown("#### 1 · Submit a request")
    st.code(f"""# default request (75 A / 25 B per 100), or post your own "request" and/or "real_profile"
curl -X POST {base}/runs -H 'content-type: application/json' -d '{{}}'
curl -X POST {base}/runs -H 'content-type: application/json' -d @- <<EOF
{{"real_profile": $(cat profile.json)}}
EOF
curl {base}/jobs/<job_id>          # → {{"status": "done", "run_id": "..."}}""", language="bash")
    st.markdown("#### 2 · Pull the dataset")
    st.code(f"""curl -o data.ndjson.gz '{base}/runs/{run_id}/dataset?format=ndjson'   # FHIR R4 Bundles, one per line
curl -o data.csv        '{base}/runs/{run_id}/dataset?format=csv'      # one row per patient
curl -o data.zip        '{base}/runs/{run_id}/dataset?format=zip'      # + HL7 v2, report, ML artifacts
curl '{base}/runs/{run_id}/patients/<patient_id>?fhir_version=R5'      # one patient, any FHIR version""",
            language="bash")
    st.markdown("#### 3 · Ask the model")
    st.code(f"""curl -X POST {base}/predict -H 'content-type: application/json' \\
  -d '{{"age": 72, "sex": "female", "race_ethnicity": "Hispanic/Latino", "cohort": "B"}}'""", language="bash")

    st.markdown("#### Try it live")
    c1, c2, c3 = st.columns(3)
    if c1.button("GET /runs"):
        try:
            st.dataframe(pd.DataFrame(requests.get(f"{base}/runs", timeout=10).json()), hide_index=True)
        except requests.RequestException as e:
            st.error(f"API not reachable: {e}")
    if c2.button(f"GET /runs/{run_id[:15]}…/dataset?format=csv"):
        try:
            res = requests.get(f"{base}/runs/{run_id}/dataset", params={"format": "csv"}, timeout=60)
            st.caption(f"HTTP {res.status_code} · {len(res.content):,} bytes")
            st.code("\n".join(res.text.splitlines()[:6]), language="text")
        except requests.RequestException as e:
            st.error(f"API not reachable: {e}")
    if c3.button("GET /requests/default"):
        try:
            st.code(json.dumps(requests.get(f"{base}/requests/default", timeout=10).json(), indent=1)[:3000],
                    language="json")
        except requests.RequestException as e:
            st.error(f"API not reachable: {e}")

"""🔁 FHIR versions · export a delivered patient as DSTU2 / STU3 / R4 / R4B / R5."""
import difflib
import json

import pandas as pd
import streamlit as st

from interface import client
from interface.common import api_url, explain, pick_run
from src.fhir.builder import build_bundle
from src.fhir.validate import validate_bundle
from src.fhir.versions import DIFFERENCES, VERSION_ORDER, VERSIONS, differences_between


def _records(run_id):
    return client.delivered_records(api_url(), run_id)


def render():
    st.subheader("🔁 The same patient in five FHIR versions")
    explain("Delivered patients are full FHIR R4 records. If the system you are testing speaks another version, "
            "the API can return a one-visit snapshot (demographics, latest vitals and labs, smoking status, chronic "
            "conditions) in DSTU2, STU3, R4B or R5. Everything except DSTU2 is checked against the official schema.")
    st.dataframe(pd.DataFrame([{"version": k, **v} for k, v in VERSIONS.items()]), hide_index=True, width="stretch")
    run_id = pick_run("fhir_run")
    if not run_id:
        return
    recs = {r["patient_id"]: r for r in _records(run_id)}
    if not recs:
        st.info("This run delivered no patients.")
        return
    c1, c2, c3 = st.columns(3)
    pid = c1.selectbox("Delivered patient", list(recs), key="fhir_pid")
    va = c2.selectbox("Version A", VERSION_ORDER, index=VERSION_ORDER.index("R4"))
    vb = c3.selectbox("Version B", VERSION_ORDER, index=VERSION_ORDER.index("R5"))
    ba, bb = build_bundle(recs[pid], va), build_bundle(recs[pid], vb)
    diffs = differences_between(va, vb)
    if va != vb and diffs:
        st.markdown(f"**{len(diffs)} differences between {va} and {vb}**")
        st.dataframe(pd.DataFrame([{"Element": d["element"], va: d[va], vb: d[vb], "Why it changed": d["why"]}
                                   for d in diffs]), hide_index=True, width="stretch")
    elif va != vb:
        st.success(f"{va} and {vb} are identical for these resources.")
    ja, jb = json.dumps(ba, indent=2), json.dumps(bb, indent=2)
    j1, j2 = st.columns(2)
    for col, v, b, j in ((j1, va, ba, ja), (j2, vb, bb, jb)):
        with col:
            res = validate_bundle(b, v)
            badge = "✅ valid" if res["valid"] else "⚠️ not schema-checked" if res["valid"] is None else "❌ invalid"
            st.markdown(f"**FHIR {v}** ({VERSIONS[v]['fhirVersion']}) · {badge}")
            if res["errors"]:
                st.error("\n".join(res["errors"]))
            st.code(j, language="json", height=460)
    if va != vb:
        with st.expander("Line-by-line diff"):
            st.code("\n".join(difflib.unified_diff(ja.splitlines(), jb.splitlines(), f"FHIR {va}", f"FHIR {vb}",
                                                   lineterm="", n=2)), language="diff")
    with st.expander("All differences across versions"):
        st.dataframe(pd.DataFrame(DIFFERENCES)[["element"] + VERSION_ORDER + ["why"]], hide_index=True,
                     width="stretch")

"""Step 01 · Profile your real data (aggregates only)."""
import json

import pandas as pd
import streamlit as st

from interface.common import explain
from src import pools
from src.config import load_config, resolve
from tools.data_quality_profile import QUALITY_KEYS, profile_paths

QUALITY_LABELS = {
    "missing_birthdate": "Birth date missing", "missing_race_ethnicity": "Race/ethnicity not recorded",
    "missing_language": "Preferred language not recorded", "missing_bmi": "No BMI observation",
    "implausible_lab": "An implausible vital/lab value", "non_ucum_unit": "A unit not UCUM-coded",
    "missing_obs_category": "An Observation without category", "reversed_period": "A period ending before it starts",
    "dangling_reference": "A reference that does not resolve",
}


def _show(profile: dict):
    n = profile["n_patients"]
    k = profile["k_min"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients profiled", f"{n:,}")
    c2.metric("Source", profile.get("source", "?"))
    c3.metric("Small cells suppressed", len(profile["suppressed"]), help=f"Counts below {k} are reported as hidden.")
    c4.metric("Complex care (2+ chronic groups)", "<k" if profile["complex_care"] is None else profile["complex_care"])

    def table(counts):
        rows = [{"group": g, "patients": "suppressed" if v is None else v,
                 "share": "–" if v is None else f"{100 * v / n:.1f}%"} for g, v in counts.items()]
        return pd.DataFrame(rows)

    a, b, c = st.columns(3)
    with a:
        st.markdown("**Race / ethnicity**")
        st.dataframe(table(profile["counts"]["race_ethnicity"]), hide_index=True, width="stretch")
        st.markdown("**Sex**")
        st.dataframe(table(profile["counts"]["sex"]), hide_index=True, width="stretch")
    with b:
        st.markdown("**Age band**")
        st.dataframe(table(profile["counts"]["age_band"]), hide_index=True, width="stretch")
        st.markdown("**Chronic condition groups**")
        st.dataframe(table(profile["counts"]["conditions"]), hide_index=True, width="stretch")
    with c:
        st.markdown("**Data-quality problems (patients affected)**")
        q = profile["quality"]
        st.dataframe(pd.DataFrame([{"problem": QUALITY_LABELS[key],
                                    "patients": "suppressed" if q.get(key) is None else q.get(key, 0),
                                    "share": "–" if q.get(key) is None else f"{100 * q.get(key, 0) / n:.1f}%"}
                                   for key in QUALITY_KEYS]), hide_index=True, width="stretch")
    with st.expander("Everything in the file (this is all that leaves your system)"):
        st.json(profile)


def render():
    st.subheader("01 · Profile your real patients")
    explain("Run this script **on your own system**. It reads your FHIR R4 export and writes `profile.json`: "
            "counts only (demographic mix, chronic conditions, vital/lab summaries, data-quality problem rates). "
            f"Any count below {load_config()['privacy']['k_min_cell_size']} is suppressed, and fewer than that many "
            "patients is refused, so no patient-level data ever leaves your system. Upload the profile and the "
            "request will match your population **and** your data quality.")

    script = resolve(load_config()["paths"]["profile_script"])
    c1, c2 = st.columns([1, 2])
    c1.download_button("⬇️ Download data_quality_profile.py", script.read_bytes(), "data_quality_profile.py",
                       "text/x-python", type="primary")
    c2.code("python data_quality_profile.py --input /path/to/fhir_export --out profile.json "
            "--source \"My EHR\"", language="bash")
    st.caption("Accepts per-patient Bundles (*.json, *.json.gz) and bulk-export NDJSON (one resource per line). "
               "Standard library only (Python 3.9+), nothing to install.")

    st.markdown("#### Upload your profile")
    up = st.file_uploader("profile.json", type=["json"])
    if up is not None:
        try:
            prof = json.load(up)
            assert "counts" in prof and "n_patients" in prof
            st.session_state["profile"] = prof
            st.success(f"Loaded a profile of {prof['n_patients']:,} patients. The **📝 Request** tab now uses it.")
        except (json.JSONDecodeError, AssertionError):
            st.error("That file is not a profile produced by data_quality_profile.py.")

    with st.expander("No data handy? Profile the Synthea pools as a demo"):
        st.caption("Runs the same script on this app's own Synthea pools, as if they were your data. Useful to see "
                   "what a profile looks like; a request matched to it reproduces Synthea's own mix.")
        names = [p["pool"] for p in pools.list_pools()]
        if st.button("Profile the pools", disabled=not names):
            with st.spinner("Profiling…"):
                prof = profile_paths([str(pools.pools_dir() / n / "patients") for n in names],
                                     k_min=load_config()["privacy"]["k_min_cell_size"],
                                     source="Demo: Synthea pools " + "+".join(names))
            st.session_state["profile"] = prof
            st.success("Demo profile loaded.")

    if "profile" in st.session_state:
        st.markdown("#### Current profile")
        _show(st.session_state["profile"])
        if st.button("Remove profile"):
            del st.session_state["profile"]
            st.rerun()

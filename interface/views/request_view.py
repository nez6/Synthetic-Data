"""📝 Request · "Test Patient Data: N Records Requested" (cohorts, resource targets, data depth)."""
import copy
import time
from datetime import date

import pandas as pd
import streamlit as st

from interface import client
from interface.common import ApiError, api_url, explain
from src.request import FAULTS, RESOURCE_ROWS, STRESS_FACTOR, apply_profile, default_request, problems

GROUP_LABELS = {"chf": "CHF", "ckd_esrd": "CKD/ESRD", "t2d": "T2D", "copd": "COPD"}


def _targets_table(cohorts):
    rows = []
    for rtype in RESOURCE_ROWS:
        row = {"FHIR resource": rtype.replace("|", " / ")}
        for c in cohorts:
            t = c["targets"].get(rtype, {"min": 0, "max": None})
            row[f"{c['name']} min"] = t["min"]
            row[f"{c['name']} max"] = t["max"]
        row["Stress factor"] = STRESS_FACTOR[rtype]
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_targets(cohorts, df):
    for _, row in df.iterrows():
        rtype = next(r for r in RESOURCE_ROWS if r.replace("|", " / ") == row["FHIR resource"])
        for c in cohorts:
            mx = row[f"{c['name']} max"]
            sev = c["targets"].get(rtype, {}).get("severity", "error")
            c["targets"][rtype] = {"min": int(row[f"{c['name']} min"]),
                                   "max": None if pd.isna(mx) else int(mx), "severity": sev}


def _pools_panel(request):
    st.markdown("##### Synthea pools (step 02)")
    try:
        by_name = {p["pool"]: p for p in client.get("/pools")}
        elig = {e["cohort"]: e["eligible"] for e in client.post("/pools/eligible", request)}
    except ApiError as e:
        st.error(str(e))
        return
    table = [{"cohort": c["name"], "pool": c["pool"],
              "people simulated": by_name.get(c["pool"], {}).get("people", 0),
              "eligible for cohort": elig.get(c["name"], 0),
              "ages": by_name.get(c["pool"], {}).get("ages", "–"),
              "history (yrs)": by_name.get(c["pool"], {}).get("years_of_history", "–")} for c in request["cohorts"]]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    st.caption("Pools are pre-built with Synthea and downloaded from Google Cloud Storage. To grow one, run "
               "`python -m src.pools build B --population 800` locally (needs Java), then "
               "`python -m src.cloud_data upload`.")


def _submit(req: dict, profile: dict | None):
    """POST /runs, then poll /jobs/{id} until the run is finished."""
    job = client.post("/runs", {"request": req, "real_profile": profile})
    bar = st.progress(0.0, "Queued…")
    while job["status"] == "running":
        time.sleep(1.5)
        job = client.get(f"/jobs/{job['job_id']}")
        bar.progress(min(float(job.get("progress") or 0), 1.0), job.get("message") or "")
    bar.empty()
    if job["status"] != "done":
        raise ApiError(job.get("error") or "run failed")
    client.runs.clear()
    return client.get(f"/runs/{job['run_id']}")


def render():
    st.subheader("📝 Test patient data request")
    explain("Describe the test data you need: how many records, which cohorts, and how many of each FHIR "
            "resource every patient should have. If a profile is loaded (tab 01) your demographic mix and data-quality "
            "problems are copied in, so the synthetic data looks like yours.")

    base = default_request()
    notes = []
    if "profile" in st.session_state:
        base, notes = apply_profile(base, st.session_state["profile"])
        st.success(f"Using the profile from **{st.session_state['profile'].get('source')}**.")
        for nt in notes:
            st.caption("• " + nt)
    req = copy.deepcopy(base)

    c1, c2, c3, c4 = st.columns(4)
    req["label"] = c1.text_input("Request name", base["label"])
    req["delivery_size"] = int(c2.number_input("Records to deliver", 10, 5000, base["delivery_size"], step=10))
    req["max_per_cohort"] = int(c3.number_input("Patients tested per cohort (max)", 20, 5000, base["max_per_cohort"],
                                                help="All eligible pool patients up to this number are validated "
                                                     "and used for the 60/20/20 split. 363 gives 80% power to show "
                                                     "> 85% when the true rate is 90%."))
    req["seed"] = int(c4.number_input("Seed", 0, 10**9, base["seed"]))

    left, right = st.columns([1, 2])
    with left:
        st.markdown("##### Requested cohorts")
        for c in req["cohorts"]:
            with st.container(border=True):
                st.markdown(f"**COHORT {c['name']}** · {c['label']}")
                a, b = st.columns(2)
                c["share"] = a.number_input("Share of delivery (%)", 0, 100, int(round(c["share"] * 100)),
                                            key=f"share{c['name']}") / 100
                ages = b.slider("Age range", 0, 100, (c["min_age"], c["max_age"]), key=f"age{c['name']}")
                c["min_age"], c["max_age"] = ages
                a, b = st.columns(2)
                c["min_condition_groups"] = int(a.number_input("Min chronic groups", 0, 4, c["min_condition_groups"],
                                                               key=f"mcg{c['name']}"))
                c["exclude_condition_groups"] = b.checkbox("Exclude chronic groups", c["exclude_condition_groups"],
                                                           key=f"ex{c['name']}")
                a, b = st.columns(2)
                c["min_history_years"] = a.number_input("Years of continuous history", 0.0, 10.0,
                                                        float(c["min_history_years"]), key=f"hy{c['name']}")
                gap = b.number_input("Max gap between visits (days, 0 = none)", 0, 3650, c["max_gap_days"] or 0,
                                     key=f"gap{c['name']}")
                c["max_gap_days"] = gap or None
                st.caption("Purpose: " + c["purpose"] + " Chronic groups: " + ", ".join(GROUP_LABELS.values()))
    with right:
        st.markdown("##### Resource targets per patient")
        edited = st.data_editor(_targets_table(req["cohorts"]), hide_index=True, width="stretch",
                                disabled=["FHIR resource", "Stress factor"], key="targets",
                                column_config={col: st.column_config.NumberColumn(col, min_value=0, step=1)
                                               for col in _targets_table(req["cohorts"]).columns
                                               if col.endswith(("min", "max"))})
        _apply_targets(req["cohorts"], edited)
        st.caption("Empty max = no upper limit ('30+'). Cohort B's allergy target is a warning only. US Core "
                   "profiles apply where they exist. Referential integrity: every reference must resolve inside its "
                   "bundle (checked by V03).")
        a, b, c = st.columns(3)
        req["split"]["train"] = a.number_input("Train %", 10, 90, 60) / 100
        req["split"]["test"] = b.number_input("Test % (tuning)", 5, 50, 20) / 100
        req["split"]["validation"] = c.number_input("Validation % (final score)", 5, 50, 20) / 100
        a, b, c = st.columns(3)
        req["acceptance"]["h0_rate"] = a.number_input("Must beat (H0)", 0.5, 0.99, 0.85, 0.01)
        req["acceptance"]["target_rate"] = b.number_input("Target pass rate", 0.5, 0.999, 0.90, 0.01)
        req["acceptance"]["alpha"] = c.number_input("Alpha", 0.001, 0.2, 0.05, 0.01)

    with st.expander("Demographic mix and data-quality faults"
                     + (" (from your profile)" if "profile" in st.session_state else " (none: Synthea's own mix)")):
        if req["demographics"]:
            st.markdown("Patients are drawn by race/ethnicity quota to match:")
            st.dataframe(pd.DataFrame([{"group": g, "target %": v} for g, v in
                                       req["demographics"]["race_ethnicity"].items()]), hide_index=True)
        else:
            st.caption("Load a profile in tab 01 to match your demographic mix.")
        st.markdown("Faults injected (% of patients), to mirror your data quality:")
        cols = st.columns(len(FAULTS))
        for col, (k, (label, effect)) in zip(cols, FAULTS.items()):
            req["faults"][k] = col.number_input(label, 0.0, 50.0, float(req["faults"].get(k, 0)), step=0.5,
                                                key=f"fault{k}", help=effect)

    _pools_panel(req)
    req["as_of"] = date.today().isoformat()
    issues = problems(req)
    for p in issues:
        st.error(p)
    if st.button("🧬 Generate & validate", type="primary", disabled=bool(issues)):
        try:
            report = _submit(req, st.session_state.get("profile"))
        except ApiError as e:
            st.error(str(e))
            return
        st.session_state["run_id"] = report["run_id"]
        o = report["cohort_scores"]["Overall"]
        st.success(f"Run **{report['run_id']}**: {report['n_tested']} patients tested, pass rate "
                   f"{o['baseline_pass_rate']:.1%} → {o['validated_pass_rate']:.1%}, {report['n_delivered']} "
                   f"delivered. Open **✅ Results** and **🧠 ML**. (POST {api_url()}/runs)")

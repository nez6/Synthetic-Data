"""✅ Results · "Four steps produced N test patients"."""
import json
from collections import Counter

import pandas as pd
import streamlit as st

from interface import client
from interface.common import ApiError, api_url, load_run, pct, pick_run
from src.request import FAULTS


def _cards(rep, req):
    s = rep["cohort_scores"]
    o = s["Overall"]
    real = (rep.get("profiles") or {}).get("real")
    cols = st.columns(4)
    with cols[0].container(border=True):
        st.caption("01 PROFILE REAL PATIENTS")
        if real:
            st.markdown(f"### {real['n_patients']:,}")
            st.write(f"patients profiled in **{real.get('source')}**. Aggregates only; "
                     f"{len(real['suppressed'])} small cells suppressed.")
        else:
            st.markdown("### –")
            st.write("No real-data profile: Synthea's own demographic mix was used.")
    with cols[1].container(border=True):
        st.caption("02 GENERATE WITH SYNTHEA")
        st.markdown(f"### {rep['n_tested']:,}")
        st.write("synthetic patients tested: " + ", ".join(f"**{rep['selection'][c]['selected']}** Cohort {c}"
                                                           for c in rep["selection"]))
        v = next(iter(rep["synthea"].values()))
        st.caption(f"Synthea {v.get('synthea_version')}, fixed seeds; jar sha256 {str(v.get('jar_sha256'))[:10]}…")
    with cols[2].container(border=True):
        st.caption("03 VALIDATE AND REPAIR")
        st.markdown(f"### {pct(o['validated_pass_rate'])}")
        st.write(f"pass rate, up from **{pct(o['baseline_pass_rate'])}**. Checks V01–V08: US Core, "
                 "references, codes, C-CDA readiness, dates, volumes; plus the independent audit.")
        st.caption(f"Script alone {pct(o['script_pass_rate'])} · audit alone {pct(o['audit_pass_rate'])}")
        st.caption(f"{o['patients_with_removals']} patients trimmed; mean {pct(o['mean_share_removed'])} of "
                   "resources removed.")
    with cols[3].container(border=True):
        st.caption("04 SELECT AND DELIVER")
        st.markdown(f"### {rep['n_delivered']:,}")
        st.write("patients delivered: " + ", ".join(f"{d['delivered']} {c}" for c, d in rep["delivery"].items())
                 + ". HTEST-labelled FHIR R4.")
        short = {c: d["requested"] - d["delivered"] for c, d in rep["delivery"].items()
                 if d["delivered"] < d["requested"]}
        if short:
            st.caption("⚠️ Short of the request: " + ", ".join(f"{n} in {c}" for c, n in short.items()))


def render():
    st.subheader("✅ Results")
    run_id = pick_run("results_run")
    if not run_id:
        return
    data = load_run(run_id)
    rep, req, records = data["report"], data["request"], data["records"]
    _cards(rep, req)

    st.markdown("#### Acceptance test per cohort")
    rows = []
    for c, s in rep["cohort_scores"].items():
        rows.append({"cohort": c, "tested": s["n"], "baseline": pct(s["baseline_pass_rate"]),
                     "after validation": pct(s["validated_pass_rate"]),
                     "95% CI": f"{pct(s['validated_ci_low'])}–{pct(s['validated_ci_high'])}",
                     f"p (H0 ≤ {pct(s['h0_rate'], 0)})": f"{s['p_value_vs_h0']:.2g}",
                     "script / audit alone": f"{pct(s['script_pass_rate'])} / {pct(s['audit_pass_rate'])}",
                     "decision": "✅ " + s["decision"] if s["reject_h0"] else "❌ " + s["decision"],
                     "McNemar fixed / broken": f"{s['mcnemar_c']} / {s['mcnemar_b']}",
                     "needed n": s["required_n"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Pass = no error from the validation script AND no failure in the independent audit, which shares no "
               f"code with the script. Alpha per cohort = {rep['alpha_per_cohort']:.3f} (Bonferroni across cohorts).")

    a, b = st.columns([3, 2])
    with a:
        st.markdown("#### Checks")
        st.dataframe(pd.DataFrame(rep["checks"]), hide_index=True, width="stretch")
    with b:
        st.markdown("#### Selection and faults")
        st.dataframe(pd.DataFrame([{"cohort": c, **{k: v for k, v in s.items() if k != "shortfall"},
                                    "shortfall": ", ".join(f"{g}: {n}" for g, n in s["shortfall"].items()) or "–"}
                                   for c, s in rep["selection"].items()]), hide_index=True, width="stretch")
        applied = rep.get("faults_applied") or {}
        if applied:
            st.dataframe(pd.DataFrame([{"fault": FAULTS[f][0], "patients": n, "effect": FAULTS[f][1]}
                                       for f, n in applied.items()]), hide_index=True, width="stretch")
        else:
            st.caption("No faults injected.")

    st.markdown("#### Patients")
    f1, f2, f3 = st.columns([1, 1, 2])
    cohort = f1.selectbox("Cohort", ["All"] + list(rep["selection"]), key="rcoh")
    show = f2.selectbox("Show", ["Failed after validation", "Repaired", "Passed", "All"], key="rshow")
    keep = {"Failed after validation": lambda r: not r["pass_after"],
            "Repaired": lambda r: bool(r["repairs"]), "Passed": lambda r: r["pass_after"],
            "All": lambda r: True}[show]
    subset = [r for r in records if keep(r) and cohort in ("All", r["cohort"])]
    if not subset:
        st.info("No patients in this group.")
    else:
        table = pd.DataFrame([{"patient": r["patient_id"][:13], "cohort": r["cohort"], "age": r["age"], "sex": r["sex"],
                               "race/ethnicity": r["race_ethnicity"],
                               "errors after": len([i for i in r["issues_after"] if i["severity"] == "error"]),
                               "repairs": len(r["repairs"]), "resources": f"{r['resources_before']} → {r['resources_after']}",
                               "first problem": next((f"{i['check_id']}: {i['message']}" for i in r["issues_after"]
                                                      if i["severity"] == "error"), "–")} for r in subset])
        st.dataframe(table, hide_index=True, width="stretch", height=260)
        pid = f3.selectbox(f"Inspect one of {len(subset)}", [r["patient_id"] for r in subset], key="rpid")
        r = next(x for x in records if x["patient_id"] == pid)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Issues before**")
            st.dataframe(pd.DataFrame(r["issues_before"] or [{"message": "none"}]), hide_index=True, height=220)
        with c2:
            st.markdown("**Repairs**")
            st.dataframe(pd.DataFrame(r["repairs"] or [{"action": "none"}]), hide_index=True, height=220)
        with c3:
            st.markdown("**Still failing**")
            fails = [f"{i['check_id']}: {i['message']}" for i in r["issues_after"] if i["severity"] == "error"]
            fails += [f"audit: {f}" for f in r["independent_failures_after"]]
            st.dataframe(pd.DataFrame({"problem": fails or ["none"]}), hide_index=True, height=220)
        if st.button("Show resource counts before → after for this patient"):
            try:
                both = client.get(f"/runs/{run_id}/patients/{pid}/before-after", timeout=120)
            except ApiError as e:
                st.error(str(e))
                return
            raw, validated = both["raw"], both["validated"]
            before = Counter(e["resource"]["resourceType"] for e in raw["entry"])
            after = Counter(e["resource"]["resourceType"] for e in validated["entry"])
            st.dataframe(pd.DataFrame([{"resource": t, "before": before[t], "after": after.get(t, 0)}
                                       for t in sorted(before)]), hide_index=True)
            st.download_button("⬇️ Validated bundle (JSON)", json.dumps(validated, indent=1), f"{pid}.json")

    st.markdown("#### Download the delivery set")
    st.caption(f"Served by GET {api_url()}/runs/{run_id}/dataset?format=…")
    d1, d2, d3 = st.columns(3)
    for col, fmt, label, name, mime in (
            (d1, "zip", "⬇️ Everything (.zip)", f"{run_id}.zip", "application/zip"),
            (d2, "csv", "⬇️ One row per patient (.csv)", f"{run_id}.csv", "text/csv"),
            (d3, "ndjson", "⬇️ FHIR R4 Bundles (.ndjson.gz)", f"{run_id}.ndjson.gz", "application/gzip")):
        key = f"dl_{fmt}_{run_id}"
        if st.session_state.get(key):
            col.download_button(label, client.dataset(api_url(), run_id, fmt), name, mime)
        elif col.button(f"Prepare {fmt}", key=f"prep_{key}"):
            st.session_state[key] = True
            st.rerun()

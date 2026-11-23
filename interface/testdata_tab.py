"""🧬 Test Data Creator tab: enter key demographics → generate, validate, score and export test patients."""
import copy
import difflib
import json
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from src.testdata import runner
from src.testdata.spec import (CONTROL_TIERS, FAULTS, LANGUAGES, NOT_RECORDED, PRESETS, RACE_OPTIONS,
                               STATES, CohortSpec)
from src.testdata.validation import CHECKS

SERIES = "#2a78d6"
INK = "#55554f"


# Runs are immutable once written, so their files can be cached by run id.
@st.cache_data(show_spinner=False, max_entries=8)
def _load_run(run_id: str) -> dict:
    return runner.load_run(run_id)


@st.cache_data(show_spinner=False, max_entries=8)
def _bundles(run_id: str, which: str) -> list:
    return runner.load_bundles(run_id, which)


@st.cache_data(show_spinner="Packaging the delivery set…", max_entries=4)
def _zip(run_id: str) -> bytes:
    return runner.zip_run(run_id)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(ci) -> str:
    return f"{100 * ci[0]:.1f}–{100 * ci[1]:.1f}%"


# ============================================================================ inputs

def _inputs():
    preset_name = st.selectbox("Start from a preset (then adjust anything below)", list(PRESETS),
                               help="Presets are illustrative starting points, not population benchmarks.")
    base: CohortSpec = copy.deepcopy(PRESETS[preset_name])
    k = preset_name  # widget keys change with the preset so its defaults load

    with st.form("td_form"):
        st.markdown("##### 1 · Cohort")
        c1, c2, c3 = st.columns(3)
        label = c1.text_input("Cohort name", base.label, key=f"label{k}")
        n = c2.number_input("Number of patients", 20, 5000, base.n, step=50, key=f"n{k}")
        seed = c3.number_input("Random seed (same seed → same patients)", 0, 10**9, base.seed, key=f"seed{k}")
        c1, c2, c3 = st.columns(3)
        ages = c1.slider("Age range", 0, 100, (base.min_age, base.max_age), key=f"age{k}")
        pct_female = c2.slider("% female", 0, 100, int(base.pct_female), key=f"fem{k}")
        states = c3.multiselect("States (address)", list(STATES), base.states, key=f"st{k}")

        st.markdown("##### 2 · Race / ethnicity mix (% of patients, should total 100)")
        cols = st.columns(len(RACE_OPTIONS))
        race_mix = {g: cols[i].number_input(g, 0.0, 100.0, float(base.race_mix.get(g, 0)), step=1.0, key=f"r{g}{k}")
                    for i, g in enumerate(RACE_OPTIONS)}
        st.caption("Hispanic/Latino is recorded as an ethnicity (US Core). 'Not recorded' leaves race and "
                   "ethnicity out of the record, which the Data Quality Index then penalises.")

        st.markdown("##### 3 · Preferred language (% of patients)")
        cols = st.columns(len(LANGUAGES))
        lang_mix = {code: cols[i].number_input(LANGUAGES[code] or NOT_RECORDED, 0.0, 100.0,
                                               float(base.language_mix.get(code, 0)), step=1.0, key=f"l{code}{k}")
                    for i, code in enumerate(LANGUAGES)}

        st.markdown("##### 4 · Clinical profile")
        cols = st.columns(3)
        control_mix = {tier: cols[i].number_input(f"% {desc}", 0.0, 100.0, float(base.control_mix[tier]),
                                                  step=1.0, key=f"c{tier}{k}")
                       for i, (tier, (desc, _, _)) in enumerate(CONTROL_TIERS.items())}
        c1, c2, c3, c4 = st.columns(4)
        bmi_mean = c1.number_input("Mean BMI", 18.0, 50.0, float(base.bmi_mean), key=f"bm{k}")
        bmi_sd = c2.number_input("BMI spread (SD)", 0.0, 15.0, float(base.bmi_sd), key=f"bs{k}")
        history = c3.number_input("Years of history", 1, 10, base.history_years, key=f"hy{k}")
        visits = c4.number_input("Visits per year", 1, 6, base.visits_per_year, key=f"vy{k}",
                                 help="Each visit records an HbA1c and a BMI.")

        with st.expander("5 · Data-quality faults to inject (% of patients)"):
            st.caption("Faults are added on purpose so the validation script has something to find. "
                       "🔧 = the script can repair it safely; ⛔ = it cannot (the record stays failed).")
            cols = st.columns(2)
            faults = {}
            for i, (fk, (flabel, repairable)) in enumerate(FAULTS.items()):
                faults[fk] = cols[i % 2].slider(f"{'🔧' if repairable else '⛔'} {flabel}", 0, 30,
                                                int(base.faults.get(fk, 0)), key=f"f{fk}{k}")

        with st.expander("6 · Acceptance criteria"):
            c1, c2, c3 = st.columns(3)
            h0 = c1.number_input("H0 pass rate (must beat)", 0.5, 0.99, base.h0_rate, step=0.01, key=f"h0{k}")
            target = c2.number_input("Target pass rate", 0.5, 0.999, base.target_rate, step=0.01, key=f"tg{k}")
            alpha = c3.number_input("Significance level (alpha)", 0.001, 0.2, base.alpha, step=0.01, key=f"al{k}")
            c1, c2, c3 = st.columns(3)
            min_a1c = c1.number_input("Min HbA1c results per patient", 0, 20, base.min_hba1c_obs, key=f"ma{k}")
            min_bmi = c2.number_input("Min BMI results per patient", 0, 20, base.min_bmi_obs, key=f"mb{k}")
            as_of = c3.date_input("Data as-of date", date.fromisoformat(base.as_of), key=f"ao{k}")
            schema = st.checkbox("Also validate every record against the official FHIR R4 schema", True,
                                 key=f"sc{k}")

        submitted = st.form_submit_button("🧬 Generate & validate test patients", type="primary")

    spec = CohortSpec(label=label, n=int(n), seed=int(seed), min_age=ages[0], max_age=ages[1],
                      pct_female=float(pct_female), race_mix=race_mix, language_mix=lang_mix, states=states,
                      control_mix=control_mix, bmi_mean=bmi_mean, bmi_sd=bmi_sd, history_years=int(history),
                      visits_per_year=int(visits), faults=faults, min_hba1c_obs=int(min_a1c),
                      min_bmi_obs=int(min_bmi), h0_rate=h0, target_rate=target, alpha=alpha,
                      as_of=as_of.isoformat())
    return submitted, spec, schema


# ============================================================================ results

def _fidelity_chart(rows: list[dict]):
    df = pd.DataFrame(rows)
    bars = alt.Chart(df).mark_bar(color=SERIES, cornerRadiusEnd=4, height=16).encode(
        y=alt.Y("category:N", title=None, sort=None),
        x=alt.X("delivered_pct:Q", title="% of delivered patients"),
        tooltip=[alt.Tooltip("category:N", title="Group"),
                 alt.Tooltip("requested_pct:Q", title="Requested %"),
                 alt.Tooltip("generated_pct:Q", title="Generated %"),
                 alt.Tooltip("delivered_pct:Q", title="Delivered %"),
                 alt.Tooltip("delivered_ci_low:Q", title="95% CI low"),
                 alt.Tooltip("delivered_ci_high:Q", title="95% CI high")])
    ci = alt.Chart(df).mark_rule(color=INK, strokeWidth=2).encode(
        y=alt.Y("category:N", sort=None), x="delivered_ci_low:Q", x2="delivered_ci_high:Q")
    req = alt.Chart(df).mark_tick(color=INK, thickness=3, size=22).encode(
        y=alt.Y("category:N", sort=None), x="requested_pct:Q",
        tooltip=[alt.Tooltip("requested_pct:Q", title="Requested %")])
    st.altair_chart((bars + ci + req).properties(height=36 * len(df) + 20), width="stretch")
    st.caption("Bar = delivered share · thin line = its 95% confidence interval · tick = what you requested.")


def _fault_outcomes(records: list[dict]) -> pd.DataFrame:
    rows = []
    for fk, (flabel, repairable) in FAULTS.items():
        hit = [r for r in records if fk in r["faults"]]
        rows.append({"fault": flabel, "repairable": "🔧 yes" if repairable else "⛔ no",
                     "patients injected": len(hit),
                     "still failing after validation": sum(not r["independent_pass_after"] for r in hit)})
    return pd.DataFrame(rows)


def _results(run_id: str):
    data = _load_run(run_id)
    rep, records = data["report"], data["records"]
    s = rep["scores"]

    st.markdown(f"#### Results · {rep['label']} · `{run_id}`")
    m = st.columns(5)
    m[0].metric("Patients generated", f"{rep['n']:,}")
    m[1].metric("Passed before validation", _pct(s["independent_before"]["rate"]))
    m[2].metric("Passed after validation", _pct(s["independent_after"]["rate"]),
                f"{100 * (s['independent_after']['rate'] - s['independent_before']['rate']):+.1f} pts")
    m[3].metric("Delivered patients", f"{rep['n_delivered']:,}")
    m[4].metric("Repairs made", f"{sum(c['repairs'] for c in rep['checks']):,}")
    if s["reject_h0"]:
        st.success(f"✅ **Accepted.** Independent pass rate {_pct(s['independent_after']['rate'])} "
                   f"(95% CI {_ci(s['independent_after']['ci'])}) is significantly above "
                   f"{_pct(s['h0_rate'])} (p = {s['p_value_vs_h0']:.2g}).")
    else:
        st.error(f"❌ **Not accepted.** Independent pass rate {_pct(s['independent_after']['rate'])} "
                 f"(95% CI {_ci(s['independent_after']['ci'])}) is not shown to be above {_pct(s['h0_rate'])} "
                 f"(p = {s['p_value_vs_h0']:.2g}). Reduce the ⛔ faults or check the failures in **Records**.")
    if not s["sample_size_adequate"]:
        st.warning(f"Sample size {s['n']} is below the {s['required_n']} needed for 80% power to show "
                   f"> {_pct(s['h0_rate'])} when the true rate is {_pct(s['target_rate'])}.")

    t_demo, t_checks, t_stats, t_rec, t_export = st.tabs(
        ["👥 Demographics", "✅ Checks & repairs", "📊 Statistics", "🔍 Records", "📦 Export"])

    with t_demo:
        st.caption("Did the delivered set keep the mix you asked for? Records that fail validation are left out, "
                   "which can shift the mix. A ⚠️ means the requested share falls outside the delivered 95% CI.")
        dims = rep["fidelity"]["dimensions"]
        dim = st.radio("Dimension", list(dims), horizontal=True, key=f"dim{run_id}")
        d = dims[dim]
        _fidelity_chart(d["rows"])
        table = pd.DataFrame(d["rows"]).drop(columns=["dimension"])
        table["ok"] = table.pop("requested_inside_ci").map({True: "✅", False: "⚠️"})
        st.dataframe(table, hide_index=True, width="stretch")
        p = d["chi2_p_delivered"]
        st.caption(f"Total variation distance (requested vs delivered): {d['tvd_delivered']:.3f} "
                   f"(0 = identical) · chi-square p = {p:.3f}" if p is not None else "")
        a = rep["age"]
        st.markdown(f"**Age**: requested {a['requested'][0]}–{a['requested'][1]} · generated "
                    f"{a['generated'][0]}–{a['generated'][1]} (mean {a['generated'][2]}) · delivered "
                    + (f"{a['delivered'][0]}–{a['delivered'][1]} (mean {a['delivered'][2]})" if a["delivered"] else "none"))

    with t_checks:
        st.caption("The validation script runs checks V01–V09, repairs only what has one certain fix, then "
                   "checks again. V09 is a warning: it is reported but does not fail a record.")
        st.dataframe(pd.DataFrame(rep["checks"]), hide_index=True, width="stretch")
        st.markdown("**What happened to each injected fault**")
        st.dataframe(_fault_outcomes(records), hide_index=True, width="stretch")

    with t_stats:
        st.markdown(
            f"""
| Score | Before validation | After validation |
|---|---|---|
| **Independent pass rate** (primary: audit + FHIR R4 schema) | {_pct(s['independent_before']['rate'])} ({_ci(s['independent_before']['ci'])}) | **{_pct(s['independent_after']['rate'])}** ({_ci(s['independent_after']['ci'])}) |
| Validation script's own pass rate | {_pct(s['script_before']['rate'])} | {_pct(s['script_after']['rate'])} |

- **Hypothesis**: H0: pass rate ≤ {_pct(s['h0_rate'])}; H1: above it (target {_pct(s['target_rate'])}). Exact one-sided binomial test: **p = {s['p_value_vs_h0']:.3g}** → {"reject H0 ✅" if s['reject_h0'] else "cannot reject H0 ❌"} at alpha = {s['alpha']}.
- **McNemar's exact test** (same records, before vs after): {s['mcnemar']['fixed']} records fixed, {s['mcnemar']['broken']} broken, one-sided p = {s['mcnemar']['p_one_sided']:.3g}.
- **Paired t-test** on the conformance score (share of checks passed): mean improvement {s['paired_t']['mean_improvement']:+.3f}, one-sided p = {s['paired_t']['p_one_sided']:.3g}.
- **Sample size**: {s['required_n']} records give 80% power to show > {_pct(s['h0_rate'])} when the true rate is {_pct(s['target_rate'])}. This run has {s['n']} ({"adequate ✅" if s['sample_size_adequate'] else "too few ⚠️"}).

The decision uses the **independent** score: a separately written audit that shares no code with the
validation script, plus official FHIR R4 schema validation. The script's own pass rate is partly
guaranteed, because it repairs what it checks.
"""
        )

    with t_rec:
        f1, f2 = st.columns([1, 2])
        which = f1.radio("Show", ["Failed", "Repaired", "Had injected faults", "All"], key=f"rf{run_id}")
        pick = {"Failed": lambda r: not r["independent_pass_after"], "Repaired": lambda r: bool(r["repairs"]),
                "Had injected faults": lambda r: bool(r["faults"]), "All": lambda r: True}[which]
        subset = [r for r in records if pick(r)]
        with f2:
            if not subset:
                st.info("No records in this group.")
            else:
                pid = st.selectbox(f"{len(subset)} patients", [r["patient_id"] for r in subset], key=f"rp{run_id}")
        if subset:
            r = next(x for x in records if x["patient_id"] == pid)
            st.markdown(f"**{pid}** · {r['race']} · {r['sex']} · {LANGUAGES.get(r['language']) or NOT_RECORDED} · "
                        f"{r['state']} · {CONTROL_TIERS[r['control']][0]} · "
                        + ("✅ delivered" if r["independent_pass_after"] else "❌ not delivered"))
            st.write("Injected faults: " + (", ".join(FAULTS[f][0] for f in r["faults"]) or "none"))
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Issues before**")
                st.dataframe(pd.DataFrame(r["issues_before"] or [{"message": "none"}]), hide_index=True)
            with c2:
                st.markdown("**Repairs**")
                st.dataframe(pd.DataFrame(r["repairs"] or [{"action": "none"}]), hide_index=True)
            with c3:
                st.markdown("**Still failing (independent audit)**")
                st.dataframe(pd.DataFrame({"failure": r["independent_failures_after"] or ["none"]}), hide_index=True)
            idx = int(pid.split("-")[1]) - 1
            before = _bundles(run_id, "baseline")[idx]
            after = _bundles(run_id, "validated")[idx]
            ja, jb = json.dumps(before, indent=2), json.dumps(after, indent=2)
            with st.expander("What the validation script changed (diff)"):
                diff = "\n".join(difflib.unified_diff(ja.splitlines(), jb.splitlines(), "before", "after",
                                                      lineterm="", n=2))
                st.code(diff or "(no changes)", language="diff")
            with st.expander("Full FHIR R4 Bundle after validation"):
                st.code(jb, language="json", height=420)

    with t_export:
        st.markdown(f"The **delivery set** is the {rep['n_delivered']:,} patients that passed the independent check. "
                    "Every Patient carries the HL7 `HTEST` (test health data) security label.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("⬇️ Download delivery set (.zip)", _zip(run_id), f"{run_id}.zip",
                               "application/zip", help="FHIR NDJSON (delivered + baseline), HL7 v2 messages, "
                                                       "spec, report and per-record results")
        with c2:
            if st.button("🎯 Use as the training dataset", key=f"use{run_id}",
                         help="Replaces the data in tabs 1 and 2 with these patients"):
                n = runner.use_as_training_data(run_id)
                st.cache_data.clear()
                st.success(f"{n:,} patients are now the training data. Open **1 · Data**, then **2 · Train**.")
        with c3:
            n_send = st.number_input("Patients to send over HL7", 1, max(1, rep["n_delivered"]),
                                     min(10, max(1, rep["n_delivered"])), key=f"ns{run_id}")
            if st.button("📤 Send to the HL7 listener", key=f"send{run_id}"):
                from src.config import load_config
                from src.hl7v2 import message as hl7
                from src.hl7v2.mllp import is_listening, send
                cfg = load_config()["hl7"]
                if not is_listening(cfg["host"], cfg["port"]):
                    st.error("The HL7 listener is not running. Start the API (`make run`).")
                else:
                    codes = {}
                    for _, msg in runner.hl7_messages(run_id)[: int(n_send)]:
                        ack = send(cfg["host"], cfg["port"], msg)
                        c = hl7.field(hl7.first(hl7.parse(ack), "MSA"), 1)
                        codes[c] = codes.get(c, 0) + 1
                    st.success("Sent. ACKs: " + ", ".join(f"{k} × {v}" for k, v in codes.items())
                               + ". See tab 5 for the inbox.")


# ============================================================================ tab

def render():
    st.subheader("🧬 Synthetic Test Data Creator")
    st.info(
        "Build a set of **test patients** with the demographics you need, then prove the set is fit for "
        "testing. Each run: **generates** US Core FHIR R4 patients matching your mix exactly → **injects** "
        "the data-quality faults you choose → runs the **validation script** (checks V01–V09, safe repairs "
        "only) → re-scores everything with an **independent audit** and the official FHIR schema → "
        "**tests** whether the pass rate beats your threshold → **exports** the patients that passed.",
        icon="💡")

    submitted, spec, schema = _inputs()
    if submitted:
        problems = spec.problems()
        blocking = [p for p in problems if "rescaled" not in p]
        for p in problems:
            (st.error if p in blocking else st.warning)(p)
        if not blocking:
            bar = st.progress(0.0, "Generating and validating…")
            report = runner.run(spec, schema_validation=schema, progress=lambda f: bar.progress(f))
            bar.empty()
            st.session_state["td_run"] = report["run_id"]

    runs = runner.list_runs()
    if not runs:
        st.caption("No runs yet. Set the demographics above and click **Generate & validate**.")
        return
    ids = [r["run_id"] for r in runs]
    current = st.session_state.get("td_run", ids[0])
    st.divider()
    chosen = st.selectbox("Run to show", ids, index=ids.index(current) if current in ids else 0,
                          format_func=lambda i: next(f"{r['created'][:16].replace('T', ' ')} · {r['label']} · "
                                                     f"{r['n']} patients · {_pct(r['independent_pass_rate'])} passed"
                                                     for r in runs if r["run_id"] == i))
    _results(chosen)
    with st.expander(f"All runs ({len(runs)})"):
        st.dataframe(pd.DataFrame(runs), hide_index=True, width="stretch")

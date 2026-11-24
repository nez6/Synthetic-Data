"""Helpers shared by the dashboard views."""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

SERIES = "#2a78d6"     # categorical slot 1
SERIES_2 = "#eb6834"   # categorical slot 2
INK = "#55554f"


def api_url() -> str:
    return st.session_state.get("api_url") or os.environ.get("API_URL", "http://localhost:8000")


def api_up() -> bool:
    try:
        return requests.get(f"{api_url()}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def pct(x, digits=1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def explain(text: str):
    st.info(text, icon="💡")


def hbar(df: pd.DataFrame, cat: str, val: str, title: str, fmt: str = ".1f", domain=None):
    x = alt.X(f"{val}:Q", title=title, scale=alt.Scale(domain=domain) if domain else alt.Undefined)
    chart = (alt.Chart(df).mark_bar(color=SERIES, cornerRadiusEnd=4, height=16)
             .encode(y=alt.Y(f"{cat}:N", sort="-x", title=None), x=x,
                     tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{val}:Q", format=fmt, title=title)])
             .properties(height=34 * len(df) + 20))
    st.altair_chart(chart, width="stretch")


def grouped_hbar(df: pd.DataFrame, cat: str, series: str, val: str, title: str, fmt=".1f", domain=None,
                 colors=(SERIES, SERIES_2)):
    """Two-series horizontal bars (legend + tooltip); `series` holds the series name per row."""
    names = list(dict.fromkeys(df[series]))
    x = alt.X(f"{val}:Q", title=title, scale=alt.Scale(domain=domain) if domain else alt.Undefined)
    chart = (alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=12)
             .encode(y=alt.Y(f"{cat}:N", title=None, sort=None), yOffset=alt.YOffset(f"{series}:N", sort=names),
                     x=x, color=alt.Color(f"{series}:N", sort=names, title=None,
                                          scale=alt.Scale(domain=names, range=list(colors[:len(names)])),
                                          legend=alt.Legend(orient="top")),
                     tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{series}:N"),
                              alt.Tooltip(f"{val}:Q", format=fmt, title=title)])
             .properties(height=max(120, 40 * df[cat].nunique() + 40)))
    st.altair_chart(chart, width="stretch")


@st.cache_data(show_spinner=False, max_entries=8)
def cached_run(run_id: str, _mtime: float):
    from src import runner
    return runner.load_run(run_id)


def load_run(run_id: str):
    from src import runner
    p = runner.runs_dir() / run_id / "report.json"
    return cached_run(run_id, p.stat().st_mtime)


def pick_run(key: str):
    from src import runner
    runs = runner.list_runs()
    if not runs:
        st.warning("No runs yet. Submit a request in the **📝 Request** tab.")
        return None
    ids = [r["run_id"] for r in runs]
    current = st.session_state.get("run_id", ids[0])
    return st.selectbox("Run", ids, index=ids.index(current) if current in ids else 0, key=key,
                        format_func=lambda i: next(f"{r['created'][:16].replace('T', ' ')} · {r['label']} · "
                                                   f"{r['tested']} tested · {r['delivered']} delivered"
                                                   for r in runs if r["run_id"] == i))

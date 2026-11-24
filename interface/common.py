"""Helpers shared by the dashboard views."""
import altair as alt
import pandas as pd
import streamlit as st

from interface import client
from interface.client import ApiError, api_url  # noqa: F401 — re-exported for the views

SERIES = "#2a78d6"     # categorical slot 1
SERIES_2 = "#eb6834"   # categorical slot 2
INK = "#55554f"


def api_up() -> bool:
    return client.health() is not None


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


def load_run(run_id: str) -> dict:
    """{report, request, records, ml} for one run, from GET /runs/{id}/full."""
    return client.run_full(api_url(), run_id)


def pick_run(key: str):
    try:
        runs = client.runs(api_url())
    except ApiError as e:
        st.error(str(e))
        return None
    if not runs:
        st.warning("No runs yet. Submit a request in the **📝 Request** tab (a fresh deployment trains one on "
                   "start-up; see the sidebar).")
        return None
    ids = [r["run_id"] for r in runs]
    current = st.session_state.get("run_id", ids[0])
    return st.selectbox("Run", ids, index=ids.index(current) if current in ids else 0, key=key,
                        format_func=lambda i: next(f"{r['created'][:16].replace('T', ' ')} · {r['label']} · "
                                                   f"{r['tested']} tested · {r['delivered']} delivered"
                                                   for r in runs if r["run_id"] == i))

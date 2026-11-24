"""HTTP client for the API. The dashboard holds no data of its own: every run, pool and inbox
row it shows is fetched from the (deployed) API through these helpers."""
import os

import requests
import streamlit as st

DEFAULT_API = "http://localhost:8000"


def _secret_api() -> str | None:
    try:
        return st.secrets.get("API_URL")
    except Exception:  # noqa: BLE001 — no secrets.toml locally
        return None


def api_url() -> str:
    return (st.session_state.get("api_url") or os.environ.get("API_URL") or _secret_api() or DEFAULT_API).rstrip("/")


class ApiError(RuntimeError):
    pass


def _call(method: str, path: str, timeout: float = 30, **kw) -> requests.Response:
    try:
        res = requests.request(method, f"{api_url()}{path}", timeout=timeout, **kw)
    except requests.RequestException as e:
        raise ApiError(f"API not reachable at {api_url()}: {e}") from e
    if res.status_code >= 400:
        try:
            detail = res.json().get("detail")
        except ValueError:
            detail = res.text[:300]
        raise ApiError(f"{method} {path} → HTTP {res.status_code}: {detail}")
    return res


def get(path: str, timeout: float = 30, **params):
    return _call("GET", path, timeout=timeout, params=params or None).json()


def get_bytes(path: str, timeout: float = 120, **params) -> bytes:
    return _call("GET", path, timeout=timeout, params=params or None).content


def post(path: str, body=None, timeout: float = 30):
    return _call("POST", path, timeout=timeout, json=body if body is not None else {}).json()


def delete(path: str, timeout: float = 30):
    return _call("DELETE", path, timeout=timeout).json()


def health() -> dict | None:
    try:
        return get("/health", timeout=5)
    except ApiError:
        return None


@st.cache_data(show_spinner=False, ttl=10)
def runs(base: str) -> list[dict]:
    return get("/runs")


@st.cache_data(show_spinner="Loading run…", max_entries=8)
def run_full(base: str, run_id: str) -> dict:
    return get(f"/runs/{run_id}/full", timeout=120)


@st.cache_data(show_spinner=False, max_entries=8)
def delivered_records(base: str, run_id: str) -> list[dict]:
    return get(f"/runs/{run_id}/delivered-records", timeout=120)


@st.cache_data(show_spinner="Downloading…", max_entries=8)
def dataset(base: str, run_id: str, fmt: str) -> bytes:
    return get_bytes(f"/runs/{run_id}/dataset", format=fmt)


@st.cache_data(show_spinner=False, max_entries=8)
def plot(base: str, run_id: str) -> bytes:
    return get_bytes(f"/runs/{run_id}/plot.png")

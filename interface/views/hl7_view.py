"""📨 HL7 v2 · send delivered patients as ORU^R01 over MLLP and inspect the listener's inbox."""
import json
import uuid

import pandas as pd
import streamlit as st

from interface import client
from interface.common import ApiError, api_url, explain, pick_run
from src.hl7v2 import message as hl7


def _records(run_id):
    return client.delivered_records(api_url(), run_id)


def _send(text: str) -> tuple[str, str]:
    """Relay through POST /hl7/send to the MLLP listener inside the API; returns (ack code, ack text)."""
    res = client.post("/hl7/send", {"message": text})
    return res["code"], res["ack"]


def render():
    h = client.health() or {}
    hl7cfg = h.get("hl7_listener") or {}
    port = hl7cfg.get("port", 2575)
    st.subheader("📨 Deliver test patients over HL7 v2")
    explain(f"Many systems still receive lab results as **HL7 v2 ORU^R01** over **MLLP** (TCP). The API runs a "
            f"MLLP listener on port {port} inside the API; this tab relays messages to it through "
            "`POST /hl7/send` (MLLP is raw TCP, which Cloud Run does not expose). Each message is parsed, mapped to FHIR R4, given a record-level quality "
            "score, scored by the model, stored in the inbox and acknowledged (AA accepted, AE error, AR rejected). "
            "Point your own HL7 interface engine at the same messages to test it.")
    listening = bool(hl7cfg.get("listening"))
    (st.success if listening else st.error)(f"Listener {'running' if listening else 'not running'} on port {port} "
                                            f"of {api_url()}" + ("" if listening else ". Start the API (`make run`)."))
    run_id = pick_run("hl7_run")
    if run_id:
        recs = _records(run_id)
        if recs:
            pid = st.selectbox("Delivered patient", [r["patient_id"] for r in recs], key="hl7_pid")
            rec = next(r for r in recs if r["patient_id"] == pid)
            ctrl = st.session_state.setdefault("hl7_ctrl", uuid.uuid4().hex[:10].upper())
            text = st.text_area("ORU^R01 (editable, one segment per line)",
                                hl7.build_oru(rec, ctrl).replace("\r", "\n"), height=200, key=f"hl7txt{pid}{ctrl}")
            with st.expander("What do these fields mean?"):
                st.dataframe(pd.DataFrame(hl7.describe(hl7.parse(text))), hide_index=True, width="stretch")
            c1, c2 = st.columns(2)
            if c1.button("📤 Send over MLLP", type="primary", disabled=not listening):
                try:
                    code, ack = _send(text)
                except ApiError as e:
                    st.error(str(e))
                    return
                msa = hl7.first(hl7.parse(ack), "MSA")
                (st.success if code == "AA" else st.error)(f"ACK **{code}** ({hl7.ACK_CODES.get(code, '?')}): "
                                                           f"{hl7.field(msa, 3)}")
                st.code(ack.replace("\r", "\n"), language="text")
                st.session_state["hl7_ctrl"] = uuid.uuid4().hex[:10].upper()
            n = c2.number_input("Send the first N delivered patients", 1, len(recs), min(10, len(recs)))
            if c2.button("📤 Send batch", disabled=not listening):
                codes = {}
                for r in recs[: int(n)]:
                    c, _ = _send(hl7.build_oru(r, uuid.uuid4().hex[:10].upper()))
                    codes[c] = codes.get(c, 0) + 1
                st.success("ACKs: " + ", ".join(f"{k} × {v}" for k, v in codes.items()))

    st.markdown("#### Inbox")
    try:
        rows = client.get("/hl7/inbox", limit=200)
    except ApiError as e:
        st.error(str(e))
        return
    if not rows:
        st.caption("No messages yet.")
        return
    st.dataframe(pd.DataFrame([{
        "id": r["id"], "received (UTC)": r["received_at"], "control id": r["control_id"], "patient": r["patient_id"],
        "ACK": r["ack_code"], "DQI": r["dqi"], "issues": ", ".join(r["issues"] or []),
        "P(passes validation)": ((r["prediction"] or {}).get("model") or {}).get("probability_pass"),
    } for r in rows]), hide_index=True, width="stretch", height=240)
    pick = st.selectbox("Inspect message", [r["id"] for r in rows])
    r = next(x for x in rows if x["id"] == pick)
    t1, t2, t3 = st.tabs(["Raw HL7", "→ FHIR R4", "→ quality and model"])
    t1.code(r["raw"].replace("\r", "\n"), language="text")
    t2.code(json.dumps(r["fhir_bundle"], indent=2) if r["fhir_bundle"] else "(rejected)", language="json")
    t3.json(r["prediction"] or {"note": r["ack_text"]})
    if st.button("🗑️ Clear inbox"):
        client.delete("/hl7/inbox")
        st.rerun()

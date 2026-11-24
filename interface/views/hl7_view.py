"""📨 HL7 v2 · send delivered patients as ORU^R01 over MLLP and inspect the listener's inbox."""
import json
import uuid

import pandas as pd
import streamlit as st

from interface.common import explain, pick_run
from src import runner
from src.config import load_config
from src.fhir.parser import bundle_to_record
from src.hl7v2 import inbox
from src.hl7v2 import message as hl7
from src.hl7v2.mllp import is_listening, send


@st.cache_data(show_spinner=False, max_entries=4)
def _records(run_id):
    return [bundle_to_record(b) for b in runner.load_delivery(run_id)]


def render():
    cfg = load_config()["hl7"]
    host, port = cfg["host"], cfg["port"]
    st.subheader("📨 Deliver test patients over HL7 v2")
    explain(f"Many systems still receive lab results as **HL7 v2 ORU^R01** over **MLLP** (TCP). The API runs a "
            f"listener on {host}:{port}: each message is parsed, mapped to FHIR R4, given a record-level quality "
            "score, scored by the model, stored in the inbox and acknowledged (AA accepted, AE error, AR rejected). "
            "Point your own HL7 interface engine at the same messages to test it.")
    listening = is_listening(host, port)
    (st.success if listening else st.error)(f"Listener {'running' if listening else 'not running'} on {host}:{port}"
                                            + ("" if listening else ". Start the API (`make run`)."))
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
                ack = send(host, port, hl7.normalize(text))
                msa = hl7.first(hl7.parse(ack), "MSA")
                code = hl7.field(msa, 1)
                (st.success if code == "AA" else st.error)(f"ACK **{code}** ({hl7.ACK_CODES.get(code, '?')}): "
                                                           f"{hl7.field(msa, 3)}")
                st.code(ack.replace("\r", "\n"), language="text")
                st.session_state["hl7_ctrl"] = uuid.uuid4().hex[:10].upper()
            n = c2.number_input("Send the first N delivered patients", 1, len(recs), min(10, len(recs)))
            if c2.button("📤 Send batch", disabled=not listening):
                codes = {}
                for r in recs[: int(n)]:
                    a = send(host, port, hl7.build_oru(r, uuid.uuid4().hex[:10].upper()))
                    c = hl7.field(hl7.first(hl7.parse(a), "MSA"), 1)
                    codes[c] = codes.get(c, 0) + 1
                st.success("ACKs: " + ", ".join(f"{k} × {v}" for k, v in codes.items()))

    st.markdown("#### Inbox")
    rows = inbox.recent(200)
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
        inbox.clear()
        st.rerun()

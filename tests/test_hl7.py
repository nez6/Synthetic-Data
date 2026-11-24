from src.hl7v2 import message as m
from src.hl7v2.listener import handle_message
from src.hl7v2.mllp import END, START, frame, send, start_in_thread

REC = dict(patient_id="pt-9", gender="male", birth_date="1980-07-15", effective="2026-04-01T09:00:00+00:00",
           race_ethnicity="Asian", language="zh", state="NY", smoking="current",
           bmi=27.4, systolic_bp=131.0, hba1c=5.6, egfr=92.0)


def test_oru_round_trip():
    segs = m.parse(m.build_oru(REC, "CTRL1"))
    assert [s[0] for s in segs] == ["MSH", "PID", "OBR", "OBX", "OBX", "OBX", "OBX", "OBX"]
    assert m.message_type(segs) == "ORU^R01" and m.field(m.first(segs, "MSH"), 10) == "CTRL1"
    out = m.oru_to_record(segs)
    for k in ("race_ethnicity", "bmi", "systolic_bp", "hba1c", "egfr", "smoking", "language", "state"):
        assert out[k] == REC[k], k
    assert out["age"] == 45


def test_hispanic_is_carried_in_ethnicity_field():
    segs = m.parse(m.build_oru({**REC, "race_ethnicity": "Hispanic/Latino"}, "C"))
    pid = m.first(segs, "PID")
    assert m.field(pid, 10) == "" and m.field(pid, 22, 1) == "2135-2"
    assert m.oru_to_record(segs)["race_ethnicity"] == "Hispanic/Latino"


def test_missing_values_are_left_out():
    segs = m.parse(m.build_oru({**REC, "bmi": None, "smoking": None, "race_ethnicity": None}, "C"))
    assert sum(s[0] == "OBX" for s in segs) == 3
    out = m.oru_to_record(segs)
    assert out["bmi"] is None and out["race_ethnicity"] is None


def test_mllp_framing():
    framed = frame("MSH|x")
    assert framed.startswith(START) and framed.endswith(END)


def test_listener_end_to_end(tmp_path, monkeypatch):
    from src.hl7v2 import inbox
    monkeypatch.setattr(inbox, "_connect", _tmp_db(tmp_path))
    server = start_in_thread("127.0.0.1", 0, handle_message)
    port = server.server_address[1]
    try:
        msa = m.first(m.parse(send("127.0.0.1", port, m.build_oru({**REC, "hba1c": 99.0}, "E2E-1"))), "MSA")
        assert m.field(msa, 1) == "AA" and m.field(msa, 2) == "E2E-1"
        stored = inbox.recent(5)[0]
        assert stored["fhir_bundle"]["resourceType"] == "Bundle"
        assert "implausible_hba1c" in stored["issues"]
        adt = m.build_oru(REC, "E2E-2").replace("ORU^R01^ORU_R01", "ADT^A08^ADT_A01")
        assert m.field(m.first(m.parse(send("127.0.0.1", port, adt)), "MSA"), 1) == "AR"
    finally:
        server.shutdown()


def _tmp_db(tmp_path):
    import sqlite3

    def connect():
        conn = sqlite3.connect(tmp_path / "inbox.db")
        conn.row_factory = sqlite3.Row
        from src.hl7v2.inbox import _SCHEMA
        conn.execute(_SCHEMA)
        return conn
    return connect

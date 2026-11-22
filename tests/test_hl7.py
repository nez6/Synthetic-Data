from src.hl7v2 import message as m
from src.hl7v2.mllp import END, START, frame

REC = dict(patient_id="pt-9", gender="male", birth_date="1980-07-15",
           effective="2026-04-01T09:00:00+00:00", race_demographic="Asian", bmi=27.4, baseline_hba1c=8.2)


def test_oru_round_trip():
    raw = m.build_oru(REC, "CTRL1")
    segs = m.parse(raw)
    assert [s[0] for s in segs] == ["MSH", "PID", "OBR", "OBX", "OBX"]
    assert m.message_type(segs) == "ORU^R01"
    assert m.field(m.first(segs, "MSH"), 10) == "CTRL1"
    out = m.oru_to_record(segs)
    assert out["race_demographic"] == "Asian"
    assert out["bmi"] == 27.4 and out["baseline_hba1c"] == 8.2
    assert out["age"] == 45


def test_hispanic_is_carried_in_ethnicity_field():
    segs = m.parse(m.build_oru({**REC, "race_demographic": "Hispanic/Latino"}, "C"))
    pid = m.first(segs, "PID")
    assert m.field(pid, 10) == "" and m.field(pid, 22, 1) == "2135-2"
    assert m.oru_to_record(segs)["race_demographic"] == "Hispanic/Latino"


def test_missing_values_are_left_out():
    segs = m.parse(m.build_oru({**REC, "race_demographic": None, "bmi": None}, "C"))
    assert sum(s[0] == "OBX" for s in segs) == 1
    out = m.oru_to_record(segs)
    assert out["race_demographic"] is None and out["bmi"] is None


def test_parse_accepts_newline_separated_text():
    raw = m.build_oru(REC, "C").replace("\r", "\n")
    assert m.oru_to_record(m.parse(raw))["baseline_hba1c"] == 8.2


def test_ack_structure():
    ack = m.parse(m.build_ack("CTRL1", "AA", "ok"))
    msa = m.first(ack, "MSA")
    assert m.field(msa, 1) == "AA" and m.field(msa, 2) == "CTRL1"


def test_mllp_framing():
    framed = frame("MSH|x")
    assert framed.startswith(START) and framed.endswith(END)

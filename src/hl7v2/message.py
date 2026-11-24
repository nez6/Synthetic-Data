"""Build and parse HL7 v2.5.1 messages (pipe-delimited) without external dependencies.

Segment layout used here (ORU^R01 = "unsolicited observation result"):
    MSH  message header        (who sent it, message type, control id, version)
    PID  patient identification (MRN, name, birth date, sex, race, address, ethnicity)
    OBR  observation request    (the order these results belong to)
    OBX  observation result     (one per vital/lab, plus smoking status as a coded result)
"""
from datetime import datetime, timezone

from src import terminology as t
from src.fhir.parser import age_on

SEGMENT_SEP = "\r"
VERSION = "2.5.1"

FIELD_NAMES = {
    "MSH": {3: "Sending application", 4: "Sending facility", 5: "Receiving application",
            6: "Receiving facility", 7: "Message date/time", 9: "Message type",
            10: "Message control ID", 11: "Processing ID", 12: "HL7 version"},
    "PID": {1: "Set ID", 3: "Patient identifier (MRN)", 5: "Patient name", 7: "Date of birth",
            8: "Administrative sex", 10: "Race", 11: "Address", 15: "Primary language", 22: "Ethnic group"},
    "OBR": {1: "Set ID", 4: "Universal service ID (test ordered)", 7: "Observation date/time"},
    "OBX": {1: "Set ID", 2: "Value type", 3: "Observation identifier (LOINC)", 5: "Observation value",
            6: "Units", 7: "Reference range", 8: "Abnormal flag", 11: "Result status",
            14: "Observation date/time"},
    "MSA": {1: "Acknowledgment code", 2: "Message control ID being acknowledged", 3: "Text message"},
}
ACK_CODES = {"AA": "Application Accept", "AE": "Application Error", "AR": "Application Reject"}


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and not v.strip())


# --------------------------------------------------------------------------- build

def build_oru(rec: dict, control_id: str, sending_app: str = "LAB_SYS", sending_facility: str = "CLINIC") -> str:
    """Build an ORU^R01 for a flat record (keys as produced by src/fhir/parser.py)."""
    effective = datetime.fromisoformat((rec.get("effective") or datetime.now(timezone.utc).isoformat())
                                       .replace("Z", "+00:00"))
    obs_ts = _ts(effective)

    race = ethnicity = ""
    group = rec.get("race_ethnicity")
    if group == t.HISPANIC_GROUP:
        ethnicity = f"{t.HISPANIC[0]}^{t.HISPANIC[1]}^CDCREC"
    elif group in t.RACE_CODES:
        code, display = t.RACE_CODES[group]
        race = f"{code}^{display}^CDCREC"
        ethnicity = f"{t.NOT_HISPANIC[0]}^{t.NOT_HISPANIC[1]}^CDCREC"

    sex = {"female": "F", "male": "M"}.get(rec.get("gender") or rec.get("sex"), "U")
    dob = str(rec["birth_date"]).replace("-", "")[:8] if not _missing(rec.get("birth_date")) else ""
    address = f"^^^{rec['state']}^^US" if not _missing(rec.get("state")) else ""
    lang = "" if _missing(rec.get("language")) else f"{rec['language']}^{t.LANGUAGES.get(rec['language'], '')}^ISO639"

    pid = ["PID", "1", "", f"{rec['patient_id']}^^^{sending_facility}^MR", "",
           f"Synthetic^Patient {rec['patient_id']}", "", dob, sex, "", race, address, "", "", "", lang] \
        + [""] * 6 + [ethnicity]

    segments = [
        ["MSH", "^~\\&", sending_app, sending_facility, "TDC_LISTENER", "TEST_DATA_CREATOR",
         _ts(datetime.now(timezone.utc)), "", "ORU^R01^ORU_R01", control_id, "P", VERSION],
        pid,
        ["OBR", "1", "", "", "24356-8^Visit results^LN", "", "", obs_ts],
    ]
    set_id = 1
    for key, lab in t.LABS.items():
        if _missing(rec.get(key)):
            continue
        segments.append(["OBX", str(set_id), "NM", f"{lab['code']}^{lab['display'][:60]}^LN", "",
                         f"{float(rec[key]):g}", f"{lab['unit']}^^UCUM", "", "", "", "", "F", "", "", obs_ts])
        set_id += 1
    if rec.get("smoking") in t.SMOKING:
        code, display = t.SMOKING[rec["smoking"]]
        segments.append(["OBX", str(set_id), "CWE", f"{t.SMOKING_LOINC[0]}^{t.SMOKING_LOINC[1]}^LN", "",
                         f"{code}^{display}^SCT", "", "", "", "", "", "F", "", "", obs_ts])
    return SEGMENT_SEP.join("|".join(seg) for seg in segments)


def build_ack(control_id: str, code: str, text: str = "", original_msh=None) -> str:
    orig = original_msh or []
    recv_app = orig[3] if len(orig) > 3 else ""
    recv_fac = orig[4] if len(orig) > 4 else ""
    msh = ["MSH", "^~\\&", "TDC_LISTENER", "TEST_DATA_CREATOR", recv_app, recv_fac,
           _ts(datetime.now(timezone.utc)), "", "ACK^R01^ACK", f"ACK-{control_id}", "P", VERSION]
    msa = ["MSA", code, control_id, text.replace("|", "/")]
    return SEGMENT_SEP.join("|".join(s) for s in (msh, msa))


# --------------------------------------------------------------------------- parse

def normalize(raw: str) -> str:
    """Accept \\r, \\n or \\r\\n segment separators (pasted text usually has \\n)."""
    return SEGMENT_SEP.join(line.strip() for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                            if line.strip())


def parse(raw: str) -> list[list[str]]:
    """Segments as lists of fields, indexed so that seg[n] == HL7 field n (MSH-1 is inserted)."""
    segments = []
    for line in normalize(raw).split(SEGMENT_SEP):
        fields = line.split("|")
        if fields[0] == "MSH":
            fields = ["MSH", "|"] + fields[1:]
        segments.append(fields)
    return segments


def field(seg: list[str], n: int, component: int = None) -> str:
    value = seg[n] if seg and len(seg) > n else ""
    if component is not None:
        parts = value.split("^")
        return parts[component - 1] if len(parts) >= component else ""
    return value


def first(segments, name):
    return next((s for s in segments if s[0] == name), None)


def message_type(segments) -> str:
    msh = first(segments, "MSH")
    return f"{field(msh, 9, 1)}^{field(msh, 9, 2)}" if msh else ""


def _hl7_ts_to_iso(ts: str):
    fmt = {8: "%Y%m%d", 12: "%Y%m%d%H%M", 14: "%Y%m%d%H%M%S"}.get(len(ts[:14])) if ts else None
    if not fmt:
        return None
    return datetime.strptime(ts[:14], fmt).replace(tzinfo=timezone.utc).isoformat()


def oru_to_record(segments) -> dict:
    """Map an ORU^R01 to the flat record shape used by the FHIR builder and the quality score."""
    pid = first(segments, "PID") or ["PID"]
    race_code, eth_code = field(pid, 10, 1), field(pid, 22, 1)
    group = t.HISPANIC_GROUP if eth_code == t.HISPANIC[0] else t.RACE_BY_CODE.get(race_code)
    dob = field(pid, 7)
    birth_date = f"{dob[:4]}-{dob[4:6]}-{dob[6:8]}" if len(dob) >= 8 else None
    lang = field(pid, 15, 1) or None

    rec = {k: None for k in t.LABS}
    rec["smoking"] = None
    effective = None
    for seg in segments:
        if seg[0] != "OBX":
            continue
        code = field(seg, 3, 1)
        effective = effective or _hl7_ts_to_iso(field(seg, 14))
        if code == t.SMOKING_LOINC[0]:
            rec["smoking"] = t.SMOKING_BY_CODE.get(field(seg, 5, 1))
        elif code in t.LAB_BY_CODE:
            try:
                rec[t.LAB_BY_CODE[code]] = float(field(seg, 5))
            except ValueError:
                pass
    obr = first(segments, "OBR")
    effective = effective or (obr and _hl7_ts_to_iso(field(obr, 7))) or datetime.now(timezone.utc).isoformat()
    sex = {"F": "female", "M": "male"}.get(field(pid, 8))
    rec.update({
        "patient_id": field(pid, 3, 1) or "unknown",
        "gender": sex, "sex": sex,
        "birth_date": birth_date,
        "effective": effective,
        "age": age_on(birth_date, datetime.fromisoformat(effective).date()) if birth_date else None,
        "race_ethnicity": group,
        "language": lang if lang in t.LANGUAGES else ("other" if lang else None),
        "state": field(pid, 11, 4) or None,
        "conditions": [],
    })
    return rec


def describe(segments) -> list[dict]:
    """Field-by-field breakdown for display."""
    rows = []
    for seg in segments:
        names = FIELD_NAMES.get(seg[0], {})
        for n in sorted(names):
            v = field(seg, n)
            if v:
                rows.append({"segment": seg[0], "field": f"{seg[0]}-{n}", "name": names[n], "value": v})
    return rows

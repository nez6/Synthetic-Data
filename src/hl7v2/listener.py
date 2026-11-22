"""HL7 v2 MLLP listener.

For every inbound ORU^R01:  parse → map to FHIR R4 Bundle → Data Quality Index →
model prediction + clinician routing → store in inbox → reply with an HL7 ACK.

Runs automatically inside the API process (see src/app.py), or standalone:
    python -m src.hl7v2.listener
"""
import time

from src.config import load_config
from src.fhir.builder import build_bundle
from src.hl7v2 import inbox, message as m
from src.hl7v2.mllp import MLLPServer
from src.inference import ModelNotReady, predict


def handle_message(raw: str) -> str:
    segments = m.parse(raw)
    msh = m.first(segments, "MSH")
    if not msh:
        return m.build_ack("", "AR", "No MSH segment: not an HL7 v2 message")
    control_id = m.field(msh, 10)
    mtype = m.message_type(segments)

    if mtype != "ORU^R01":
        inbox.save(control_id=control_id, message_type=mtype, ack_code="AR",
                   ack_text=f"Unsupported message type {mtype}", raw=raw)
        return m.build_ack(control_id, "AR", f"Unsupported message type {mtype}; send ORU^R01", msh)

    record = m.oru_to_record(segments)
    bundle = build_bundle(record, "R4")
    try:
        result = predict(record)
        code, text = "AA", f"Processed. DQI={result['data_quality_index']}"
    except ModelNotReady as e:
        result, code, text = None, "AE", str(e)

    inbox.save(control_id=control_id, message_type=mtype, patient_id=record["patient_id"],
               ack_code=code, ack_text=text,
               dqi=result["data_quality_index"] if result else None,
               issues=result["data_quality_issues"] if result else None,
               prediction=result, fhir_bundle=bundle, raw=raw)
    return m.build_ack(control_id, code, text, msh)


def main():
    cfg = load_config()["hl7"]
    server = MLLPServer(cfg["host"], cfg["port"], handle_message)
    print(f"HL7 v2 MLLP listener on {cfg['host']}:{cfg['port']} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()

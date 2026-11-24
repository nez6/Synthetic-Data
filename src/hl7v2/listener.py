"""HL7 v2 MLLP listener.

For every inbound ORU^R01:  parse → map to a FHIR R4 Bundle → record-level Data Quality Index →
(if a model has been trained) probability the record passes validation → store in the inbox →
reply with an HL7 ACK.

Runs inside the API process (config: hl7.start_with_api), or standalone:
    python -m src.hl7v2.listener
"""
from src.config import load_config
from src.fhir.builder import build_bundle
from src.hl7v2 import inbox, message as m
from src.hl7v2.mllp import MLLPServer
from src.quality import score_record


def _predict(record):
    try:
        from src.inference import ModelNotReady, predict_pass
        return predict_pass([record])["predictions"][0]
    except (ImportError, ModelNotReady):
        return None


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
    quality = score_record(record)
    prediction = _predict(record)
    text = f"Processed. DQI={quality['score']}"
    inbox.save(control_id=control_id, message_type=mtype, patient_id=record["patient_id"], ack_code="AA",
               ack_text=text, dqi=quality["score"], issues=quality["issues"],
               prediction={"quality": quality, "model": prediction, "record": record},
               fhir_bundle=bundle, raw=raw)
    return m.build_ack(control_id, "AA", text, msh)


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

"""Command-line HL7 sender.

    python -m src.hl7v2.send --count 10          # 10 delivered patients from the latest run
    python -m src.hl7v2.send --file message.hl7
"""
import argparse
import uuid

from src.config import load_config
from src.fhir.parser import bundle_to_record
from src.hl7v2 import message as m
from src.hl7v2.mllp import send


def delivered_messages(count: int, run_id: str | None = None) -> list[str]:
    from src import runner
    runs = runner.list_runs()
    if not runs:
        raise SystemExit("No run yet: submit a test-data request first.")
    bundles = runner.load_delivery(run_id or runs[0]["run_id"])[:count]
    return [m.build_oru(bundle_to_record(b), control_id=uuid.uuid4().hex[:10].upper()) for b in bundles]


def main():
    cfg = load_config()["hl7"]
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=cfg["port"])
    p.add_argument("--file")
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--run")
    args = p.parse_args()
    messages = [open(args.file).read()] if args.file else delivered_messages(args.count, args.run)
    for msg in messages:
        ack = send(args.host, args.port, m.normalize(msg))
        msa = m.first(m.parse(ack), "MSA")
        print(f"{m.field(msa, 1)}  {m.field(msa, 2)}  {m.field(msa, 3)}")


if __name__ == "__main__":
    main()

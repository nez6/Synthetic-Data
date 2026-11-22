"""Command-line HL7 sender.

    python -m src.hl7v2.send                    # send 1 random synthetic patient
    python -m src.hl7v2.send --count 10
    python -m src.hl7v2.send --file samples/hl7/oru_r01.hl7
"""
import argparse
import random
import uuid

from src.config import load_config
from src.data_ingestion import generate_records
from src.hl7v2 import message as m
from src.hl7v2.mllp import send


def random_message() -> str:
    rec = generate_records(1, seed=random.randint(0, 10**9), id_prefix="hl7")[0]
    return m.build_oru(rec, control_id=uuid.uuid4().hex[:10].upper())


def main():
    cfg = load_config()["hl7"]
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=cfg["port"])
    p.add_argument("--file")
    p.add_argument("--count", type=int, default=1)
    args = p.parse_args()

    messages = [open(args.file).read()] if args.file else [random_message() for _ in range(args.count)]
    for msg in messages:
        ack = send(args.host, args.port, m.normalize(msg))
        msa = m.first(m.parse(ack), "MSA")
        print(f"{m.field(msa, 1)}  {m.field(msa, 2)}  {m.field(msa, 3)}")


if __name__ == "__main__":
    main()

"""Run the Synthetic Test Data Creator from the command line.

    python -m src.testdata.cli --preset "Older adults (65+)" --n 400
    python -m src.testdata.cli --spec my_cohort.json --use-as-training
"""
import argparse
import json

from src.testdata import runner
from src.testdata.spec import PRESETS, CohortSpec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=list(PRESETS), default="Balanced clinic population")
    p.add_argument("--spec", help="JSON file with CohortSpec fields (overrides the preset)")
    p.add_argument("--n", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--no-schema", action="store_true", help="skip official FHIR R4 schema validation")
    p.add_argument("--use-as-training", action="store_true")
    args = p.parse_args()

    spec = CohortSpec.from_dict(json.load(open(args.spec))) if args.spec else PRESETS[args.preset]
    if args.n:
        spec.n = args.n
    if args.seed is not None:
        spec.seed = args.seed
    for problem in spec.problems():
        print("note:", problem)

    report = runner.run(spec, schema_validation=not args.no_schema)
    s = report["scores"]
    print(f"run {report['run_id']}: {report['n']} generated, {report['n_delivered']} delivered")
    print(f"independent pass rate {s['independent_before']['rate']:.1%} → {s['independent_after']['rate']:.1%} "
          f"(95% CI {s['independent_after']['ci'][0]:.1%}–{s['independent_after']['ci'][1]:.1%})")
    print(f"H0 ≤ {s['h0_rate']:.0%}: p = {s['p_value_vs_h0']:.3g} → {'ACCEPT' if s['reject_h0'] else 'NOT ACCEPTED'}")
    print(f"files in {runner.runs_dir() / report['run_id']}")
    if args.use_as_training:
        print(f"{runner.use_as_training_data(report['run_id'])} patients set as the training dataset")


if __name__ == "__main__":
    main()

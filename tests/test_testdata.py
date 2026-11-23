import ast
import copy
from collections import Counter
from pathlib import Path

import pytest

from src.fhir.parser import bundle_to_record
from src.fhir.validate import validate_bundle
from src.testdata import runner
from src.testdata.audit import audit_bundle
from src.testdata.generator import allocate, generate
from src.testdata.spec import FAULTS, PRESETS, CohortSpec
from src.testdata.stats import mcnemar_exact, one_sided_binomial, required_n, wilson_ci
from src.testdata.validation import run_validation


def _clean(n=60, **kw):
    spec = copy.deepcopy(PRESETS["Clean reference cohort (no faults)"])
    spec.n = n
    for k, v in kw.items():
        setattr(spec, k, v)
    return spec


def test_allocate_is_exact():
    import random
    out = allocate(500, {"a": 18, "b": 40, "c": 42}, random.Random(1))
    assert Counter(out) == {"a": 90, "b": 200, "c": 210}


def test_requested_demographics_are_hit_exactly():
    spec = CohortSpec(n=500, faults={k: 0 for k in FAULTS})
    _, truth = generate(spec)
    races = Counter(t["race"] for t in truth)
    assert races["Black/African American"] == 90 and races["White"] == 200
    assert Counter(t["sex"] for t in truth)["female"] == 260
    assert set(t["state"] for t in truth) == set(spec.states)


def test_ages_inside_requested_range():
    spec = _clean(min_age=65, max_age=70)
    bundles, _ = generate(spec)
    ages = [bundle_to_record(b)["age"] for b in bundles]
    # age at the latest visit can be one less than age on the as-of date
    assert min(ages) >= 64 and max(ages) <= 70


def test_clean_cohort_passes_everything_and_is_valid_r4():
    spec = _clean()
    bundles, _ = generate(spec)
    for b in bundles[:10]:
        assert validate_bundle(b, "R4")["valid"]
    assert all(run_validation(b, spec).passed_before for b in bundles)


def test_same_seed_same_patients():
    assert generate(_clean(n=20))[0] == generate(_clean(n=20))[0]


@pytest.mark.parametrize("fault", list(FAULTS))
def test_each_fault_is_caught_and_repaired_only_if_safe(fault):
    spec = _clean(n=40, faults={k: (50 if k == fault else 0) for k in FAULTS})
    bundles, truth = generate(spec)
    rules = runner._audit_rules(spec)
    from datetime import date
    as_of = date.fromisoformat(spec.as_of)
    hit = [i for i, t in enumerate(truth) if fault in t["faults"]]
    assert len(hit) == 20
    for i in hit:
        res = run_validation(bundles[i], spec)
        assert not res.passed_before, fault
        assert audit_bundle(bundles[i], rules, as_of), f"audit missed {fault}"
        repairable = FAULTS[fault][1]
        assert res.passed == repairable, fault
        assert (not audit_bundle(res.bundle, rules, as_of)) == repairable, fault


def test_validation_never_modifies_input():
    spec = CohortSpec(n=30)
    bundles, _ = generate(spec)
    snapshot = copy.deepcopy(bundles)
    for b in bundles:
        run_validation(b, spec)
    assert bundles == snapshot


def test_record_without_synthetic_identifier_is_left_untouched():
    spec = _clean(n=1)
    b = generate(spec)[0][0]
    b["entry"][0]["resource"]["identifier"] = [{"system": "https://hospital.example/mrn", "value": "123"}]
    b["entry"][0]["resource"]["meta"].pop("security")
    res = run_validation(b, spec)
    assert res.repairs == [] and not res.passed


def test_audit_shares_no_code_with_validation():
    tree = ast.parse(Path("src/testdata/audit.py").read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and ("validation" in m or "generator" in m or m.startswith("src")) for m in imported), imported


def test_stats():
    lo, hi = wilson_ci(90, 100)
    assert lo < 0.9 < hi
    assert one_sided_binomial(95, 100, 0.85) < 0.01
    assert one_sided_binomial(85, 100, 0.85) > 0.4
    assert mcnemar_exact([False, False, True], [True, True, True]) == {"fixed": 2, "broken": 0,
                                                                     "p_one_sided": 0.25}
    assert 250 < required_n(0.85, 0.90) < 320


def test_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "runs_dir", lambda: tmp_path)
    spec = CohortSpec(n=300)
    report = runner.run(spec)
    s = report["scores"]
    assert s["independent_after"]["rate"] > s["independent_before"]["rate"]
    assert s["mcnemar"]["broken"] == 0
    assert report["n_delivered"] == s["independent_after"]["passed"]
    race = report["fidelity"]["dimensions"]["Race / ethnicity"]["rows"]
    assert all(r["generated_pct"] == r["requested_pct"] for r in race)
    run_id = report["run_id"]
    assert runner.list_runs()[0]["run_id"] == run_id
    msgs = runner.hl7_messages(run_id)
    assert len(msgs) == report["n_delivered"] and msgs[0][1].startswith("MSH|")
    assert len(runner.zip_run(run_id)) > 1000
    for b in runner.load_bundles(run_id):
        assert any(s.get("code") == "HTEST" for s in b["entry"][0]["resource"]["meta"]["security"])

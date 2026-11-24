import random

from src.quality import score_record
from src.request import apply_profile, default_request, problems
from src.selection import eligible, inject, plan_faults, select_cohort


def _row(i, age, race, groups=(), alive=True):
    return {"patient_id": f"p{i:04d}", "age": age, "alive": alive, "sex": "female", "race_ethnicity": race,
            "condition_groups": list(groups), "file": f"patients/p{i}.json.gz", "counts": {}}


def test_default_request_matches_the_slide():
    req = default_request()
    a, b = req["cohorts"]
    assert (a["share"], b["share"]) == (0.75, 0.25) and req["delivery_size"] == 100
    assert a["targets"]["Encounter"] == {"min": 2, "max": 5}
    assert b["targets"]["Observation"] == {"min": 500, "max": None}
    assert b["min_history_years"] == 3 and b["min_condition_groups"] == 2
    assert problems(req) == []


def test_problems_are_reported():
    req = default_request()
    req["cohorts"][0]["share"] = 0.5
    req["split"]["train"] = 0.9
    assert len(problems(req)) == 2


def test_eligibility_follows_the_reference_rules():
    a, b = default_request()["cohorts"]
    assert eligible(_row(1, 40, "White"), a)
    assert not eligible(_row(2, 40, "White", ["t2d"]), a)          # A excludes chronic groups
    assert not eligible(_row(3, 70, "White", ["t2d"]), b)          # B needs 2+
    assert eligible(_row(4, 70, "White", ["t2d", "ckd_esrd"]), b)
    assert not eligible(_row(5, 70, "White", ["t2d", "ckd_esrd"], alive=False), b)


def test_selection_uses_race_quota_and_reports_shortfall():
    a = default_request()["cohorts"][0]
    rows = [_row(i, 30, "White") for i in range(80)] + [_row(100 + i, 30, "Asian") for i in range(5)]
    chosen, rep = select_cohort(rows, a, 40, {"race_ethnicity": {"White": 50, "Asian": 50}}, seed=1)
    assert len(chosen) == 40
    assert sum(r["race_ethnicity"] == "Asian" for r in chosen) == 5
    assert rep["shortfall"] == {"Asian": 15}


def test_apply_profile_copies_mix_and_fault_rates():
    profile = {"k_min": 11, "n_patients": 200, "source": "test",
               "counts": {"sex": {"female": 120, "male": 80},
                          "race_ethnicity": {"White": 150, "Asian": None, "Hispanic/Latino": 45}},
               "quality": {"missing_birthdate": 0, "missing_race_ethnicity": 20, "dangling_reference": None,
                           "missing_obs_category": 0, "reversed_period": 0}}
    req, notes = apply_profile(default_request(), profile)
    assert req["faults"]["missing_race_ethnicity"] == 10.0
    assert req["faults"]["dangling_reference"] == 2.75          # suppressed → k/2 = 5.5 of 200
    assert round(sum(req["demographics"]["race_ethnicity"].values())) == 100
    assert len(notes) == 2


def test_fault_plan_has_exact_counts():
    plan = plan_faults(200, {"missing_birthdate": 5, "reversed_period": 10}, seed=3)
    assert sum("missing_birthdate" in p for p in plan) == 10 and sum("reversed_period" in p for p in plan) == 20


def test_inject_breaks_what_it_says():
    bundle = {"entry": [{"resource": {"resourceType": "Patient", "id": "p", "birthDate": "1970-01-01",
                                      "extension": [{"url": "x/us-core-race"}]}},
                        {"resource": {"resourceType": "Observation", "id": "o", "category": [{}],
                                      "encounter": {"reference": "urn:uuid:e"}}}]}
    rng = random.Random(1)
    assert inject(bundle, "missing_birthdate", rng) and "birthDate" not in bundle["entry"][0]["resource"]
    assert inject(bundle, "missing_race_ethnicity", rng) and bundle["entry"][0]["resource"]["extension"] == []
    assert inject(bundle, "dangling_reference", rng)
    assert inject(bundle, "missing_obs_category", rng) and "category" not in bundle["entry"][1]["resource"]


def test_record_quality_score():
    good = {"race_ethnicity": "Asian", "birth_date": "1980-01-01", "sex": "female",
            "bmi": 25, "systolic_bp": 120, "hba1c": 5.5, "egfr": 90}
    assert score_record(good)["score"] == 100
    bad = {**good, "race_ethnicity": None, "hba1c": 99}
    assert score_record(bad)["issues"] == ["missing_race_ethnicity", "implausible_hba1c"]

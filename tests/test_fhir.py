import pytest

from src.fhir.builder import build_bundle
from src.fhir.parser import bundle_to_record, detect_version
from src.fhir.validate import validate_bundle
from src.fhir.versions import DIFFERENCES, VERSION_ORDER

BASE = dict(patient_id="pt-1", gender="female", birth_date="1974-05-02",
            effective="2026-03-04T10:30:00+00:00", race_demographic="Black/African American",
            bmi=31.2, baseline_hba1c=9.8,
            recommended_medication_plan="Insulin Therapy Intensive + Metformin",
            recommended_meal_plan="Strict Carb Counting / Low Glycemic Index")

RECORDS = [
    BASE,
    {**BASE, "patient_id": "pt-2", "race_demographic": "Hispanic/Latino", "bmi": None},
    {**BASE, "patient_id": "pt-3", "race_demographic": None, "baseline_hba1c": 99.0},
]


@pytest.mark.parametrize("version", ["STU3", "R4", "R4B", "R5"])
@pytest.mark.parametrize("rec", RECORDS, ids=lambda r: r["patient_id"])
def test_bundles_pass_official_schema(version, rec):
    result = validate_bundle(build_bundle(rec, version), version)
    assert result["valid"], result["errors"]


def test_dstu2_is_reported_as_not_validated():
    assert validate_bundle(build_bundle(BASE, "DSTU2"), "DSTU2")["valid"] is None


@pytest.mark.parametrize("version", VERSION_ORDER)
@pytest.mark.parametrize("rec", RECORDS, ids=lambda r: r["patient_id"])
def test_round_trip_every_version(version, rec):
    out = bundle_to_record(build_bundle(rec, version))
    for key in ("race_demographic", "bmi", "baseline_hba1c", "recommended_medication_plan"):
        assert out[key] == rec[key]
    assert out["age"] == 51


@pytest.mark.parametrize("version,expected", [("DSTU2", "DSTU2"), ("STU3", "STU3"), ("R4", "R4/R4B"),
                                              ("R4B", "R4/R4B"), ("R5", "R5")])
def test_detect_version(version, expected):
    assert detect_version(build_bundle(BASE, version))[0] == expected


def test_wrong_version_shape_is_rejected():
    errors = validate_bundle(build_bundle(BASE, "R4"), "R5")["errors"]
    assert any("Encounter" in e and "class" in e for e in errors)
    assert validate_bundle(build_bundle(BASE, "STU3"), "R4")["valid"] is False


def test_missing_data_is_omitted_not_nulled():
    bundle = build_bundle(RECORDS[2], "R4")
    patient = bundle["entry"][0]["resource"]
    assert "extension" not in patient
    assert '"value": null' not in str(bundle)


def test_differences_table_covers_every_version():
    for row in DIFFERENCES:
        assert all(v in row for v in VERSION_ORDER)

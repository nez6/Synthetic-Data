import pytest

from src.fhir.builder import build_bundle
from src.fhir.parser import bundle_to_record, detect_version
from src.fhir.validate import validate_bundle
from src.fhir.versions import DIFFERENCES, VERSION_ORDER

BASE = dict(patient_id="pt-1", gender="female", birth_date="1960-05-02", effective="2026-03-04T10:30:00+00:00",
            race_ethnicity="Black/African American", language="es", state="MA", smoking="former",
            bmi=31.2, systolic_bp=142.0, hba1c=7.9, egfr=48.0, conditions=["hypertension", "t2d", "ckd"])
RECORDS = [
    BASE,
    {**BASE, "patient_id": "pt-2", "race_ethnicity": "Hispanic/Latino", "bmi": None, "conditions": []},
    {**BASE, "patient_id": "pt-3", "race_ethnicity": None, "smoking": None, "language": None},
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
    for key in ("race_ethnicity", "bmi", "systolic_bp", "hba1c", "egfr", "smoking", "language", "state"):
        assert out[key] == rec[key], key
    assert out["conditions"] == sorted(rec["conditions"])
    assert out["age"] == 65
    assert out["complex_care"] == (rec["conditions"] == BASE["conditions"])


@pytest.mark.parametrize("version,expected", [("DSTU2", "DSTU2"), ("STU3", "STU3"), ("R4", "R4/R4B"),
                                              ("R4B", "R4/R4B"), ("R5", "R5")])
def test_detect_version(version, expected):
    assert detect_version(build_bundle(BASE, version))[0] == expected


def test_wrong_version_shape_is_rejected():
    errors = validate_bundle(build_bundle(BASE, "R4"), "R5")["errors"]
    assert any("Encounter" in e and "class" in e for e in errors)
    assert validate_bundle(build_bundle(BASE, "STU3"), "R4")["valid"] is False


def test_condition_shape_differs_by_version():
    def cond(v):
        return next(e["resource"] for e in build_bundle(BASE, v)["entry"] if e["resource"]["resourceType"] == "Condition")
    assert isinstance(cond("STU3")["clinicalStatus"], str) and isinstance(cond("R4")["clinicalStatus"], dict)
    assert "patient" in cond("DSTU2") and "subject" in cond("R5")


def test_differences_table_covers_every_version():
    for row in DIFFERENCES:
        assert all(v in row for v in VERSION_ORDER)

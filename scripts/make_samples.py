"""Write example files to samples/: one patient as FHIR in every version, plus HL7 v2 messages."""
import json
from pathlib import Path

from src.fhir.builder import build_bundle
from src.fhir.versions import VERSION_ORDER
from src.hl7v2.message import build_oru

ROOT = Path(__file__).resolve().parent.parent
REC = dict(patient_id="pt-sample", gender="female", birth_date="1974-05-02",
           effective="2026-03-04T10:30:00+00:00", race_demographic="Black/African American",
           bmi=31.2, baseline_hba1c=9.8,
           recommended_medication_plan="Insulin Therapy Intensive + Metformin",
           recommended_meal_plan="Strict Carb Counting / Low Glycemic Index")

for v in VERSION_ORDER:
    (ROOT / f"samples/fhir/patient-bundle-{v}.json").write_text(json.dumps(build_bundle(REC, v), indent=2) + "\n")

labs = {k: v for k, v in REC.items() if not k.startswith("recommended")}
(ROOT / "samples/hl7/oru_r01.hl7").write_text(build_oru(labs, "SAMPLE0001").replace("\r", "\n") + "\n")
(ROOT / "samples/hl7/oru_r01_missing_data.hl7").write_text(
    build_oru({**labs, "race_demographic": None, "bmi": None, "baseline_hba1c": 99.0},
              "SAMPLE0002").replace("\r", "\n") + "\n")
print("Wrote samples/fhir/*.json and samples/hl7/*.hl7")

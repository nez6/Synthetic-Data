"""Shared clinical code systems used by both the FHIR and HL7 v2 layers."""

LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"
CDC_RACE_ETHNICITY = "urn:oid:2.16.840.1.113883.6.238"  # CDC Race & Ethnicity (CDCREC)

HBA1C = {"code": "4548-4", "display": "Hemoglobin A1c/Hemoglobin.total in Blood",
         "unit": "%", "category": "laboratory"}
BMI = {"code": "39156-5", "display": "Body mass index (BMI) [Ratio]",
       "unit": "kg/m2", "category": "vital-signs"}

# Demographic group used by the model  ->  (CDC race code, display)
RACE_CODES = {
    "American Indian/Alaska Native": ("1002-5", "American Indian or Alaska Native"),
    "Asian": ("2028-9", "Asian"),
    "Black/African American": ("2054-5", "Black or African American"),
    "White": ("2106-3", "White"),
}
RACE_BY_CODE = {code: group for group, (code, _) in RACE_CODES.items()}

# In US Core (and OMB standards) Hispanic/Latino is an *ethnicity*, not a race.
HISPANIC = ("2135-2", "Hispanic or Latino")
NOT_HISPANIC = ("2186-5", "Not Hispanic or Latino")
HISPANIC_GROUP = "Hispanic/Latino"

DEMOGRAPHIC_GROUPS = list(RACE_CODES) + [HISPANIC_GROUP]

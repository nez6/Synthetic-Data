"""Shared clinical code systems used by the generator, FHIR builder/parser and HL7 v2 layers."""

LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"
SNOMED = "http://snomed.info/sct"
ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
BCP47 = "urn:ietf:bcp:47"
CDC_RACE_ETHNICITY = "urn:oid:2.16.840.1.113883.6.238"  # CDC Race & Ethnicity (CDCREC)

# Vitals and labs recorded at each visit. `synonyms` are non-UCUM unit spellings that
# the validation script may safely rewrite to the UCUM code.
LABS = {
    "bmi": {"code": "39156-5", "display": "Body mass index (BMI) [Ratio]", "unit": "kg/m2",
            "category": "vital-signs", "synonyms": ["kg/m^2", "kg/m²"]},
    "systolic_bp": {"code": "8480-6", "display": "Systolic blood pressure", "unit": "mm[Hg]",
                    "category": "vital-signs", "synonyms": ["mmHg", "mm Hg"]},
    "hba1c": {"code": "4548-4", "display": "Hemoglobin A1c/Hemoglobin.total in Blood", "unit": "%",
              "category": "laboratory", "synonyms": ["percent"]},
    "egfr": {"code": "33914-3",
             "display": "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum or Plasma "
                        "by Creatinine-based formula (MDRD)",
             "unit": "mL/min/{1.73_m2}", "category": "laboratory", "synonyms": ["mL/min/1.73m2", "ml/min"]},
}
LAB_BY_CODE = {v["code"]: k for k, v in LABS.items()}

SMOKING_LOINC = ("72166-2", "Tobacco smoking status")
SMOKING = {
    "never": ("266919005", "Never smoked tobacco (finding)"),
    "former": ("8517006", "Ex-smoker (finding)"),
    "current": ("449868002", "Smokes tobacco daily (finding)"),
}
SMOKING_BY_CODE = {code: k for k, (code, _) in SMOKING.items()}

# Chronic conditions: SNOMED CT + ICD-10-CM, and a first-line RxNorm ingredient (or None)
CONDITIONS = {
    "hypertension": {"label": "Hypertension", "snomed": ("59621000", "Essential hypertension (disorder)"),
                     "icd10": ("I10", "Essential (primary) hypertension"), "med": ("29046", "lisinopril")},
    "t2d": {"label": "Type 2 diabetes", "snomed": ("44054006", "Diabetes mellitus type 2 (disorder)"),
            "icd10": ("E11.9", "Type 2 diabetes mellitus without complications"), "med": ("6809", "metformin")},
    "ckd": {"label": "Chronic kidney disease", "snomed": ("433144002", "Chronic kidney disease stage 3 (disorder)"),
            "icd10": ("N18.30", "Chronic kidney disease, stage 3 unspecified"), "med": None},
    "chf": {"label": "Heart failure", "snomed": ("88805009", "Chronic congestive heart failure (disorder)"),
            "icd10": ("I50.9", "Heart failure, unspecified"), "med": ("4603", "furosemide")},
    "copd": {"label": "COPD", "snomed": ("13645005", "Chronic obstructive lung disease (disorder)"),
             "icd10": ("J44.9", "Chronic obstructive pulmonary disease, unspecified"), "med": ("435", "albuterol")},
}
CONDITION_BY_CODE = {}
for _key, _c in CONDITIONS.items():
    CONDITION_BY_CODE[_c["snomed"][0]] = _key
    CONDITION_BY_CODE[_c["icd10"][0]] = _key

# The ML target (as in the reference project's Cohort B): 2+ of these chronic condition groups.
COMPLEX_CARE_GROUPS = ("t2d", "ckd", "chf", "copd")

RACE_CODES = {
    "American Indian/Alaska Native": ("1002-5", "American Indian or Alaska Native"),
    "Asian": ("2028-9", "Asian"),
    "Black/African American": ("2054-5", "Black or African American"),
    "Native Hawaiian/Pacific Islander": ("2076-8", "Native Hawaiian or Other Pacific Islander"),
    "White": ("2106-3", "White"),
}
RACE_BY_CODE = {code: group for group, (code, _) in RACE_CODES.items()}
# In US Core (and OMB standards) Hispanic/Latino is an *ethnicity*, not a race.
HISPANIC = ("2135-2", "Hispanic or Latino")
NOT_HISPANIC = ("2186-5", "Not Hispanic or Latino")
HISPANIC_GROUP = "Hispanic/Latino"
DEMOGRAPHIC_GROUPS = list(RACE_CODES) + [HISPANIC_GROUP]

LANGUAGES = {"en-US": "English", "es": "Spanish", "zh": "Chinese", "vi": "Vietnamese", "other": "Other"}
OTHER_LANGUAGE_CODES = ["fr", "ht", "ar", "ru", "ko", "pt"]


def complex_care(conditions) -> bool:
    return sum(c in conditions for c in COMPLEX_CARE_GROUPS) >= 2

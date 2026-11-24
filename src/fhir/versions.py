"""Metadata about the FHIR versions this project can produce, and what changes between them."""

VERSIONS = {
    "DSTU2": {"fhirVersion": "1.0.2", "released": "2015",
              "notes": "First widely deployed release. US data used the Argonaut profiles."},
    "STU3": {"fhirVersion": "3.0.2", "released": "2017",
             "notes": "US Core replaces Argonaut. Many single fields become lists. Observation.context appears."},
    "R4": {"fhirVersion": "4.0.1", "released": "2019",
           "notes": "First 'normative' release; the version US regulation (ONC Cures Act) requires today."},
    "R4B": {"fhirVersion": "4.3.0", "released": "2022",
            "notes": "Interim release. Changes are in medication-definition resources; "
                     "Patient/Observation/Encounter/Condition are identical to R4."},
    "R5": {"fhirVersion": "5.0.0", "released": "2023",
           "notes": "Latest release. Encounter was substantially redesigned."},
}
VERSION_ORDER = list(VERSIONS)

# Each row: what the element looks like in each version, for the resources this project uses.
DIFFERENCES = [
    {
        "element": "Race / ethnicity extension",
        "why": "US-specific demographics moved from the Argonaut project into the official US Core guide.",
        "DSTU2": "argo-race / argo-ethnicity (fhir.org/guides/argonaut)",
        "STU3": "us-core-race / us-core-ethnicity",
        "R4": "us-core-race / us-core-ethnicity",
        "R4B": "us-core-race / us-core-ethnicity",
        "R5": "us-core-race / us-core-ethnicity",
    },
    {
        "element": "Observation.category",
        "why": "Became a list (an Observation can be in several categories); code system moved to terminology.hl7.org in R4.",
        "DSTU2": "single object, system hl7.org/fhir/observation-category",
        "STU3": "list, system hl7.org/fhir/observation-category",
        "R4": "list, system terminology.hl7.org/CodeSystem/observation-category",
        "R4B": "list, system terminology.hl7.org/CodeSystem/observation-category",
        "R5": "list, system terminology.hl7.org/CodeSystem/observation-category",
    },
    {
        "element": "Observation → Encounter link",
        "why": "STU3 renamed it 'context' (to allow EpisodeOfCare); R4 reverted to 'encounter'.",
        "DSTU2": "encounter", "STU3": "context", "R4": "encounter", "R4B": "encounter", "R5": "encounter",
    },
    {
        "element": "Observation free-text note",
        "why": "R4 replaced a plain string with Annotation (author + time + text).",
        "DSTU2": "comments (string)", "STU3": "comment (string)",
        "R4": "note[] (Annotation)", "R4B": "note[] (Annotation)", "R5": "note[] (Annotation)",
    },
    {
        "element": "Encounter → Patient link",
        "why": "Renamed so Encounter can reference a Group as well as a Patient.",
        "DSTU2": "patient", "STU3": "subject", "R4": "subject", "R4B": "subject", "R5": "subject",
    },
    {
        "element": "Encounter.class",
        "why": "Plain code → Coding (STU3) → list of CodeableConcept (R5).",
        "DSTU2": '"ambulatory" (code)',
        "STU3": "Coding, system hl7.org/fhir/v3/ActCode",
        "R4": "Coding, system terminology.hl7.org/CodeSystem/v3-ActCode",
        "R4B": "Coding, system terminology.hl7.org/CodeSystem/v3-ActCode",
        "R5": "list of CodeableConcept",
    },
    {
        "element": "Encounter.status for a completed visit",
        "why": "R5 redesigned the Encounter status value set.",
        "DSTU2": "finished", "STU3": "finished", "R4": "finished", "R4B": "finished", "R5": "completed",
    },
    {
        "element": "Encounter time span",
        "why": "R5 separates actual vs planned times.",
        "DSTU2": "period", "STU3": "period", "R4": "period", "R4B": "period", "R5": "actualPeriod",
    },
    {
        "element": "Condition → Patient link",
        "why": "Renamed so a Condition can be recorded against a Group as well as a Patient.",
        "DSTU2": "patient", "STU3": "subject", "R4": "subject", "R4B": "subject", "R5": "subject",
    },
    {
        "element": "Condition.clinicalStatus / verificationStatus",
        "why": "R4 turned plain codes into CodeableConcepts so they can carry a code system.",
        "DSTU2": "plain codes", "STU3": "plain codes",
        "R4": "CodeableConcept (terminology.hl7.org)", "R4B": "CodeableConcept (terminology.hl7.org)",
        "R5": "CodeableConcept (terminology.hl7.org)",
    },
    {
        "element": "Condition.category",
        "why": "Became a list in STU3 ('diagnosis' → 'problem-list-item'); code system moved in R4.",
        "DSTU2": "single object, 'diagnosis'", "STU3": "list, hl7.org/fhir/condition-category",
        "R4": "list, terminology.hl7.org/CodeSystem/condition-category",
        "R4B": "list, terminology.hl7.org/CodeSystem/condition-category",
        "R5": "list, terminology.hl7.org/CodeSystem/condition-category",
    },
    {
        "element": "Bundle.timestamp",
        "why": "Added in R4 to record when the Bundle was assembled.",
        "DSTU2": "(does not exist)", "STU3": "(does not exist)",
        "R4": "present", "R4B": "present", "R5": "present",
    },
]


def differences_between(a: str, b: str) -> list[dict]:
    return [d for d in DIFFERENCES if d[a] != d[b]]

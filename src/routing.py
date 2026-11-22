"""Rule-based clinician routing layer applied on top of model predictions."""


def recommend_provider(hba1c: float, bmi: float, predicted_medication: str) -> dict:
    if hba1c >= 9.5:
        return {
            "specialty": "Endocrinology & Metabolic Specialist Liaison",
            "urgency": "High - Immediate Consult Required",
            "clinical_rationale": "Severe HbA1c elevation warrants immediate complex glycemic management protocols.",
        }
    if "Combination" in predicted_medication or bmi > 32.0:
        return {
            "specialty": "Internal Medicine Team & Registered Dietitian",
            "urgency": "Routine Care Path Tracking",
            "clinical_rationale": "Multi-therapeutic management required to balance metabolic baseline and structural weight metrics.",
        }
    return {
        "specialty": "Primary Care Physician & Behavioral Health Coach",
        "urgency": "Preventative Track Monitoring",
        "clinical_rationale": "Early-stage management alignment using preventative lifestyle frameworks.",
    }

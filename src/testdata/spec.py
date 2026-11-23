"""Cohort specification: the key demographics, clinical profile and fault rates for a test-data run."""
from dataclasses import asdict, dataclass, field
from datetime import date

from src import terminology as t

NOT_RECORDED = "Not recorded"

RACE_OPTIONS = t.DEMOGRAPHIC_GROUPS + [NOT_RECORDED]
LANGUAGES = {"en-US": "English", "es": "Spanish", "vi": "Vietnamese", "zh": "Chinese", NOT_RECORDED: None}
CONTROL_TIERS = {
    "well_controlled": ("HbA1c < 7.0%", 5.7, 6.9),
    "above_goal": ("HbA1c 7.0–9.0%", 7.0, 9.0),
    "poorly_controlled": ("HbA1c > 9.0%", 9.1, 12.5),
}
STATES = {
    "MA": [("Boston", "021"), ("Worcester", "016"), ("Springfield", "011")],
    "NY": [("Bronx", "104"), ("Buffalo", "142"), ("Rochester", "146")],
    "NC": [("Durham", "277"), ("Raleigh", "276"), ("Charlotte", "282")],
    "FL": [("Miami", "331"), ("Tampa", "336"), ("Orlando", "328")],
    "MS": [("Jackson", "392"), ("Gulfport", "395"), ("Hattiesburg", "394")],
    "TX": [("Houston", "770"), ("San Antonio", "782"), ("El Paso", "799")],
    "AZ": [("Phoenix", "850"), ("Tucson", "857"), ("Flagstaff", "860")],
    "CA": [("Los Angeles", "900"), ("Fresno", "937"), ("Oakland", "946")],
}

# fault key -> (label, repairable by the validation script?)
FAULTS = {
    "missing_bmi": ("No BMI observations", False),
    "implausible_hba1c": ("Latest HbA1c entered as 99%", False),
    "missing_birthdate": ("Patient.birthDate missing", False),
    "non_ucum_unit": ("HbA1c unit written as 'percent' (not UCUM)", True),
    "missing_obs_category": ("An Observation has no category", True),
    "reversed_period": ("An Encounter ends before it starts", True),
    "dangling_reference": ("An Observation points to a missing Encounter", True),
    "missing_htest_label": ("Patient lacks the HTEST test-data label", True),
}


@dataclass
class CohortSpec:
    label: str = "Balanced clinic population"
    n: int = 500
    seed: int = 20261122
    min_age: int = 18
    max_age: int = 85
    pct_female: float = 52.0
    race_mix: dict = field(default_factory=lambda: {
        "American Indian/Alaska Native": 3, "Asian": 10, "Black/African American": 18,
        "White": 40, "Hispanic/Latino": 22, NOT_RECORDED: 7})
    language_mix: dict = field(default_factory=lambda: {"en-US": 80, "es": 12, "vi": 2, "zh": 3, NOT_RECORDED: 3})
    states: list = field(default_factory=lambda: ["MA", "NC", "TX"])
    control_mix: dict = field(default_factory=lambda: {"well_controlled": 45, "above_goal": 35, "poorly_controlled": 20})
    bmi_mean: float = 31.0
    bmi_sd: float = 6.0
    history_years: int = 2
    visits_per_year: int = 2
    faults: dict = field(default_factory=lambda: {
        "missing_bmi": 3, "implausible_hba1c": 2, "missing_birthdate": 1, "non_ucum_unit": 3,
        "missing_obs_category": 3, "reversed_period": 2, "dangling_reference": 2, "missing_htest_label": 2})
    # Acceptance criteria (as in the reference project's testing plan)
    min_hba1c_obs: int = 2
    min_bmi_obs: int = 1
    h0_rate: float = 0.85
    target_rate: float = 0.90
    alpha: float = 0.05
    as_of: str = field(default_factory=lambda: date.today().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CohortSpec":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def problems(self) -> list[str]:
        out = []
        if self.min_age > self.max_age:
            out.append("Minimum age is above maximum age.")
        for name, mix in (("Race/ethnicity", self.race_mix), ("Language", self.language_mix),
                          ("Diabetes control", self.control_mix)):
            total = sum(mix.values())
            if total <= 0:
                out.append(f"{name} mix is all zero.")
            elif abs(total - 100) > 0.01:
                out.append(f"{name} mix adds up to {total:g}%, not 100% (it will be rescaled).")
        if not self.states:
            out.append("Pick at least one state.")
        return out


PRESETS = {
    "Balanced clinic population": CohortSpec(),
    "Community health center (higher risk)": CohortSpec(
        label="Community health center (higher risk)", min_age=25, max_age=75, pct_female=55,
        race_mix={"American Indian/Alaska Native": 5, "Asian": 5, "Black/African American": 30,
                  "White": 15, "Hispanic/Latino": 40, NOT_RECORDED: 5},
        language_mix={"en-US": 55, "es": 35, "vi": 3, "zh": 3, NOT_RECORDED: 4},
        states=["NY", "TX", "MS"], control_mix={"well_controlled": 25, "above_goal": 40, "poorly_controlled": 35},
        bmi_mean=34, bmi_sd=7,
        faults={"missing_bmi": 8, "implausible_hba1c": 4, "missing_birthdate": 2, "non_ucum_unit": 6,
                "missing_obs_category": 5, "reversed_period": 3, "dangling_reference": 4, "missing_htest_label": 3}),
    "Older adults (65+)": CohortSpec(
        label="Older adults (65+)", min_age=65, max_age=90, pct_female=50,
        race_mix={"American Indian/Alaska Native": 2, "Asian": 10, "Black/African American": 15,
                  "White": 55, "Hispanic/Latino": 15, NOT_RECORDED: 3},
        language_mix={"en-US": 85, "es": 8, "vi": 2, "zh": 3, NOT_RECORDED: 2},
        states=["FL", "AZ", "MA"], control_mix={"well_controlled": 50, "above_goal": 35, "poorly_controlled": 15},
        bmi_mean=29, bmi_sd=5, history_years=4, visits_per_year=3),
    "Clean reference cohort (no faults)": CohortSpec(
        label="Clean reference cohort (no faults)", faults={k: 0 for k in FAULTS}),
}

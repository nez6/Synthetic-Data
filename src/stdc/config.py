"""Loads config/emrs.json: test settings, Synthea settings, cohorts and EMRs."""

import json
from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class Target:
    """Allowed resources of one type per patient. max None means open-ended ('30+')."""
    min: int
    max: int | None
    severity: str = "error"  # "error" fails the record, "warning" is reported only


@dataclass(frozen=True)
class CohortSpec:
    name: str
    label: str
    share: float
    min_age: int
    max_age: int
    min_condition_groups: int
    exclude_condition_groups: bool
    min_history_years: float
    max_gap_days: int | None
    targets: dict  # resource type (or "A|B" for combined types) -> Target


@dataclass(frozen=True)
class SyntheaConfig:
    version: str
    reference_date: date
    state: str
    data_as_of: date | None  # last date the committed data can contain; defaults to reference_date
    pools: dict  # cohort name -> {"population", "min_age", "max_age", "years_of_history"}


@dataclass(frozen=True)
class EmrConfig:
    key: str
    display_name: str
    enabled: bool
    sample_size: int
    seed: int
    profile_source: str  # "none" or "cdm"
    cdm: dict = field(default_factory=dict)
    cohort_sample_sizes: dict | None = None  # cohort -> n; None = split sample_size by cohort share

    def counts(self, cohorts):
        """Patients to test per cohort."""
        if self.cohort_sample_sizes:
            return {c.name: int(self.cohort_sample_sizes[c.name]) for c in cohorts}
        counts, assigned = {}, 0
        for i, c in enumerate(cohorts):
            counts[c.name] = self.sample_size - assigned if i == len(cohorts) - 1 else round(self.sample_size * c.share)
            assigned += counts[c.name]
        return counts


@dataclass(frozen=True)
class TestConfig:
    k_min_cell_size: int
    h0_rate: float
    target_rate: float
    alpha: float
    power: float
    correct_for_multiple_emrs: bool
    per_cohort_tests: bool
    population_weights: dict  # cohort -> weight, or empty when not yet estimated
    population_benchmarks: tuple
    forbidden_output_schemas: tuple
    synthea: SyntheaConfig
    condition_groups: dict  # group -> {"snomed": [...], "icd10_prefixes": [...]}
    cohorts: tuple
    emrs: tuple

    @property
    def as_of_date(self):
        return self.synthea.data_as_of or self.synthea.reference_date

    @property
    def enabled_emrs(self):
        return tuple(e for e in self.emrs if e.enabled)

    def emr(self, key):
        for e in self.emrs:
            if e.key == key:
                return e
        raise KeyError(f"EMR '{key}' is not in the config")

    def cohort(self, name):
        for c in self.cohorts:
            if c.name == name:
                return c
        raise KeyError(f"cohort '{name}' is not in the config")

    def alpha_for_emr(self):
        """Per-EMR significance level, Bonferroni-corrected when enabled."""
        k = len(self.enabled_emrs) if self.correct_for_multiple_emrs else 1
        return self.alpha / max(k, 1)

    def alpha_for_cohort(self):
        """Per-cohort significance level: also divided by the number of cohorts tested."""
        return self.alpha_for_emr() / (len(self.cohorts) if self.per_cohort_tests else 1)


def _target(v):
    return Target(min=int(v["min"]), max=None if v.get("max") is None else int(v["max"]),
                  severity=v.get("severity", "error"))


def _cohort(d):
    return CohortSpec(
        name=d["name"], label=d["label"], share=float(d["share"]),
        min_age=int(d["min_age"]), max_age=int(d["max_age"]),
        min_condition_groups=int(d.get("min_condition_groups", 0)),
        exclude_condition_groups=bool(d.get("exclude_condition_groups", False)),
        min_history_years=float(d.get("min_history_years", 0)),
        max_gap_days=d.get("max_gap_days"),
        targets={k: _target(v) for k, v in d["targets"].items()},
    )


def config_from_dict(d):
    h, s = d["hypothesis"], d["synthea"]
    cohorts = tuple(_cohort(c) for c in d["cohorts"])
    total = sum(c.share for c in cohorts)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"cohort shares sum to {total}, expected 1.0")
    return TestConfig(
        k_min_cell_size=int(d["k_min_cell_size"]),
        h0_rate=float(h["h0_rate"]), target_rate=float(h["target_rate"]),
        alpha=float(h["alpha"]), power=float(h["power"]),
        correct_for_multiple_emrs=bool(h["correct_for_multiple_emrs"]),
        per_cohort_tests=bool(h.get("per_cohort_tests", False)),
        population_weights={k: float(v) for k, v in d.get("population_weights", {}).items()
                            if k in {c.name for c in cohorts} and v is not None},
        population_benchmarks=tuple(d.get("population_benchmarks", [])),
        forbidden_output_schemas=tuple(d.get("forbidden_output_schemas", [])),
        synthea=SyntheaConfig(
            version=s["version"], reference_date=date.fromisoformat(s["reference_date"]),
            data_as_of=date.fromisoformat(s["data_as_of"]) if s.get("data_as_of") else None,
            state=s.get("state", "Massachusetts"), pools=s["pools"],
        ),
        condition_groups=d["condition_groups"],
        cohorts=cohorts,
        emrs=tuple(EmrConfig(
            key=e["key"], display_name=e["display_name"], enabled=e.get("enabled", True),
            sample_size=int(e.get("sample_size") or sum(e["cohort_sample_sizes"].values())),
            seed=int(e["seed"]), cohort_sample_sizes=e.get("cohort_sample_sizes"),
            profile_source=e.get("profile_source", "none"), cdm=e.get("cdm", {}),
        ) for e in d["emrs"]),
    )


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return config_from_dict(json.load(f))

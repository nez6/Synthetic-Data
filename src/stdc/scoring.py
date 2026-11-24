"""Scores Synthea bundles before and after validation.

  record pass rate       share of bundles with no error-level issue (primary;
                         McNemar's test before vs after)
  conformance score      0-1 per record: share of checks V01-V08 with no error
                         (paired t-test before vs after)
  realism (optional)     similarity to an aggregate real-data profile, only
                         when the EMR has a CDM profile configured
"""

import statistics
from dataclasses import dataclass

from . import profile as prof
from .validation import conformance_score, errors, validate


@dataclass(frozen=True)
class RecordScore:
    passed: bool
    n_errors: int
    n_warnings: int
    conformance: float
    failed_checks: tuple


def score_record(bundle, ctx):
    issues = validate(bundle, ctx)
    errs = errors(issues)
    return RecordScore(
        passed=not errs,
        n_errors=len(errs),
        n_warnings=len(issues) - len(errs),
        conformance=conformance_score(issues),
        failed_checks=tuple(sorted({i.check_id for i in errs})),
    )


def total_variation_distance(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def quantile_distance(qs_a, qs_b):
    """Mean relative gap between matching quantiles, 0 = identical."""
    gaps = [abs(a - b) / max(abs(a), abs(b), 1.0) for a, b in zip(qs_a, qs_b)]
    return statistics.fmean(gaps)


def realism_scores(features, reference):
    """Similarity of a set of patients to a reference profile. 1.0 = identical."""
    synthetic = prof.build_profile_from_features(features, k_min=1)
    out = {}
    for name in ("gender", "condition_group"):
        out[f"tvd_{name}"] = total_variation_distance(synthetic["categorical"][name],
                                                      reference["categorical"].get(name, {}))
    for name, qs in reference["numeric"].items():
        if name in synthetic["numeric"]:
            out[f"qdist_{name}"] = quantile_distance(synthetic["numeric"][name], qs)
    out["categorical_similarity"] = 1 - statistics.fmean(v for k, v in out.items() if k.startswith("tvd_"))
    out["distribution_similarity"] = 1 - statistics.fmean(v for k, v in out.items() if k.startswith("qdist_"))
    ref = reference["correlation"]
    syn = prof.correlation_matrix(features, ref["features"])
    k = len(ref["features"])
    out["correlation_similarity"] = 1 - statistics.fmean(
        abs(syn[i][j] - ref["matrix"][i][j]) for i in range(k) for j in range(i + 1, k)) / 2
    return out

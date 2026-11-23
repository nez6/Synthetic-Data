"""Hypothesis tests used to accept a test-data set (same design as the reference project).

    H0: independent pass rate <= h0_rate (default 85%)   H1: > h0_rate, target 90%
    Exact one-sided binomial test, Wilson 95% CI
    McNemar's exact test: pass/fail before vs after validation (same records)
    Paired t-test: conformance score before vs after
"""
import math
from statistics import NormalDist

from scipy.stats import binomtest, chisquare, ttest_rel


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    ci = binomtest(successes, n).proportion_ci(confidence, method="wilson")
    return (float(ci.low), float(ci.high))


def one_sided_binomial(successes: int, n: int, p0: float) -> float:
    return float(binomtest(successes, n, p0, alternative="greater").pvalue) if n else 1.0


def mcnemar_exact(before: list[bool], after: list[bool]) -> dict:
    b = sum(1 for x, y in zip(before, after) if not x and y)   # fixed by validation
    c = sum(1 for x, y in zip(before, after) if x and not y)   # broken by validation
    p = float(binomtest(b, b + c, 0.5, alternative="greater").pvalue) if b + c else 1.0
    return {"fixed": b, "broken": c, "p_one_sided": p}


def paired_t(before: list[float], after: list[float]) -> dict:
    diffs = [a - b for a, b in zip(after, before)]
    mean = sum(diffs) / len(diffs) if diffs else 0.0
    if len(diffs) < 2 or all(d == diffs[0] for d in diffs):
        return {"mean_improvement": mean, "p_one_sided": 0.0 if mean > 0 else 1.0}
    res = ttest_rel(after, before, alternative="greater")
    return {"mean_improvement": mean, "t": float(res.statistic), "p_one_sided": float(res.pvalue)}


def required_n(p0: float, p1: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Records needed to show a pass rate above p0 when the true rate is p1."""
    z = NormalDist().inv_cdf
    num = z(1 - alpha) * math.sqrt(p0 * (1 - p0)) + z(power) * math.sqrt(p1 * (1 - p1))
    return math.ceil((num / (p1 - p0)) ** 2)


def goodness_of_fit(observed: dict, expected_share: dict) -> float | None:
    """Chi-square p-value: do observed counts follow the requested shares?"""
    keys = [k for k in expected_share if expected_share[k] > 0]
    n = sum(observed.get(k, 0) for k in keys)
    if n == 0 or len(keys) < 2:
        return None
    total = sum(expected_share[k] for k in keys)
    exp = [n * expected_share[k] / total for k in keys]
    return float(chisquare([observed.get(k, 0) for k in keys], exp).pvalue)


def _rate(flags):
    n = len(flags)
    return {"rate": sum(flags) / n if n else 0.0, "ci": wilson_ci(sum(flags), n), "passed": sum(flags)}


def evaluate(script_before: list[bool], script_after: list[bool],
             independent_before: list[bool], independent_after: list[bool],
             conf_before: list[float], conf_after: list[float],
             h0_rate: float, target_rate: float, alpha: float) -> dict:
    """The decision uses the *independent* pass rate after validation (primary score)."""
    n = len(independent_after)
    p = one_sided_binomial(sum(independent_after), n, h0_rate)
    need = required_n(h0_rate, target_rate, alpha)
    return {
        "n": n,
        "script_before": _rate(script_before),
        "script_after": _rate(script_after),
        "independent_before": _rate(independent_before),
        "independent_after": _rate(independent_after),
        "h0_rate": h0_rate, "target_rate": target_rate, "alpha": alpha,
        "p_value_vs_h0": p,
        "reject_h0": p < alpha,
        "mcnemar": mcnemar_exact(independent_before, independent_after),
        "paired_t": paired_t(conf_before, conf_after),
        "required_n": need,
        "sample_size_adequate": n >= need,
    }

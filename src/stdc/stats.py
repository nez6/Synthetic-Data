"""Hypothesis tests, confidence intervals and sample sizes (standard library only).

Hypotheses for each EMR:
    H0: validated record pass rate <= h0_rate (default 0.85)
    H1: validated record pass rate >  h0_rate, target 0.90

Supporting paired comparisons on the same synthetic records:
    McNemar's exact test   pass/fail before vs after validation
    Paired t-test          record conformance before vs after validation
"""

import math
import statistics
from statistics import NormalDist

_N01 = NormalDist()


def z(p):
    return _N01.inv_cdf(p)


# ------------------------------------------------------------ distributions

def binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p), computed in log space."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k, n + 1):
        log_term = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                    + i * lp + (n - i) * lq)
        total += math.exp(log_term)
    return min(total, 1.0)


def _betacf(a, b, x, max_iter=300, eps=3e-14):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def t_sf(t, df):
    """P(T > t) for Student's t with df degrees of freedom."""
    tail = 0.5 * betainc(df / 2, 0.5, df / (df + t * t))
    return tail if t > 0 else 1 - tail


# ---------------------------------------------------------- intervals/tests

def wilson_ci(successes, n, confidence=0.95):
    if n == 0:
        return (0.0, 1.0)
    zz = z(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + zz * zz / n
    centre = (p + zz * zz / (2 * n)) / denom
    half = zz * math.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def one_sided_binomial_test(successes, n, p0):
    """Exact p-value for H0: rate <= p0 against H1: rate > p0."""
    return binom_sf(successes, n, p0)


def mcnemar_exact(before, after):
    """Exact McNemar test on paired pass/fail outcomes.

    b = passed before, failed after; c = failed before, passed after.
    Returns one-sided (improvement) and two-sided p-values.
    """
    if len(before) != len(after):
        raise ValueError("paired samples must be the same length")
    b = sum(1 for x, y in zip(before, after) if x and not y)
    c = sum(1 for x, y in zip(before, after) if not x and y)
    n = b + c
    one_sided = binom_sf(c, n, 0.5) if n else 1.0
    lower = 1 - binom_sf(c + 1, n, 0.5) if n else 1.0
    two_sided = min(1.0, 2 * min(one_sided, lower)) if n else 1.0
    return {"b": b, "c": c, "p_one_sided": one_sided, "p_two_sided": two_sided}


def paired_t_test(before, after):
    """One-sided paired t-test for H1: mean(after - before) > 0."""
    if len(before) != len(after):
        raise ValueError("paired samples must be the same length")
    d = [y - x for x, y in zip(before, after)]
    n = len(d)
    mean = statistics.fmean(d)
    sd = statistics.stdev(d) if n > 1 else 0.0
    if sd == 0:
        return {"mean_diff": mean, "t": math.inf if mean > 0 else 0.0, "df": n - 1,
                "p_one_sided": 0.0 if mean > 0 else 1.0}
    t = mean / (sd / math.sqrt(n))
    return {"mean_diff": mean, "t": t, "df": n - 1, "p_one_sided": t_sf(t, n - 1)}


def bonferroni(alpha, k):
    return alpha / max(k, 1)


# ------------------------------------------------------------- sample sizes

def n_one_proportion(p0, p1, alpha=0.05, power=0.80):
    """Patients needed to show rate > p0 when the true rate is p1 (one-sided)."""
    num = z(1 - alpha) * math.sqrt(p0 * (1 - p0)) + z(power) * math.sqrt(p1 * (1 - p1))
    return math.ceil((num / (p1 - p0)) ** 2)


def n_ci_half_width(p, half_width, confidence=0.95):
    """Patients needed to estimate p to within +/- half_width."""
    zz = z(1 - (1 - confidence) / 2)
    return math.ceil(zz * zz * p * (1 - p) / half_width ** 2)


def n_correlation(r0, r1, alpha=0.05, power=0.80):
    """Pairs needed to show correlation r1 exceeds r0 (Fisher z, one-sided)."""
    return math.ceil(((z(1 - alpha) + z(power)) / (math.atanh(r1) - math.atanh(r0))) ** 2 + 3)


# ---------------------------------------------------------------- decision

def evaluate(baseline_pass, validated_pass, baseline_fid, validated_fid,
             h0_rate, target_rate, alpha, power=0.80):
    """Run every test for one EMR and return a flat result dictionary."""
    n = len(validated_pass)
    x0, x1 = sum(baseline_pass), sum(validated_pass)
    b_lo, b_hi = wilson_ci(x0, n)
    v_lo, v_hi = wilson_ci(x1, n)
    p_h0 = one_sided_binomial_test(x1, n, h0_rate)
    mc = mcnemar_exact(baseline_pass, validated_pass)
    pt = paired_t_test(baseline_fid, validated_fid)
    required = n_one_proportion(h0_rate, target_rate, alpha, power)
    reject = p_h0 < alpha
    return {
        "n": n,
        "baseline_pass_rate": x0 / n,
        "baseline_ci_low": b_lo,
        "baseline_ci_high": b_hi,
        "validated_pass_rate": x1 / n,
        "validated_ci_low": v_lo,
        "validated_ci_high": v_hi,
        "h0_rate": h0_rate,
        "target_rate": target_rate,
        "alpha": alpha,
        "p_value_vs_h0": p_h0,
        "reject_h0": reject,
        "target_met": x1 / n >= target_rate,
        "mcnemar_b": mc["b"],
        "mcnemar_c": mc["c"],
        "mcnemar_p_one_sided": mc["p_one_sided"],
        "baseline_mean_conformance": statistics.fmean(baseline_fid),
        "validated_mean_conformance": statistics.fmean(validated_fid),
        "conformance_mean_diff": pt["mean_diff"],
        "paired_t": pt["t"],
        "paired_t_p_one_sided": pt["p_one_sided"],
        "required_n": required,
        "sample_size_adequate": n >= required,
        "decision": ("PASS: validated rate is above H0" if reject
                     else "FAIL: cannot show validated rate is above H0"),
    }


def binom_two_sided(successes, n, p0):
    """Exact two-sided binomial p-value (doubling the smaller tail)."""
    upper = binom_sf(successes, n, p0)
    lower = 1 - binom_sf(successes + 1, n, p0)
    return min(1.0, 2 * min(upper, lower))


def stratified_rate(strata, weights, h0_rate, confidence=0.95):
    """Population-weighted pass rate from independent stratum samples.

    strata maps stratum -> (successes, n); weights maps stratum -> population share
    (normalised here). Uses the standard stratified estimator sum(w * p) with variance
    sum(w^2 * p(1-p) / n), a normal-approximation interval, and a one-sided z-test
    against h0_rate.
    """
    total = sum(weights[s] for s in strata)
    est, var = 0.0, 0.0
    for s, (x, n) in strata.items():
        w = weights[s] / total
        p = x / n
        est += w * p
        var += w * w * p * (1 - p) / n
    se = math.sqrt(var)
    zz = z(1 - (1 - confidence) / 2)
    p_value = (1 - _N01.cdf((est - h0_rate) / se)) if se > 0 else (0.0 if est > h0_rate else 1.0)
    return {"estimate": est, "se": se, "ci_low": max(0.0, est - zz * se),
            "ci_high": min(1.0, est + zz * se), "p_value_vs_h0": p_value}

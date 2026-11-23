"""Demographic fidelity: did we get the cohort that was asked for?

Compares the requested mix with (a) the generated records and (b) the delivered records
(those that passed). Dropping failed records can shift the mix, which is exactly the kind
of hidden bias the Data Quality work in this project is about.
"""
from datetime import date

from src.testdata.spec import CONTROL_TIERS, LANGUAGES, NOT_RECORDED, CohortSpec
from src.testdata.stats import goodness_of_fit, wilson_ci


def _shares(values: list) -> dict:
    n = len(values)
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return {k: c / n for k, c in out.items()} if n else {}


def _dimension(name, requested: dict, generated: list, delivered: list, label=lambda k: k) -> dict:
    total = sum(requested.values()) or 1
    req = {k: v / total for k, v in requested.items()}
    gen, dlv = _shares(generated), _shares(delivered)
    counts_dlv = {k: delivered.count(k) for k in req}
    rows = []
    for k in req:
        lo, hi = wilson_ci(counts_dlv[k], len(delivered))
        rows.append({
            "dimension": name, "category": label(k),
            "requested_pct": round(100 * req[k], 1),
            "generated_pct": round(100 * gen.get(k, 0), 1),
            "delivered_pct": round(100 * dlv.get(k, 0), 1),
            "delivered_ci_low": round(100 * lo, 1), "delivered_ci_high": round(100 * hi, 1),
            "requested_inside_ci": lo <= req[k] <= hi,
        })
    tvd = 0.5 * sum(abs(req.get(k, 0) - dlv.get(k, 0)) for k in set(req) | set(dlv))
    return {"rows": rows, "tvd_delivered": round(tvd, 4),
            "chi2_p_delivered": goodness_of_fit(counts_dlv, req)}


def compare(spec: CohortSpec, truth: list[dict], passed: list[bool]) -> dict:
    delivered = [t for t, ok in zip(truth, passed) if ok]

    def col(key, rows):
        return [r[key] for r in rows]

    dims = {
        "Race / ethnicity": _dimension("Race / ethnicity", spec.race_mix, col("race", truth), col("race", delivered)),
        "Sex": _dimension("Sex", {"female": spec.pct_female, "male": 100 - spec.pct_female},
                          col("sex", truth), col("sex", delivered), str.capitalize),
        "Preferred language": _dimension("Preferred language", spec.language_mix, col("language", truth),
                                         col("language", delivered),
                                         lambda k: LANGUAGES[k] or NOT_RECORDED),
        "State": _dimension("State", {s: 1 for s in spec.states}, col("state", truth), col("state", delivered)),
        "Diabetes control": _dimension("Diabetes control", spec.control_mix, col("control", truth),
                                       col("control", delivered), lambda k: CONTROL_TIERS[k][0]),
    }
    return {"dimensions": dims, "n_generated": len(truth), "n_delivered": len(delivered)}


def age_summary(bundles: list[dict], as_of: date) -> list[int]:
    ages = []
    for b in bundles:
        p = next((e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Patient"), {})
        try:
            bd = date.fromisoformat(p["birthDate"])
            ages.append(as_of.year - bd.year - ((as_of.month, as_of.day) < (bd.month, bd.day)))
        except (KeyError, ValueError):
            pass
    return ages

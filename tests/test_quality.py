import math

import pandas as pd

from src.quality import score_dataframe, score_record


def test_clean_record_scores_100():
    assert score_record("Asian", 27.0, 7.2) == {"score": 100, "issues": []}


def test_missing_race_and_bmi():
    result = score_record(None, math.nan, 7.2)
    assert result["score"] == 50
    assert set(result["issues"]) == {"missing_race", "missing_bmi"}


def test_blank_race_counts_as_missing():
    assert score_record("  ", 27.0, 7.2)["score"] == 75


def test_out_of_bounds_hba1c():
    result = score_record("White", 27.0, 99.0)
    assert result["score"] == 50
    assert result["issues"] == ["out_of_bounds_hba1c"]


def test_score_floors_at_zero():
    assert score_record(None, 200.0, 99.0)["score"] == 0


def test_score_dataframe_adds_column():
    df = pd.DataFrame({"race_demographic": ["Asian", None],
                       "bmi": [25.0, None], "baseline_hba1c": [6.0, 6.0]})
    assert score_dataframe(df)["data_quality_index"].tolist() == [100, 50]

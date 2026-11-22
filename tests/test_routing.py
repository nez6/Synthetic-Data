from src.routing import recommend_provider


def test_severe_hba1c_routes_to_endocrinology():
    assert "Endocrinology" in recommend_provider(10.1, 25.0, "Monotherapy")["specialty"]


def test_combination_therapy_routes_to_internal_medicine():
    assert "Internal Medicine" in recommend_provider(8.0, 25.0, "Combination Oral")["specialty"]


def test_high_bmi_routes_to_internal_medicine():
    assert "Internal Medicine" in recommend_provider(6.5, 34.0, "Monotherapy")["specialty"]


def test_default_routes_to_primary_care():
    assert "Primary Care" in recommend_provider(6.5, 25.0, "Monotherapy")["specialty"]

"""End-to-end smoke tests for the ingestion pipeline."""

from __future__ import annotations

from src.ingestion.clean import CHURN_PROCESSED_SCHEMA, clean
from src.ingestion.loader import load_raw, write_raw
from src.ingestion.synthetic import generate_churn_data


def test_synthetic_round_trip_through_csv(tmp_path):
    """Generate -> write -> read -> clean. All schemas must validate."""
    raw = generate_churn_data(n_rows=200, seed=0)

    raw_csv = tmp_path / "raw.csv"
    write_raw(raw, raw_csv)

    raw_loaded = load_raw(raw_csv)
    assert len(raw_loaded) == 200

    processed = clean(raw_loaded)
    assert len(processed) == 200
    CHURN_PROCESSED_SCHEMA.validate(processed)


def test_clean_handles_blank_total_charges():
    """tenure=0 rows must come out with total_charges == 0.0."""
    raw = generate_churn_data(n_rows=500, seed=0)

    # synthetic data should already include some tenure=0 rows; if not, force one
    if (raw["tenure"] == 0).sum() == 0:
        raw.loc[raw.index[0], "tenure"] = 0
        raw.loc[raw.index[0], "TotalCharges"] = ""

    processed = clean(raw)
    zero_tenure_mask = raw["tenure"].to_numpy() == 0
    assert (processed.loc[zero_tenure_mask, "total_charges"] == 0.0).all()


def test_clean_target_is_bool():
    raw = generate_churn_data(n_rows=50, seed=0)
    processed = clean(raw)
    assert processed["churn"].dtype == bool


def test_clean_no_internet_service_collapses_to_false():
    """Customers with InternetService='No' should have all internet sub-services as False."""
    raw = generate_churn_data(n_rows=500, seed=0)
    processed = clean(raw)
    no_internet = raw["InternetService"].to_numpy() == "No"
    if no_internet.sum() > 0:
        for col in (
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
        ):
            assert (~processed.loc[no_internet, col]).all(), f"{col} should be False"

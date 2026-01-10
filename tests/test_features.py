"""Tests for feature engineering."""

from __future__ import annotations

import pandera.errors as pae
import pytest

from src.features.build import build_features
from src.features.schema import CHURN_FEATURES_SCHEMA, TENURE_BUCKETS
from src.ingestion.clean import clean
from src.ingestion.synthetic import generate_churn_data


def _fresh_processed(n_rows: int = 200, seed: int = 0):
    return clean(generate_churn_data(n_rows=n_rows, seed=seed))


def test_features_validate_against_schema():
    features = build_features(_fresh_processed())
    CHURN_FEATURES_SCHEMA.validate(features)


def test_services_count_in_zero_to_six():
    features = build_features(_fresh_processed())
    assert (features["services_count"] >= 0).all()
    assert (features["services_count"] <= 6).all()


def test_tenure_buckets_cover_all_rows():
    features = build_features(_fresh_processed())
    assert features["tenure_bucket"].isin(TENURE_BUCKETS).all()


def test_avg_monthly_spend_zero_for_zero_tenure():
    """Customers with tenure=0 have total_charges=0 → avg_monthly_spend == 0."""
    processed = _fresh_processed(n_rows=500)
    if (processed["tenure"] == 0).sum() == 0:
        processed.loc[processed.index[0], "tenure"] = 0
        processed.loc[processed.index[0], "total_charges"] = 0.0
    features = build_features(processed)
    zero_tenure = processed["tenure"] == 0
    assert (features.loc[zero_tenure, "avg_monthly_spend"] == 0.0).all()


def test_is_new_customer_matches_tenure_le_3():
    features = build_features(_fresh_processed())
    assert (features["is_new_customer"] == (features["tenure"] <= 3)).all()


def test_features_rejects_unprocessed_input():
    """build_features must fail if handed raw (un-cleaned) data."""
    raw = generate_churn_data(n_rows=10, seed=0)
    with pytest.raises((pae.SchemaError, pae.SchemaErrors, KeyError)):
        build_features(raw)


def test_monthly_charges_per_service_no_division_by_zero():
    """Customers with zero services should still produce a finite value."""
    features = build_features(_fresh_processed(n_rows=500))
    assert features["monthly_charges_per_service"].notna().all()
    assert (features["monthly_charges_per_service"] > 0).all()

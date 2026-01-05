"""Tests for the raw churn data schema.

Each test exercises a different way upstream data can break the contract:
extra columns, missing columns, invalid categorical, out-of-range numeric.
These are the four most common production data failure modes.
"""

from __future__ import annotations

import pandera.errors as pae
import pytest

from src.ingestion.schema import CHURN_RAW_SCHEMA
from src.ingestion.synthetic import generate_churn_data


def test_synthetic_data_validates_against_schema():
    df = generate_churn_data(n_rows=200, seed=0)
    validated = CHURN_RAW_SCHEMA.validate(df)
    assert len(validated) == 200


def test_schema_rejects_unknown_column():
    df = generate_churn_data(n_rows=10, seed=0)
    df["surprise_new_column"] = 1
    with pytest.raises((pae.SchemaError, pae.SchemaErrors)):
        CHURN_RAW_SCHEMA.validate(df)


def test_schema_rejects_missing_column():
    df = generate_churn_data(n_rows=10, seed=0).drop(columns=["Churn"])
    with pytest.raises((pae.SchemaError, pae.SchemaErrors)):
        CHURN_RAW_SCHEMA.validate(df)


def test_schema_rejects_invalid_categorical():
    df = generate_churn_data(n_rows=10, seed=0)
    df.loc[0, "Contract"] = "Lifetime"
    with pytest.raises((pae.SchemaError, pae.SchemaErrors)):
        CHURN_RAW_SCHEMA.validate(df)


def test_schema_rejects_negative_monthly_charges():
    df = generate_churn_data(n_rows=10, seed=0)
    df.loc[0, "MonthlyCharges"] = -5.0
    with pytest.raises((pae.SchemaError, pae.SchemaErrors)):
        CHURN_RAW_SCHEMA.validate(df)


def test_schema_rejects_duplicate_customer_id():
    df = generate_churn_data(n_rows=10, seed=0)
    df.loc[1, "customerID"] = df.loc[0, "customerID"]
    with pytest.raises((pae.SchemaError, pae.SchemaErrors)):
        CHURN_RAW_SCHEMA.validate(df)


def test_synthetic_total_charges_blank_for_zero_tenure():
    """Real Telco data has '' for new customers; synthetic must reproduce that."""
    df = generate_churn_data(n_rows=500, seed=0)
    zero_tenure = df[df["tenure"] == 0]
    if len(zero_tenure) > 0:
        assert (zero_tenure["TotalCharges"] == "").all()


def test_synthetic_churn_rate_is_realistic():
    """Sanity: churn rate should land in the 15-40% band like the real dataset."""
    df = generate_churn_data(n_rows=2000, seed=0)
    churn_rate = (df["Churn"] == "Yes").mean()
    assert 0.15 <= churn_rate <= 0.40, f"churn rate {churn_rate:.2%} out of band"

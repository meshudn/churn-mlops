"""Schema for the post-feature-engineering dataset.

Extends the processed schema with derived signal columns. Anything downstream
of feature engineering (training, evaluation, serving) should depend on this
schema, not on the processed schema — that's how a feature store boundary
is enforced.
"""

from __future__ import annotations

import pandera.pandas as pa

from src.ingestion.clean import CHURN_PROCESSED_SCHEMA

TENURE_BUCKETS = ["0-12", "13-24", "25-48", "49-72"]

# add_columns returns a NEW schema with both the processed columns and the
# derived ones. Composing schemas this way keeps the processed contract as
# the single source of truth for upstream-facing columns.
CHURN_FEATURES_SCHEMA = CHURN_PROCESSED_SCHEMA.add_columns(
    {
        "services_count": pa.Column(
            int,
            [
                pa.Check.greater_than_or_equal_to(0),
                pa.Check.less_than_or_equal_to(6),
            ],
        ),
        "tenure_bucket": pa.Column(str, pa.Check.isin(TENURE_BUCKETS)),
        "is_new_customer": pa.Column(bool),
        "avg_monthly_spend": pa.Column(float, pa.Check.greater_than_or_equal_to(0)),
        "monthly_charges_per_service": pa.Column(float, pa.Check.greater_than(0)),
    }
)


__all__ = ["CHURN_FEATURES_SCHEMA", "TENURE_BUCKETS"]

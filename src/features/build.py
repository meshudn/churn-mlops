"""Feature engineering: derive new signals from the processed dataset.

The features are picked for *business meaning*, not statistical magic:

- ``tenure_bucket``: discretizes tenure into onboarding / early / mid / mature
  bands. Models often pick up non-linear effects better via explicit bins.
- ``services_count``: how many add-on services the customer pays for. More
  services typically means more entanglement with the brand → lower churn.
- ``is_new_customer``: tenure ≤ 3 months. New-customer churn dynamics differ
  from established-customer dynamics; the explicit flag helps the model.
- ``avg_monthly_spend``: realized average. Differs from ``monthly_charges``
  when the customer recently changed plans — captures price-tier drift.
- ``monthly_charges_per_service``: pricing efficiency. Customers paying a lot
  per service may be primed to leave for a competitor.

Encoding (one-hot, scaling, target encoding) is intentionally **not** done
here. Doing it now would couple feature storage to a specific model. Instead,
encoding lives in the training-time sklearn ``Pipeline`` so the persisted
model object knows how to encode at serving time, eliminating train/serve skew.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.ingestion.clean import CHURN_PROCESSED_SCHEMA, load_processed

from .schema import CHURN_FEATURES_SCHEMA, TENURE_BUCKETS

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = _REPO_ROOT / "data" / "processed" / "churn.csv"
DEFAULT_OUTPUT = _REPO_ROOT / "data" / "features" / "churn_features.csv"

_SERVICE_FLAGS = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

# Right-closed bins matching TENURE_BUCKETS. -0.5 lets tenure=0 fall in the first bin.
_TENURE_BIN_EDGES = [-0.5, 12, 24, 48, 72]


def _bucket_tenure(tenure: pd.Series) -> pd.Series:
    return pd.cut(tenure, bins=_TENURE_BIN_EDGES, labels=TENURE_BUCKETS).astype(str)


def build_features(processed: pd.DataFrame) -> pd.DataFrame:
    """Derive features from a validated processed frame.

    Input must satisfy :data:`src.ingestion.clean.CHURN_PROCESSED_SCHEMA`.
    Output is validated against :data:`CHURN_FEATURES_SCHEMA`.
    """
    CHURN_PROCESSED_SCHEMA.validate(processed)
    df = processed.copy()

    df["services_count"] = df[_SERVICE_FLAGS].sum(axis=1).astype(int)
    df["tenure_bucket"] = _bucket_tenure(df["tenure"])
    df["is_new_customer"] = df["tenure"] <= 3

    df["avg_monthly_spend"] = (df["total_charges"] / df["tenure"].clip(lower=1)).round(2)
    df["monthly_charges_per_service"] = (
        df["monthly_charges"] / df["services_count"].clip(lower=1)
    ).round(2)

    return CHURN_FEATURES_SCHEMA.validate(df)


def main() -> int:
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[featurize] reading  {DEFAULT_INPUT}")
    processed = load_processed(DEFAULT_INPUT)

    features = build_features(processed)
    features.to_csv(DEFAULT_OUTPUT, index=False)
    print(
        f"[featurize] wrote    {DEFAULT_OUTPUT}  "
        f"({len(features)} rows, {len(features.columns)} cols)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Load engineered features and produce a stratified train/test split.

The split is stratified on the target so the train and test sets share the
same churn rate — important when the positive class is the rarer one
(~16-26% churn in this dataset).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.schema import CHURN_FEATURES_SCHEMA
from src.ingestion.clean import PROCESSED_BOOL_COLUMNS

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = _REPO_ROOT / "data" / "features" / "churn_features.csv"

# Bool columns to coerce after CSV round-trip: the processed bools + is_new_customer.
FEATURE_BOOL_COLUMNS = PROCESSED_BOOL_COLUMNS + ["is_new_customer"]

TARGET = "churn"


def load_features(path: str | Path = DEFAULT_FEATURES) -> pd.DataFrame:
    """Load features CSV and restore bool dtypes lost in CSV serialization."""
    df = pd.read_csv(path)
    for col in FEATURE_BOOL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({"True": True, "False": False}).astype(bool)
    return CHURN_FEATURES_SCHEMA.validate(df)


def train_test_features(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    path: str | Path = DEFAULT_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load features, drop the target column from X, return stratified split."""
    df = load_features(path)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


__all__ = ["load_features", "train_test_features", "DEFAULT_FEATURES", "TARGET"]

"""sklearn Pipeline factory.

The Pipeline wraps a ColumnTransformer (encoding) and a model. Persisting the
fitted pipeline persists the fitted encoder, so serving-time encoding uses
literally the same Python objects as training-time. This eliminates train/serve
skew by construction — see docs/03 section 6.

Adding a new model kind:
1. Append to ``ModelKind``.
2. Add a branch in ``_make_estimator``.
3. The Pipeline construction is shared — no other code needs to change.
"""

from __future__ import annotations

from typing import Literal

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLS = [
    "gender",
    "internet_service",
    "contract",
    "payment_method",
    "tenure_bucket",
]

NUMERIC_COLS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "services_count",
    "avg_monthly_spend",
    "monthly_charges_per_service",
]

# Bools pass through — already 0/1 once pandas casts to int internally.
BOOL_COLS = [
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "paperless_billing",
    "is_new_customer",
]

ModelKind = Literal["logreg", "random_forest"]
ALL_MODEL_KINDS: tuple[ModelKind, ...] = ("logreg", "random_forest")


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            ("scale", StandardScaler(), NUMERIC_COLS),
            ("bool_pass", "passthrough", BOOL_COLS),
        ],
        remainder="drop",
    )


def _make_estimator(kind: ModelKind, random_state: int = 42) -> BaseEstimator:
    if kind == "logreg":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"unknown model kind: {kind}")


def make_pipeline(kind: ModelKind = "logreg", *, random_state: int = 42) -> Pipeline:
    """Build a fresh, unfitted Pipeline of preprocess + model."""
    return Pipeline(
        [
            ("preprocess", _make_preprocessor()),
            ("model", _make_estimator(kind, random_state=random_state)),
        ]
    )


__all__ = [
    "make_pipeline",
    "ModelKind",
    "ALL_MODEL_KINDS",
    "CATEGORICAL_COLS",
    "NUMERIC_COLS",
    "BOOL_COLS",
]

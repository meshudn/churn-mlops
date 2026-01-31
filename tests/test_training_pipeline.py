"""Tests for the sklearn training Pipeline factory."""

from __future__ import annotations

import pytest

from src.features.build import build_features
from src.ingestion.clean import clean
from src.ingestion.synthetic import generate_churn_data
from src.training.pipeline import (
    ALL_MODEL_KINDS,
    BOOL_COLS,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    make_pipeline,
)


def _features_X_y(n_rows: int = 200, seed: int = 0):
    df = build_features(clean(generate_churn_data(n_rows=n_rows, seed=seed)))
    return df.drop(columns=["churn"]), df["churn"]


@pytest.mark.parametrize("kind", ALL_MODEL_KINDS)
def test_pipeline_can_fit_and_predict(kind):
    X, y = _features_X_y()
    pipe = make_pipeline(kind)
    pipe.fit(X, y)
    pred = pipe.predict(X.head(10))
    assert len(pred) == 10
    proba = pipe.predict_proba(X.head(10))
    assert proba.shape == (10, 2)
    # probabilities sum to 1
    assert (abs(proba.sum(axis=1) - 1.0) < 1e-6).all()


def test_pipeline_columns_cover_features():
    """Every feature column should be claimed by exactly one transformer."""
    X, _ = _features_X_y()
    claimed = set(CATEGORICAL_COLS + NUMERIC_COLS + BOOL_COLS)
    columns = set(X.columns)
    missing_from_pipeline = columns - claimed
    referenced_but_absent = claimed - columns
    assert not missing_from_pipeline, (
        f"feature columns not handled by any transformer: {missing_from_pipeline}"
    )
    assert not referenced_but_absent, (
        f"transformer references nonexistent columns: {referenced_but_absent}"
    )


def test_make_pipeline_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_pipeline("xgboost")  # type: ignore[arg-type]

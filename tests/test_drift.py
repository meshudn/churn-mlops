"""Tests for drift detection."""

from __future__ import annotations

import pandas as pd

from src.features.build import build_features
from src.ingestion.clean import clean
from src.ingestion.synthetic import generate_churn_data
from src.monitoring.drift import compute_drift, generate_drifted_current


def _features_no_target(seed: int, n_rows: int = 500) -> pd.DataFrame:
    df = build_features(clean(generate_churn_data(n_rows=n_rows, seed=seed)))
    return df.drop(columns=["churn"])


def test_no_drift_when_same_seed():
    """Identical samples should not drift."""
    ref = _features_no_target(seed=0)
    cur = _features_no_target(seed=0)
    result = compute_drift(ref, cur)
    # Drift share with identical data should be ~0
    assert result.overall["drift_share"] == 0.0
    assert result.overall["dataset_drifted"] is False


def test_drift_detected_when_distributions_differ():
    """A deliberately shifted current batch should trip drift detection."""
    ref = _features_no_target(seed=0, n_rows=1000)
    cur = generate_drifted_current(n_rows=1000, seed=7).drop(columns=["churn"])
    result = compute_drift(ref, cur)
    # We forced fiber + monthly_charges to shift; one of those should drift
    drifted_cols = {c["column"] for c in result.columns if c["drifted"]}
    assert (
        "internet_service" in drifted_cols
        or "monthly_charges" in drifted_cols
    ), f"expected drift in internet_service or monthly_charges, got: {drifted_cols}"


def test_drift_result_shape():
    ref = _features_no_target(seed=0)
    cur = _features_no_target(seed=0)
    result = compute_drift(ref, cur)

    assert isinstance(result.overall, dict)
    for key in ("drifted_columns", "drift_share", "drift_share_threshold", "dataset_drifted"):
        assert key in result.overall

    assert isinstance(result.columns, list)
    assert len(result.columns) > 0
    for c in result.columns:
        assert "column" in c
        assert "method" in c
        assert "score" in c
        assert "threshold" in c
        assert "drifted" in c


def test_drift_columns_sorted_most_drifted_first():
    """Drifted columns should land at the top regardless of which method Evidently picks."""
    ref = _features_no_target(seed=0, n_rows=1500)
    cur = generate_drifted_current(n_rows=1500, seed=7).drop(columns=["churn"])
    result = compute_drift(ref, cur)
    drifted = [c for c in result.columns if c["drifted"]]
    not_drifted = [c for c in result.columns if not c["drifted"]]
    if drifted and not_drifted:
        first_drifted_idx = result.columns.index(drifted[0])
        first_clean_idx = result.columns.index(not_drifted[0])
        assert first_drifted_idx < first_clean_idx, (
            "drifted columns should sort before clean ones"
        )


def test_dataset_drifted_threshold():
    """Lowering threshold should make the same drift register as 'dataset_drifted'."""
    ref = _features_no_target(seed=0, n_rows=600)
    cur = generate_drifted_current(n_rows=600, seed=7).drop(columns=["churn"])
    strict = compute_drift(ref, cur, drift_share_threshold=0.05)
    lenient = compute_drift(ref, cur, drift_share_threshold=0.95)
    # If the dataset truly drifted, strict will say yes and lenient will say no.
    assert strict.overall["dataset_drifted"] >= lenient.overall["dataset_drifted"]

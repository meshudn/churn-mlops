"""Smoke tests for train_one — fits each model kind, asserts MLflow logging works."""

from __future__ import annotations

import mlflow
import pytest

from src.features.build import build_features
from src.ingestion.clean import clean
from src.ingestion.synthetic import generate_churn_data
from src.training.pipeline import ALL_MODEL_KINDS
from src.training.train import train_one


def _stratified_halves(n_rows: int = 200, seed: int = 0):
    df = build_features(clean(generate_churn_data(n_rows=n_rows, seed=seed)))
    y = df["churn"]
    X = df.drop(columns=["churn"])
    # crude 50/50 split is fine for a smoke test
    half = len(df) // 2
    return X.iloc[:half], X.iloc[half:], y.iloc[:half], y.iloc[half:]


@pytest.mark.parametrize("kind", ALL_MODEL_KINDS)
def test_train_one_logs_run_and_returns_metrics(tmp_path, kind, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    mlflow.set_experiment("test-train")

    X_train, X_test, y_train, y_test = _stratified_halves()
    result = train_one(kind, X_train, X_test, y_train, y_test)

    assert result["run_id"], "run_id should be populated"
    assert result["model_kind"] == kind

    metrics = result["metrics"]
    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert set(metrics.keys()) == expected_keys
    for v in metrics.values():
        assert 0.0 <= v <= 1.0

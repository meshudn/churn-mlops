"""Tests for the compare-then-promote decision logic.

These run against a real (file-based) MLflow tracking URI in a tmp dir so we
exercise the same Registry code paths the production script uses. Logging a
tiny sklearn model per fake "run" keeps the test fast (~1s per case).
"""

from __future__ import annotations

import json

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation.promote import decide_promotion
from src.evaluation.registry import get_production


@pytest.fixture
def mlflow_tmp(tmp_path, monkeypatch):
    """Tracking URI + working dir pointed at a tmp path."""
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    mlflow.set_experiment("test-promote")
    yield tmp_path


def _log_run_with_metric(metric_name: str, metric_value: float) -> str:
    with mlflow.start_run() as run:
        mlflow.log_metric(metric_name, metric_value)
        # log a real (tiny) sklearn model so the registry has something to register
        clf = LogisticRegression(max_iter=10).fit(
            np.array([[0, 0], [1, 1], [0, 1], [1, 0]]),
            np.array([0, 1, 0, 1]),
        )
        mlflow.sklearn.log_model(sk_model=clf, artifact_path="model")
        return run.info.run_id


def _write_metrics_file(tmp_path, runs: list[dict]) -> str:
    path = tmp_path / "train_metrics.json"
    with open(path, "w") as f:
        json.dump({"experiment": "test-promote", "runs": runs}, f)
    return str(path)


def test_bootstrap_first_run(mlflow_tmp):
    run_id = _log_run_with_metric("f1", 0.40)
    metrics_file = _write_metrics_file(
        mlflow_tmp,
        [{"run_id": run_id, "model_kind": "logreg", "metrics": {"f1": 0.40}}],
    )

    decision = decide_promotion(
        metrics_file, metric="f1", model_name="test-model", apply=True,
    )

    assert decision["decision"] == "bootstrap"
    assert decision["promoted_version"] == "1"
    assert decision["candidate"]["model_kind"] == "logreg"

    prod = get_production(model_name="test-model")
    assert prod is not None and prod.version == "1"


def test_better_candidate_promotes(mlflow_tmp):
    # establish a v1 production model with f1=0.30
    first_run = _log_run_with_metric("f1", 0.30)
    decide_promotion(
        _write_metrics_file(
            mlflow_tmp,
            [{"run_id": first_run, "model_kind": "logreg", "metrics": {"f1": 0.30}}],
        ),
        metric="f1",
        model_name="test-model",
        apply=True,
    )

    # candidate at f1=0.45 should win
    second_run = _log_run_with_metric("f1", 0.45)
    decision = decide_promotion(
        _write_metrics_file(
            mlflow_tmp,
            [{"run_id": second_run, "model_kind": "random_forest", "metrics": {"f1": 0.45}}],
        ),
        metric="f1",
        model_name="test-model",
        apply=True,
    )

    assert decision["decision"] == "promote"
    assert decision["promoted_version"] == "2"
    assert decision["candidate"]["model_kind"] == "random_forest"
    assert get_production(model_name="test-model").version == "2"


def test_worse_candidate_rejected(mlflow_tmp):
    first_run = _log_run_with_metric("f1", 0.45)
    decide_promotion(
        _write_metrics_file(
            mlflow_tmp,
            [{"run_id": first_run, "model_kind": "logreg", "metrics": {"f1": 0.45}}],
        ),
        metric="f1",
        model_name="test-model",
        apply=True,
    )

    # candidate at f1=0.30 should be rejected; production stays at v1
    second_run = _log_run_with_metric("f1", 0.30)
    decision = decide_promotion(
        _write_metrics_file(
            mlflow_tmp,
            [{"run_id": second_run, "model_kind": "logreg", "metrics": {"f1": 0.30}}],
        ),
        metric="f1",
        model_name="test-model",
        apply=True,
    )

    assert decision["decision"] == "reject"
    assert decision["promoted_version"] is None
    assert get_production(model_name="test-model").version == "1"


def test_equal_candidate_rejected(mlflow_tmp):
    """Strict >: equal metrics get rejected. The incumbent has the burden of staying."""
    first_run = _log_run_with_metric("f1", 0.40)
    decide_promotion(
        _write_metrics_file(
            mlflow_tmp,
            [{"run_id": first_run, "model_kind": "logreg", "metrics": {"f1": 0.40}}],
        ),
        metric="f1",
        model_name="test-model",
        apply=True,
    )

    second_run = _log_run_with_metric("f1", 0.40)
    decision = decide_promotion(
        _write_metrics_file(
            mlflow_tmp,
            [{"run_id": second_run, "model_kind": "logreg", "metrics": {"f1": 0.40}}],
        ),
        metric="f1",
        model_name="test-model",
        apply=True,
    )

    assert decision["decision"] == "reject"
    assert get_production(model_name="test-model").version == "1"


def test_picks_best_of_multiple_runs_in_round(mlflow_tmp):
    """Within a training round, the highest-metric run is the candidate."""
    rid_a = _log_run_with_metric("f1", 0.30)
    rid_b = _log_run_with_metric("f1", 0.50)  # this should be picked

    decision = decide_promotion(
        _write_metrics_file(
            mlflow_tmp,
            [
                {"run_id": rid_a, "model_kind": "logreg", "metrics": {"f1": 0.30}},
                {"run_id": rid_b, "model_kind": "random_forest", "metrics": {"f1": 0.50}},
            ],
        ),
        metric="f1",
        model_name="test-model",
        apply=True,
    )

    assert decision["candidate"]["run_id"] == rid_b
    assert decision["candidate"]["model_kind"] == "random_forest"

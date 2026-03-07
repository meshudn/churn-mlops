"""Tests for the inference API.

Strategy: stand up a real but tiny MLflow registry in a tmp dir, train a
small model, register + promote it, then drive the API through TestClient.
This exercises the same code path the production container would, so a
green test really does mean the wiring works.
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import pytest
from fastapi.testclient import TestClient

from src.deployment.api import create_app
from src.deployment.settings import APISettings
from src.evaluation.registry import register_run, set_production
from src.features.build import build_features
from src.ingestion.clean import clean
from src.ingestion.synthetic import generate_churn_data
from src.training.train import train_one

_VALID_RECORD = {
    "customerID": "1234-ABCDE",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.50,
    "TotalCharges": "1572.00",
}


def _train_and_register(model_name: str = "test-churn-model") -> None:
    """Train a logreg + register + promote to `production` in the active tracking URI."""
    mlflow.set_experiment("test-api")

    df = build_features(clean(generate_churn_data(n_rows=400, seed=0)))
    y = df["churn"]
    X = df.drop(columns=["churn"])
    half = len(df) // 2
    result = train_one(
        "logreg",
        X.iloc[:half],
        X.iloc[half:],
        y.iloc[:half],
        y.iloc[half:],
    )
    rv = register_run(result["run_id"], model_name=model_name)
    set_production(rv.version, model_name=model_name)


@pytest.fixture
def loaded_client(tmp_path, monkeypatch):
    """TestClient with model loaded from a real tmp registry."""
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    _train_and_register("test-churn-model")

    settings = APISettings(
        model_name="test-churn-model",
        mlflow_tracking_uri=tracking_uri,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def empty_client(tmp_path, monkeypatch):
    """TestClient against an empty registry — model load should fail gracefully."""
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"file://{tmp_path}/mlruns"
    settings = APISettings(
        model_name="nonexistent-model",
        mlflow_tracking_uri=tracking_uri,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


def test_health_when_model_loaded(loaded_client):
    r = loaded_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "1"


def test_health_when_model_missing(empty_client):
    r = empty_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["model_version"] is None


def test_info_when_model_loaded(loaded_client):
    r = loaded_client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "test-churn-model"
    assert body["model_version"] == "1"
    assert body["threshold"] == 0.5


def test_info_when_model_missing_returns_503(empty_client):
    r = empty_client.get("/info")
    assert r.status_code == 503


def test_predict_returns_probability_and_decision(loaded_client):
    r = loaded_client.post("/predict", json=_VALID_RECORD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["customer_id"] == "1234-ABCDE"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["predicted_churn"] in (True, False)
    assert body["model_version"] == "1"
    # decision is consistent with threshold
    above = body["churn_probability"] >= body["threshold"]
    assert body["predicted_churn"] is above


def test_predict_rejects_unknown_field(loaded_client):
    bad = dict(_VALID_RECORD, surprise_field="oops")
    r = loaded_client.post("/predict", json=bad)
    assert r.status_code == 422  # pydantic validation


def test_predict_rejects_invalid_categorical(loaded_client):
    bad = dict(_VALID_RECORD, Contract="Lifetime")
    r = loaded_client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_zero_monthly_charges(loaded_client):
    """Field has gt=0; zero or negative values should be rejected at the API."""
    bad = dict(_VALID_RECORD, MonthlyCharges=0)
    r = loaded_client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_when_model_missing_returns_503(empty_client):
    r = empty_client.post("/predict", json=_VALID_RECORD)
    assert r.status_code == 503


def test_predict_batch(loaded_client):
    records = [_VALID_RECORD, dict(_VALID_RECORD, customerID="9999-ZZZZZ", tenure=60)]
    r = loaded_client.post("/predict/batch", json={"records": records})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["customer_id"] == "1234-ABCDE"
    assert body["predictions"][1]["customer_id"] == "9999-ZZZZZ"

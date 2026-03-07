"""FastAPI inference service for churn prediction.

Endpoints:
- ``GET /health``        — liveness probe; reports whether a model is loaded.
- ``GET /info``          — currently-served model name, version, run id.
- ``POST /predict``      — single customer record -> churn probability + decision.
- ``POST /predict/batch``— list of records, processed as one DataFrame for efficiency.

Model loading happens once at startup via FastAPI's ``lifespan``. Restarting
the service is what picks up a new ``production`` alias — see Phase 5's
docs/05 §4 on aliases-as-deploy-pointers.

The persisted ``Pipeline`` object includes its fitted encoder, so the
preprocessing here is identical to training. No translation layer at the
API boundary; therefore no train/serve skew.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.features.build import build_features
from src.ingestion.clean import clean

from .model_loader import LoadedModel, load_production_model
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ChurnRecord,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from .settings import APISettings


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings: APISettings = app.state.settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    try:
        loaded = load_production_model(
            settings.model_name, alias=settings.model_alias,
        )
        print(
            f"[api] loaded {loaded.name} v{loaded.version} "
            f"(run {loaded.run_id[:8]}) from {settings.mlflow_tracking_uri}"
        )
    except Exception as e:
        # Start in degraded mode rather than failing — health endpoint will
        # report model_loaded=False, /predict will 503. Lets the liveness
        # probe distinguish "container is running but unconfigured" from
        # "container died." Common pattern for ops simplicity.
        print(f"[api] WARNING: model load failed: {e}")
        loaded = None

    app.state.model = loaded
    yield


def create_app(settings: APISettings | None = None) -> FastAPI:
    app = FastAPI(
        title="churn-mlops inference",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.settings = settings or APISettings()
    app.state.model = None  # populated by lifespan

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        loaded: LoadedModel | None = app.state.model
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded is not None,
            model_name=app.state.settings.model_name,
            model_version=loaded.version if loaded else None,
        )

    @app.get("/info", response_model=ModelInfoResponse)
    def info() -> ModelInfoResponse:
        loaded = _require_model(app)
        s: APISettings = app.state.settings
        return ModelInfoResponse(
            model_name=s.model_name,
            model_version=loaded.version,
            model_run_id=loaded.run_id,
            threshold=s.threshold,
            tracking_uri=s.mlflow_tracking_uri,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(record: ChurnRecord) -> PredictionResponse:
        loaded = _require_model(app)
        s: APISettings = app.state.settings
        return _predict_records([record], loaded, s)[0]

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(req: BatchPredictionRequest) -> BatchPredictionResponse:
        loaded = _require_model(app)
        s: APISettings = app.state.settings
        results = _predict_records(req.records, loaded, s)
        return BatchPredictionResponse(predictions=results, count=len(results))

    return app


# ─── internals ────────────────────────────────────────────────────────────


def _require_model(app: FastAPI) -> LoadedModel:
    loaded: LoadedModel | None = app.state.model
    if loaded is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not loaded; check /health for details",
        )
    return loaded


def _features_for_records(records: Iterable[ChurnRecord]) -> pd.DataFrame:
    """Run raw -> clean -> features on a list of customer records.

    The ``Churn`` column is required by the raw schema but unused by the
    feature transformations; we inject a placeholder ``"No"`` so validation
    passes, and drop the resulting ``churn`` column before predict-time.
    """
    rows = []
    for r in records:
        d = r.model_dump()
        d["Churn"] = "No"
        rows.append(d)
    raw_df = pd.DataFrame(rows)
    processed = clean(raw_df)
    features = build_features(processed)
    return features.drop(columns=["churn"])


def _predict_records(
    records: list[ChurnRecord],
    loaded: LoadedModel,
    s: APISettings,
) -> list[PredictionResponse]:
    features = _features_for_records(records)
    probs = loaded.pipeline.predict_proba(features)[:, 1]
    return [
        PredictionResponse(
            customer_id=record.customerID,
            churn_probability=float(p),
            predicted_churn=bool(p >= s.threshold),
            threshold=s.threshold,
            model_name=loaded.name,
            model_version=loaded.version,
        )
        for record, p in zip(records, probs, strict=True)
    ]


# Default app for `uvicorn src.deployment.api:app`
app = create_app()

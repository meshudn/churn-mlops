"""Pydantic models for the inference API.

Field names mirror the upstream Telco schema verbatim (``customerID``,
``MonthlyCharges``, ``SeniorCitizen``). Translating to snake_case here would
require a translation layer at the API boundary — a famous source of
train/serve skew. We keep one set of names from upstream all the way to the
sklearn Pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChurnRecord(BaseModel):
    """One customer record (raw Telco format), no target."""

    model_config = ConfigDict(extra="forbid")

    customerID: str
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(gt=0)
    # Telco quirk: string with "" for new customers (tenure=0). We keep the
    # API surface honest about this rather than making the caller guess.
    TotalCharges: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    customer_id: str
    churn_probability: float = Field(ge=0.0, le=1.0)
    predicted_churn: bool
    threshold: float
    model_name: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    records: list[ChurnRecord]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str
    model_version: str | None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    model_run_id: str
    threshold: float
    tracking_uri: str


__all__ = [
    "ChurnRecord",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "HealthResponse",
    "ModelInfoResponse",
]

"""Pandera schema for raw Telco-style customer churn data.

This schema models the data *exactly as it arrives from upstream* — including
known quirks. Most notably, ``TotalCharges`` is delivered as a string with
empty values for brand-new customers (tenure = 0); this is a famous data
quality trap in the real Telco dataset and we keep it in the raw schema on
purpose. Cleaning up that column happens in :mod:`src.ingestion.clean`.

The split between *raw schema* and *processed schema* is a core MLOps idea:
- raw schema = the contract with whoever produces the data (it documents what
  ugliness we accept and forces upstream changes to fail loudly).
- processed schema (added later) = the contract with whoever consumes the data
  for training and inference.
"""

from __future__ import annotations

import pandera.pandas as pa

YES_NO = ["Yes", "No"]
INTERNET_SCOPED = ["Yes", "No", "No internet service"]
PHONE_SCOPED = ["Yes", "No", "No phone service"]
INTERNET_SERVICE = ["DSL", "Fiber optic", "No"]
CONTRACT = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHOD = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


CHURN_RAW_SCHEMA = pa.DataFrameSchema(
    columns={
        "customerID": pa.Column(str, unique=True, nullable=False),
        "gender": pa.Column(str, pa.Check.isin(["Male", "Female"])),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1])),
        "Partner": pa.Column(str, pa.Check.isin(YES_NO)),
        "Dependents": pa.Column(str, pa.Check.isin(YES_NO)),
        "tenure": pa.Column(
            int,
            [
                pa.Check.greater_than_or_equal_to(0),
                pa.Check.less_than_or_equal_to(72),
            ],
        ),
        "PhoneService": pa.Column(str, pa.Check.isin(YES_NO)),
        "MultipleLines": pa.Column(str, pa.Check.isin(PHONE_SCOPED)),
        "InternetService": pa.Column(str, pa.Check.isin(INTERNET_SERVICE)),
        "OnlineSecurity": pa.Column(str, pa.Check.isin(INTERNET_SCOPED)),
        "OnlineBackup": pa.Column(str, pa.Check.isin(INTERNET_SCOPED)),
        "DeviceProtection": pa.Column(str, pa.Check.isin(INTERNET_SCOPED)),
        "TechSupport": pa.Column(str, pa.Check.isin(INTERNET_SCOPED)),
        "StreamingTV": pa.Column(str, pa.Check.isin(INTERNET_SCOPED)),
        "StreamingMovies": pa.Column(str, pa.Check.isin(INTERNET_SCOPED)),
        "Contract": pa.Column(str, pa.Check.isin(CONTRACT)),
        "PaperlessBilling": pa.Column(str, pa.Check.isin(YES_NO)),
        "PaymentMethod": pa.Column(str, pa.Check.isin(PAYMENT_METHOD)),
        "MonthlyCharges": pa.Column(float, pa.Check.greater_than(0)),
        # Intentionally str: in the real upstream CSV this column is delivered
        # as a string and is empty ("") for customers with tenure = 0.
        "TotalCharges": pa.Column(str, nullable=False),
        "Churn": pa.Column(str, pa.Check.isin(YES_NO)),
    },
    strict=True,
    coerce=False,
    ordered=False,
)

__all__ = [
    "CHURN_RAW_SCHEMA",
    "YES_NO",
    "INTERNET_SCOPED",
    "PHONE_SCOPED",
    "INTERNET_SERVICE",
    "CONTRACT",
    "PAYMENT_METHOD",
]

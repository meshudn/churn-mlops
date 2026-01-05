"""Synthetic Telco-style churn data generator.

Produces a DataFrame conforming to :data:`src.ingestion.schema.CHURN_RAW_SCHEMA`
with marginal distributions and a target churn rate (~26%) calibrated to the
real public Telco Customer Churn dataset. The generator is deterministic given
``seed``.

Why synthetic? It keeps the project self-contained (no flaky external URLs
or licensing concerns) while remaining swappable: the loader accepts any CSV
that satisfies the same schema, so the real dataset is a one-flag substitute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CONTRACT, INTERNET_SERVICE, PAYMENT_METHOD


def _generate_unique_ids(rng: np.random.Generator, n: int) -> list[str]:
    """Telco-style customer IDs like '7590-VHVEG', guaranteed unique."""
    seen: set[str] = set()
    ids: list[str] = []
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    while len(ids) < n:
        digits = int(rng.integers(10000, 99999))
        suffix = "".join(rng.choice(letters, 5))
        cid = f"{digits}-{suffix}"
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)
    return ids


def _internet_scoped(
    rng: np.random.Generator,
    internet_service: np.ndarray,
    p_yes: float,
) -> np.ndarray:
    """Yes/No fields that become 'No internet service' when the customer has none."""
    n = len(internet_service)
    return np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes]),
    )


def generate_churn_data(n_rows: int = 1000, *, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic raw churn dataset.

    The data is shaped to look like the IBM Telco Customer Churn dataset so
    everything downstream (cleaning, features, training) can be swapped to
    real data without code changes.
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    customer_ids = _generate_unique_ids(rng, n)

    gender = rng.choice(["Male", "Female"], n, p=[0.50, 0.50])
    senior_citizen = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure = rng.integers(0, 73, n)

    phone_service = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n, p=[0.42, 0.58]),
    )

    internet_service = rng.choice(INTERNET_SERVICE, n, p=[0.34, 0.44, 0.22])
    online_security = _internet_scoped(rng, internet_service, 0.29)
    online_backup = _internet_scoped(rng, internet_service, 0.34)
    device_protection = _internet_scoped(rng, internet_service, 0.34)
    tech_support = _internet_scoped(rng, internet_service, 0.29)
    streaming_tv = _internet_scoped(rng, internet_service, 0.38)
    streaming_movies = _internet_scoped(rng, internet_service, 0.39)

    contract = rng.choice(CONTRACT, n, p=[0.55, 0.21, 0.24])
    paperless_billing = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = rng.choice(PAYMENT_METHOD, n, p=[0.34, 0.23, 0.22, 0.21])

    # MonthlyCharges depends on internet tier
    monthly_charges = np.where(
        internet_service == "No",
        rng.uniform(18.25, 25.0, n),
        np.where(
            internet_service == "Fiber optic",
            rng.uniform(70.0, 118.75, n),
            rng.uniform(35.0, 70.0, n),
        ),
    ).round(2)

    # TotalCharges is delivered as a string; "" for new customers (tenure 0)
    total_numeric = (monthly_charges * tenure + rng.normal(0, 5, n)).clip(min=0)
    total_charges = np.where(
        tenure == 0,
        "",
        np.array([f"{x:.2f}" for x in total_numeric]),
    )

    # Plausible churn signal: month-to-month, fiber, low tenure, senior all increase risk.
    churn_logits = (
        -1.5
        + 0.8 * (np.array(contract) == "Month-to-month")
        + 0.3 * (senior_citizen == 1)
        + 0.6 * (np.array(internet_service) == "Fiber optic")
        - 0.03 * tenure
        + rng.normal(0, 0.5, n)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-churn_logits))
    churn = np.where(rng.uniform(0, 1, n) < churn_prob, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender.astype(str),
            "SeniorCitizen": senior_citizen.astype(int),
            "Partner": partner.astype(str),
            "Dependents": dependents.astype(str),
            "tenure": tenure.astype(int),
            "PhoneService": phone_service.astype(str),
            "MultipleLines": multiple_lines.astype(str),
            "InternetService": internet_service.astype(str),
            "OnlineSecurity": online_security.astype(str),
            "OnlineBackup": online_backup.astype(str),
            "DeviceProtection": device_protection.astype(str),
            "TechSupport": tech_support.astype(str),
            "StreamingTV": streaming_tv.astype(str),
            "StreamingMovies": streaming_movies.astype(str),
            "Contract": contract.astype(str),
            "PaperlessBilling": paperless_billing.astype(str),
            "PaymentMethod": payment_method.astype(str),
            "MonthlyCharges": monthly_charges.astype(float),
            "TotalCharges": total_charges.astype(str),
            "Churn": churn.astype(str),
        }
    )
    return df


__all__ = ["generate_churn_data"]

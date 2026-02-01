"""Produce the *processed* churn dataset from a validated raw frame.

Cleaning rules applied here:

- ``TotalCharges`` (string with ``""`` for new customers) → ``float`` with
  imputed ``0.0`` for those rows. They have not been billed yet, so 0 is the
  faithful value, not an arbitrary fillna.
- ``SeniorCitizen`` (0/1 int) → ``bool``.
- All ``Yes``/``No`` columns → ``bool``. "No internet service" and
  "No phone service" collapse to ``False`` because the absence of a service
  is semantically equivalent to not having it on. Keeping them as separate
  categories adds noise without adding signal at this stage.
- ``customerID`` is dropped — it's an identifier, not a feature.
- Column names are normalized to ``snake_case`` so every downstream module
  speaks one convention.

The processed schema below is the *contract with feature engineering* in
Phase 3. If a future cleaning change wants to break it, that's a deliberate
breaking change and downstream consumers will notice immediately.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from .schema import CHURN_RAW_SCHEMA, CONTRACT, INTERNET_SERVICE, PAYMENT_METHOD

CHURN_PROCESSED_SCHEMA = pa.DataFrameSchema(
    columns={
        "gender": pa.Column(str, pa.Check.isin(["Male", "Female"])),
        "senior_citizen": pa.Column(bool),
        "partner": pa.Column(bool),
        "dependents": pa.Column(bool),
        "tenure": pa.Column(
            int,
            [
                pa.Check.greater_than_or_equal_to(0),
                pa.Check.less_than_or_equal_to(72),
            ],
        ),
        "phone_service": pa.Column(bool),
        "multiple_lines": pa.Column(bool),
        "internet_service": pa.Column(str, pa.Check.isin(INTERNET_SERVICE)),
        "online_security": pa.Column(bool),
        "online_backup": pa.Column(bool),
        "device_protection": pa.Column(bool),
        "tech_support": pa.Column(bool),
        "streaming_tv": pa.Column(bool),
        "streaming_movies": pa.Column(bool),
        "contract": pa.Column(str, pa.Check.isin(CONTRACT)),
        "paperless_billing": pa.Column(bool),
        "payment_method": pa.Column(str, pa.Check.isin(PAYMENT_METHOD)),
        "monthly_charges": pa.Column(float, pa.Check.greater_than(0)),
        "total_charges": pa.Column(float, pa.Check.greater_than_or_equal_to(0)),
        "churn": pa.Column(bool),
    },
    strict=True,
    coerce=False,
    ordered=False,
)

_YES_NO_LIKE = {
    "Yes": True,
    "No": False,
    "No internet service": False,
    "No phone service": False,
}

_YES_NO_COLUMNS: list[tuple[str, str]] = [
    ("Partner", "partner"),
    ("Dependents", "dependents"),
    ("PhoneService", "phone_service"),
    ("MultipleLines", "multiple_lines"),
    ("OnlineSecurity", "online_security"),
    ("OnlineBackup", "online_backup"),
    ("DeviceProtection", "device_protection"),
    ("TechSupport", "tech_support"),
    ("StreamingTV", "streaming_tv"),
    ("StreamingMovies", "streaming_movies"),
    ("PaperlessBilling", "paperless_billing"),
    ("Churn", "churn"),
]


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Transform a validated raw frame into the processed frame.

    The input must satisfy :data:`CHURN_RAW_SCHEMA`. The output is validated
    against :data:`CHURN_PROCESSED_SCHEMA` before return — defense in depth
    so cleaning bugs surface here rather than three steps downstream.
    """
    CHURN_RAW_SCHEMA.validate(raw)
    df = raw.copy()

    # TotalCharges: empty string → 0.0 (new customers, never billed)
    df["total_charges"] = pd.to_numeric(
        df["TotalCharges"].replace("", "0"), errors="raise"
    ).astype(float)
    df["monthly_charges"] = df["MonthlyCharges"].astype(float)

    df["senior_citizen"] = df["SeniorCitizen"].astype(bool)

    for src_col, dst_col in _YES_NO_COLUMNS:
        df[dst_col] = df[src_col].map(_YES_NO_LIKE).astype(bool)

    df["gender"] = df["gender"].astype(str)
    df["tenure"] = df["tenure"].astype(int)
    df["internet_service"] = df["InternetService"].astype(str)
    df["contract"] = df["Contract"].astype(str)
    df["payment_method"] = df["PaymentMethod"].astype(str)

    keep = list(CHURN_PROCESSED_SCHEMA.columns.keys())
    out = df[keep]
    return CHURN_PROCESSED_SCHEMA.validate(out)


PROCESSED_BOOL_COLUMNS: list[str] = [
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
    "churn",
]


def coerce_bool_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Idempotent bool coercion that fails loudly on unexpected values.

    pandas read_csv may return either bool dtype (auto-detected from
    ``True``/``False`` text) or object dtype (mixed-case strings, etc.). This
    helper handles both and raises if a column contains values that are not
    bool-like, instead of the previous silent NaN-then-True corruption.
    """
    for col in columns:
        if df[col].dtype == bool:
            continue
        mapped = df[col].map({"True": True, "False": False, True: True, False: False})
        if mapped.isna().any():
            bad = df.loc[mapped.isna(), col].unique().tolist()
            raise ValueError(
                f"column {col!r} contains non-bool-like values: {bad[:5]}"
            )
        df[col] = mapped.astype(bool)
    return df


def load_processed(path: str | Path) -> pd.DataFrame:
    """Read a processed-format CSV and restore dtypes lost in serialization."""
    df = pd.read_csv(path)
    df = coerce_bool_columns(df, PROCESSED_BOOL_COLUMNS)
    return CHURN_PROCESSED_SCHEMA.validate(df)


__all__ = [
    "clean",
    "coerce_bool_columns",
    "load_processed",
    "CHURN_PROCESSED_SCHEMA",
    "PROCESSED_BOOL_COLUMNS",
]

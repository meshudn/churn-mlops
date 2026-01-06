"""Read and write raw churn CSVs, validating against the raw schema each time.

Putting validation on both *read* and *write* sides means a corrupted file can
never silently enter the system: if upstream changes the format, ingestion
fails; if our own pipeline tries to write garbage, that fails too.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import CHURN_RAW_SCHEMA


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read a CSV from ``csv_path`` and validate against the raw schema."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"TotalCharges": str, "customerID": str})
    # Empty TotalCharges cells round-trip through pandas as NaN; the raw
    # schema expects them as the empty string the upstream system actually
    # sends, so we restore that exact contract here.
    df["TotalCharges"] = df["TotalCharges"].fillna("")
    return CHURN_RAW_SCHEMA.validate(df)


def write_raw(df: pd.DataFrame, csv_path: str | Path) -> Path:
    """Validate against the raw schema then write to ``csv_path``."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    CHURN_RAW_SCHEMA.validate(df)
    df.to_csv(csv_path, index=False)
    return csv_path


__all__ = ["load_raw", "write_raw"]

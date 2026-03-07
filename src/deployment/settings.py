"""API configuration via pydantic-settings.

All knobs are env-driven (prefix ``CHURN_API_``) so the same Docker image runs
in dev / staging / prod with no code change. Defaults match the
single-developer local setup.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHURN_API_",
        env_file=".env",
        extra="ignore",
        # Allow our own ``model_*`` fields without pydantic warnings.
        protected_namespaces=(),
    )

    model_name: str = Field(default="churn-model")
    model_alias: str = Field(default="production")
    mlflow_tracking_uri: str = Field(
        default_factory=lambda: f"file://{Path.cwd()}/mlruns"
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


__all__ = ["APISettings"]

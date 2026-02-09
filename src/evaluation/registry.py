"""MLflow Model Registry helpers — alias-based, not the deprecated stages API.

We use the MLflow 2.x **alias** API (``set_registered_model_alias``) instead of
the older "stages" API (``transition_model_version_stage``). Aliases are
mutable pointers (``production`` -> version N) that decouple the deployment
target from the model version. Stages are being deprecated; new code should
use aliases.

Why aliases:
- Multiple aliases per version (``production``, ``canary``, ``shadow``) without
  needing extra concepts.
- Reading the production version is one call; promoting is one call.
- Audit trail is preserved (versions are immutable; you only move pointers).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

DEFAULT_MODEL_NAME = "churn-model"
PRODUCTION_ALIAS = "production"


@dataclass(frozen=True)
class RegisteredVersion:
    """Lightweight, JSON-serializable view of an MLflow ModelVersion."""

    name: str
    version: str
    run_id: str


def get_production(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    client: MlflowClient | None = None,
) -> RegisteredVersion | None:
    """Return the version currently aliased ``production``, or None."""
    client = client or MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except MlflowException:
        # Either the model name is unregistered or the alias is unset — both
        # collapse to "no current production" for our purposes. The base class
        # is caught (rather than RestException) so the file-store backend used
        # in tests behaves the same as the REST backend used in production.
        return None
    return RegisteredVersion(name=mv.name, version=str(mv.version), run_id=mv.run_id)


def register_run(
    run_id: str,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    artifact_path: str = "model",
) -> RegisteredVersion:
    """Register a run's logged model as a new version of ``model_name``."""
    mv = mlflow.register_model(
        model_uri=f"runs:/{run_id}/{artifact_path}",
        name=model_name,
    )
    return RegisteredVersion(name=mv.name, version=str(mv.version), run_id=run_id)


def set_production(
    version: str,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    client: MlflowClient | None = None,
) -> None:
    """Point the ``production`` alias at ``version``."""
    client = client or MlflowClient()
    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, version=version)


def get_run_metric(
    run_id: str,
    metric: str,
    *,
    client: MlflowClient | None = None,
) -> float | None:
    """Return a logged metric for a run, or None if the run/metric is missing."""
    client = client or MlflowClient()
    try:
        run = client.get_run(run_id)
    except MlflowException:
        return None
    return run.data.metrics.get(metric)


__all__ = [
    "DEFAULT_MODEL_NAME",
    "PRODUCTION_ALIAS",
    "RegisteredVersion",
    "get_production",
    "get_run_metric",
    "register_run",
    "set_production",
]

"""Load the production model from the MLflow registry.

The deploy unit is the alias (``production``), not a specific version. When
Phase 5's evaluate stage flips the alias to a new version, restarting the
service is what picks it up. That separation is the whole point of using a
registry: deployment is "restart with a different alias target," not "rebuild
and ship a new container."
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import mlflow.sklearn

from src.evaluation.registry import get_production


@dataclass
class LoadedModel:
    """A loaded sklearn Pipeline plus the registry metadata it came from."""

    pipeline: object  # the fitted sklearn.Pipeline
    name: str
    version: str
    run_id: str


def load_production_model(
    model_name: str,
    *,
    alias: str = "production",
) -> LoadedModel:
    """Resolve the alias and load the underlying sklearn Pipeline."""
    prod = get_production(model_name=model_name)
    if prod is None:
        raise RuntimeError(
            f"no model {model_name!r} with alias {alias!r} in registry at "
            f"{mlflow.get_tracking_uri()}"
        )

    pipeline = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
    return LoadedModel(
        pipeline=pipeline,
        name=prod.name,
        version=prod.version,
        run_id=prod.run_id,
    )


__all__ = ["LoadedModel", "load_production_model"]

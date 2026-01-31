"""Train all configured models and log everything to MLflow.

The script is the canonical entry point Phase 5 (evaluation) and Phase 6
(CI/CD) call. Every run logs to a single MLflow experiment so cross-run
comparison ("did random_forest beat logreg this time?") is one query away.

What gets logged per run:
- Params: model_kind, random_state, train/test sizes, positive rate.
- Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC.
- Artifacts: the fitted sklearn Pipeline (with encoder!), feature importances.
- Tags: implicit (run_name = model_kind, source = python script).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .data import train_test_features
from .pipeline import ALL_MODEL_KINDS, ModelKind, make_pipeline

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_OUT = _REPO_ROOT / "metrics" / "train_metrics.json"


def _evaluate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
    }


def _feature_importances(pipe: Pipeline) -> dict[str, float] | None:
    """Pair encoded feature names with model weights, when the model exposes them."""
    try:
        feature_names = pipe.named_steps["preprocess"].get_feature_names_out().tolist()
    except Exception:
        return None

    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        weights = model.feature_importances_
    elif hasattr(model, "coef_"):
        weights = np.abs(model.coef_).ravel()
    else:
        return None

    return {name: float(w) for name, w in zip(feature_names, weights, strict=True)}


def train_one(
    kind: ModelKind,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit one model kind, log to MLflow, return a compact summary dict."""
    with mlflow.start_run(run_name=kind) as run:
        mlflow.log_param("model_kind", kind)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("test_positive_rate", float(y_test.mean()))

        pipe = make_pipeline(kind, random_state=random_state)
        pipe.fit(X_train, y_train)

        metrics = _evaluate(pipe, X_test, y_test)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        importances = _feature_importances(pipe)
        if importances is not None:
            mlflow.log_dict(importances, "feature_importances.json")

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=X_train.head(3),
        )

        return {
            "run_id": run.info.run_id,
            "model_kind": kind,
            "metrics": metrics,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train churn models and log to MLflow.")
    parser.add_argument("--experiment", default="churn-baseline")
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    mlflow.set_experiment(args.experiment)

    X_train, X_test, y_train, y_test = train_test_features(
        test_size=args.test_size, random_state=args.random_state,
    )
    print(
        f"[train] train={len(X_train)} test={len(X_test)}  "
        f"positives_train={int(y_train.sum())} positives_test={int(y_test.sum())}"
    )

    runs: list[dict[str, Any]] = []
    for kind in ALL_MODEL_KINDS:
        print(f"[train] training {kind}...")
        result = train_one(
            kind, X_train, X_test, y_train, y_test, random_state=args.random_state,
        )
        runs.append(result)
        m = result["metrics"]
        print(
            f"[train]   {kind}: f1={m['f1']:.3f} pr_auc={m['pr_auc']:.3f} "
            f"roc_auc={m['roc_auc']:.3f}"
        )

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": args.experiment,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "runs": runs,
    }
    with open(args.metrics_out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[train] summary -> {args.metrics_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

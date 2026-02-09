"""Compare-then-promote: deploy a new model only if it beats the current one.

The decision rule is intentionally strict: *strictly greater* on the chosen
metric. Equal-or-worse candidates get rejected, even by a tiny margin. The
asymmetry exists because changing production is a real-world risk (rollouts,
retraining cost, audit trail) — the burden of proof is on the candidate, not
the incumbent. Phase 5's tutorial walks through alternatives (margin
thresholds, multi-metric guards, soak periods) and why they belong in CI/CD
rather than here.

Inputs:
- ``metrics/train_metrics.json`` — produced by ``src.training.train``,
  contains one entry per model kind from the latest training round.
- The MLflow Registry — read for the current ``production`` version.

Outputs:
- ``metrics/promotion_decision.json`` — the verdict (cache: false in dvc.yaml
  so it's git-tracked alongside the metric snapshot).
- A registry side effect (when ``--apply`` is true): the production alias
  moves to a newly-registered version on a "promote"/"bootstrap" decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

from .registry import (
    DEFAULT_MODEL_NAME,
    get_production,
    get_run_metric,
    register_run,
    set_production,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_IN = _REPO_ROOT / "metrics" / "train_metrics.json"
DEFAULT_DECISION_OUT = _REPO_ROOT / "metrics" / "promotion_decision.json"


def _best_candidate(metrics_summary: dict, metric: str) -> dict | None:
    """Pick the highest-``metric`` run from the latest training round."""
    runs = metrics_summary.get("runs") or []
    if not runs:
        return None
    return max(runs, key=lambda r: r["metrics"][metric])


def decide_promotion(
    metrics_path: str | Path = DEFAULT_METRICS_IN,
    *,
    metric: str = "f1",
    model_name: str = DEFAULT_MODEL_NAME,
    apply: bool = True,
    client: MlflowClient | None = None,
) -> dict[str, Any]:
    """Compare the best candidate from this training round to current production.

    Returns a JSON-serializable verdict. With ``apply=True``, also mutates the
    MLflow Registry: registering the candidate as a new version and pointing
    the ``production`` alias at it on a positive decision.
    """
    client = client or MlflowClient()
    metrics_path = Path(metrics_path)

    with open(metrics_path) as f:
        summary = json.load(f)

    candidate = _best_candidate(summary, metric)
    if candidate is None:
        return {
            "decision": "no_op",
            "reason": f"no runs in {metrics_path}",
            "metric": metric,
            "model_name": model_name,
        }

    candidate_run_id: str = candidate["run_id"]
    candidate_metric: float = candidate["metrics"][metric]
    candidate_kind: str = candidate["model_kind"]

    current = get_production(model_name=model_name, client=client)

    if current is None:
        # Bootstrap: there is no production model yet.
        decision = "bootstrap"
        reason = (
            f"no current production; registering {candidate_kind} run "
            f"as the first version with {metric}={candidate_metric:.4f}"
        )
        promoted_version = None
        if apply:
            registered = register_run(candidate_run_id, model_name=model_name)
            set_production(registered.version, model_name=model_name, client=client)
            promoted_version = registered.version
        return {
            "decision": decision,
            "reason": reason,
            "metric": metric,
            "model_name": model_name,
            "candidate": {
                "run_id": candidate_run_id,
                "model_kind": candidate_kind,
                metric: candidate_metric,
            },
            "current_production": None,
            "promoted_version": promoted_version,
        }

    current_metric = get_run_metric(current.run_id, metric, client=client)

    candidate_better = (
        current_metric is None or candidate_metric > current_metric
    )

    if candidate_better:
        decision = "promote"
        reason = (
            f"candidate {candidate_kind} {metric}={candidate_metric:.4f} > "
            f"current production {metric}={current_metric}"
        )
        promoted_version = None
        if apply:
            registered = register_run(candidate_run_id, model_name=model_name)
            set_production(registered.version, model_name=model_name, client=client)
            promoted_version = registered.version
    else:
        decision = "reject"
        reason = (
            f"candidate {candidate_kind} {metric}={candidate_metric:.4f} "
            f"<= current production {metric}={current_metric:.4f}"
        )
        promoted_version = None

    return {
        "decision": decision,
        "reason": reason,
        "metric": metric,
        "model_name": model_name,
        "candidate": {
            "run_id": candidate_run_id,
            "model_kind": candidate_kind,
            metric: candidate_metric,
        },
        "current_production": {
            "run_id": current.run_id,
            "version": current.version,
            metric: current_metric,
        },
        "promoted_version": promoted_version,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decide whether to promote the best run from the latest training round.",
    )
    parser.add_argument("--metrics-in", type=Path, default=DEFAULT_METRICS_IN)
    parser.add_argument("--decision-out", type=Path, default=DEFAULT_DECISION_OUT)
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the decision without applying registry changes.",
    )
    args = parser.parse_args(argv)

    decision = decide_promotion(
        args.metrics_in,
        metric=args.metric,
        model_name=args.model_name,
        apply=not args.dry_run,
    )

    print(f"[promote] {decision['decision'].upper()}: {decision['reason']}")
    if decision.get("promoted_version"):
        print(f"[promote] promoted version -> {decision['promoted_version']}")

    args.decision_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.decision_out, "w") as f:
        json.dump(decision, f, indent=2, sort_keys=True)
    print(f"[promote] decision -> {args.decision_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

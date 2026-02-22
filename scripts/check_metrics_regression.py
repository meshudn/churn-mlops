"""Compare a PR's metrics/train_metrics.json against main and write a markdown delta.

CI runs ``dvc repro train`` on the PR branch, then this script:

1. Reads ``metrics/train_metrics.json`` produced on the PR.
2. Reads the same file from a base ref (default ``origin/main``) via
   ``git show``.
3. Picks the best run on the chosen metric (default ``f1``) from each side.
4. Writes a markdown summary to ``metrics/_delta.md`` for the comment step.
5. Exits non-zero (failing CI) iff the regression on the metric exceeds
   ``--tolerance``.

The tolerance encodes the "margin threshold" guardrail discussed in
docs/05 §7.1: a candidate must beat the baseline by enough to be worth the
deployment cost. For PRs the polarity is inverted — we want to *block*
merges that regress by more than the tolerance.

Usage:
    python scripts/check_metrics_regression.py
    python scripts/check_metrics_regression.py --metric pr_auc --tolerance 0.01
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_METRIC = "f1"
DEFAULT_TOLERANCE = 0.005
DEFAULT_BASE_REF = "origin/main"

PR_METRICS_PATH = Path("metrics/train_metrics.json")
DELTA_OUT = Path("metrics/_delta.md")


def best_metric(summary: dict, metric: str) -> tuple[float | None, str | None]:
    """Return (best_value, model_kind_for_best_run) or (None, None)."""
    runs = summary.get("runs") or []
    if not runs:
        return None, None
    best = max(runs, key=lambda r: r["metrics"][metric])
    return best["metrics"][metric], best.get("model_kind")


def get_base_metrics(ref: str) -> dict | None:
    try:
        out = subprocess.check_output(
            ["git", "show", f"{ref}:metrics/train_metrics.json"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return json.loads(out)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def render_markdown(
    *,
    metric: str,
    tolerance: float,
    pr_best: float | None,
    pr_kind: str | None,
    base_best: float | None,
    base_kind: str | None,
    base_ref: str,
) -> tuple[str, bool]:
    """Return (markdown_body, regressed)."""
    lines = ["## Metric delta vs main", ""]

    if pr_best is None:
        lines.append(f"PR has no `{metric}` runs in `{PR_METRICS_PATH}`.")
        return "\n".join(lines), True  # missing metrics is itself a regression

    if base_best is None:
        lines.append(
            f"No baseline metric on `{base_ref}` yet. "
            f"PR best **{metric}** = `{pr_best:.4f}` ({pr_kind})."
        )
        return "\n".join(lines), False

    delta = pr_best - base_best
    direction = "UP" if delta >= 0 else "DOWN"
    lines += [
        f"| | best `{metric}` | model |",
        "|---|---|---|",
        f"| main (`{base_ref}`) | `{base_best:.4f}` | {base_kind} |",
        f"| this PR | `{pr_best:.4f}` | {pr_kind} |",
        f"| delta | `{delta:+.4f}` ({direction}) | |",
        "",
    ]
    regressed = delta < -tolerance
    if regressed:
        lines.append(
            f"**Regression beyond tolerance** "
            f"(`{tolerance:.4f}` on `{metric}`). Merge blocked."
        )
    else:
        lines.append("No significant regression.")
    return "\n".join(lines), regressed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare PR metrics against base ref.")
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Max allowed regression on `metric` before failing.",
    )
    parser.add_argument("--base-ref", default=DEFAULT_BASE_REF)
    args = parser.parse_args(argv)

    if not PR_METRICS_PATH.exists():
        print(f"::error::no PR metrics at {PR_METRICS_PATH}", file=sys.stderr)
        return 2

    pr_summary = json.loads(PR_METRICS_PATH.read_text())
    pr_best, pr_kind = best_metric(pr_summary, args.metric)

    base_summary = get_base_metrics(args.base_ref)
    if base_summary is not None:
        base_best, base_kind = best_metric(base_summary, args.metric)
    else:
        base_best, base_kind = None, None

    body, regressed = render_markdown(
        metric=args.metric,
        tolerance=args.tolerance,
        pr_best=pr_best,
        pr_kind=pr_kind,
        base_best=base_best,
        base_kind=base_kind,
        base_ref=args.base_ref,
    )

    DELTA_OUT.parent.mkdir(parents=True, exist_ok=True)
    DELTA_OUT.write_text(body)
    print(body)
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())

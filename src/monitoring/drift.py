"""Data drift detection between a reference and a current feature dataset.

Wraps Evidently AI's ``DataDriftPreset`` which:

- Picks an appropriate test per column type (K-S for numeric, Z-test or
  chi-square for categorical depending on cardinality).
- Reports per-column drift via p-values; below threshold => column drifted.
- Aggregates into a "drift share" — fraction of columns that drifted — and
  flags the dataset as drifted when that share exceeds ``drift_share``.

The CLI also generates a "drifted" current batch on the fly when no current
file is given. That batch is the same synthetic generator with a deliberate
shift (more fiber-optic customers, higher charges) so the demo end-to-end
shows drift being *detected*, not just plumbing-tested.

Outputs:
- ``metrics/drift_report.json``  — small structured summary committed to git.
- ``metrics/drift_report.html``  — Evidently's full visual report (gitignored).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.training.data import load_features

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = _REPO_ROOT / "data" / "features" / "churn_features.csv"
DEFAULT_REPORT_JSON = _REPO_ROOT / "metrics" / "drift_report.json"
DEFAULT_REPORT_HTML = _REPO_ROOT / "metrics" / "drift_report.html"

# Evidently's per-column metric names look like:
#   ValueDrift(column=monthly_charges,method=K-S p_value,threshold=0.05)
#   ValueDrift(column=tenure,method=Wasserstein distance (normed),threshold=0.1)
#   ValueDrift(column=gender,method=Jensen-Shannon distance,threshold=0.1)
# Different methods are picked per column type and sample size. The ones
# whose name ends in " p_value" are inverted (small value => drift); for the
# distance-based methods, large value => drift.
_VALUE_DRIFT_RE = re.compile(
    r"ValueDrift\(column=(?P<col>[^,]+),"
    r"method=(?P<method>[^,]+),"
    r"threshold=(?P<thr>[\d.]+)\)"
)


def _is_drifted_for_method(method: str, value: float, threshold: float) -> bool:
    """True iff this column's value indicates drift, given the method's polarity."""
    if method.endswith(" p_value"):
        return value < threshold
    return value > threshold


@dataclass
class DriftResult:
    """JSON-serializable drift summary plus the underlying Evidently snapshot."""

    overall: dict[str, Any]
    columns: list[dict[str, Any]]
    snapshot: Any  # evidently.core.report.Snapshot

    def to_serializable(self) -> dict[str, Any]:
        return {"overall": self.overall, "columns": self.columns}


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    drift_share_threshold: float = 0.5,
) -> DriftResult:
    """Run DataDriftPreset and return a typed summary."""
    report = Report(metrics=[DataDriftPreset(drift_share=drift_share_threshold)])
    snapshot = report.run(reference_data=reference, current_data=current)
    raw = snapshot.dict()

    overall: dict[str, Any] = {}
    columns: list[dict[str, Any]] = []

    for m in raw.get("metrics", []):
        name = m.get("metric_name", "")
        value = m.get("value")

        if name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            share = float(value["share"])
            overall = {
                "drifted_columns": int(value["count"]),
                "drift_share": share,
                "drift_share_threshold": drift_share_threshold,
                "dataset_drifted": share > drift_share_threshold,
            }
            continue

        parsed = _VALUE_DRIFT_RE.match(name)
        if parsed:
            threshold = float(parsed["thr"])
            score = float(value)
            method = parsed["method"]
            columns.append(
                {
                    "column": parsed["col"],
                    "method": method,
                    "score": score,
                    "threshold": threshold,
                    "drifted": _is_drifted_for_method(method, score, threshold),
                }
            )

    # Most-drifted first. Distance methods are larger when more drifted; p-value
    # methods are smaller. Sort by signed distance from threshold so both kinds
    # of metrics surface their most-drifted columns at the top.
    def _sort_key(c: dict[str, Any]) -> float:
        if c["method"].endswith(" p_value"):
            return c["score"] - c["threshold"]  # negative when drifted
        return c["threshold"] - c["score"]      # negative when drifted

    columns.sort(key=_sort_key)
    return DriftResult(overall=overall, columns=columns, snapshot=snapshot)


def generate_drifted_current(
    *,
    n_rows: int = 2000,
    seed: int = 7,
    fiber_boost: float = 0.25,
) -> pd.DataFrame:
    """Synthesize a 'current' feature batch with a deliberate distribution shift.

    Mimics a marketing campaign that pushed more customers onto fiber-optic
    plans at higher monthly charges — the kind of upstream change that should
    trigger a drift alert. The returned frame matches CHURN_FEATURES_SCHEMA
    (with ``churn`` still present; callers may drop it for X-only drift).
    """
    from src.features.build import build_features
    from src.ingestion.clean import clean
    from src.ingestion.synthetic import generate_churn_data

    df_raw = generate_churn_data(n_rows=n_rows, seed=seed)
    n_boost = int(fiber_boost * len(df_raw))
    boost_idx = df_raw.sample(n=n_boost, random_state=seed).index
    df_raw.loc[boost_idx, "InternetService"] = "Fiber optic"
    # Fiber customers in the synthetic generator pay 70-118.75; nudge boosted
    # rows toward the upper end so MonthlyCharges shows a drift signal too.
    df_raw.loc[boost_idx, "MonthlyCharges"] = 95.0

    return build_features(clean(df_raw))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect data drift between a reference and a current dataset."
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--current",
        type=Path,
        default=None,
        help="Path to current features CSV. If omitted, a drifted batch is "
        "generated on the fly for the demo.",
    )
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-html", type=Path, default=DEFAULT_REPORT_HTML)
    parser.add_argument(
        "--drift-share-threshold",
        type=float,
        default=0.5,
        help="Dataset-level drift threshold (fraction of columns drifted).",
    )
    parser.add_argument("--current-seed", type=int, default=7)
    args = parser.parse_args(argv)

    print(f"[drift] reference: {args.reference}")
    reference = load_features(args.reference).drop(columns=["churn"])

    if args.current:
        print(f"[drift] current  : {args.current}")
        current = load_features(args.current).drop(columns=["churn"])
    else:
        print(f"[drift] current  : (synthetic drifted batch, seed={args.current_seed})")
        current = generate_drifted_current(seed=args.current_seed).drop(columns=["churn"])

    print(f"[drift] reference rows: {len(reference)}, current rows: {len(current)}")
    result = compute_drift(
        reference, current, drift_share_threshold=args.drift_share_threshold
    )

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_html.parent.mkdir(parents=True, exist_ok=True)
    result.snapshot.save_html(str(args.report_html))
    with open(args.report_json, "w") as f:
        json.dump(result.to_serializable(), f, indent=2, sort_keys=True)

    overall = result.overall
    print(
        f"[drift] drifted_columns={overall['drifted_columns']} "
        f"drift_share={overall['drift_share']:.2f} "
        f"dataset_drifted={overall['dataset_drifted']}"
    )
    print(f"[drift] json -> {args.report_json}")
    print(f"[drift] html -> {args.report_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

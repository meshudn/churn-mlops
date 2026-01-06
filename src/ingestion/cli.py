"""CLI entry point for the data ingestion pipeline.

Examples::

    # generate synthetic data and process it (default)
    uv run python -m src.ingestion.cli

    # use a real CSV that satisfies the raw schema
    uv run python -m src.ingestion.cli --source path/to/Telco-Customer-Churn.csv

The CLI is the single source of truth for how data lands on disk. In Phase 3
DVC will wrap this same command as a pipeline stage, so changing CLI flags
is the equivalent of changing a pipeline interface — keep it stable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .clean import clean
from .loader import load_raw, write_raw
from .synthetic import generate_churn_data

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_OUT = _REPO_ROOT / "data" / "raw" / "churn.csv"
DEFAULT_PROCESSED_OUT = _REPO_ROOT / "data" / "processed" / "churn.csv"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest and clean churn data.")
    parser.add_argument(
        "--source",
        default="synthetic",
        help="Either 'synthetic' (use built-in generator) or a path to a CSV file.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=2000,
        help="Number of rows to generate (only used when --source=synthetic).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the synthetic generator.",
    )
    parser.add_argument(
        "--raw-out",
        type=Path,
        default=DEFAULT_RAW_OUT,
        help=f"Where to write the raw CSV (default: {DEFAULT_RAW_OUT}).",
    )
    parser.add_argument(
        "--processed-out",
        type=Path,
        default=DEFAULT_PROCESSED_OUT,
        help=f"Where to write the processed CSV (default: {DEFAULT_PROCESSED_OUT}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.source == "synthetic":
        print(f"[ingest] generating {args.n_rows} synthetic rows (seed={args.seed})")
        raw = generate_churn_data(n_rows=args.n_rows, seed=args.seed)
    else:
        src = Path(args.source)
        print(f"[ingest] loading {src}")
        raw = load_raw(src)

    write_raw(raw, args.raw_out)
    print(f"[ingest] raw  -> {args.raw_out}  ({len(raw)} rows)")

    processed = clean(raw)
    args.processed_out.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.processed_out, index=False)
    print(f"[ingest] proc -> {args.processed_out}  ({len(processed)} rows)")
    print(f"[ingest] churn rate = {processed['churn'].mean():.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

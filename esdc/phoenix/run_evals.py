#!/usr/bin/env python3
"""ESDC Phoenix Evaluation Runner.

Fetches spans from a Phoenix server and runs tool evaluators against them.

Usage:
    python -m esdc.phoenix.run_evals [--project iris]
        [--evaluators tool_selection,tool_invocation,tool_response_handling]
    python -m esdc.phoenix.run_evals --spans-file spans.csv
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("esdc.phoenix.run_evals")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phoenix evaluations on ESDC traces"
    )
    parser.add_argument(
        "--project",
        default="iris",
        help="Phoenix project name (default: iris)",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:6006",
        help="Phoenix server endpoint (default: http://localhost:6006)",
    )
    parser.add_argument(
        "--spans-file",
        default=None,
        help="Path to a CSV/Parquet file with spans (skips fetching from Phoenix)",
    )
    parser.add_argument(
        "--evaluators",
        default="tool_selection,tool_invocation,tool_response_handling",
        help="Comma-separated list of evaluators to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default="eval_results.csv",
        help="Output CSV file path (default: eval_results.csv)",
    )
    return parser.parse_args()


def fetch_spans_from_phoenix(project: str, endpoint: str) -> pd.DataFrame:
    from phoenix.client import Client
    from phoenix.client.types.spans import SpanQuery

    client = Client(base_url=endpoint)
    query = (
        SpanQuery().where("span_kind == 'LLM'").select("input.value", "output.value")
    )
    df = client.spans.get_spans_dataframe(query=query, project_name=project)
    logger.info("Fetched %d spans from project '%s'", len(df), project)
    return df


def load_spans_from_file(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    logger.info("Loaded %d spans from %s", len(df), path)
    return df


def main() -> None:
    args = parse_args()
    evaluators = [e.strip() for e in args.evaluators.split(",")]

    from esdc.configs import Config
    from esdc.phoenix.phoenix_evals import run_evaluations

    Config.init_config()

    if args.spans_file:
        spans_df = load_spans_from_file(args.spans_file)
    else:
        try:
            spans_df = fetch_spans_from_phoenix(args.project, args.endpoint)
        except Exception as e:
            logger.error("Failed to fetch spans from Phoenix: %s", e)
            logger.info("Make sure Phoenix server is running at %s", args.endpoint)
            sys.exit(1)

    if spans_df.empty:
        logger.warning("No spans found. Nothing to evaluate.")
        sys.exit(0)

    results = run_evaluations(
        spans_df, evaluators=evaluators, project_name=args.project
    )

    if not results:
        logger.warning("No evaluation results produced.")
        sys.exit(0)

    combined = pd.concat(results.values(), axis=1)
    combined.to_csv(args.output, index=False)
    logger.info("Results saved to %s", args.output)

    for name, df in results.items():
        score_col = f"{name}_score"
        if score_col in df.columns:
            scores = df[score_col]
            labels = scores.apply(
                lambda x: x.get("label") if isinstance(x, dict) else x
            )
            correct = (labels == "correct").sum()
            total = len(labels)
            pct = correct / total * 100 if total else 0
            logger.info("%s: %d/%d correct (%.1f%%)", name, correct, total, pct)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Legacy-compatible entrypoint for the unified fundamentals reliability pipeline.

This shim preserves historical CLI usage while delegating all logic to
`Models.data_pipeline.FundamentalsPipeline`.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from Models.data_pipeline import (
    FreshnessThresholds,
    FundamentalsPipeline,
    PipelineRunResult,
    build_default_config,
    build_default_paths,
)

logger = logging.getLogger(__name__)


def _build_pipeline(args: argparse.Namespace) -> FundamentalsPipeline:
    config = build_default_config(
        enforce_freshness_gate=not args.disable_freshness_gate,
        allow_stale_override=args.allow_stale_override,
    )
    if args.request_delay is not None:
        config = replace(config, request_delay_seconds=args.request_delay)
    if args.max_retries is not None:
        config = replace(config, max_retries=args.max_retries)
    thresholds = FreshnessThresholds(
        max_median_staleness_days=args.max_median_staleness_days,
        max_pct_over_120_days=args.max_pct_over_120_days,
        min_q4_coverage_pct=args.min_q4_coverage_pct,
        max_max_staleness_days=args.max_max_staleness_days,
        grace_days=args.grace_days,
    )
    return FundamentalsPipeline(
        paths=build_default_paths(),
        config=config,
        thresholds=thresholds,
    )


def run_full_pipeline(
    tickers: list[str] | None = None,
    force: bool = False,
    skip_fetch: bool = False,
    diagnostics_only: bool = False,
    pipeline: FundamentalsPipeline | None = None,
) -> PipelineRunResult:
    runner = pipeline or FundamentalsPipeline(paths=build_default_paths(), config=build_default_config())
    return runner.run(
        tickers=tickers,
        force=force,
        merge_only=skip_fetch,
        diagnostics_only=diagnostics_only,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BIST fundamentals pipeline: fetch -> normalize -> merge -> diagnostics",
    )
    parser.add_argument("--tickers", nargs="+", default=None, help="Specific tickers to process")
    parser.add_argument("--force", action="store_true", help="Force fresh fetch/merge (ignore cache state)")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip remote fetch and merge from existing raw cache",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Only compute staleness metrics/quality reports on current consolidated data",
    )
    parser.add_argument("--request-delay", type=float, default=None, help="Seconds between ticker requests")
    parser.add_argument("--max-retries", type=int, default=None, help="Max retries per ticker fetch")
    parser.add_argument(
        "--disable-freshness-gate",
        action="store_true",
        help="Do not block on stale data (still emits alerts)",
    )
    parser.add_argument(
        "--allow-stale-override",
        action="store_true",
        help="Keep freshness gate enabled but continue execution when violated",
    )
    parser.add_argument("--max-median-staleness-days", type=int, default=120)
    parser.add_argument("--max-pct-over-120-days", type=float, default=0.90)
    parser.add_argument("--min-q4-coverage-pct", type=float, default=0.10)
    parser.add_argument("--max-max-staleness-days", type=int, default=500)
    parser.add_argument("--grace-days", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pipeline = _build_pipeline(args)
    result = run_full_pipeline(
        tickers=args.tickers,
        force=args.force,
        skip_fetch=args.merge_only,
        diagnostics_only=args.diagnostics_only,
        pipeline=pipeline,
    )
    if result.merged_bundle is not None:
        metrics = result.merged_bundle.quality_metrics
        logger.info("Fundamentals pipeline completed.")
        logger.info(f"Quality metrics: {metrics}")
        if result.merged_bundle.warnings:
            logger.info("Freshness warnings:")
            for warning in result.merged_bundle.warnings:
                logger.info(f"  - {warning}")
    elif result.raw_bundle is not None:
        logger.info(
            "Fetch stage completed.",
            f"tickers={len(result.raw_bundle.raw_by_ticker)}",
            f"errors={len(result.raw_bundle.errors)}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

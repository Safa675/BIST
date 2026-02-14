#!/usr/bin/env python3
"""
Legacy-compatible Q4 2025 refresh entrypoint.

All heavy lifting is now delegated to the unified typed fundamentals pipeline in
`Models.data_pipeline`.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from datetime import datetime, timezone

from Models.data_pipeline import (
    FreshnessThresholds,
    FundamentalsPipeline,
    PipelineRunResult,
    RawDataBundle,
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


def run_fetch(
    tickers: list[str] | None = None,
    max_tickers: int | None = None,
    pipeline: FundamentalsPipeline | None = None,
) -> tuple[dict, list]:
    runner = pipeline or FundamentalsPipeline(paths=build_default_paths(), config=build_default_config())
    result = runner.run(tickers=tickers, fetch_only=True, max_tickers=max_tickers)
    raw_bundle = result.raw_bundle or RawDataBundle(
        raw_by_ticker={},
        errors=[],
        source_name="unknown",
        fetched_at=datetime.now(timezone.utc),
    )
    return raw_bundle.raw_by_ticker, raw_bundle.errors


def run_merge(
    raw_data: dict[str, dict] | None = None,
    pipeline: FundamentalsPipeline | None = None,
    tickers: list[str] | None = None,
    max_tickers: int | None = None,
    force: bool = False,
) -> str:
    runner = pipeline or FundamentalsPipeline(paths=build_default_paths(), config=build_default_config())
    if raw_data is None:
        result = runner.run(
            tickers=tickers,
            merge_only=True,
            force=force,
            max_tickers=max_tickers,
        )
    else:
        raw_bundle = RawDataBundle(
            raw_by_ticker=raw_data,
            errors=[],
            source_name="legacy_in_memory_raw",
            fetched_at=datetime.now(timezone.utc),
        )
        result = runner.process_raw_bundle(raw_bundle=raw_bundle, force_merge=force)
    if result.merged_bundle is not None:
        logger.info("Merge complete.", result.merged_bundle.merge_stats)
    return str(runner.paths.consolidated_parquet)


def run_diagnostics(pipeline: FundamentalsPipeline | None = None) -> PipelineRunResult:
    runner = pipeline or FundamentalsPipeline(paths=build_default_paths(), config=build_default_config())
    result = runner.run(diagnostics_only=True)
    if result.merged_bundle is not None:
        logger.info("Diagnostics complete.", result.merged_bundle.quality_metrics)
        if result.merged_bundle.warnings:
            logger.info("Freshness warnings:")
            for warning in result.merged_bundle.warnings:
                logger.info(f"  - {warning}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BIST Q4 2025 fundamental data refresh pipeline")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch raw data")
    parser.add_argument("--merge-only", action="store_true", help="Only merge raw cache into consolidated parquet")
    parser.add_argument("--diagnose-only", action="store_true", help="Only run staleness diagnostics")
    parser.add_argument("--tickers", nargs="+", default=None, help="Specific tickers")
    parser.add_argument("--max-tickers", type=int, default=None, help="Limit number of tickers")
    parser.add_argument("--force", action="store_true", help="Force fetch/merge and bypass fingerprint cache hits")
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

    if args.diagnose_only:
        run_diagnostics(pipeline=pipeline)
        return 0

    if args.merge_only:
        run_merge(
            pipeline=pipeline,
            tickers=args.tickers,
            max_tickers=args.max_tickers,
            force=args.force,
        )
        run_diagnostics(pipeline=pipeline)
        return 0

    raw_data, errors = run_fetch(
        tickers=args.tickers,
        max_tickers=args.max_tickers,
        pipeline=pipeline,
    )
    logger.info(f"Fetch completed. tickers={len(raw_data)} errors={len(errors)}")

    if not args.fetch_only:
        run_merge(
            raw_data=raw_data,
            pipeline=pipeline,
            tickers=args.tickers,
            max_tickers=args.max_tickers,
            force=args.force,
        )
        run_diagnostics(pipeline=pipeline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

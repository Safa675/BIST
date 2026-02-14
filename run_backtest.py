#!/usr/bin/env python3
"""
Robust backtest entrypoint for the BIST project.

Features:
- Stable CLI/config loading
- Working-directory independent path resolution
- Loud, actionable failures for missing inputs/empty data windows
- Version display, dry-run mode, verbose/quiet logging
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from Models.portfolio_engine import PortfolioEngine, load_signal_configs

PROJECT_ROOT = Path(__file__).resolve().parent


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    @classmethod
    def disable(cls) -> None:
        """Disable colors (for non-TTY output)."""
        cls.RESET = ""
        cls.BOLD = ""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.MAGENTA = ""
        cls.CYAN = ""


def _get_version() -> str:
    """Get package version from metadata."""
    try:
        return version("bist-quant")
    except PackageNotFoundError:
        return "0.1.0-dev"


def _setup_logging(verbose: bool, quiet: bool) -> None:
    """Configure logging based on verbosity flags."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        force=True,
    )


def _abs_path(path_str: str | None, default: Path) -> Path:
    if path_str is None:
        return default.resolve()
    return Path(path_str).expanduser().resolve()


def _resolve_regime_outputs_dir(project_root: Path, cli_value: str | None) -> Path:
    if cli_value:
        out_dir = _abs_path(cli_value, project_root / "regime_filter" / "outputs")
        if not out_dir.exists():
            raise FileNotFoundError(
                f"Regime outputs directory does not exist: {out_dir}\n"
                "Pass a valid --regime-outputs path that contains regime_features.csv."
            )
        return out_dir

    candidates = [
        project_root / "Simple Regime Filter" / "outputs",
        project_root / "regime_filter" / "outputs",
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        cands = "\n".join(f"- {p}" for p in candidates)
        raise FileNotFoundError(
            "No regime outputs directory found.\n"
            f"Checked:\n{cands}\n"
            "Run the regime pipeline first to generate outputs."
        )
    return found


def _validate_required_inputs(data_dir: Path, regime_outputs_dir: Path) -> list[str]:
    """Validate required input files exist. Returns list of issues for dry-run mode."""
    issues: list[str] = []

    price_csv = data_dir / "bist_prices_full.csv"
    price_parquet = data_dir / "bist_prices_full.parquet"
    if not price_csv.exists() and not price_parquet.exists():
        issues.append(f"Missing price file: expected {price_parquet} or {price_csv}")

    xautry_file = data_dir / "xau_try_2013_2026.csv"
    if not xautry_file.exists():
        issues.append(f"Missing gold benchmark file: {xautry_file}")

    xu100_file = data_dir / "xu100_prices.csv"
    if not xu100_file.exists():
        issues.append(f"Missing XU100 benchmark file: {xu100_file}")

    regime_features_file = regime_outputs_dir / "regime_features.csv"
    if not regime_features_file.exists():
        issues.append(f"Missing regime features file: {regime_features_file}")

    return issues


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BIST backtests with robust path and input validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s momentum              # Run momentum signal backtest
  %(prog)s all --start-date 2020-01-01  # Run all signals from 2020
  %(prog)s --list-signals        # List available signals
  %(prog)s momentum --dry-run    # Validate inputs without running
""",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    parser.add_argument(
        "signal",
        nargs="?",
        default=None,
        help="Signal/factor name to run, or 'all'.",
    )
    parser.add_argument(
        "--factor",
        type=str,
        default=None,
        help="Alias for positional signal argument.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2018-01-01",
        help="Backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="Backtest end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (default: <project>/data).",
    )
    parser.add_argument(
        "--regime-outputs",
        type=str,
        default=None,
        help="Path to regime outputs directory (contains regime_features.csv).",
    )
    parser.add_argument(
        "--list-signals",
        action="store_true",
        help="List available signal configs and exit.",
    )
    parser.add_argument(
        "--use-config-timeline",
        action="store_true",
        help=(
            "Respect per-signal timeline in config files. "
            "Default behavior enforces CLI --start-date/--end-date."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show what would run without executing.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress INFO logging (only show warnings and errors).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output.",
    )
    return parser.parse_args()


def _validate_dates(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
    except Exception as exc:
        raise ValueError(f"Invalid date format: {exc}") from exc
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("Invalid date provided. Use YYYY-MM-DD.")
    if end_ts <= start_ts:
        raise ValueError(
            f"Invalid date range: start={start_ts.date()} end={end_ts.date()}. "
            "end-date must be after start-date."
        )
    return start_ts, end_ts


def _validate_loaded_universe(
    engine: PortfolioEngine, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> None:
    if engine.prices is None or engine.prices.empty:
        raise ValueError(
            "No price data loaded (empty price table). "
            "Check data files under data/ and run update_prices.py."
        )
    if engine.open_df is None or engine.open_df.empty:
        raise ValueError("Open-price panel is empty; cannot run backtest.")
    if engine.close_df is None or engine.close_df.empty:
        raise ValueError("Close-price panel is empty; cannot run backtest.")
    if engine.close_df.shape[1] == 0:
        raise ValueError("Universe is empty (0 tickers) after panel construction.")

    window = engine.close_df.loc[
        (engine.close_df.index >= start_ts) & (engine.close_df.index <= end_ts)
    ]
    if window.empty:
        raise ValueError(
            f"No price rows available in requested date range: {start_ts.date()}..{end_ts.date()}."
        )
    if window.notna().sum().sum() == 0:
        raise ValueError(
            f"Requested date range has only NaN prices: {start_ts.date()}..{end_ts.date()}."
        )


def _print_header(
    signal: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    data_dir: Path,
    regime_outputs_dir: Path,
    timeline_mode: str,
) -> None:
    """Print colorized header."""
    c = Colors
    print(f"\n{c.BOLD}{c.CYAN}{'=' * 80}{c.RESET}")
    print(f"{c.BOLD}{c.CYAN}BIST BACKTEST RUNNER{c.RESET}")
    print(f"{c.BOLD}{c.CYAN}{'=' * 80}{c.RESET}")
    print(f"{c.BOLD}project_root:{c.RESET} {PROJECT_ROOT}")
    print(f"{c.BOLD}signal:{c.RESET} {c.GREEN}{signal}{c.RESET}")
    print(f"{c.BOLD}date_range:{c.RESET} {c.YELLOW}{start_ts.date()}..{end_ts.date()}{c.RESET}")
    print(f"{c.BOLD}data_dir:{c.RESET} {data_dir}")
    print(f"{c.BOLD}regime_outputs:{c.RESET} {regime_outputs_dir}")
    print(f"{c.BOLD}timeline_mode:{c.RESET} {timeline_mode}")
    print()


def _print_dry_run_summary(
    signal: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    data_dir: Path,
    regime_outputs_dir: Path,
    issues: list[str],
    signal_names: list[str],
) -> int:
    """Print dry-run summary and return exit code."""
    c = Colors
    print(f"\n{c.BOLD}{c.MAGENTA}{'=' * 80}{c.RESET}")
    print(f"{c.BOLD}{c.MAGENTA}DRY RUN - Validation Summary{c.RESET}")
    print(f"{c.BOLD}{c.MAGENTA}{'=' * 80}{c.RESET}\n")

    print(f"{c.BOLD}Configuration:{c.RESET}")
    print(f"  Signal:     {c.GREEN}{signal}{c.RESET}")
    print(f"  Date range: {c.YELLOW}{start_ts.date()} to {end_ts.date()}{c.RESET}")
    print(f"  Data dir:   {data_dir}")
    print(f"  Regime dir: {regime_outputs_dir}")
    print()

    if signal == "all":
        print(f"{c.BOLD}Signals to run ({len(signal_names)}):{c.RESET}")
        for name in signal_names[:10]:
            print(f"  - {name}")
        if len(signal_names) > 10:
            print(f"  ... and {len(signal_names) - 10} more")
        print()

    if issues:
        print(f"{c.BOLD}{c.RED}Validation FAILED:{c.RESET}")
        for issue in issues:
            print(f"  {c.RED}x{c.RESET} {issue}")
        print()
        return 1

    print(f"{c.BOLD}{c.GREEN}Validation PASSED{c.RESET}")
    print(f"  {c.GREEN}+{c.RESET} All required files exist")
    print(f"  {c.GREEN}+{c.RESET} Date range is valid")
    print(f"  {c.GREEN}+{c.RESET} Signal configuration loaded")
    print()
    print(f"{c.CYAN}Ready to execute. Remove --dry-run to run the backtest.{c.RESET}")
    return 0


def main() -> int:
    args = _parse_args()

    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    # Setup logging
    _setup_logging(args.verbose, args.quiet)
    logger = logging.getLogger(__name__)

    configs = load_signal_configs()
    signal_names = sorted(configs.keys())

    if args.list_signals:
        c = Colors
        print(f"\n{c.BOLD}Available signals ({len(signal_names)}):{c.RESET}")
        for name in signal_names:
            print(f"  {c.GREEN}-{c.RESET} {name}")
        return 0

    if not configs:
        raise RuntimeError("No signal configs loaded from Models/configs.")

    signal_to_run = args.signal or args.factor or "all"
    if signal_to_run != "all" and signal_to_run not in signal_names:
        available_preview = ", ".join(signal_names[:30])
        raise ValueError(
            f"Unknown signal: {signal_to_run}\n"
            f"Available (first 30): {available_preview}\n"
            "Use --list-signals to print all."
        )

    start_ts, end_ts = _validate_dates(args.start_date, args.end_date)
    data_dir = _abs_path(args.data_dir, PROJECT_ROOT / "data")

    # Handle regime outputs directory
    validation_issues: list[str] = []
    try:
        regime_outputs_dir = _resolve_regime_outputs_dir(PROJECT_ROOT, args.regime_outputs)
    except FileNotFoundError as e:
        if args.dry_run:
            regime_outputs_dir = PROJECT_ROOT / "regime_filter" / "outputs"
            validation_issues.append(str(e).split("\n")[0])
        else:
            raise

    if not data_dir.exists():
        if args.dry_run:
            validation_issues.append(f"Data directory does not exist: {data_dir}")
            return _print_dry_run_summary(
                signal_to_run,
                start_ts,
                end_ts,
                data_dir,
                regime_outputs_dir,
                validation_issues,
                signal_names if signal_to_run == "all" else [signal_to_run],
            )
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    # Validate inputs
    validation_issues.extend(_validate_required_inputs(data_dir, regime_outputs_dir))

    # Handle dry-run mode
    if args.dry_run:
        return _print_dry_run_summary(
            signal_to_run,
            start_ts,
            end_ts,
            data_dir,
            regime_outputs_dir,
            validation_issues,
            signal_names if signal_to_run == "all" else [signal_to_run],
        )

    # In normal mode, raise on validation issues
    if validation_issues:
        for issue in validation_issues:
            logger.error(issue)
        raise FileNotFoundError("Missing required input files. Run with --dry-run for details.")

    timeline_mode = "config" if args.use_config_timeline else "cli_override"
    _print_header(signal_to_run, start_ts, end_ts, data_dir, regime_outputs_dir, timeline_mode)

    logger.debug("Loading portfolio engine...")
    engine = PortfolioEngine(
        data_dir=data_dir,
        regime_model_dir=regime_outputs_dir,
        start_date=str(start_ts.date()),
        end_date=str(end_ts.date()),
    )

    logger.debug("Loading all data...")
    engine.load_all_data()
    _validate_loaded_universe(engine, start_ts, end_ts)

    if signal_to_run == "all":
        if not args.use_config_timeline:
            for cfg in engine.signal_configs.values():
                tl = dict(cfg.get("timeline", {}))
                tl["start_date"] = str(start_ts.date())
                tl["end_date"] = str(end_ts.date())
                cfg["timeline"] = tl
        logger.info(f"Running all {len(signal_names)} signals...")
        engine.run_all_factors()
    else:
        override_cfg = None
        if not args.use_config_timeline:
            base_cfg = engine.signal_configs.get(signal_to_run)
            if base_cfg is None:
                raise ValueError(f"Signal config not found at runtime: {signal_to_run}")
            override_cfg = copy.deepcopy(base_cfg)
            tl = dict(override_cfg.get("timeline", {}))
            tl["start_date"] = str(start_ts.date())
            tl["end_date"] = str(end_ts.date())
            override_cfg["timeline"] = tl
        logger.info(f"Running signal: {signal_to_run}")
        engine.run_factor(signal_to_run, override_config=override_cfg)

    c = Colors
    print(f"\n{c.BOLD}{c.GREEN}Backtest completed successfully.{c.RESET}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        c = Colors
        print(f"{c.RED}x Backtest failed: {exc}{c.RESET}", file=sys.stderr)
        raise SystemExit(2)

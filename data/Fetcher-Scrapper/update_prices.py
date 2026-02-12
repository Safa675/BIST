#!/usr/bin/env python3
"""
Incremental price updater for BIST strategy data.

Updates both CSV and Parquet files to the latest available date:
  1. bist_prices_full.csv/.parquet   – all BIST stock OHLCV
  2. xu100_prices.csv/.parquet       – XU100 index OHLCV
  3. xau_try_2013_2026.csv/.parquet  – XAU/TRY (gold in lira)

Both CSV and Parquet versions are kept in sync for flexibility.
Parquet files are used for faster data loading in the portfolio engine.

Usage:
    python data/update_prices.py            # update all files
    python data/update_prices.py --dry-run  # show what would be fetched

Schedule with cron (every weekday at 18:45 Istanbul time):
    45 18 * * 1-5  cd /home/safa/Documents/Models/BIST && python data/update_prices.py >> data/update.log 2>&1
"""

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent
BIST_PRICES = DATA_DIR / "bist_prices_full.csv"
BIST_PRICES_PARQUET = DATA_DIR / "bist_prices_full.parquet"
XU100_PRICES = DATA_DIR / "xu100_prices.csv"
XU100_PRICES_PARQUET = DATA_DIR / "xu100_prices.parquet"
XAU_TRY_PRICES = DATA_DIR / "xau_try_2013_2026.csv"
XAU_TRY_PARQUET = DATA_DIR / "xau_try_2013_2026.parquet"

MNYET_URL = "https://finans.mynet.com/borsa/hisseler/"

BIST_COLS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
XU100_COLS = BIST_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def last_date_in_csv(path: Path, date_col: str = "Date") -> dt.date:
    """Read only the Date column and return the latest date."""
    df = pd.read_csv(path, usecols=[date_col], parse_dates=[date_col])
    return df[date_col].max().date()


def _fallback_bist_tickers_from_existing() -> list[str]:
    """Fallback ticker universe from local historical file."""
    if not BIST_PRICES.exists():
        return []

    try:
        existing = pd.read_csv(BIST_PRICES, usecols=["Ticker"])
    except Exception:
        return []

    tickers = (
        existing["Ticker"]
        .dropna()
        .astype(str)
        .str.replace(".IS", "", regex=False)
        .str.upper()
        .unique()
        .tolist()
    )
    return sorted(t for t in tickers if t.isalpha())


def fetch_bist_tickers() -> list[str]:
    try:
        tables = pd.read_html(MNYET_URL)
        if not tables:
            raise RuntimeError(f"No tables found at {MNYET_URL}")
        tickers = (
            tables[0]["Hisseler"]
            .astype(str)
            .str.split()
            .str[0]
            .dropna()
            .unique()
            .tolist()
        )
        return sorted(t for t in tickers if t.isalpha())
    except Exception as exc:
        print(f"  Warning: could not fetch ticker list from {MNYET_URL}: {exc}")
        fallback = _fallback_bist_tickers_from_existing()
        if fallback:
            print(f"  Using {len(fallback)} tickers from local price history fallback")
            return fallback
        raise RuntimeError("Ticker universe fetch failed and no local fallback is available") from exc


# ---------------------------------------------------------------------------
# 1. BIST all-stock prices
# ---------------------------------------------------------------------------

def update_bist_prices(dry_run: bool = False) -> None:
    print("\n" + "=" * 60)
    print("BIST STOCK PRICES")
    print("=" * 60)

    last = last_date_in_csv(BIST_PRICES)
    # Start from the day after the last row (yfinance start is inclusive)
    start = (last + dt.timedelta(days=1)).isoformat()
    end = dt.date.today().isoformat()

    print(f"  Last date in CSV : {last}")
    print(f"  Fetch window     : {start}  ->  {end}")

    if start >= end:
        print("  Already up to date.")
        return

    if dry_run:
        print("  [dry-run] Would fetch BIST prices.")
        return

    tickers = fetch_bist_tickers()
    print(f"  Ticker list      : {len(tickers)} BIST tickers")

    yf_tickers = [f"{t}.IS" for t in tickers]
    data = yf.download(
        yf_tickers,
        start=start,
        end=end,
        progress=True,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
    )

    if data is None or data.empty:
        print("  No new data returned by yfinance.")
        return

    records = []
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in yf_tickers:
            if ticker not in data.columns.get_level_values(0):
                continue
            df_t = data[ticker].copy().reset_index()
            df_t["Ticker"] = ticker
            records.append(df_t)
    else:
        df_single = data.copy().reset_index()
        df_single["Ticker"] = yf_tickers[0]
        records.append(df_single)

    if not records:
        print("  No records after parsing.")
        return

    new_df = pd.concat(records, ignore_index=True)
    new_df = new_df.rename(columns=str.title)
    # Ensure column order matches existing file
    for col in BIST_COLS:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df[BIST_COLS]
    new_df["Date"] = pd.to_datetime(new_df["Date"])

    # Drop rows that are entirely NaN for OHLCV (holiday artifacts)
    ohlcv = ["Open", "High", "Low", "Close", "Volume"]
    new_df = new_df.dropna(subset=ohlcv, how="all")

    if new_df.empty:
        print("  No valid new rows after cleanup.")
        return

    # Append
    existing = pd.read_csv(BIST_PRICES, parse_dates=["Date"])
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    combined.to_csv(BIST_PRICES, index=False)
    
    # Also save as parquet for faster loading
    combined.to_parquet(BIST_PRICES_PARQUET, index=False)
    print(f"  ✅ Parquet updated: {BIST_PRICES_PARQUET.name}")

    new_last = combined["Date"].max().date()
    print(f"  Appended {len(new_df)} rows  ->  new last date: {new_last}")
    print(f"  Total rows: {len(combined)}")


# ---------------------------------------------------------------------------
# 2. XU100 index prices
# ---------------------------------------------------------------------------

def update_xu100_prices(dry_run: bool = False) -> None:
    print("\n" + "=" * 60)
    print("XU100 INDEX PRICES")
    print("=" * 60)

    last = last_date_in_csv(XU100_PRICES)
    start = (last + dt.timedelta(days=1)).isoformat()
    end = dt.date.today().isoformat()

    print(f"  Last date in CSV : {last}")
    print(f"  Fetch window     : {start}  ->  {end}")

    if start >= end:
        print("  Already up to date.")
        return

    if dry_run:
        print("  [dry-run] Would fetch XU100 prices.")
        return

    for ticker in ("XU100.IS", "XU100"):
        data = yf.download(
            ticker,
            start=start,
            end=end,
            progress=True,
            auto_adjust=False,
            threads=True,
        )
        if data is not None and not data.empty:
            break
    else:
        print("  No XU100 data returned.")
        return

    # Flatten MultiIndex columns that yfinance may return for single tickers
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    new_df = data.copy().reset_index()
    new_df["Ticker"] = "XU100.IS"
    new_df = new_df.rename(columns=str.title)
    for col in XU100_COLS:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df[XU100_COLS]
    new_df["Date"] = pd.to_datetime(new_df["Date"])

    if new_df.empty:
        print("  No valid new rows.")
        return

    existing = pd.read_csv(XU100_PRICES, parse_dates=["Date"])
    # Remove the header artifact row if present (,,Xu100.Is,...)
    existing = existing[existing["Date"].notna()]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date"], keep="last")
    combined = combined.sort_values("Date").reset_index(drop=True)
    combined.to_csv(XU100_PRICES, index=False)
    
    # Also save as parquet for faster loading
    combined.to_parquet(XU100_PRICES_PARQUET, index=False)
    print(f"  ✅ Parquet updated: {XU100_PRICES_PARQUET.name}")

    new_last = combined["Date"].max().date()
    print(f"  Appended {len(new_df)} rows  ->  new last date: {new_last}")


# ---------------------------------------------------------------------------
# 3. XAU/TRY (gold price in Turkish lira)
# ---------------------------------------------------------------------------

def update_xau_try(dry_run: bool = False) -> None:
    print("\n" + "=" * 60)
    print("XAU/TRY PRICES")
    print("=" * 60)

    last = last_date_in_csv(XAU_TRY_PRICES)
    start = (last + dt.timedelta(days=1)).isoformat()
    end = dt.date.today().isoformat()

    print(f"  Last date in CSV : {last}")
    print(f"  Fetch window     : {start}  ->  {end}")

    if start >= end:
        print("  Already up to date.")
        return

    if dry_run:
        print("  [dry-run] Would fetch XAU/TRY prices.")
        return

    # Download gold (USD) and USD/TRY
    def _get_close(ticker, start, end):
        raw = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw["Close"]

    try:
        xau = _get_close("XAUUSD=X", start, end)
    except Exception:
        xau = _get_close("GC=F", start, end)

    usd_try = _get_close("USDTRY=X", start, end)

    new_df = pd.concat([xau, usd_try], axis=1)
    new_df.columns = ["XAU_USD", "USD_TRY"]
    new_df["XAU_TRY"] = new_df["XAU_USD"] * new_df["USD_TRY"]
    new_df = new_df.dropna()
    new_df.index.name = "Date"

    if new_df.empty:
        print("  No valid new rows.")
        return

    existing = pd.read_csv(XAU_TRY_PRICES, parse_dates=["Date"], index_col="Date")
    combined = pd.concat([existing, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined.to_csv(XAU_TRY_PRICES)
    
    # Also save as parquet for faster loading
    combined.to_parquet(XAU_TRY_PARQUET)
    print(f"  ✅ Parquet updated: {XAU_TRY_PARQUET.name}")

    new_last = combined.index.max().date()
    print(f"  Appended {len(new_df)} rows  ->  new last date: {new_last}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Incrementally update BIST price data files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without downloading.",
    )
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"BIST DATA UPDATER  —  {dt.datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'=' * 60}")

    failures: list[tuple[str, Exception]] = []

    steps = [
        ("BIST STOCK PRICES", update_bist_prices),
        ("XU100 INDEX PRICES", update_xu100_prices),
        ("XAU/TRY PRICES", update_xau_try),
    ]
    for label, step in steps:
        try:
            step(dry_run=args.dry_run)
        except Exception as exc:
            failures.append((label, exc))
            print(f"\n  ERROR in {label}: {exc}")

    print("\n" + "=" * 60)
    if failures:
        print(f"UPDATES COMPLETED WITH {len(failures)} FAILURE(S)")
        for label, exc in failures:
            print(f"  - {label}: {exc}")
        print("=" * 60)
        return 1

    print("ALL UPDATES COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

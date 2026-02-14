"""
Fetch XU100 index prices from Yahoo Finance.

Usage:
    python "data/Fetch Scripts/xu100_fetcher.py" --start 2013-01-01 --end 2026-12-31 --out data/xu100_prices.csv
"""

import logging
import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf
logger = logging.getLogger(__name__)


YFINANCE_XU100_TICKERS: tuple[str, ...] = ("XU100.IS", "XU100")


def _download_single_ticker(
    ticker: str,
    start: str,
    end: Optional[str],
    progress: bool = True,
) -> Optional[pd.DataFrame]:
    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=progress,
        auto_adjust=False,
        threads=True,
    )
    if data is None or data.empty:
        return None

    df = data.copy().reset_index()
    df["Ticker"] = ticker
    df = df.rename(columns=str.title)
    cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return df[cols]


def download_xu100_prices(
    start: str,
    end: Optional[str],
    tickers: Iterable[str] = YFINANCE_XU100_TICKERS,
    progress: bool = True,
) -> tuple[pd.DataFrame, str]:
    for ticker in tickers:
        df = _download_single_ticker(ticker, start=start, end=end, progress=progress)
        if df is not None:
            return df, ticker

    raise RuntimeError(
        f"No XU100 data returned from yfinance using tickers: {', '.join(tickers)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch XU100 index prices via Yahoo Finance.")
    parser.add_argument(
        "--start",
        default="2013-04-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument("--end", default=None, help="End date (2026-01-25), defaults to today")
    parser.add_argument(
        "--out",
        default="data/xu100_prices.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    end_date = args.end or dt.date.today().isoformat()

    logger.info(f"Downloading XU100 prices from {args.start} to {end_date}...")
    prices, ticker_used = download_xu100_prices(
        start=args.start,
        end=end_date,
        progress=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out_path, index=False)
    logger.info(f"Saved {len(prices)} rows for {ticker_used} to {out_path}")


if __name__ == "__main__":
    main()

"""
Batch market cap/share fetcher for BIST tickers via isyatirimhisse.

What it does:
- Pulls daily market data for each ticker between START_DATE and END_DATE.
- Extracts market cap (PD) and shares outstanding (SERMAYE) columns.
- Resamples to quarter-end values (last trading day of each quarter).
- Saves per-ticker Excel (daily + quarterly) and CSV (quarterly only).
- Builds a combined quarterly CSV across all tickers.

Run: python isyatirim_marketcaps.py
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from isyatirimhisse import fetch_stock_data


START_DATE = "01-01-2016"
END_DATE = "31-12-2026"
OUTPUT_DIR = Path("isyatirim_marketcaps")

# Full ticker universe provided
TICKERS: List[str] = [
    "THYAO"
]


def pick_date_column(df: pd.DataFrame) -> str:
    # Common possibilities (library/website can change column naming)
    exact_matches = {"date", "tarih"}
    contains_matches = {"date", "tarih"}
    lower_cols = {col: str(col).lower() for col in df.columns}

    for col, low in lower_cols.items():
        if low in exact_matches:
            return col

    for col, low in lower_cols.items():
        if any(token in low for token in contains_matches):
            return col

    raise ValueError(f"Could not find a date column. Columns: {list(df.columns)}")


def pick_market_cap_column(df: pd.DataFrame) -> str:
    # Prefer total market cap (PD) over free-float alternatives.
    candidates = [
        "PD", "pd", "Pd",
        "market_cap", "MarketCap", "marketcap",
        "piyasa_degeri", "piyasadegeri",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    lower_cols = {col: str(col).lower() for col in df.columns}
    for col, low in lower_cols.items():
        if any(token in low for token in ["pd", "marketcap", "market_cap", "piyasa"]) and "hao" not in low:
            return col

    raise ValueError(f"Could not find a market cap column. Columns: {list(df.columns)}")


def pick_shares_column(df: pd.DataFrame) -> str:
    candidates = [
        "SERMAYE", "sermaye", "Sermaye",
        "shares_outstanding", "shares", "outstanding_shares",
        "outstanding", "kapanan_sermaye",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    lower_cols = {col: str(col).lower() for col in df.columns}
    for col, low in lower_cols.items():
        if any(token in low for token in ["sermaye", "share", "outstanding"]):
            return col

    raise ValueError(f"Could not find a shares outstanding column. Columns: {list(df.columns)}")


def fetch_and_process(ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """
    Returns (daily_df, quarterly_df, error_message)
    """
    try:
        df_daily = fetch_stock_data(
            symbols=ticker,
            start_date=START_DATE,
            end_date=END_DATE,
            save_to_excel=False,
        )
    except Exception as exc:  # noqa: BLE001
        return None, None, f"fetch failed: {exc}"

    if df_daily is None or len(df_daily) == 0:
        return None, None, "no data returned"

    try:
        date_col = pick_date_column(df_daily)
        df_daily[date_col] = pd.to_datetime(df_daily[date_col], dayfirst=True, errors="coerce")
        df_daily = df_daily.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

        mcap_col = pick_market_cap_column(df_daily)
        shares_col = pick_shares_column(df_daily)

        numeric_cols = [mcap_col, shares_col]
        df_daily[numeric_cols] = df_daily[numeric_cols].apply(pd.to_numeric, errors="coerce")

        selected = df_daily[numeric_cols].ffill()
        if selected.dropna(how="all").empty:
            return df_daily, None, "no usable market cap or shares data"

        df_quarterly = selected.resample("Q").last()
        df_quarterly = df_quarterly.rename(columns={
            mcap_col: f"{ticker}_MarketCap_QuarterEnd",
            shares_col: f"{ticker}_SharesOutstanding_QuarterEnd",
        })
        df_quarterly.index.name = "QuarterEnd"
    except Exception as exc:  # noqa: BLE001
        return df_daily, None, f"processing failed: {exc}"

    return df_daily, df_quarterly, None


def save_outputs(ticker: str, daily: pd.DataFrame, quarterly: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"{ticker}_2016_2026"
    out_xlsx = OUTPUT_DIR / f"{base_name}_daily_and_quarterly.xlsx"
    out_csv = OUTPUT_DIR / f"{base_name}_quarterly.csv"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        daily.to_excel(writer, sheet_name="daily")
        quarterly.to_excel(writer, sheet_name="quarterly")

    quarterly.to_csv(out_csv, encoding="utf-8-sig")


def main() -> None:
    successes: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}

    print(f"Fetching {len(TICKERS)} tickers from {START_DATE} to {END_DATE}...")
    for i, ticker in enumerate(TICKERS, start=1):
        print(f"[{i}/{len(TICKERS)}] {ticker} ...", end="", flush=True)
        daily, quarterly, err = fetch_and_process(ticker)
        if err:
            errors[ticker] = err
            print(f" failed ({err})")
            continue

        save_outputs(ticker, daily, quarterly)
        successes[ticker] = quarterly
        print(f" ok (daily {len(daily):,} rows, quarterly {len(quarterly):,} rows)")

    if successes:
        combined = pd.concat(successes.values(), axis=1)
        combined.index.name = "QuarterEnd"
        combined.to_csv(OUTPUT_DIR / "all_tickers_quarterly_2016_2026.csv", encoding="utf-8-sig")
        print(f"\nCombined quarterly saved to {OUTPUT_DIR/'all_tickers_quarterly_2016_2026.csv'} "
              f"with shape {combined.shape}")

    if errors:
        print("\nCompleted with errors:")
        for t, msg in errors.items():
            print(f"- {t}: {msg}")
    else:
        print("\nCompleted without errors.")


if __name__ == "__main__":
    main()

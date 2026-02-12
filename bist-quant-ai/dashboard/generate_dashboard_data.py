#!/usr/bin/env python3
"""
Dashboard Data Generator

Generates JSON data for the BIST Factor Models Dashboard.
Reads results from all factor models and creates a consolidated view,
including custom portfolio performance vs XU030 and XU100.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_FILE = Path(__file__).resolve()
DASHBOARD_DIR = SCRIPT_FILE.parent
PROJECT_ROOT = DASHBOARD_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "Models"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR_CANDIDATES = [
    MODELS_DIR / "results",
    PROJECT_ROOT / "signals" / "results",
    MODELS_DIR / "signals" / "results",
]
SIMPLE_REGIME_OUTPUTS_DIR = PROJECT_ROOT / "Simple Regime Filter" / "outputs"
LEGACY_REGIME_OUTPUTS_DIR = PROJECT_ROOT / "Regime Filter" / "outputs"
MANUAL_HOLDINGS_FILE = DASHBOARD_DIR / "manual_portfolio_holdings.csv"
MANUAL_TRADES_FILE = DASHBOARD_DIR / "manual_trades.csv"
MANUAL_PORTFOLIO_DAILY_FILE = DASHBOARD_DIR / "manual_portfolio_daily.csv"
PUBLIC_DASHBOARD_FILE = PROJECT_ROOT / "bist-quant-ai" / "public" / "data" / "dashboard_data.json"


def _resolve_results_dir():
    """Resolve first available results directory from known locations."""
    for candidate in RESULTS_DIR_CANDIDATES:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return RESULTS_DIR_CANDIDATES[0]


RESULTS_DIR = _resolve_results_dir()

# Allow importing sibling modules in Models/
sys.path.insert(0, str(MODELS_DIR))

try:
    from analyze_2026_daily import (
        _annualized_tracking_error,
        _information_ratio,
        _load_xu030_from_yfinance,
    )
except Exception:
    _annualized_tracking_error = None
    _information_ratio = None
    _load_xu030_from_yfinance = None


def _env_int(name, default, minimum=1):
    """Read positive integer from environment with a safe fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _env_flag(name, default=False):
    """Read boolean-ish environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_csv_columns(csv_path):
    """Read only CSV headers to discover available columns."""
    try:
        return pd.read_csv(csv_path, nrows=0).columns.tolist()
    except Exception:
        return []


def _read_parquet_columns(parquet_path):
    """Read parquet schema column names with graceful fallback."""
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(parquet_path).names)
    except Exception:
        return []


def _dedupe_tickers(tickers):
    """Normalize and dedupe tickers while preserving order."""
    seen = set()
    result = []
    for raw in tickers or []:
        ticker = _normalize_ticker(raw)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        result.append(ticker)
    return result


def _truncate_list(values, limit, keep_tail=False):
    """Return a bounded list slice."""
    seq = list(values or [])
    if limit <= 0 or len(seq) <= limit:
        return seq
    if keep_tail:
        return seq[-limit:]
    return seq[:limit]


def _read_csv_with_date_index(csv_path):
    """Read CSV and set the first date-like column as datetime index."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    date_col = next((c for c in ("date", "Date", "DATE") if c in df.columns), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    return df.set_index(date_col)


def _extract_metric(content, patterns, default=0.0):
    """Extract first matching numeric metric using regex patterns."""
    if isinstance(patterns, str):
        patterns = [patterns]

    for pattern in patterns:
        match = re.search(pattern, content, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                continue
    return default


def _to_float_or_none(value):
    """Convert number-like value to float, but keep missing values as None."""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(num) or np.isinf(num):
        return None
    return num


def _to_int_or_none(value):
    """Convert numeric-like value to int or None."""
    num = _to_float_or_none(value)
    if num is None:
        return None
    return int(round(num))


def _to_datetime_or_none(value):
    """Parse value to pandas Timestamp or return None."""
    if value is None:
        return None
    raw = str(value).strip()
    iso_like = bool(re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", raw))
    if iso_like:
        ts = pd.to_datetime(raw, errors="coerce", dayfirst=False)
        if pd.isna(ts):
            ts = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    else:
        ts = pd.to_datetime(raw, errors="coerce", dayfirst=True)
        if pd.isna(ts):
            ts = pd.to_datetime(raw, errors="coerce", dayfirst=False)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _normalize_ticker(raw):
    """Normalize ticker to BIST core code (e.g., AKBNK from AKBNK.IS)."""
    if raw is None or pd.isna(raw):
        return ""
    ticker = str(raw).strip().upper()
    ticker = ticker.replace(".IS", "")
    return ticker


def _ensure_manual_input_files():
    """Create manual input templates if missing."""
    if not MANUAL_HOLDINGS_FILE.exists():
        MANUAL_HOLDINGS_FILE.write_text(
            "date,ticker,weight,quantity,avg_cost,currency,note\n",
            encoding="utf-8",
        )

    if not MANUAL_TRADES_FILE.exists():
        MANUAL_TRADES_FILE.write_text(
            "datetime,ticker,side,quantity,price,fee,currency,note\n",
            encoding="utf-8",
        )

    if not MANUAL_PORTFOLIO_DAILY_FILE.exists():
        MANUAL_PORTFOLIO_DAILY_FILE.write_text(
            "date,portfolio_return,xu030_return,xu100_return,"
            "portfolio_cumulative,xu030_cumulative,xu100_cumulative\n",
            encoding="utf-8",
        )

    # Backward-compat: expand legacy files that only had a subset of columns.
    _ensure_csv_schema(
        MANUAL_HOLDINGS_FILE,
        required_columns=["date", "ticker", "weight", "quantity", "avg_cost", "currency", "note"],
        default_values={"currency": "TRY"},
    )
    _ensure_csv_schema(
        MANUAL_TRADES_FILE,
        required_columns=["datetime", "ticker", "side", "quantity", "price", "fee", "currency", "note"],
        default_values={"currency": "TRY"},
    )


def _ensure_csv_schema(csv_path, required_columns, default_values=None):
    """Ensure CSV has required columns; append missing columns preserving existing data."""
    default_values = default_values or {}
    if not csv_path.exists():
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return

    changed = False
    for col in required_columns:
        if col not in df.columns:
            df[col] = default_values.get(col, "")
            changed = True

    if not changed:
        return

    ordered_cols = [c for c in required_columns if c in df.columns] + [c for c in df.columns if c not in required_columns]
    df = df[ordered_cols]
    df.to_csv(csv_path, index=False)


def _load_manual_holdings():
    """Load manually maintained holdings with optional date-based snapshots."""
    notes = []
    rows = []

    if not MANUAL_HOLDINGS_FILE.exists():
        return {
            "file": str(MANUAL_HOLDINGS_FILE.relative_to(PROJECT_ROOT)),
            "rows": [],
            "weights": {},
            "tickers": [],
            "notes": ["Manual holdings file not found."],
        }

    df = pd.read_csv(MANUAL_HOLDINGS_FILE)
    if df.empty:
        return {
            "file": str(MANUAL_HOLDINGS_FILE.relative_to(PROJECT_ROOT)),
            "rows": [],
            "weights": {},
            "tickers": [],
            "notes": ["Manual holdings file is empty."],
        }

    ticker_col = next((c for c in df.columns if c.lower() == "ticker"), None)
    if ticker_col is None:
        return {
            "file": str(MANUAL_HOLDINGS_FILE.relative_to(PROJECT_ROOT)),
            "rows": [],
            "weights": {},
            "tickers": [],
            "notes": ["'ticker' column is required in manual_portfolio_holdings.csv"],
        }

    date_col = next((c for c in df.columns if c.lower() in ("date", "as_of", "asof", "snapshot_date")), None)
    weight_col = next((c for c in df.columns if c.lower() == "weight"), None)
    qty_col = next((c for c in df.columns if c.lower() in ("quantity", "qty")), None)
    avg_cost_col = next((c for c in df.columns if c.lower() in ("avg_cost", "average_cost", "cost")), None)
    ccy_col = next((c for c in df.columns if c.lower() in ("currency", "ccy")), None)
    note_col = next((c for c in df.columns if c.lower() in ("note", "notes", "comment")), None)

    parsed_rows = []
    for _, row in df.iterrows():
        ticker = _normalize_ticker(row.get(ticker_col))
        if not ticker:
            continue

        snapshot_date = _to_datetime_or_none(row.get(date_col)) if date_col else None
        raw_weight = _to_float_or_none(row.get(weight_col)) if weight_col else None
        if raw_weight is not None and raw_weight > 1:
            raw_weight = raw_weight / 100.0

        parsed_rows.append(
            {
                "date": snapshot_date,
                "ticker": ticker,
                "raw_weight": raw_weight,
                "quantity": _to_int_or_none(row.get(qty_col)) if qty_col else None,
                "avg_cost": _to_float_or_none(row.get(avg_cost_col)) if avg_cost_col else None,
                "currency": str(row.get(ccy_col)).strip() if ccy_col and pd.notna(row.get(ccy_col)) else "TRY",
                "note": str(row.get(note_col)).strip() if note_col and pd.notna(row.get(note_col)) else "",
            }
        )

    if not parsed_rows:
        notes.append("No valid holdings rows found in manual holdings file.")

    has_dated_rows = bool(date_col and any(r["date"] is not None for r in parsed_rows))
    normalized_weights = {}
    as_of_date = None
    daily_weights = {}

    if has_dated_rows:
        valid_dates = [r["date"] for r in parsed_rows if r["date"] is not None]
        latest_ts = max(valid_dates) if valid_dates else None
        if latest_ts is not None:
            as_of_date = latest_ts.strftime("%Y-%m-%d")

        missing_date_rows = 0
        if latest_ts is not None:
            for parsed_row in parsed_rows:
                if parsed_row["date"] is None:
                    parsed_row["date"] = latest_ts
                    missing_date_rows += 1
        if missing_date_rows:
            notes.append(f"{missing_date_rows} holdings rows had no date; assigned to latest snapshot {as_of_date}.")

        snapshots = {}
        for parsed_row in parsed_rows:
            if parsed_row["date"] is None:
                continue
            date_key = parsed_row["date"].strftime("%Y-%m-%d")
            snapshots.setdefault(date_key, []).append(parsed_row)

        equal_weight_days = []
        for date_key, snapshot_rows in sorted(snapshots.items()):
            weighted = {}
            for snapshot_row in snapshot_rows:
                ticker = snapshot_row["ticker"]
                raw_weight = _to_float_or_none(snapshot_row.get("raw_weight"))
                if raw_weight is not None and raw_weight > 0:
                    weighted[ticker] = weighted.get(ticker, 0.0) + raw_weight

            total_weight = sum(weighted.values())
            if total_weight > 0:
                snapshot_weights = {k: v / total_weight for k, v in weighted.items()}
            else:
                unique_tickers = sorted({item["ticker"] for item in snapshot_rows})
                if unique_tickers:
                    eq = 1.0 / len(unique_tickers)
                    snapshot_weights = {t: eq for t in unique_tickers}
                    equal_weight_days.append(date_key)
                else:
                    snapshot_weights = {}

            daily_weights[date_key] = snapshot_weights

        if equal_weight_days:
            preview = ", ".join(equal_weight_days[:5])
            suffix = " ..." if len(equal_weight_days) > 5 else ""
            notes.append(f"Weights missing on snapshot date(s) {preview}{suffix}; equal-weight assumed per day.")

        latest_rows = snapshots.get(as_of_date, []) if as_of_date else []
        normalized_weights = daily_weights.get(as_of_date, {}) if as_of_date else {}
        for parsed_row in latest_rows:
            rows.append(
                {
                    "date": as_of_date,
                    "ticker": parsed_row["ticker"],
                    "weight": _to_float_or_none(normalized_weights.get(parsed_row["ticker"])),
                    "quantity": parsed_row["quantity"],
                    "avg_cost": parsed_row["avg_cost"],
                    "currency": parsed_row["currency"],
                    "note": parsed_row["note"],
                }
            )

        if len(daily_weights) > 1 and as_of_date:
            notes.append(
                f"Loaded {len(daily_weights)} dated holdings snapshots; "
                f"showing latest snapshot {as_of_date} in holdings table."
            )
    else:
        if date_col:
            notes.append("Date column found but no valid dates parsed; treating holdings as a single snapshot.")

        weighted = {}
        for parsed_row in parsed_rows:
            ticker = parsed_row["ticker"]
            raw_weight = _to_float_or_none(parsed_row.get("raw_weight"))
            if raw_weight is not None and raw_weight > 0:
                weighted[ticker] = weighted.get(ticker, 0.0) + raw_weight

            rows.append(
                {
                    "date": None,
                    "ticker": ticker,
                    "weight": _to_float_or_none(raw_weight),
                    "quantity": parsed_row["quantity"],
                    "avg_cost": parsed_row["avg_cost"],
                    "currency": parsed_row["currency"],
                    "note": parsed_row["note"],
                }
            )

        total_weight = sum(weighted.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in weighted.items()}
        else:
            unique_tickers = sorted({r["ticker"] for r in rows})
            if unique_tickers:
                eq = 1.0 / len(unique_tickers)
                normalized_weights = {t: eq for t in unique_tickers}
                notes.append("Weights missing; equal-weight portfolio assumed from manual holdings.")
            else:
                normalized_weights = {}

        for row in rows:
            row["weight"] = _to_float_or_none(normalized_weights.get(row["ticker"]))

    qty_tickers = sorted({r["ticker"] for r in rows if r.get("quantity") is not None and r.get("quantity") > 0})
    latest_price_by_ticker = {}
    market_value_date = None
    if qty_tickers:
        close_panel = _load_close_panel_for_tickers(qty_tickers)
        if not close_panel.empty:
            market_value_date = close_panel.index.max().strftime("%Y-%m-%d")
            latest_row = close_panel.ffill().iloc[-1]
            latest_price_by_ticker = {t: _to_float_or_none(latest_row.get(t)) for t in qty_tickers}
        else:
            notes.append("No market prices available to compute total portfolio market value from quantities.")

    total_cost_value = 0.0
    total_market_value = 0.0
    has_cost_value = False
    has_market_value = False
    with_qty_count = 0
    with_cost_count = 0
    with_market_price_count = 0
    for row in rows:
        quantity = row.get("quantity")
        avg_cost = row.get("avg_cost")
        ticker = row.get("ticker")
        if quantity is not None and quantity > 0:
            with_qty_count += 1
        if quantity is not None and avg_cost is not None:
            with_cost_count += 1

        cost_value = None
        if quantity is not None and avg_cost is not None:
            cost_value = _to_float_or_none(quantity * avg_cost)
            if cost_value is not None:
                total_cost_value += cost_value
                has_cost_value = True

        latest_price = _to_float_or_none(latest_price_by_ticker.get(ticker))
        market_value = None
        if quantity is not None and latest_price is not None:
            market_value = _to_float_or_none(quantity * latest_price)
            if market_value is not None:
                total_market_value += market_value
                has_market_value = True
                with_market_price_count += 1

        row["last_price"] = latest_price
        row["cost_value"] = cost_value
        row["market_value"] = market_value

    return {
        "file": str(MANUAL_HOLDINGS_FILE.relative_to(PROJECT_ROOT)),
        "rows": rows,
        "weights": {k: _to_float_or_none(v) for k, v in normalized_weights.items()},
        "daily_weights": {d: {k: _to_float_or_none(v) for k, v in w.items()} for d, w in daily_weights.items()},
        "as_of_date": as_of_date,
        "tickers": sorted(normalized_weights.keys()),
        "summary": {
            "positions": int(len(sorted(normalized_weights.keys()))),
            "positions_with_qty": int(with_qty_count),
            "positions_with_cost": int(with_cost_count),
            "positions_with_market_price": int(with_market_price_count),
            "total_cost_value": _to_float_or_none(total_cost_value) if has_cost_value else None,
            "total_market_value": _to_float_or_none(total_market_value) if has_market_value else None,
            "market_value_date": market_value_date,
            "currency": "TRY",
        },
        "notes": notes,
    }


def _load_manual_trades():
    """Load manually maintained trade log with date+hour details."""
    if not MANUAL_TRADES_FILE.exists():
        return {
            "file": str(MANUAL_TRADES_FILE.relative_to(PROJECT_ROOT)),
            "rows": [],
            "summary": {},
            "notes": ["Manual trades file not found."],
        }

    df = pd.read_csv(MANUAL_TRADES_FILE)
    if df.empty:
        return {
            "file": str(MANUAL_TRADES_FILE.relative_to(PROJECT_ROOT)),
            "rows": [],
            "summary": {},
            "notes": ["Manual trades file is empty."],
        }

    ticker_col = next((c for c in df.columns if c.lower() == "ticker"), None)
    side_col = next((c for c in df.columns if c.lower() in ("side", "action", "type")), None)
    qty_col = next((c for c in df.columns if c.lower() in ("quantity", "qty")), None)
    price_col = next((c for c in df.columns if c.lower() in ("price", "fill_price")), None)
    fee_col = next((c for c in df.columns if c.lower() in ("fee", "fees", "commission")), None)
    ccy_col = next((c for c in df.columns if c.lower() in ("currency", "ccy")), None)
    note_col = next((c for c in df.columns if c.lower() in ("note", "notes", "comment")), None)
    datetime_col = next((c for c in df.columns if c.lower() in ("datetime", "timestamp", "date_time")), None)
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    time_col = next((c for c in df.columns if c.lower() == "time"), None)

    rows = []
    for _, row in df.iterrows():
        raw_dt = None
        if datetime_col:
            raw_dt = row.get(datetime_col)
        elif date_col:
            date_str = str(row.get(date_col)).strip()
            time_str = str(row.get(time_col)).strip() if time_col else "00:00:00"
            raw_dt = f"{date_str} {time_str}"

        dt = _to_datetime_or_none(raw_dt)
        ticker = _normalize_ticker(row.get(ticker_col)) if ticker_col else ""
        side = str(row.get(side_col)).strip().upper() if side_col and pd.notna(row.get(side_col)) else ""
        side = {"B": "BUY", "S": "SELL"}.get(side, side)
        quantity = _to_int_or_none(row.get(qty_col)) if qty_col else None
        price = _to_float_or_none(row.get(price_col)) if price_col else None
        fee = _to_float_or_none(row.get(fee_col)) if fee_col else None
        currency = str(row.get(ccy_col)).strip() if ccy_col and pd.notna(row.get(ccy_col)) else "TRY"
        note = str(row.get(note_col)).strip() if note_col and pd.notna(row.get(note_col)) else ""

        if not ticker and dt is None and quantity is None and price is None:
            continue

        gross = None
        if quantity is not None and price is not None:
            gross = float(quantity * price)

        rows.append(
            {
                "datetime": dt.strftime("%Y-%m-%d %H:%M:%S") if dt is not None else None,
                "date": dt.strftime("%Y-%m-%d") if dt is not None else None,
                "time": dt.strftime("%H:%M:%S") if dt is not None else None,
                "ticker": ticker or None,
                "side": side or None,
                "quantity": quantity,
                "price": price,
                "gross_amount": gross,
                "fee": fee,
                "currency": currency,
                "note": note,
            }
        )

    rows = sorted(rows, key=lambda x: (x["datetime"] or ""), reverse=True)
    buy_count = sum(1 for r in rows if r.get("side") == "BUY")
    sell_count = sum(1 for r in rows if r.get("side") == "SELL")

    return {
        "file": str(MANUAL_TRADES_FILE.relative_to(PROJECT_ROOT)),
        "rows": rows,
        "summary": {
            "total_trades": len(rows),
            "buy_trades": buy_count,
            "sell_trades": sell_count,
            "last_trade_at": rows[0]["datetime"] if rows else None,
        },
        "notes": [],
    }


def _load_current_holdings(holdings_file):
    """
    Load latest holdings supporting both formats:
    1) Long: date,ticker,weight,...
    2) Wide matrix: date as index, tickers as columns
    """
    holdings_df = pd.read_csv(holdings_file)
    if holdings_df.empty:
        return []

    lower_map = {c.lower(): c for c in holdings_df.columns}
    has_long_schema = {"date", "ticker", "weight"}.issubset(lower_map.keys())

    if has_long_schema:
        date_col = lower_map["date"]
        ticker_col = lower_map["ticker"]
        weight_col = lower_map["weight"]

        holdings_df[date_col] = pd.to_datetime(holdings_df[date_col], errors="coerce")
        holdings_df[weight_col] = pd.to_numeric(holdings_df[weight_col], errors="coerce")
        holdings_df = holdings_df.dropna(subset=[date_col, ticker_col, weight_col])

        if holdings_df.empty:
            return []

        latest_date = holdings_df[date_col].max()
        latest = holdings_df[holdings_df[date_col] == latest_date]
        latest = latest[latest[weight_col] > 0].sort_values(weight_col, ascending=False)
        tickers = latest[ticker_col].astype(str).tolist()
        return list(dict.fromkeys(tickers))

    # Fallback for old wide-format holdings files
    wide_df = _read_csv_with_date_index(holdings_file)
    if wide_df.empty:
        return []

    latest_row = wide_df.iloc[-1]
    numeric_weights = pd.to_numeric(latest_row, errors="coerce")
    return [
        str(ticker)
        for ticker, weight in numeric_weights.items()
        if pd.notna(weight) and weight > 0
    ]


def _load_close_panel_for_tickers(tickers):
    """Load close/adj-close panel for selected tickers from local BIST price data."""
    tickers = _dedupe_tickers(tickers)
    if not tickers:
        return pd.DataFrame()

    parquet_file = DATA_DIR / "bist_prices_full.parquet"
    csv_file = DATA_DIR / "bist_prices_full.csv"
    ticker_filter = tickers + [f"{t}.IS" for t in tickers]

    if parquet_file.exists():
        available_cols = set(_read_parquet_columns(parquet_file))
        required_cols = {"Date", "Ticker"}
        if not required_cols.issubset(available_cols):
            return pd.DataFrame()

        price_cols = [c for c in ("Adj Close", "Close") if c in available_cols]
        if not price_cols:
            return pd.DataFrame()

        use_cols = ["Date", "Ticker"] + price_cols
        try:
            df = pd.read_parquet(parquet_file, columns=use_cols, filters=[("Ticker", "in", ticker_filter)])
        except Exception:
            df = pd.read_parquet(parquet_file, columns=use_cols)
    elif csv_file.exists():
        available_cols = set(_read_csv_columns(csv_file))
        required_cols = {"Date", "Ticker"}
        if not required_cols.issubset(available_cols):
            return pd.DataFrame()

        price_cols = [c for c in ("Adj Close", "Close") if c in available_cols]
        if not price_cols:
            return pd.DataFrame()

        use_cols = ["Date", "Ticker"] + price_cols
        df = pd.read_csv(csv_file, usecols=use_cols)
    else:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.replace(".IS", "", regex=False)

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df = df[df["Ticker"].isin(tickers)]
    df = df.dropna(subset=["Date", "Ticker", price_col])
    if df.empty:
        return pd.DataFrame()

    panel = df.pivot_table(index="Date", columns="Ticker", values=price_col, aggfunc="last").sort_index()
    return panel


def _load_xu100_returns(start_date, end_date):
    """Load XU100 daily returns from local CSV."""
    xu100_file = DATA_DIR / "xu100_prices.csv"
    if not xu100_file.exists():
        return pd.Series(dtype=float)

    df = pd.read_csv(xu100_file)
    if df.empty:
        return pd.Series(dtype=float)

    date_col = next((c for c in ("Date", "date", "DATE") if c in df.columns), None)
    if date_col is None:
        return pd.Series(dtype=float)

    price_col = next((c for c in ("Adj Close", "Close", "close", "adj_close") if c in df.columns), None)
    if price_col is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return pd.Series(dtype=float)
        price_col = numeric_cols[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[date_col, price_col]).sort_values(date_col)

    prices = df.set_index(date_col)[price_col].astype(float)
    returns = prices.pct_change(fill_method=None).dropna()
    returns = returns[(returns.index >= start_date) & (returns.index <= end_date)]
    returns.name = "xu100_return"
    return returns


def _configure_yfinance_cache(cache_dir):
    """Point yfinance cache DBs to a writable folder."""
    try:
        import yfinance as yf

        cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_dir))
    except Exception:
        pass


def _extract_close_from_yf_download(df):
    """Extract close-like series from yfinance download output."""
    if df is None or getattr(df, "empty", True):
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        for field in ("Adj Close", "Close"):
            if field in df.columns.get_level_values(0):
                sub = df[field]
                if isinstance(sub, pd.DataFrame):
                    return pd.to_numeric(sub.iloc[:, 0], errors="coerce")
                return pd.to_numeric(sub, errors="coerce")
    else:
        for field in ("Adj Close", "Close"):
            if field in df.columns:
                return pd.to_numeric(df[field], errors="coerce")

    return pd.Series(dtype=float)


def _load_xu030_returns(start_date, end_date):
    """Load XU030 daily returns via analyzer helper (Yahoo Finance)."""
    local_candidates = [
        DATA_DIR / "xu030_prices.parquet",
        DATA_DIR / "xu030_prices.csv",
        DATA_DIR / "xu30_prices.parquet",
        DATA_DIR / "xu30_prices.csv",
    ]
    for local_file in local_candidates:
        if not local_file.exists():
            continue
        try:
            if local_file.suffix.lower() == ".parquet":
                df = pd.read_parquet(local_file)
            else:
                df = pd.read_csv(local_file)
            if df.empty:
                continue

            date_col = next((c for c in ("Date", "date", "DATE") if c in df.columns), None)
            if date_col is None:
                continue

            price_col = next((c for c in ("Adj Close", "Close", "close", "adj_close") if c in df.columns), None)
            if price_col is None:
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                if not numeric_cols:
                    continue
                price_col = numeric_cols[0]

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
            df = df.dropna(subset=[date_col, price_col]).sort_values(date_col)
            if df.empty:
                continue

            prices = df.set_index(date_col)[price_col].astype(float)
            series = prices.pct_change(fill_method=None).dropna()
            series = series[(series.index >= start_date) & (series.index <= end_date)]
            series.name = "xu030_return"
            if not series.empty:
                return series, None
        except Exception:
            continue

    if not _env_flag("DASHBOARD_ALLOW_REMOTE_XU030", default=True):
        return pd.Series(dtype=float), "XU030 local file missing and remote fetch disabled (set DASHBOARD_ALLOW_REMOTE_XU030=1)"

    _configure_yfinance_cache(DASHBOARD_DIR / ".yfinance_cache")
    symbols = ("XU030.IS", "^XU030", "XU030")
    remote_errors = []

    # 1) Analyzer helper path (if available)
    if _load_xu030_from_yfinance is not None:
        year = int(start_date.year)
        for symbol in symbols:
            try:
                series, err = _load_xu030_from_yfinance(symbol, year)
            except Exception as exc:
                remote_errors.append(f"{symbol}: helper error: {exc}")
                continue
            if series is None:
                remote_errors.append(f"{symbol}: {err or 'no data'}")
                continue

            series = pd.to_numeric(series, errors="coerce").dropna()
            series.index = pd.to_datetime(series.index, errors="coerce")
            series = series[series.index.notna()].sort_index()
            series = series[(series.index >= start_date) & (series.index <= end_date)]
            series.name = "xu030_return"
            if not series.empty:
                return series, None
            remote_errors.append(f"{symbol}: helper returned no rows in requested date range")
    else:
        remote_errors.append("helper unavailable")

    # 2) Direct yfinance fallback
    try:
        import yfinance as yf
    except Exception as exc:
        return pd.Series(dtype=float), f"XU030 loader unavailable: yfinance import failed: {exc}"

    yf_start = (pd.Timestamp(start_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    yf_end = (pd.Timestamp(end_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    for symbol in symbols:
        try:
            raw = yf.download(
                symbol,
                start=yf_start,
                end=yf_end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            remote_errors.append(f"{symbol}: direct download failed: {exc}")
            continue

        close = _extract_close_from_yf_download(raw).dropna()
        if close.empty:
            remote_errors.append(f"{symbol}: no close series from direct download")
            continue

        close.index = pd.to_datetime(close.index, errors="coerce")
        close = close[close.index.notna()].sort_index()
        series = close.pct_change(fill_method=None).dropna()
        series = series[(series.index >= start_date) & (series.index <= end_date)]
        series.name = "xu030_return"
        if not series.empty:
            return series, None
        remote_errors.append(f"{symbol}: direct download returned no rows in requested date range")

    return pd.Series(dtype=float), "XU030 data not available; " + " | ".join(remote_errors[-6:])


def _compute_excess_return_pct(signal_returns, benchmark_returns):
    """Return excess cumulative return in percentage points over overlapping dates."""
    if signal_returns is None or benchmark_returns is None:
        return None

    signal_clean = pd.to_numeric(signal_returns, errors="coerce").dropna()
    benchmark_clean = pd.to_numeric(benchmark_returns, errors="coerce").dropna()
    if signal_clean.empty or benchmark_clean.empty:
        return None

    common_index = signal_clean.index.intersection(benchmark_clean.index)
    if common_index.empty:
        return None

    signal_common = signal_clean.reindex(common_index).dropna()
    benchmark_common = benchmark_clean.reindex(common_index).dropna()
    common_index = signal_common.index.intersection(benchmark_common.index)
    if common_index.empty:
        return None

    signal_cum = (1.0 + signal_common.reindex(common_index)).prod() - 1.0
    benchmark_cum = (1.0 + benchmark_common.reindex(common_index)).prod() - 1.0
    return _to_float_or_none((signal_cum - benchmark_cum) * 100.0)


def _compact_dashboard_payload(signals, holdings, manual_holdings, manual_trades, custom_portfolio):
    """Bound payload sections so dashboard rendering remains fast."""
    max_signals = _env_int("DASHBOARD_MAX_SIGNALS", 40)
    max_holdings_per_signal = _env_int("DASHBOARD_MAX_HOLDINGS_PER_SIGNAL", 200)
    max_manual_holdings_rows = _env_int("DASHBOARD_MAX_MANUAL_HOLDINGS_ROWS", 250)
    max_manual_trades_rows = _env_int("DASHBOARD_MAX_MANUAL_TRADES_ROWS", 250)
    max_custom_daily_rows = _env_int("DASHBOARD_MAX_CUSTOM_DAILY_ROWS", 370)

    payload_notes = []
    dashboard_signals = list(signals or [])
    if len(dashboard_signals) > max_signals:
        dashboard_signals = dashboard_signals[:max_signals]
        payload_notes.append(f"Signals truncated to top {max_signals} by CAGR.")

    selected_names = [s.get("name") for s in dashboard_signals if s.get("name")]
    dashboard_holdings = {}
    holdings_meta = {}
    for name in selected_names:
        tickers = _dedupe_tickers((holdings or {}).get(name, []))
        total_count = len(tickers)
        shown = _truncate_list(tickers, max_holdings_per_signal)
        dashboard_holdings[name] = shown
        holdings_meta[name] = {
            "total_count": total_count,
            "shown_count": len(shown),
            "truncated": total_count > len(shown),
        }

    if any(meta.get("truncated") for meta in holdings_meta.values()):
        payload_notes.append(f"Holdings preview limited to {max_holdings_per_signal} tickers per signal.")

    dashboard_manual_holdings = dict(manual_holdings or {})
    dashboard_manual_holdings.pop("daily_weights", None)
    mh_rows = list((dashboard_manual_holdings.get("rows") or []))
    if len(mh_rows) > max_manual_holdings_rows:
        dashboard_manual_holdings["rows"] = _truncate_list(mh_rows, max_manual_holdings_rows)
        payload_notes.append(f"Manual holdings rows limited to {max_manual_holdings_rows}.")
    else:
        dashboard_manual_holdings["rows"] = mh_rows

    dashboard_manual_trades = dict(manual_trades or {})
    mt_rows = list((dashboard_manual_trades.get("rows") or []))
    if len(mt_rows) > max_manual_trades_rows:
        dashboard_manual_trades["rows"] = _truncate_list(mt_rows, max_manual_trades_rows)
        payload_notes.append(f"Manual trades rows limited to {max_manual_trades_rows}.")
    else:
        dashboard_manual_trades["rows"] = mt_rows

    dashboard_custom_portfolio = dict(custom_portfolio or {})
    cp_daily = list((dashboard_custom_portfolio.get("daily") or []))
    if len(cp_daily) > max_custom_daily_rows:
        dashboard_custom_portfolio["daily"] = _truncate_list(cp_daily, max_custom_daily_rows, keep_tail=True)
        payload_notes.append(f"Custom portfolio daily history limited to {max_custom_daily_rows} recent rows.")
    else:
        dashboard_custom_portfolio["daily"] = cp_daily

    return (
        dashboard_signals,
        dashboard_holdings,
        dashboard_manual_holdings,
        dashboard_manual_trades,
        dashboard_custom_portfolio,
        holdings_meta,
        payload_notes,
    )


def _build_custom_portfolio_performance(manual_holdings=None):
    """Compute daily custom portfolio performance vs XU030 and XU100."""
    today = pd.Timestamp(datetime.now().date())
    # User-requested anchor: start portfolio history from 2026-02-05.
    start_date = pd.Timestamp(year=2026, month=2, day=5)
    if start_date > today:
        start_date = pd.Timestamp(year=today.year, month=2, day=5)
        if start_date > today:
            start_date = pd.Timestamp(year=today.year - 1, month=2, day=5)
    end_date = today

    notes = list((manual_holdings or {}).get("notes", [])) if isinstance(manual_holdings, dict) else []
    source_file = MANUAL_HOLDINGS_FILE
    raw_daily_weights = (manual_holdings or {}).get("daily_weights", {}) if isinstance(manual_holdings, dict) else {}
    has_dated_snapshots = isinstance(raw_daily_weights, dict) and len(raw_daily_weights) > 0
    daily_weight_snapshots = {}
    if has_dated_snapshots:
        for raw_date, raw_weights in raw_daily_weights.items():
            snapshot_date = _to_datetime_or_none(raw_date)
            if snapshot_date is None or not isinstance(raw_weights, dict):
                continue

            weighted = {}
            for ticker, raw_weight in raw_weights.items():
                normalized_ticker = _normalize_ticker(ticker)
                weight = _to_float_or_none(raw_weight)
                if not normalized_ticker or weight is None or weight <= 0:
                    continue
                weighted[normalized_ticker] = weighted.get(normalized_ticker, 0.0) + float(weight)

            total_weight = sum(weighted.values())
            if total_weight <= 0:
                continue

            key = pd.Timestamp(snapshot_date).normalize()
            daily_weight_snapshots[key] = {k: v / total_weight for k, v in weighted.items()}

    manual_weights = (manual_holdings or {}).get("weights", {}) if isinstance(manual_holdings, dict) else {}
    static_weighted = {}
    for ticker, raw_weight in (manual_weights or {}).items():
        normalized_ticker = _normalize_ticker(ticker)
        weight = _to_float_or_none(raw_weight)
        if not normalized_ticker or weight is None or weight <= 0:
            continue
        static_weighted[normalized_ticker] = static_weighted.get(normalized_ticker, 0.0) + float(weight)

    static_total_weight = sum(static_weighted.values())
    static_weights = (
        {k: v / static_total_weight for k, v in static_weighted.items()}
        if static_total_weight > 0
        else {}
    )

    if not daily_weight_snapshots and static_weights:
        daily_weight_snapshots[start_date] = static_weights

    if not daily_weight_snapshots:
        notes.append(
            f"No valid manual holdings weights found. Fill {MANUAL_HOLDINGS_FILE.relative_to(PROJECT_ROOT)}."
        )
        return {
            "file": str(source_file.relative_to(PROJECT_ROOT)),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "tickers": [],
            "weights": {},
            "missing_tickers": [],
            "summary": {},
            "daily": [],
            "notes": notes,
        }

    snapshot_dates = sorted(daily_weight_snapshots.keys())
    latest_snapshot_date = snapshot_dates[-1]
    weights = daily_weight_snapshots[latest_snapshot_date]
    tickers = sorted({ticker for snapshot in daily_weight_snapshots.values() for ticker in snapshot.keys()})

    if has_dated_snapshots:
        notes.append(f"Using {len(snapshot_dates)} dated holdings snapshot(s) from manual_portfolio_holdings.csv.")
        if snapshot_dates[0] > start_date:
            notes.append(
                f"Holdings snapshots start on {snapshot_dates[0].date()}; "
                "portfolio returns begin once weights are available."
            )
        if snapshot_dates[-1] < end_date:
            notes.append(
                f"No newer holdings snapshot after {snapshot_dates[-1].date()}; latest weights are carried forward."
            )

    close_panel = _load_close_panel_for_tickers(tickers)

    if close_panel.empty:
        return {
            "file": str(source_file.relative_to(PROJECT_ROOT)),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "tickers": tickers,
            "weights": {k: _to_float_or_none(v) for k, v in weights.items()},
            "missing_tickers": tickers,
            "summary": {},
            "daily": [],
            "notes": notes + ["No local price data found for portfolio tickers in data/bist_prices_full.*"],
        }

    # Compute returns on full history first, then slice to requested window.
    returns_panel = close_panel.pct_change(fill_method=None)
    window_close = close_panel[(close_panel.index >= start_date) & (close_panel.index <= end_date)]
    returns_panel = returns_panel[(returns_panel.index >= start_date) & (returns_panel.index <= end_date)]
    if window_close.empty:
        max_date = close_panel.index.max()
        notes.append(
            f"No market rows in requested range ({start_date.date()} to {end_date.date()}). "
            f"Latest available date is {max_date.date()}."
        )
    elif window_close.index.max() < end_date:
        notes.append(
            f"Market data available up to {window_close.index.max().date()} (requested end: {end_date.date()})."
        )

    returns_panel = returns_panel.reindex(columns=tickers)
    snapshot_matrix = (
        pd.DataFrame.from_dict(daily_weight_snapshots, orient="index")
        .sort_index()
        .reindex(columns=returns_panel.columns, fill_value=0.0)
    )
    aligned_weights = snapshot_matrix.reindex(returns_panel.index, method="ffill").fillna(0.0)
    available_weight = aligned_weights.where(returns_panel.notna(), 0.0).sum(axis=1)
    weighted_sum = returns_panel.mul(aligned_weights).sum(axis=1, min_count=1)
    portfolio_return = weighted_sum / available_weight.replace(0.0, np.nan)
    portfolio_return = portfolio_return.dropna().rename("portfolio_return")

    if portfolio_return.empty:
        return {
            "file": str(source_file.relative_to(PROJECT_ROOT)),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "tickers": tickers,
            "weights": {k: _to_float_or_none(v) for k, v in weights.items()},
            "missing_tickers": tickers,
            "summary": {},
            "daily": [],
            "notes": notes + ["Could not compute portfolio returns in requested date range."],
        }

    active_tickers = [c for c in window_close.columns if window_close[c].notna().any()]
    missing_tickers = [t for t in tickers if t not in active_tickers]

    xu100_return = _load_xu100_returns(start_date, end_date)
    xu030_return, xu030_err = _load_xu030_returns(start_date, end_date)
    if xu030_err:
        notes.append(f"XU030 unavailable: {xu030_err}")

    daily = pd.concat([portfolio_return, xu030_return, xu100_return], axis=1).sort_index()
    daily = daily[(daily.index >= start_date) & (daily.index <= end_date)]
    if "portfolio_return" in daily.columns:
        first_valid_portfolio = daily["portfolio_return"].dropna()
        if not first_valid_portfolio.empty:
            daily = daily[daily.index >= first_valid_portfolio.index.min()]

    # If benchmark data extends beyond local portfolio/XU100 prices, keep continuity by
    # treating trailing unknown returns as flat (0.0) instead of missing.
    for col, label in (("portfolio_return", "Portfolio"), ("xu100_return", "XU100")):
        if col not in daily.columns:
            continue
        valid = daily[col].dropna()
        if valid.empty:
            continue
        last_valid = valid.index.max()
        stale_mask = daily.index > last_valid
        stale_count = int(stale_mask.sum())
        if stale_count > 0:
            daily.loc[stale_mask, col] = 0.0
            notes.append(
                f"{label} data unavailable after {last_valid.date()}; "
                f"set trailing {stale_count} day(s) return to 0.0."
            )

    for return_col, cum_col in (
        ("portfolio_return", "portfolio_cumulative"),
        ("xu030_return", "xu030_cumulative"),
        ("xu100_return", "xu100_cumulative"),
    ):
        if return_col not in daily.columns:
            continue
        series = daily[return_col].dropna()
        if series.empty:
            daily[cum_col] = np.nan
            continue
        cumulative = (1.0 + series).cumprod() - 1.0
        daily[cum_col] = cumulative.reindex(daily.index).ffill()

    summary = {
        "obs_days": int(daily["portfolio_return"].dropna().shape[0]) if "portfolio_return" in daily.columns else 0,
        "portfolio_cumulative": _to_float_or_none(daily.get("portfolio_cumulative", pd.Series(dtype=float)).dropna().iloc[-1])
        if "portfolio_cumulative" in daily.columns and not daily["portfolio_cumulative"].dropna().empty
        else None,
        "xu030_cumulative": _to_float_or_none(daily.get("xu030_cumulative", pd.Series(dtype=float)).dropna().iloc[-1])
        if "xu030_cumulative" in daily.columns and not daily["xu030_cumulative"].dropna().empty
        else None,
        "xu100_cumulative": _to_float_or_none(daily.get("xu100_cumulative", pd.Series(dtype=float)).dropna().iloc[-1])
        if "xu100_cumulative" in daily.columns and not daily["xu100_cumulative"].dropna().empty
        else None,
    }

    if {"portfolio_return", "xu100_return"}.issubset(daily.columns):
        aligned = daily[["portfolio_return", "xu100_return"]].dropna()
        if not aligned.empty:
            excess = aligned["portfolio_return"] - aligned["xu100_return"]
            summary["outperform_days_vs_xu100"] = int((excess > 0).sum())
            summary["outperform_rate_vs_xu100"] = _to_float_or_none((excess > 0).mean())
            if _annualized_tracking_error is not None:
                summary["tracking_error_vs_xu100"] = _to_float_or_none(_annualized_tracking_error(excess))
            if _information_ratio is not None:
                summary["information_ratio_vs_xu100"] = _to_float_or_none(_information_ratio(excess))

    if {"portfolio_return", "xu030_return"}.issubset(daily.columns):
        aligned = daily[["portfolio_return", "xu030_return"]].dropna()
        if not aligned.empty:
            excess = aligned["portfolio_return"] - aligned["xu030_return"]
            summary["outperform_days_vs_xu030"] = int((excess > 0).sum())
            summary["outperform_rate_vs_xu030"] = _to_float_or_none((excess > 0).mean())
            if _annualized_tracking_error is not None:
                summary["tracking_error_vs_xu030"] = _to_float_or_none(_annualized_tracking_error(excess))
            if _information_ratio is not None:
                summary["information_ratio_vs_xu030"] = _to_float_or_none(_information_ratio(excess))

    daily_records = []
    ordered_cols = [
        "portfolio_return",
        "xu030_return",
        "xu100_return",
        "portfolio_cumulative",
        "xu030_cumulative",
        "xu100_cumulative",
    ]
    for dt, row in daily.iterrows():
        item = {"date": dt.strftime("%Y-%m-%d")}
        for col in ordered_cols:
            item[col] = _to_float_or_none(row[col]) if col in row else None
        daily_records.append(item)

    # Persist daily portfolio history so it is available between dashboard refreshes.
    export_df = pd.DataFrame(daily_records)
    if not export_df.empty:
        export_df.to_csv(MANUAL_PORTFOLIO_DAILY_FILE, index=False)

    return {
        "file": str(source_file.relative_to(PROJECT_ROOT)),
        "daily_file": str(MANUAL_PORTFOLIO_DAILY_FILE.relative_to(PROJECT_ROOT)),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "tickers": tickers,
        "weights": {k: _to_float_or_none(v) for k, v in weights.items()},
        "missing_tickers": missing_tickers,
        "summary": summary,
        "daily": daily_records,
        "notes": notes,
    }


def load_regime_data():
    """Load current regime and regime distribution."""
    try:
        candidate_files = [
            SIMPLE_REGIME_OUTPUTS_DIR / "regime_features.csv",
            SIMPLE_REGIME_OUTPUTS_DIR / "simplified_regimes.csv",
            SIMPLE_REGIME_OUTPUTS_DIR / "enhanced_regimes.csv",
            LEGACY_REGIME_OUTPUTS_DIR / "simplified_regimes.csv",
            LEGACY_REGIME_OUTPUTS_DIR / "enhanced_regimes.csv",
            LEGACY_REGIME_OUTPUTS_DIR / "regime_features.csv",
        ]

        for regime_file in candidate_files:
            if not regime_file.exists():
                continue

            regime_df = _read_csv_with_date_index(regime_file)
            if regime_df.empty:
                continue

            regime_col = next(
                (c for c in ("regime_label", "simplified_regime", "regime", "detailed_regime") if c in regime_df.columns),
                None,
            )
            if regime_col is None:
                continue

            regimes = regime_df[regime_col].dropna().astype(str)
            if regimes.empty:
                continue

            latest_regime = regimes.iloc[-1]
            recent_regimes = regimes.tail(252)
            regime_dist = (recent_regimes.value_counts(normalize=True) * 100).to_dict()
            return latest_regime, regime_dist

        print(
            "Warning: No valid regime data found in "
            f"{SIMPLE_REGIME_OUTPUTS_DIR} or {LEGACY_REGIME_OUTPUTS_DIR}"
        )
        return "Unknown", {}
    except Exception as exc:
        print(f"Warning: Could not load regime data: {exc}")
        return "Unknown", {}


def load_xu100_data():
    """Load XU100 benchmark YTD and trading day count."""
    try:
        xu100_file = DATA_DIR / "xu100_prices.csv"
        xu100_df = pd.read_csv(xu100_file)
        if xu100_df.empty:
            return 0.0, 0

        date_col = next((c for c in ("Date", "date", "DATE") if c in xu100_df.columns), xu100_df.columns[0])
        price_col = next((c for c in ("Adj Close", "Close", "close", "adj_close") if c in xu100_df.columns), None)

        if price_col is None:
            numeric_cols = xu100_df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric price column found in xu100_prices.csv")
            price_col = numeric_cols[0]

        xu100_df[date_col] = pd.to_datetime(xu100_df[date_col], errors="coerce")
        xu100_df[price_col] = pd.to_numeric(xu100_df[price_col], errors="coerce")
        xu100_df = xu100_df.dropna(subset=[date_col, price_col]).sort_values(date_col)

        if xu100_df.empty:
            return 0.0, 0

        current_year = datetime.now().year
        ytd_data = xu100_df[xu100_df[date_col].dt.year == current_year]
        ytd_return = ((ytd_data[price_col].iloc[-1] / ytd_data[price_col].iloc[0]) - 1) * 100 if len(ytd_data) > 0 else 0.0

        return float(ytd_return), int(len(xu100_df))
    except Exception as exc:
        print(f"Warning: Could not load XU100 data: {exc}")
        return 0.0, 0


def load_signal_results():
    """Load summary metrics and holdings from all signal result folders."""
    signals = []
    holdings = {}

    if not RESULTS_DIR.exists():
        candidates = ", ".join(str(p) for p in RESULTS_DIR_CANDIDATES)
        print(f"Warning: Results directory not found: {RESULTS_DIR} (candidates: {candidates})")
        return signals, holdings

    current_year = datetime.now().year
    ytd_start = pd.Timestamp(year=current_year, month=1, day=1)
    ytd_end = pd.Timestamp(datetime.now().date())
    xu100_ytd_returns = _load_xu100_returns(ytd_start, ytd_end)
    latest_market_date = None
    if not xu100_ytd_returns.empty:
        latest_market_date = pd.Timestamp(xu100_ytd_returns.index.max()).normalize()
    xu030_ytd_returns, xu030_err = _load_xu030_returns(ytd_start, ytd_end)
    if xu030_err:
        print(f"Warning: {xu030_err}")

    for signal_dir in RESULTS_DIR.iterdir():
        if not signal_dir.is_dir():
            continue

        signal_name = signal_dir.name
        summary_file = signal_dir / "summary.txt"

        metrics = {
            "name": signal_name,
            "enabled": True,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "ytd": 0.0,
            "beta": None,
            "excess_vs_xu030": None,
            "excess_vs_xu100": None,
            "last_rebalance": None,
        }

        if summary_file.exists():
            try:
                content = summary_file.read_text(encoding="utf-8")
                metrics["cagr"] = _extract_metric(content, r"\bCAGR\s*:\s*([+-]?\d+(?:\.\d+)?)\s*%", default=0.0)
                metrics["sharpe"] = _extract_metric(
                    content,
                    [r"\bSharpe\s+Ratio\s*:\s*([+-]?\d+(?:\.\d+)?)", r"\bSharpe\s*:\s*([+-]?\d+(?:\.\d+)?)"],
                    default=0.0,
                )
                metrics["max_dd"] = _extract_metric(
                    content,
                    r"\bMax\s+Drawdown\s*:\s*([+-]?\d+(?:\.\d+)?)\s*%",
                    default=0.0,
                )
                metrics["beta"] = _extract_metric(
                    content,
                    r"\bBeta\s*:\s*([+-]?\d+(?:\.\d+)?)",
                    default=None,
                )
            except Exception as exc:
                print(f"Warning: Could not parse summary for {signal_name}: {exc}")

        returns_file = signal_dir / "returns.csv"
        if returns_file.exists():
            try:
                returns_df = _read_csv_with_date_index(returns_file)
                ytd_data = returns_df[returns_df.index.year == current_year]

                if len(ytd_data) > 0 and "Return" in ytd_data.columns:
                    ytd_returns = pd.to_numeric(ytd_data["Return"], errors="coerce").dropna()
                    if not ytd_returns.empty:
                        metrics["ytd"] = ((1 + ytd_returns).prod() - 1) * 100
                        metrics["excess_vs_xu100"] = _compute_excess_return_pct(ytd_returns, xu100_ytd_returns)
                        metrics["excess_vs_xu030"] = _compute_excess_return_pct(ytd_returns, xu030_ytd_returns)
                        signal_date = pd.Timestamp(ytd_returns.index[-1]).normalize()
                        rebalance_date = signal_date
                        # If benchmark data is one market session ahead, display that latest session date.
                        if latest_market_date is not None:
                            lag_days = int((latest_market_date - signal_date).days)
                            if 0 <= lag_days <= 1:
                                rebalance_date = latest_market_date
                        metrics["last_rebalance"] = rebalance_date.strftime("%Y-%m-%d")
            except Exception as exc:
                print(f"Warning: Could not load returns for {signal_name}: {exc}")

        holdings_file = signal_dir / "holdings.csv"
        current_holdings = []
        if holdings_file.exists():
            try:
                current_holdings = _load_current_holdings(holdings_file)
            except Exception as exc:
                print(f"Warning: Could not load holdings for {signal_name}: {exc}")

        signals.append(metrics)
        holdings[signal_name] = current_holdings

    signals.sort(key=lambda x: x["cagr"], reverse=True)
    return signals, holdings


def generate_dashboard_data():
    """Generate dashboard JSON payload file."""
    print("Generating dashboard data...")
    print(f"  Results dir: {RESULTS_DIR}")
    _ensure_manual_input_files()

    current_regime, regime_dist = load_regime_data()
    print(f"  Current regime: {current_regime}")

    xu100_ytd, trading_days = load_xu100_data()
    print(f"  XU100 YTD: {xu100_ytd:.2f}%")

    signals, holdings = load_signal_results()
    print(f"  Loaded {len(signals)} signals")

    manual_holdings = _load_manual_holdings()
    custom_portfolio = _build_custom_portfolio_performance(manual_holdings=manual_holdings)
    manual_trades = _load_manual_trades()
    print(f"  Custom portfolio tickers: {len(custom_portfolio.get('tickers', []))}")
    print(f"  Manual trades: {len(manual_trades.get('rows', []))}")
    (
        dashboard_signals,
        dashboard_holdings,
        dashboard_manual_holdings,
        dashboard_manual_trades,
        dashboard_custom_portfolio,
        holdings_meta,
        payload_notes,
    ) = _compact_dashboard_payload(
        signals=signals,
        holdings=holdings,
        manual_holdings=manual_holdings,
        manual_trades=manual_trades,
        custom_portfolio=custom_portfolio,
    )

    dashboard_data = {
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_regime": current_regime,
        "regime_distribution": regime_dist,
        "xu100_ytd": xu100_ytd,
        "trading_days": trading_days,
        "active_signals": len([s for s in signals if s["enabled"]]),
        "displayed_signals": len(dashboard_signals),
        "signals": dashboard_signals,
        "holdings": dashboard_holdings,
        "holdings_meta": holdings_meta,
        "manual_holdings": dashboard_manual_holdings,
        "manual_trades": dashboard_manual_trades,
        "custom_portfolio": dashboard_custom_portfolio,
        "payload_notes": payload_notes,
    }

    serialized = json.dumps(dashboard_data, indent=2)
    output_json = DASHBOARD_DIR / "dashboard_data.json"
    output_json.write_text(serialized, encoding="utf-8")

    PUBLIC_DASHBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_DASHBOARD_FILE.write_text(serialized, encoding="utf-8")

    print(f"Saved JSON: {output_json}")
    print(f"Synced JSON: {PUBLIC_DASHBOARD_FILE}")
    print("")
    print("Summary:")
    print(f"  - Active Signals: {dashboard_data['active_signals']}")
    print(f"  - Displayed Signals: {dashboard_data['displayed_signals']}")
    print(f"  - Current Regime: {current_regime}")
    print(f"  - XU100 YTD: {xu100_ytd:.2f}%")
    if dashboard_custom_portfolio.get("summary", {}).get("portfolio_cumulative") is not None:
        pct = dashboard_custom_portfolio["summary"]["portfolio_cumulative"] * 100
        print(f"  - Oksuz Picks YTD: {pct:.2f}%")
    if payload_notes:
        print("  - Payload Notes:")
        for note in payload_notes:
            print(f"      * {note}")
    print("")
    print(f"Use dashboard app frontend at: {DASHBOARD_DIR / 'app' / 'frontend'}")

    return dashboard_data


if __name__ == "__main__":
    generate_dashboard_data()

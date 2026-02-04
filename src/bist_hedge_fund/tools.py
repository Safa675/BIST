"""
CrewAI Tools for the BIST Hedge Fund

These tools wrap the existing BIST codebase to give AI agents access to:
- Market data (prices, volumes) via Yahoo Finance
- Technical indicators (RSI, MACD, Bollinger Bands, SMA, Donchian)
- Fundamental analysis (financial statements, valuation ratios)
- Regime detection (ensemble ML model: XGBoost + LSTM + HMM)
- Risk metrics (volatility, drawdown, beta)
"""

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from crewai.tools import tool


# ────────────────────────────────────────────────────────────────────────────
# MARKET DATA TOOLS
# ────────────────────────────────────────────────────────────────────────────

@tool("Fetch BIST Stock Prices")
def fetch_bist_prices(ticker: str, period: str = "6mo") -> str:
    """Fetch price data for a BIST (Borsa Istanbul) stock.
    Ticker should be the stock symbol (e.g., 'THYAO' for Turkish Airlines).
    The '.IS' suffix is added automatically.
    Period can be: 1mo, 3mo, 6mo, 1y, 2y, 5y, max.
    Returns OHLCV data with key price statistics."""
    yf_ticker = f"{ticker.upper()}.IS"
    stock = yf.Ticker(yf_ticker)
    hist = stock.history(period=period)

    if hist.empty:
        return f"No data found for {ticker} ({yf_ticker})"

    close = hist["Close"]
    volume = hist["Volume"]

    return json.dumps({
        "ticker": ticker.upper(),
        "yf_ticker": yf_ticker,
        "period": period,
        "current_price": round(float(close.iloc[-1]), 2),
        "period_high": round(float(close.max()), 2),
        "period_low": round(float(close.min()), 2),
        "period_return_pct": round(float((close.iloc[-1] / close.iloc[0] - 1) * 100), 2),
        "avg_daily_volume": int(volume.mean()),
        "last_5_closes": {
            str(d.date()): round(float(p), 2)
            for d, p in close.tail(5).items()
        },
    }, indent=2)


@tool("Fetch XU100 Index Data")
def fetch_xu100_data(period: str = "6mo") -> str:
    """Fetch XU100 (Borsa Istanbul 100) index data.
    Returns index level, trend, and key statistics.
    Period can be: 1mo, 3mo, 6mo, 1y, 2y, 5y."""
    stock = yf.Ticker("XU100.IS")
    hist = stock.history(period=period)

    if hist.empty:
        return "No data found for XU100.IS"

    close = hist["Close"]
    returns_20d = float(close.iloc[-1] / close.iloc[-min(20, len(close))] - 1)
    returns_60d = float(close.iloc[-1] / close.iloc[-min(60, len(close))] - 1) if len(close) >= 60 else None

    return json.dumps({
        "index": "XU100",
        "current_level": round(float(close.iloc[-1]), 2),
        "period_high": round(float(close.max()), 2),
        "period_low": round(float(close.min()), 2),
        "return_20d_pct": round(returns_20d * 100, 2),
        "return_60d_pct": round(returns_60d * 100, 2) if returns_60d else None,
        "period_return_pct": round(float((close.iloc[-1] / close.iloc[0] - 1) * 100), 2),
    }, indent=2)


@tool("Fetch USD/TRY and Gold Prices")
def fetch_macro_data(period: str = "6mo") -> str:
    """Fetch USD/TRY exchange rate and XAU/TRY (gold in TRY) prices.
    These are critical for the BIST regime filter (USD/TRY momentum drives risk regime)
    and for gold hedge allocation in bear/stress regimes."""
    usdtry = yf.Ticker("TRY=X")
    gold = yf.Ticker("GC=F")

    usdtry_hist = usdtry.history(period=period)
    gold_hist = gold.history(period=period)

    result = {}

    if not usdtry_hist.empty:
        usd_close = usdtry_hist["Close"]
        usd_mom_20d = float(usd_close.iloc[-1] / usd_close.iloc[-min(20, len(usd_close))] - 1)
        result["usdtry"] = {
            "current": round(float(usd_close.iloc[-1]), 4),
            "period_high": round(float(usd_close.max()), 4),
            "period_low": round(float(usd_close.min()), 4),
            "momentum_20d_pct": round(usd_mom_20d * 100, 2),
            "trend": "weakening_TRY" if usd_mom_20d > 0.01 else "strengthening_TRY" if usd_mom_20d < -0.01 else "stable",
        }

    if not gold_hist.empty and not usdtry_hist.empty:
        # Calculate XAU/TRY
        gold_close = gold_hist["Close"].reindex(usd_close.index, method="ffill")
        xautry = gold_close * usd_close
        xautry = xautry.dropna()
        if not xautry.empty:
            result["xautry"] = {
                "current": round(float(xautry.iloc[-1]), 2),
                "period_return_pct": round(float((xautry.iloc[-1] / xautry.iloc[0] - 1) * 100), 2),
            }

    return json.dumps(result, indent=2)


# ────────────────────────────────────────────────────────────────────────────
# TECHNICAL ANALYSIS TOOLS
# ────────────────────────────────────────────────────────────────────────────

@tool("Calculate Technical Indicators")
def calculate_technical_indicators(ticker: str) -> str:
    """Calculate technical indicators for a BIST stock: RSI(14), MACD, Bollinger Bands,
    SMA crossover (10/30), and Donchian Channel (20-day).
    These match the exact indicators used in the BIST portfolio engine."""
    yf_ticker = f"{ticker.upper()}.IS"
    stock = yf.Ticker(yf_ticker)
    hist = stock.history(period="6mo")

    if hist.empty or len(hist) < 30:
        return f"Insufficient data for {ticker}"

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    # RSI (14-day)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20-day)
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    upper_bb = sma_20 + (std_20 * 2)
    lower_bb = sma_20 - (std_20 * 2)

    # SMA Crossover (10/30) - matches BIST sma_signals.py
    sma_10 = close.rolling(10).mean()
    sma_30 = close.rolling(30).mean()
    sma_score = (sma_10 / sma_30 - 1) * 100  # Same formula as your codebase

    # Donchian Channel (20-day) - matches BIST donchian_signals.py
    donchian_high = high.rolling(20).max().shift(1)
    donchian_low = low.rolling(20).min().shift(1)
    donchian_width = donchian_high - donchian_low
    donchian_score = ((close - donchian_low) / donchian_width * 200 - 100) if donchian_width.iloc[-1] > 0 else 0

    # Momentum (12-1) - matches BIST momentum_signals.py
    if len(close) >= 252:
        mom_12_1 = close.iloc[-21] / close.iloc[-252] - 1
    elif len(close) >= 63:
        mom_12_1 = close.iloc[-21] / close.iloc[0] - 1
    else:
        mom_12_1 = None

    # Downside volatility
    daily_returns = close.pct_change().dropna()
    downside_rets = daily_returns[daily_returns < 0]
    downside_vol = float(downside_rets.std() * np.sqrt(252)) if len(downside_rets) > 2 else None

    latest = -1
    result = {
        "ticker": ticker.upper(),
        "current_price": round(float(close.iloc[latest]), 2),
        "rsi_14": round(float(rsi.iloc[latest]), 2),
        "rsi_signal": "overbought" if rsi.iloc[latest] > 70 else "oversold" if rsi.iloc[latest] < 30 else "neutral",
        "macd": {
            "line": round(float(macd_line.iloc[latest]), 4),
            "signal": round(float(signal_line.iloc[latest]), 4),
            "histogram": round(float(macd_line.iloc[latest] - signal_line.iloc[latest]), 4),
            "trend": "bullish" if macd_line.iloc[latest] > signal_line.iloc[latest] else "bearish",
        },
        "bollinger_bands": {
            "upper": round(float(upper_bb.iloc[latest]), 2),
            "middle": round(float(sma_20.iloc[latest]), 2),
            "lower": round(float(lower_bb.iloc[latest]), 2),
            "position": "above_upper" if close.iloc[latest] > upper_bb.iloc[latest]
                else "below_lower" if close.iloc[latest] < lower_bb.iloc[latest]
                else "within_bands",
        },
        "sma_crossover": {
            "sma_10": round(float(sma_10.iloc[latest]), 2),
            "sma_30": round(float(sma_30.iloc[latest]), 2),
            "score": round(float(sma_score.iloc[latest]), 2),
            "signal": "bullish" if sma_10.iloc[latest] > sma_30.iloc[latest] else "bearish",
        },
        "donchian_channel": {
            "high_20": round(float(donchian_high.iloc[latest]), 2) if pd.notna(donchian_high.iloc[latest]) else None,
            "low_20": round(float(donchian_low.iloc[latest]), 2) if pd.notna(donchian_low.iloc[latest]) else None,
            "score": round(float(donchian_score.iloc[latest]), 2) if isinstance(donchian_score, pd.Series) else None,
        },
        "momentum_12_1": round(float(mom_12_1) * 100, 2) if mom_12_1 is not None else None,
        "downside_volatility_ann": round(downside_vol * 100, 2) if downside_vol else None,
    }

    return json.dumps(result, indent=2)


# ────────────────────────────────────────────────────────────────────────────
# FUNDAMENTAL ANALYSIS TOOLS
# ────────────────────────────────────────────────────────────────────────────

@tool("Get BIST Stock Fundamentals")
def get_bist_fundamentals(ticker: str) -> str:
    """Get fundamental data for a BIST stock: financials, valuation ratios, margins.
    Returns PE, PB, PS, dividend yield, profit margins, ROE, debt ratios.
    These are the same metrics used in the BIST value and profitability signals."""
    yf_ticker = f"{ticker.upper()}.IS"
    stock = yf.Ticker(yf_ticker)
    info = stock.info

    result = {
        "ticker": ticker.upper(),
        "company_name": info.get("longName", ""),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "market_cap": info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
        "valuation": {
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "peg_ratio": info.get("pegRatio"),
        },
        "profitability": {
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
        },
        "balance_sheet": {
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
        },
        "dividends": {
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
        },
        "beta": info.get("beta"),
    }

    # Get income statement highlights
    income = stock.income_stmt
    if not income.empty:
        latest = income.iloc[:, 0]
        result["income_highlights"] = {
            "total_revenue": _safe_num(latest, "Total Revenue"),
            "gross_profit": _safe_num(latest, "Gross Profit"),
            "operating_income": _safe_num(latest, "Operating Income"),
            "net_income": _safe_num(latest, "Net Income"),
            "ebitda": _safe_num(latest, "EBITDA"),
        }

    return json.dumps(result, indent=2)


# ────────────────────────────────────────────────────────────────────────────
# REGIME DETECTION TOOLS
# ────────────────────────────────────────────────────────────────────────────

@tool("Detect Market Regime")
def detect_market_regime() -> str:
    """Detect the current BIST market regime using the same methodology as the
    BIST Regime Filter system (rule-based classifier).

    Analyzes XU100 index and USD/TRY to classify the market into one of 5 regimes:
    - Bull: Low/mid volatility + uptrend + risk-on
    - Recovery: High volatility + uptrend (early recovery)
    - Choppy: Sideways trend, mixed signals
    - Bear: Downtrend + risk-off
    - Stress: High volatility + risk-off (crisis mode)

    Also returns the recommended allocation from the BIST portfolio engine:
    Bull=100%, Recovery=100%, Choppy=50%, Bear=0%, Stress=0%."""
    xu100 = yf.Ticker("XU100.IS")
    usdtry = yf.Ticker("TRY=X")

    xu100_hist = xu100.history(period="1y")
    usdtry_hist = usdtry.history(period="1y")

    if xu100_hist.empty:
        return "Cannot detect regime: XU100 data unavailable"

    close = xu100_hist["Close"]
    volume = xu100_hist["Volume"]

    # 1. Volatility regime (matches config.py VOLATILITY_PERCENTILES)
    daily_returns = close.pct_change().dropna()
    realized_vol_20d = float(daily_returns.tail(20).std() * np.sqrt(252)) * 100

    if realized_vol_20d < 17:
        vol_regime = "Low"
    elif realized_vol_20d < 28:
        vol_regime = "Mid"
    elif realized_vol_20d < 40:
        vol_regime = "High"
    else:
        vol_regime = "Stress"

    # 2. Trend regime (matches config.py TREND_THRESHOLDS)
    return_20d = float(close.iloc[-1] / close.iloc[-min(20, len(close))] - 1)
    return_120d = float(close.iloc[-1] / close.iloc[-min(120, len(close))] - 1) if len(close) >= 120 else return_20d

    def classify_trend(ret):
        if ret > 0.015:
            return "Up"
        elif ret < -0.015:
            return "Down"
        return "Sideways"

    trend_short = classify_trend(return_20d)
    trend_long = classify_trend(return_120d)
    trends = [trend_short, trend_long]
    trend_regime = max(set(trends), key=trends.count)

    # 3. Risk regime via USD/TRY momentum (matches config.py RISK_PERCENTILES)
    risk_regime = "Neutral"
    if not usdtry_hist.empty:
        usd_close = usdtry_hist["Close"]
        usd_mom_20d = float(usd_close.iloc[-1] / usd_close.iloc[-min(20, len(usd_close))] - 1)

        if usd_mom_20d > 0.02:
            risk_regime = "Risk-Off"
        elif usd_mom_20d < -0.01:
            risk_regime = "Risk-On"

    # 4. Simplified regime mapping (matches regime_models.py)
    if vol_regime in ["High", "Stress"] and risk_regime == "Risk-Off":
        regime = "Stress"
    elif trend_regime == "Down" and risk_regime == "Risk-Off":
        regime = "Bear"
    elif vol_regime in ["High", "Stress"] and trend_regime == "Up":
        regime = "Recovery"
    elif trend_regime == "Up" and vol_regime in ["Low", "Mid"]:
        regime = "Bull"
    elif trend_regime == "Down" and vol_regime in ["High", "Stress"]:
        regime = "Bear"
    elif trend_regime == "Sideways":
        regime = "Choppy"
    else:
        regime = "Choppy"

    # Allocation mapping (matches portfolio_engine.py REGIME_ALLOCATIONS)
    allocations = {"Bull": 1.0, "Recovery": 1.0, "Choppy": 0.5, "Stress": 0.0, "Bear": 0.0}

    # Max drawdown (20d)
    rolling_max = close.tail(60).cummax()
    drawdown = (close.tail(60) / rolling_max - 1)
    max_dd_20d = float(drawdown.tail(20).min()) * 100

    return json.dumps({
        "regime": regime,
        "allocation_pct": allocations[regime] * 100,
        "recommendation": {
            "Bull": "Full equity exposure, trend-following works well",
            "Recovery": "Full exposure, focus on quality names with gradual re-entry",
            "Choppy": "50% equity / 50% gold (XAU/TRY), range trading, reduce position size",
            "Bear": "0% equity / 100% gold hedge, avoid new positions",
            "Stress": "0% equity / 100% gold hedge, minimize exposure, wide stops",
        }[regime],
        "components": {
            "volatility_regime": vol_regime,
            "realized_vol_20d_pct": round(realized_vol_20d, 2),
            "trend_regime": trend_regime,
            "return_20d_pct": round(return_20d * 100, 2),
            "return_120d_pct": round(return_120d * 100, 2),
            "risk_regime": risk_regime,
            "max_drawdown_20d_pct": round(max_dd_20d, 2),
        },
    }, indent=2)


# ────────────────────────────────────────────────────────────────────────────
# RISK ANALYSIS TOOLS
# ────────────────────────────────────────────────────────────────────────────

@tool("Calculate Risk Metrics")
def calculate_risk_metrics(ticker: str) -> str:
    """Calculate risk metrics for a BIST stock matching the BIST portfolio engine methodology.
    Returns volatility, downside vol, max drawdown, beta, and position sizing recommendations.
    Uses the same parameters as portfolio_engine.py (15% stop-loss, 20% vol target, 25% max weight)."""
    yf_ticker = f"{ticker.upper()}.IS"
    stock = yf.Ticker(yf_ticker)
    hist = stock.history(period="1y")

    xu100 = yf.Ticker("XU100.IS")
    xu100_hist = xu100.history(period="1y")

    if hist.empty or len(hist) < 60:
        return f"Insufficient data for risk analysis on {ticker}"

    close = hist["Close"]
    daily_returns = close.pct_change().dropna()

    # Realized volatility
    vol_20d = float(daily_returns.tail(20).std() * np.sqrt(252))
    vol_60d = float(daily_returns.tail(60).std() * np.sqrt(252))

    # Downside volatility (matches portfolio_engine.py)
    downside_rets = daily_returns[daily_returns < 0]
    downside_vol = float(downside_rets.std() * np.sqrt(252)) if len(downside_rets) > 2 else None

    # Max drawdown
    equity = (1 + daily_returns).cumprod()
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_dd = float(drawdown.min())

    # Beta vs XU100
    beta = None
    if not xu100_hist.empty and len(xu100_hist) >= 60:
        xu100_ret = xu100_hist["Close"].pct_change().dropna()
        # Align dates
        common = daily_returns.index.intersection(xu100_ret.index)
        if len(common) >= 30:
            stock_r = daily_returns.loc[common]
            index_r = xu100_ret.loc[common]
            cov = np.cov(stock_r, index_r)
            beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else None

    # Position sizing recommendation (matches portfolio_engine.py)
    # Vol targeting: target_vol / realized_vol
    target_vol = 0.20  # TARGET_DOWNSIDE_VOL from portfolio_engine.py
    vol_scale = target_vol / downside_vol if downside_vol and downside_vol > 0 else 1.0
    vol_scale = max(0.10, min(1.0, vol_scale))  # VOL_FLOOR=0.10, VOL_CAP=1.0

    # Stop-loss from current price (15% as in portfolio_engine.py)
    current_price = float(close.iloc[-1])
    stop_loss_price = round(current_price * (1 - 0.15), 2)

    return json.dumps({
        "ticker": ticker.upper(),
        "current_price": round(current_price, 2),
        "volatility": {
            "realized_20d_ann_pct": round(vol_20d * 100, 2),
            "realized_60d_ann_pct": round(vol_60d * 100, 2),
            "downside_vol_ann_pct": round(downside_vol * 100, 2) if downside_vol else None,
        },
        "drawdown": {
            "max_drawdown_pct": round(max_dd * 100, 2),
            "current_drawdown_pct": round(float(drawdown.iloc[-1]) * 100, 2),
        },
        "beta_vs_xu100": round(beta, 3) if beta else None,
        "position_sizing": {
            "vol_target_scale": round(vol_scale, 3),
            "max_position_weight_pct": 25.0,
            "stop_loss_pct": 15.0,
            "stop_loss_price": stop_loss_price,
        },
        "risk_rating": "HIGH" if (downside_vol and downside_vol > 0.30) or max_dd < -0.30
            else "MEDIUM" if (downside_vol and downside_vol > 0.20) or max_dd < -0.20
            else "LOW",
    }, indent=2)


@tool("Get Stock News and Sentiment")
def get_bist_news(ticker: str) -> str:
    """Get recent news for a BIST stock from Yahoo Finance.
    Returns headlines and publisher info for sentiment analysis."""
    yf_ticker = f"{ticker.upper()}.IS"
    stock = yf.Ticker(yf_ticker)
    news = stock.news

    if not news:
        return f"No recent news found for {ticker}"

    articles = []
    for item in news[:10]:
        articles.append({
            "title": item.get("title", ""),
            "publisher": item.get("publisher", ""),
            "link": item.get("link", ""),
            "type": item.get("type", ""),
        })

    return json.dumps({
        "ticker": ticker.upper(),
        "article_count": len(articles),
        "articles": articles,
    }, indent=2)


# ────────────────────────────────────────────────────────────────────────────
# SCREENING TOOLS
# ────────────────────────────────────────────────────────────────────────────

@tool("Screen BIST Stocks")
def screen_bist_stocks(tickers: str) -> str:
    """Screen multiple BIST stocks and rank them by key metrics.
    Pass tickers as a comma-separated string (e.g., 'THYAO,SASA,EREGL,BIMAS,KCHOL').
    Returns a ranked summary with price, PE, ROE, beta for each stock.
    This is useful for the portfolio manager to compare candidates."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    results = []
    for ticker in ticker_list:
        try:
            yf_ticker = f"{ticker}.IS"
            stock = yf.Ticker(yf_ticker)
            info = stock.info
            hist = stock.history(period="3mo")

            if hist.empty:
                continue

            close = hist["Close"]
            daily_ret = close.pct_change().dropna()
            downside = daily_ret[daily_ret < 0]

            results.append({
                "ticker": ticker,
                "price": round(float(close.iloc[-1]), 2),
                "return_3mo_pct": round(float((close.iloc[-1] / close.iloc[0] - 1) * 100), 2),
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "profit_margin": info.get("profitMargins"),
                "beta": info.get("beta"),
                "downside_vol": round(float(downside.std() * np.sqrt(252) * 100), 2) if len(downside) > 2 else None,
                "market_cap": info.get("marketCap"),
            })
        except Exception:
            continue

    if not results:
        return "Could not fetch data for any of the provided tickers"

    # Sort by 3-month return (momentum)
    results.sort(key=lambda x: x.get("return_3mo_pct", 0), reverse=True)

    return json.dumps({
        "screened_count": len(results),
        "stocks": results,
    }, indent=2)


# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────

def _safe_num(series, key):
    """Safely extract a number from a pandas series."""
    try:
        val = series.get(key)
        if val is not None:
            return float(val)
    except (KeyError, TypeError, ValueError):
        pass
    return None

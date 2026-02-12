"""BIST Daily Portfolio Dashboard API."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


DASHBOARD_DATA_PATH = Path(__file__).parent.parent.parent / "dashboard_data.json"

app = FastAPI(
    title="BIST Daily Portfolio Dashboard API",
    description="Serves daily dashboard data from bist-quant-ai/dashboard/dashboard_data.json",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_dashboard_data() -> Dict[str, Any]:
    """Load dashboard payload from disk."""
    try:
        with DASHBOARD_DATA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Error loading dashboard data: {exc}")
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================================
# SCHEMAS
# ============================================================================


class SignalData(BaseModel):
    name: str
    enabled: bool
    cagr: float
    sharpe: float
    beta: Optional[float] = None
    max_dd: float
    ytd: float
    excess_vs_xu030: Optional[float] = None
    excess_vs_xu100: Optional[float] = None
    last_rebalance: str
    holdings: List[str] = []


class MarketRegimeResponse(BaseModel):
    regime: str
    distribution: Dict[str, float]
    xu100_ytd: float


class PortfolioHolding(BaseModel):
    ticker: str
    weight: float
    quantity: int
    avg_cost: float
    last_price: float
    cost_value: float
    market_value: float
    pnl: float
    pnl_pct: float


class PortfolioTrade(BaseModel):
    datetime: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    ticker: Optional[str] = None
    side: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    gross_amount: Optional[float] = None
    fee: Optional[float] = None
    currency: Optional[str] = None
    note: Optional[str] = None


class PortfolioDailyRow(BaseModel):
    date: Optional[str] = None
    portfolio_return: Optional[float] = None
    xu030_return: Optional[float] = None
    xu100_return: Optional[float] = None
    portfolio_cumulative: Optional[float] = None
    xu030_cumulative: Optional[float] = None
    xu100_cumulative: Optional[float] = None


class PortfolioSummary(BaseModel):
    positions: int
    total_cost: float
    total_market_value: float
    total_pnl: float
    total_pnl_pct: float
    daily_return: float
    xu100_daily_return: float
    score_start_date: Optional[str] = None
    score_end_date: Optional[str] = None
    score_days: int = 0


# ============================================================================
# BASIC
# ============================================================================


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "name": "BIST Daily Portfolio Dashboard API",
        "status": "active",
        "version": "1.0.0",
    }


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============================================================================
# DASHBOARD DATA
# ============================================================================


@app.get("/api/dashboard")
async def get_full_dashboard() -> Dict[str, Any]:
    """Return full dashboard payload."""
    data = load_dashboard_data()
    if not data:
        raise HTTPException(status_code=500, detail="Could not load dashboard data")
    return data


@app.get("/api/regime")
async def get_market_regime() -> MarketRegimeResponse:
    """Return current market regime summary."""
    data = load_dashboard_data()
    return MarketRegimeResponse(
        regime=data.get("current_regime", "Unknown"),
        distribution=data.get("regime_distribution", {}),
        xu100_ytd=_safe_float(data.get("xu100_ytd"), 0.0),
    )


@app.get("/api/signals")
async def get_all_signals() -> List[SignalData]:
    """Return all signal metrics and current holdings list."""
    data = load_dashboard_data()
    signals = data.get("signals", [])
    holdings = data.get("holdings", {})

    result: List[SignalData] = []
    for sig in signals:
        signal_name = sig.get("name", "")
        result.append(
            SignalData(
                name=signal_name,
                enabled=bool(sig.get("enabled", True)),
                cagr=_safe_float(sig.get("cagr"), 0.0),
                sharpe=_safe_float(sig.get("sharpe"), 0.0),
                beta=_safe_float(sig.get("beta"), 0.0)
                if sig.get("beta") is not None
                else None,
                max_dd=_safe_float(sig.get("max_dd"), 0.0),
                ytd=_safe_float(sig.get("ytd"), 0.0),
                excess_vs_xu030=_safe_float(sig.get("excess_vs_xu030"), 0.0)
                if sig.get("excess_vs_xu030") is not None
                else None,
                excess_vs_xu100=_safe_float(sig.get("excess_vs_xu100"), 0.0)
                if sig.get("excess_vs_xu100") is not None
                else None,
                last_rebalance=str(sig.get("last_rebalance") or ""),
                holdings=list(holdings.get(signal_name, [])),
            )
        )

    return result


@app.get("/api/signals/{signal_name}")
async def get_signal_detail(signal_name: str) -> Dict[str, Any]:
    """Return one signal with holdings metadata."""
    data = load_dashboard_data()
    signals = data.get("signals", [])
    holdings = data.get("holdings", {})
    holdings_meta = data.get("holdings_meta", {})

    signal = next((s for s in signals if s.get("name") == signal_name), None)
    if signal is None:
        raise HTTPException(status_code=404, detail=f"Signal {signal_name} not found")

    return {
        **signal,
        "holdings": holdings.get(signal_name, []),
        "holdings_meta": holdings_meta.get(signal_name, {}),
    }


# ============================================================================
# PORTFOLIO
# ============================================================================


@app.get("/api/portfolio/holdings")
async def get_portfolio_holdings() -> List[PortfolioHolding]:
    """Return manual portfolio holdings with PnL fields."""
    data = load_dashboard_data()
    manual = data.get("manual_holdings", {})
    rows = manual.get("rows", [])

    result: List[PortfolioHolding] = []
    for row in rows:
        cost_value = _safe_float(row.get("cost_value"), 0.0)
        market_value = _safe_float(row.get("market_value"), 0.0)
        pnl = market_value - cost_value
        pnl_pct = (pnl / cost_value * 100.0) if cost_value > 0 else 0.0

        result.append(
            PortfolioHolding(
                ticker=str(row.get("ticker", "")),
                weight=_safe_float(row.get("weight"), 0.0),
                quantity=int(round(_safe_float(row.get("quantity"), 0.0))),
                avg_cost=_safe_float(row.get("avg_cost"), 0.0),
                last_price=_safe_float(row.get("last_price"), 0.0),
                cost_value=cost_value,
                market_value=market_value,
                pnl=pnl,
                pnl_pct=pnl_pct,
            )
        )

    result.sort(key=lambda item: item.pnl, reverse=True)
    return result


@app.get("/api/portfolio/trades")
async def get_recent_trades(limit: int = 200) -> List[PortfolioTrade]:
    """Return latest manual trade rows."""
    data = load_dashboard_data()
    trades = data.get("manual_trades", {})
    rows = list(trades.get("rows", []))

    result: List[PortfolioTrade] = []
    for row in rows[: max(limit, 1)]:
        result.append(PortfolioTrade(**row))

    return result


@app.get("/api/portfolio/daily")
async def get_portfolio_daily(limit: int = 240, latest_first: bool = True) -> List[PortfolioDailyRow]:
    """Return custom portfolio daily history."""
    data = load_dashboard_data()
    custom = data.get("custom_portfolio", {})
    rows = list(custom.get("daily", []))

    if latest_first:
        rows = list(reversed(rows))

    rows = rows[: max(limit, 1)]
    return [PortfolioDailyRow(**row) for row in rows]


@app.get("/api/portfolio/summary")
async def get_portfolio_summary() -> PortfolioSummary:
    """Return dashboard-ready portfolio summary metrics."""
    data = load_dashboard_data()

    manual = data.get("manual_holdings", {})
    manual_summary = manual.get("summary", {})

    custom = data.get("custom_portfolio", {})
    custom_summary = custom.get("summary", {})
    daily = custom.get("daily", [])

    total_cost = _safe_float(manual_summary.get("total_cost_value"), 0.0)
    total_market = _safe_float(manual_summary.get("total_market_value"), total_cost)
    total_pnl = total_market - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100.0) if total_cost > 0 else 0.0

    latest = daily[-1] if daily else {}

    return PortfolioSummary(
        positions=int(_safe_float(manual_summary.get("positions"), 0.0)),
        total_cost=total_cost,
        total_market_value=total_market,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        daily_return=_safe_float(latest.get("portfolio_return"), 0.0) * 100.0,
        xu100_daily_return=_safe_float(latest.get("xu100_return"), 0.0) * 100.0,
        score_start_date=custom.get("start_date"),
        score_end_date=custom.get("end_date"),
        score_days=int(_safe_float(custom_summary.get("obs_days"), 0.0)),
    )


# ============================================================================
# STATS
# ============================================================================


@app.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    """Return aggregate dashboard stats."""
    data = load_dashboard_data()

    signals = data.get("signals", [])
    active_signals = [s for s in signals if bool(s.get("enabled", True)) and _safe_float(s.get("cagr"), 0.0) > 0]

    avg_cagr = (
        sum(_safe_float(s.get("cagr"), 0.0) for s in active_signals) / len(active_signals)
        if active_signals
        else 0.0
    )
    avg_sharpe = (
        sum(_safe_float(s.get("sharpe"), 0.0) for s in active_signals) / len(active_signals)
        if active_signals
        else 0.0
    )
    avg_ytd = (
        sum(_safe_float(s.get("ytd"), 0.0) for s in active_signals) / len(active_signals)
        if active_signals
        else 0.0
    )

    best = max(active_signals, key=lambda x: _safe_float(x.get("cagr"), 0.0)) if active_signals else {}

    return {
        "last_update": data.get("last_update", ""),
        "trading_days": int(_safe_float(data.get("trading_days"), 0.0)),
        "active_signals": len(active_signals),
        "current_regime": data.get("current_regime", "Unknown"),
        "xu100_ytd": _safe_float(data.get("xu100_ytd"), 0.0),
        "avg_strategy_cagr": round(avg_cagr, 2),
        "avg_strategy_sharpe": round(avg_sharpe, 2),
        "avg_strategy_ytd": round(avg_ytd, 2),
        "best_strategy": {
            "name": best.get("name", ""),
            "cagr": _safe_float(best.get("cagr"), 0.0),
            "sharpe": _safe_float(best.get("sharpe"), 0.0),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# Vercel AI Accelerator Application — Quant AI Platform

## Company / Project Name
**Quant AI Platform**

## Tagline
AI-Powered Multi-Agent Trading Intelligence for Emerging Markets

---

## What are you building? (250 words)

We're building the first AI-native quantitative trading platform designed for emerging markets. Our system combines 34+ proven factor models with a multi-agent AI architecture that autonomously manages portfolios, monitors risk, and explains every decision in natural language.

The platform has three AI agents that collaborate:
- **Portfolio Manager** — Manages factor allocations and rebalancing using signals with up to 102% CAGR (backtested on BIST, 2013-2026)
- **Risk Manager** — Monitors drawdowns, regime shifts, and volatility targeting using an ensemble detector (XGBoost + LSTM + HMM)
- **Market Analyst** — Analyzes sector rotations, macro drivers, and market conditions

Unlike traditional quant platforms that are black boxes, our agents explain WHY they make decisions using plain language — democratizing institutional-grade tools for retail investors and small prop trading firms in emerging markets.

Our core engine is proven: 34 factor models across value, momentum, quality, breakout, and macro strategies. The top signal achieves 102% CAGR with a 2.92 Sharpe ratio across 10+ years of live backtesting on Borsa Istanbul, including transaction costs, slippage, and regime-based risk management.

The web platform is built on Next.js with the Vercel AI SDK powering agent orchestration. The regime detection pipeline automatically switches between equities and gold during market stress — reducing drawdowns by ~30% historically.

We're starting with BIST (Turkey) and expanding to other emerging markets where retail investors lack access to sophisticated quantitative tools.

---

## How does AI feature in your product?

AI is the core of our product across three layers:

### 1. Multi-Agent AI System (Vercel AI SDK)
Three specialized agents coordinate using the Vercel AI SDK:
- Agents share context about market state, holdings, and risk metrics
- Each agent has a distinct persona and expertise domain
- Natural language explanations make complex quant decisions accessible

### 2. Regime Detection (ML Pipeline)
An ensemble of three models classifies market regimes in real-time:
- **XGBoost** — Feature-based classification using 50+ market indicators
- **LSTM** — Temporal pattern recognition on price sequences
- **Hidden Markov Model** — Probabilistic state transitions
- Consensus voting determines the final regime (Bull/Bear/Recovery/Stress)

### 3. Factor Signal Generation
Machine learning enhances traditional factor investing:
- Dynamic signal combination based on regime context
- Volatility-adjusted position sizing using realized downside vol
- Optimal rebalancing timing detection

---

## What stage are you at?

**Working Product with Real Data**

- ✅ 34+ factor models backtested on 10+ years of BIST data
- ✅ Top signal: 102% CAGR, 2.92 Sharpe, 0.36 Beta
- ✅ Live regime detection pipeline (ensembled ML models)
- ✅ Next.js web platform with interactive dashboard
- ✅ Multi-agent AI chat interface
- ✅ Real-time signal monitoring and portfolio analytics
- 🔜 Live trading integration (paper trading → live)
- 🔜 Multi-market expansion (beyond BIST)

---

## What makes you different?

### 1. Explainable AI for Quant Trading
No other platform combines factor investing with conversational AI agents that explain reasoning. Traditional quant is a black box.

### 2. Emerging Market Focus
Most quant platforms target US/EU markets. We built from scratch for emerging markets where data is messier, markets are less efficient (more alpha), and retail investors have fewer tools.

### 3. Regime-Adaptive Risk Management
Our ML ensemble automatically detects market regime changes and rotates to gold — reducing max drawdowns by ~30%. This is institutional-grade risk management made accessible.

### 4. Proven Performance
Not theoretical — our backtests include:
- Transaction costs (realistic slippage)
- Liquidity filters (only tradeable stocks)
- Survivorship bias treatment
- 10+ years of out-of-sample data

### 5. Multi-Agent Architecture
Three specialized agents (portfolio, risk, analyst) that share context and collaborate. Each has domain expertise and provides explainable, actionable insights.

---

## Team

**Founder / Solo Developer**

- Built the entire quantitative research framework from scratch
- 10+ years of market data processing and analysis
- Full-stack development: Python backend + Next.js/React frontend
- Deep expertise in factor investing, portfolio construction, and risk management
- Passionate about democratizing finance in emerging markets

---

## Why Vercel AI Accelerator?

1. **AI SDK Integration** — Our multi-agent system is built directly on the Vercel AI SDK. We need credits and optimization guidance to scale LLM calls for real-time portfolio management.

2. **Deployment & Scale** — Vercel is our deployment platform. As we expand to multi-market coverage, we need the infrastructure to handle real-time data streams and agent coordination.

3. **Community & Mentorship** — We're a solo founder building something ambitious. Access to AI engineering expertise and the Vercel ecosystem would accelerate our path to production.

4. **Vision Alignment** — We're using AI to make complex financial tools accessible. This aligns with Vercel's mission of making powerful technology accessible to everyone.

---

## Technical Architecture

```
┌─────────────────────────────────────────┐
│              Next.js Frontend           │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ │
│  │Dashboard │ │ Signals  │ │ Holdings │ │
│  └─────────┘ └──────────┘ └──────────┘ │
│                    │                     │
│         ┌──────────┴──────────┐         │
│         │  AI Agent Orchestrator│         │
│         │   (Vercel AI SDK)    │         │
│         └──────────┬──────────┘         │
│    ┌───────────────┼───────────────┐    │
│    │               │               │    │
│ ┌──┴──┐      ┌────┴────┐    ┌────┴──┐ │
│ │Port.│      │  Risk   │    │Market │ │
│ │Mgr  │      │  Mgr    │    │Analyst│ │
│ └──┬──┘      └────┬────┘    └────┬──┘ │
│    └───────────────┼───────────────┘    │
│                    │                     │
│         ┌──────────┴──────────┐         │
│         │   Quant Engine      │         │
│         │  (34+ Factor Models)│         │
│         └──────────┬──────────┘         │
│                    │                     │
│         ┌──────────┴──────────┐         │
│         │  Regime Detection   │         │
│         │  (XGBoost+LSTM+HMM) │         │
│         └─────────────────────┘         │
└─────────────────────────────────────────┘
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Top Signal CAGR | 102.18% |
| Best Sharpe Ratio | 2.92 |
| Average CAGR (34 signals) | 61.2% |
| Average Sharpe (34 signals) | 2.01 |
| Top Signal Beta | 0.36 |
| Top Signal Alpha | 82.66% ann. |
| Backtest Period | 2013-2026 (10+ years) |
| Max Drawdown (top signal) | -31.47% |
| Market Covered | BIST (Borsa Istanbul) |
| Factor Models | 34+ |
| AI Agents | 3 (Portfolio, Risk, Analyst) |

---

## Links

- **Live Demo**: [To be deployed on Vercel]
- **GitHub**: [Repository link]
- **Dashboard**: /dashboard
- **Landing Page**: /

---

*Application prepared for the Vercel AI Accelerator — February 2026*

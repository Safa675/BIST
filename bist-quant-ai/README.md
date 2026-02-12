# Quant AI Platform

> **AI-Powered Multi-Agent Trading Intelligence for Emerging Markets**

[![Next.js](https://img.shields.io/badge/Next.js-16-black?style=flat-square&logo=next.js)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?style=flat-square&logo=typescript)](https://typescriptlang.org)
[![Vercel AI SDK](https://img.shields.io/badge/Vercel-AI%20SDK-black?style=flat-square&logo=vercel)](https://sdk.vercel.ai)

## Overview

Quant AI is an institutional-grade quantitative trading platform that combines **34+ proven factor models** with a **multi-agent AI system** for portfolio management, risk monitoring, and market analysis. Built for emerging markets, starting with Borsa Istanbul (BIST).

### Key Metrics

| Metric | Value |
|--------|-------|
| Top Signal CAGR | **102.18%** |
| Best Sharpe Ratio | **2.92** |
| Average CAGR (34 signals) | **61.2%** |
| Top Signal Alpha | **82.66%** ann. |
| Top Signal Beta | **0.36** |
| Backtest Period | **2013–2026** |

## Architecture

```
┌──────────────────────────────────────────┐
│           Next.js Frontend               │
│  Landing Page • Dashboard • AI Agents    │
├──────────────────────────────────────────┤
│        AI Agent Orchestrator             │
│  Portfolio Manager • Risk Manager        │
│        Market Analyst                    │
├──────────────────────────────────────────┤
│         Quant Engine (34+ Factors)       │
│  Value • Momentum • Quality • Breakout   │
│  Macro Hedge • Sector Rotation           │
├──────────────────────────────────────────┤
│        Regime Detection Pipeline         │
│  XGBoost + LSTM + HMM Ensemble           │
└──────────────────────────────────────────┘
```

## Multi-Agent AI System

Three specialized agents collaborate to manage your portfolio:

- 🎯 **Portfolio Manager** — Factor allocations, rebalancing, position sizing
- 🛡️ **Risk Manager** — Drawdown monitoring, vol-targeting, regime-based risk management
- 🧠 **Market Analyst** — BIST trends, sector rotation, macro analysis

## Factor Models

The platform runs 34+ factor models including:

- **Breakout Value** — Donchian breakout × value fundamentals (102% CAGR)
- **Small Cap Momentum** — Size factor × cross-sectional momentum (94% CAGR)
- **Trend Value** — Trend following × value overlay (88% CAGR)
- **Five Factor Rotation** — Dynamic multi-factor allocation (87% CAGR)
- And 30+ more across value, momentum, quality, macro, and sector strategies

## Tech Stack

- **Frontend**: Next.js 16 (App Router), React 19, TypeScript
- **Styling**: Custom CSS design system (glassmorphism, dark theme)
- **Charts**: Recharts
- **AI**: Vercel AI SDK (multi-agent orchestration)
- **Backend**: Python quantitative engine
- **ML**: XGBoost, LSTM, HMM (regime detection)
- **Icons**: Lucide React

## Getting Started

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm run start
```

The app runs at `http://localhost:3000`.

### Required Environment Variables (Agent APIs)

Create `.env.local` with:

```bash
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
AZURE_OPENAI_API_KEY=<your-azure-openai-key>
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
AZURE_OPENAI_API_VERSION=2024-10-21
```

The app now performs startup checks and will fail fast if required Azure OpenAI vars are missing.

### Agent Diagnostics

- `GET /api/agents/health` runs a lightweight live Azure OpenAI connectivity/deployment check.
- Agent APIs emit structured JSON logs (request, Azure call, response, errors) with request IDs and latency.

## Project Structure

```
src/
├── app/
│   ├── page.tsx              # Landing page
│   ├── dashboard/page.tsx    # Trading dashboard
│   ├── agents/page.tsx       # AI Agents showcase
│   ├── globals.css           # Design system
│   ├── layout.tsx            # Root layout
│   └── api/
│       ├── signals/route.ts  # Signal data API
│       └── agents/
│           ├── portfolio/route.ts
│           ├── risk/route.ts
│           └── analyst/route.ts
├── components/
│   ├── Navbar.tsx            # Glass navigation
│   ├── SignalTable.tsx       # Sortable signal table
│   ├── EquityChart.tsx       # Interactive equity curves
│   ├── RegimeIndicator.tsx   # Market regime badge
│   ├── PortfolioView.tsx     # Holdings grid
│   └── AgentChat.tsx         # AI multi-agent chat
└── lib/
    └── agents/
        └── orchestrator.ts   # Agent coordination logic
public/
└── data/
    ├── dashboard_data.json   # Aggregated signal metrics
    └── equity_curves.json    # Historical equity curves
```

## Deployment

Deployed on [Vercel](https://vercel.com):

```bash
npx vercel
```

## License

Proprietary — All rights reserved.

---

*Built with ❤️ for emerging markets*

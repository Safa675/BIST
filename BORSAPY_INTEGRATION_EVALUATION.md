# Borsapy & Borsaci Integration Evaluation

**Date**: February 2026
**Purpose**: Comprehensive evaluation of integration opportunities between your existing implementation and the borsapy/borsa-mcp/borsaci ecosystem

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Gap Analysis](#gap-analysis)
4. [Integration Opportunities](#integration-opportunities)
5. [Borsaci Multi-Agent Architecture](#borsaci-multi-agent-architecture)
6. [Detailed Implementation Plan](#detailed-implementation-plan)
7. [Architecture Recommendations](#architecture-recommendations)
8. [Risk Assessment](#risk-assessment)
9. [Priority Matrix](#priority-matrix)

---

## Executive Summary

### What You Have (Already Integrated)

| Component | Status | Quality |
|-----------|--------|---------|
| **Borsapy Client** | ✅ Production | 720 lines, comprehensive wrapper |
| **18 Technical Indicators** | ✅ Production | RSI, MACD, BB, ATR, ADX, Supertrend, etc. |
| **Real-Time Quotes** | ✅ Production | TTL caching, portfolio snapshots |
| **Portfolio Analytics** | ✅ Production | Sharpe, Sortino, VaR, CVaR, etc. |
| **54+ Trading Signals** | ✅ Production | Technical, fundamental, sector rotation |
| **Borsa-MCP Client** | ✅ Production | 26 tools, JSON-RPC 2.0, SSE support |
| **Tool Orchestrator** | ✅ Production | Iterative function calling, retry logic |
| **4 LLM Agents** | ✅ Production | Portfolio, Risk, Analyst, Research |

### What's Blocked/Broken

| Feature | Issue | Impact |
|---------|-------|--------|
| **Stock Screener** | SSL_CERTIFICATE_VERIFY_FAILED | Cannot filter stocks by fundamentals |
| **Financial Statements** | Returns empty DataFrames | No balance sheet/income stmt data |
| **Dividends** | Returns empty data | No dividend history |

### Key Integration Opportunities

| Feature | Source | Value |
|---------|--------|-------|
| **Multi-Agent Orchestration** | Borsaci | Planning → Execution → Validation → Synthesis |
| **Terminal Candlestick Charts** | Borsaci | CLI visualization via plotext |
| **TEFAS Funds (836+)** | Borsa-MCP | Turkish investment fund analysis |
| **Crypto (295+ pairs)** | Borsa-MCP | BtcTurk + Coinbase integration |
| **US Stocks** | Borsa-MCP | NYSE/NASDAQ comparison |
| **Sector Comparison** | Borsa-MCP | Cross-sector analysis |
| **Pivot Points** | Borsa-MCP | Support/resistance levels |

---

## Current State Analysis

### Python Backend (BIST/)

```
BIST/
├── data/Fetcher-Scrapper/
│   ├── borsapy_client.py      # ✅ Core wrapper (720 lines)
│   ├── realtime_stream.py     # ✅ Real-time quotes
│   └── realtime_api.py        # ✅ CLI interface
├── Models/
│   ├── common/
│   │   ├── data_loader.py     # ✅ BorsapyAdapter integration
│   │   ├── portfolio_analytics.py  # ✅ Risk metrics
│   │   └── borsapy_adapter.py # ✅ Adapter pattern
│   └── signals/
│       ├── borsapy_indicators.py  # ✅ 18 indicators (1,076 lines)
│       └── [54+ signal files]     # ✅ Trading signals
```

**Strengths:**
- Centralized borsapy wrapper with caching (in-memory + disk)
- Batch processing for multiple tickers
- Multi-indicator panel building optimized for backtesting
- Direct Python library access (faster than HTTP)

**Weaknesses:**
- SSL certificate issue blocks screener
- Financial statements return empty
- No TEFAS fund integration
- Limited FX/crypto coverage

### TypeScript Frontend (bist-quant-ai/)

```
bist-quant-ai/src/lib/agents/
├── borsa-mcp-client.ts        # ✅ MCP integration (828 lines)
├── tool-orchestrator.ts       # ✅ Function calling (471 lines)
├── orchestrator.ts            # ✅ Base agent logic
├── policy.ts                  # ✅ Tool policies per agent
└── logging.ts                 # ✅ Telemetry
```

**Strengths:**
- Full Borsa-MCP integration with 26 tools
- Iterative function calling loop
- Retry logic with exponential backoff
- Token usage tracking
- Tool definition caching (5-min TTL)

**Weaknesses:**
- Single iteration loop (no parallel task execution)
- No planning agent for complex queries
- No validation step for tool results
- No conversation history persistence

---

## Gap Analysis

### Feature Gaps vs Borsa-MCP

| Feature | Your Implementation | Borsa-MCP | Gap Level |
|---------|---------------------|-----------|-----------|
| Stock Screener | ⚠️ SSL blocked | ✅ 23 presets | 🔴 Critical |
| Financial Statements | ⚠️ Empty returns | ✅ Full data | 🔴 Critical |
| TEFAS Funds | ❌ None | ✅ 836+ funds | 🟡 Medium |
| Crypto Markets | ❌ Basic only | ✅ BtcTurk + Coinbase | 🟡 Medium |
| FX/Commodities | ❌ USD/TRY only | ✅ 65 pairs | 🟡 Medium |
| US Stocks | ❌ None | ✅ NYSE/NASDAQ | 🟡 Medium |
| Sector Comparison | ❌ None | ✅ Full support | 🟢 Low |
| Pivot Points | ❌ None | ✅ Support/resistance | 🟢 Low |
| Symbol Search | ❌ None | ✅ Multi-asset | 🟢 Low |

### Agent Architecture Gaps vs Borsaci

| Feature | Your Implementation | Borsaci | Gap Level |
|---------|---------------------|---------|-----------|
| Planning Agent | ❌ None | ✅ Gemini 2.5 Pro | 🔴 Critical |
| Task Decomposition | ❌ None | ✅ Topological sorting | 🔴 Critical |
| Parallel Execution | ❌ Sequential only | ✅ Dependency-aware | 🟡 Medium |
| Validation Agent | ❌ None | ✅ Retry logic | 🟡 Medium |
| Chart Generation | ❌ None | ✅ plotext terminal | 🟡 Medium |
| Conversation History | ❌ None | ✅ Persistent | 🟢 Low |
| Loop Detection | ❌ None | ✅ Max steps/retries | 🟢 Low |

---

## Integration Opportunities

### 1. Immediate Fixes (Use Borsa-MCP as Fallback)

Your borsapy has SSL/data issues. Use MCP as a reliable fallback:

```python
# borsapy_client.py - Add fallback pattern

class BorsapyClient:
    def __init__(self, cache_dir=None, use_mcp_fallback=True):
        self.use_mcp_fallback = use_mcp_fallback
        self._mcp_endpoint = "https://borsamcp.fastmcp.app/mcp"

    def screen_stocks(self, **filters) -> pd.DataFrame:
        """Screen stocks with MCP fallback for SSL issues."""
        try:
            # Try direct borsapy first
            return bp.screen_stocks(**filters)
        except SSLError:
            if not self.use_mcp_fallback:
                raise
            # Fallback to MCP
            return self._screen_via_mcp(filters)

    def _screen_via_mcp(self, filters: dict) -> pd.DataFrame:
        """Use Borsa-MCP for screening when borsapy fails."""
        import httpx

        response = httpx.post(
            self._mcp_endpoint,
            json={
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/call",
                "params": {
                    "name": "screen_securities",
                    "arguments": filters
                }
            },
            headers={"Content-Type": "application/json"},
            timeout=15.0
        )

        result = response.json()
        if "error" in result:
            raise Exception(result["error"]["message"])

        return pd.DataFrame(result["result"]["data"])
```

### 2. Multi-Agent Architecture from Borsaci

Borsaci's 4-agent architecture is highly relevant for your system:

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Query                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PLANNING AGENT                                │
│  • Gemini 2.5 Pro (reasoning-heavy)                             │
│  • Decomposes query into atomic tasks                           │
│  • Builds dependency graph via topological sort                 │
│  • Output: Task plan with execution order                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ACTION AGENT                                  │
│  • Gemini 2.5 Flash (tool calling)                              │
│  • Executes MCP tools per task                                  │
│  • Parallel execution for independent tasks                     │
│  • Respects dependency order                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION AGENT                              │
│  • Gemini 2.5 Flash                                             │
│  • Verifies task completion                                     │
│  • Handles retries (max 5 per task)                             │
│  • Detects loops and failures                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANSWER AGENT                                  │
│  • Gemini 2.5 Flash                                             │
│  • Synthesizes findings into narrative                          │
│  • Generates charts via plotext (outside LLM)                   │
│  • Turkish language support                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation for Your System:**

```typescript
// src/lib/agents/planning-agent.ts

interface TaskNode {
  id: string;
  description: string;
  tool: string;
  params: Record<string, unknown>;
  dependencies: string[];  // IDs of tasks that must complete first
  status: "pending" | "running" | "completed" | "failed";
  result?: unknown;
}

interface ExecutionPlan {
  tasks: TaskNode[];
  executionOrder: string[][];  // Groups of tasks that can run in parallel
}

export async function createExecutionPlan(
  query: string,
  context: AgentContext
): Promise<ExecutionPlan> {
  const planningPrompt = `You are a financial analysis planning agent.

Given the user query, decompose it into atomic tasks that can be executed
using the available Borsa MCP tools. For each task, specify:
1. A unique task ID
2. Which tool to call
3. The parameters for the tool
4. Which other tasks it depends on (by ID)

Available tools:
- screen_securities: Filter stocks by fundamentals/technicals
- get_financial_statements: Balance sheet, income, cash flow
- get_financial_ratios: Valuation and health metrics
- get_technical_analysis: RSI, MACD, Bollinger, etc.
- get_sector_comparison: Compare sectors
- get_fund_data: TEFAS fund information
- get_news: KAP announcements

User Query: ${query}

Respond with a JSON array of tasks.`;

  // Call Azure OpenAI to get the plan
  const response = await callAzureOpenAI(planningPrompt);
  const tasks = JSON.parse(response) as TaskNode[];

  // Topological sort to determine execution order
  const executionOrder = topologicalSort(tasks);

  return { tasks, executionOrder };
}

function topologicalSort(tasks: TaskNode[]): string[][] {
  const inDegree = new Map<string, number>();
  const graph = new Map<string, string[]>();

  // Initialize
  for (const task of tasks) {
    inDegree.set(task.id, task.dependencies.length);
    graph.set(task.id, []);
  }

  // Build adjacency list
  for (const task of tasks) {
    for (const dep of task.dependencies) {
      const edges = graph.get(dep) || [];
      edges.push(task.id);
      graph.set(dep, edges);
    }
  }

  // Kahn's algorithm with level tracking
  const levels: string[][] = [];
  let queue = tasks
    .filter(t => t.dependencies.length === 0)
    .map(t => t.id);

  while (queue.length > 0) {
    levels.push([...queue]);
    const nextQueue: string[] = [];

    for (const nodeId of queue) {
      for (const neighbor of (graph.get(nodeId) || [])) {
        const newDegree = (inDegree.get(neighbor) || 1) - 1;
        inDegree.set(neighbor, newDegree);
        if (newDegree === 0) {
          nextQueue.push(neighbor);
        }
      }
    }

    queue = nextQueue;
  }

  return levels;
}
```

### 3. Parallel Task Execution

Borsaci achieves 50-70% performance improvement via dependency-aware parallelization:

```typescript
// src/lib/agents/parallel-executor.ts

export async function executeWithDependencies(
  plan: ExecutionPlan,
  requestId: string
): Promise<Map<string, MCPToolResult>> {
  const results = new Map<string, MCPToolResult>();
  const taskMap = new Map(plan.tasks.map(t => [t.id, t]));

  for (const level of plan.executionOrder) {
    // Execute all tasks in this level in parallel
    const promises = level.map(async (taskId) => {
      const task = taskMap.get(taskId)!;

      // Inject results from dependencies into params
      const enrichedParams = { ...task.params };
      for (const depId of task.dependencies) {
        const depResult = results.get(depId);
        if (depResult?.success && depResult.data) {
          enrichedParams[`_dep_${depId}`] = depResult.data;
        }
      }

      const result = await executeMCPTool({
        tool: task.tool,
        params: enrichedParams
      }, requestId);

      return { taskId, result };
    });

    const levelResults = await Promise.all(promises);

    for (const { taskId, result } of levelResults) {
      results.set(taskId, result);
    }
  }

  return results;
}
```

### 4. TEFAS Fund Integration

Add TEFAS fund analysis capability:

```python
# Models/common/fund_analyzer.py

from dataclasses import dataclass
from typing import Optional
import httpx
import pandas as pd

@dataclass
class FundMetrics:
    fund_code: str
    name: str
    category: str
    nav: float
    return_1m: float
    return_3m: float
    return_1y: float
    expense_ratio: float
    aum: float

class TEFASAnalyzer:
    """Analyze Turkish investment funds via Borsa-MCP."""

    def __init__(self):
        self._mcp_endpoint = "https://borsamcp.fastmcp.app/mcp"

    def get_fund_data(
        self,
        fund_code: Optional[str] = None,
        category: Optional[str] = None,
        sort_by: str = "return_1y"
    ) -> pd.DataFrame:
        """
        Fetch TEFAS fund data.

        Args:
            fund_code: Specific fund code (e.g., "IPB", "AK1")
            category: Filter by category (equity, bond, money_market, mixed)
            sort_by: Sort metric (return_1m, return_3m, return_1y, nav)
        """
        params = {"sort_by": sort_by}
        if fund_code:
            params["fund_code"] = fund_code
        if category:
            params["category"] = category

        result = self._call_mcp("get_fund_data", params)
        return pd.DataFrame(result)

    def compare_funds(
        self,
        fund_codes: list[str],
        period: str = "1y"
    ) -> pd.DataFrame:
        """Compare multiple funds side-by-side."""
        results = []
        for code in fund_codes:
            data = self.get_fund_data(fund_code=code)
            if not data.empty:
                results.append(data.iloc[0])

        return pd.DataFrame(results)

    def get_top_performers(
        self,
        category: str,
        period: str = "1y",
        limit: int = 10
    ) -> pd.DataFrame:
        """Get top performing funds in a category."""
        data = self.get_fund_data(
            category=category,
            sort_by=f"return_{period}"
        )
        return data.head(limit)

    def _call_mcp(self, tool: str, params: dict) -> dict:
        response = httpx.post(
            self._mcp_endpoint,
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "tools/call",
                "params": {"name": tool, "arguments": params}
            },
            timeout=15.0
        )
        result = response.json()
        if "error" in result:
            raise Exception(result["error"]["message"])
        return result.get("result", {}).get("content", [{}])[0]
```

### 5. Terminal Chart Generation (from Borsaci)

Add CLI-based chart visualization:

```python
# scripts/cli_charts.py

import plotext as plt
import pandas as pd
from datetime import datetime, timedelta

def plot_candlestick(
    symbol: str,
    data: pd.DataFrame,
    period: str = "1M"
) -> None:
    """
    Plot candlestick chart in terminal.

    Args:
        symbol: Stock ticker
        data: DataFrame with Date, Open, High, Low, Close
        period: Display period (1W, 1M, 3M, 1Y)
    """
    plt.clear_figure()
    plt.theme("dark")

    # Filter data by period
    days = {"1W": 7, "1M": 30, "3M": 90, "1Y": 365}[period]
    cutoff = datetime.now() - timedelta(days=days)
    df = data[data["Date"] >= cutoff].copy()

    # Prepare candlestick data
    dates = [d.strftime("%m/%d") for d in df["Date"]]
    opens = df["Open"].tolist()
    closes = df["Close"].tolist()
    highs = df["High"].tolist()
    lows = df["Low"].tolist()

    # Plot
    plt.candlestick(dates, {"Open": opens, "Close": closes,
                            "High": highs, "Low": lows})
    plt.title(f"{symbol} - {period}")
    plt.xlabel("Date")
    plt.ylabel("Price (TRY)")
    plt.show()

def plot_indicator_panel(
    symbol: str,
    data: pd.DataFrame,
    indicators: list[str] = ["RSI", "MACD"]
) -> None:
    """Plot price with technical indicators."""
    plt.clear_figure()
    plt.theme("dark")

    # Price subplot
    plt.subplot(len(indicators) + 1, 1, 1)
    plt.plot(data["Close"].tolist(), label="Close")
    plt.title(f"{symbol} Price")

    # Indicator subplots
    for i, indicator in enumerate(indicators, 2):
        plt.subplot(len(indicators) + 1, 1, i)
        if indicator in data.columns:
            plt.plot(data[indicator].tolist(), label=indicator)
            plt.title(indicator)

    plt.show()

def plot_sector_comparison(sectors: pd.DataFrame) -> None:
    """Bar chart comparing sector performance."""
    plt.clear_figure()
    plt.theme("dark")

    names = sectors["sector"].tolist()
    returns = sectors["return_1m"].tolist()

    colors = ["green" if r > 0 else "red" for r in returns]
    plt.bar(names, returns, color=colors)
    plt.title("Sector Performance (1M)")
    plt.xlabel("Sector")
    plt.ylabel("Return %")
    plt.show()
```

### 6. Validation Agent

Add result validation to prevent hallucination:

```typescript
// src/lib/agents/validation-agent.ts

interface ValidationResult {
  isValid: boolean;
  issues: string[];
  suggestedRetry?: {
    tool: string;
    params: Record<string, unknown>;
  };
}

export async function validateToolResult(
  task: TaskNode,
  result: MCPToolResult,
  context: AgentContext
): Promise<ValidationResult> {
  if (!result.success) {
    return {
      isValid: false,
      issues: [`Tool ${task.tool} failed: ${result.error}`],
      suggestedRetry: {
        tool: task.tool,
        params: task.params
      }
    };
  }

  // Validate data structure based on tool
  const data = result.data as Record<string, unknown>;

  switch (task.tool) {
    case "screen_securities":
      return validateScreenerResult(data);
    case "get_financial_statements":
      return validateFinancialsResult(data);
    case "get_technical_analysis":
      return validateTechnicalResult(data);
    default:
      return { isValid: true, issues: [] };
  }
}

function validateScreenerResult(data: unknown): ValidationResult {
  const issues: string[] = [];

  if (!data || typeof data !== "object") {
    issues.push("Screener returned empty or invalid data");
  }

  const content = (data as { content?: unknown[] })?.content;
  if (!Array.isArray(content) || content.length === 0) {
    issues.push("No stocks matched the screening criteria");
  }

  return {
    isValid: issues.length === 0,
    issues
  };
}

function validateFinancialsResult(data: unknown): ValidationResult {
  const issues: string[] = [];
  const content = (data as { content?: Array<{ text?: string }> })?.content;

  if (!content || content.length === 0) {
    issues.push("Financial statements returned empty");
  }

  // Check for common issues
  const text = content?.[0]?.text || "";
  if (text.includes("No financial data available")) {
    issues.push("No financial data available for this stock");
  }

  return {
    isValid: issues.length === 0,
    issues
  };
}

function validateTechnicalResult(data: unknown): ValidationResult {
  const issues: string[] = [];
  const content = (data as { content?: Array<{ text?: string }> })?.content;
  const text = content?.[0]?.text || "";

  // Validate RSI range
  const rsiMatch = text.match(/RSI[:\s]+(\d+\.?\d*)/i);
  if (rsiMatch) {
    const rsi = parseFloat(rsiMatch[1]);
    if (rsi < 0 || rsi > 100) {
      issues.push(`Invalid RSI value: ${rsi} (must be 0-100)`);
    }
  }

  return {
    isValid: issues.length === 0,
    issues
  };
}
```

---

## Detailed Implementation Plan

### Phase 1: Critical Fixes (Week 1)

| Task | Description | Files |
|------|-------------|-------|
| **1.1** | Add MCP fallback for SSL-blocked screener | `borsapy_client.py` |
| **1.2** | Add MCP fallback for empty financials | `borsapy_client.py` |
| **1.3** | Test MCP endpoint reliability | New test file |
| **1.4** | Add retry logic with exponential backoff | `borsapy_client.py` |

**Deliverable**: Working screener and financials via MCP fallback

### Phase 2: Agent Enhancement (Week 2-3)

| Task | Description | Files |
|------|-------------|-------|
| **2.1** | Implement Planning Agent | `planning-agent.ts` |
| **2.2** | Add topological sort for dependencies | `planning-agent.ts` |
| **2.3** | Implement parallel task execution | `parallel-executor.ts` |
| **2.4** | Add Validation Agent | `validation-agent.ts` |
| **2.5** | Update tool orchestrator to use new agents | `tool-orchestrator.ts` |

**Deliverable**: Multi-agent orchestration with 50-70% performance improvement

### Phase 3: New Data Sources (Week 4)

| Task | Description | Files |
|------|-------------|-------|
| **3.1** | Add TEFAS fund analyzer | `fund_analyzer.py` |
| **3.2** | Integrate fund data with portfolio analytics | `portfolio_analytics.py` |
| **3.3** | Add crypto market integration | `crypto_client.py` |
| **3.4** | Expand FX coverage (65 pairs) | `borsapy_client.py` |

**Deliverable**: Access to 836+ TEFAS funds, crypto, and full FX

### Phase 4: Visualization (Week 5)

| Task | Description | Files |
|------|-------------|-------|
| **4.1** | Add plotext terminal charts | `cli_charts.py` |
| **4.2** | Integrate charts with CLI | `realtime_api.py` |
| **4.3** | Add sector comparison visualizations | `cli_charts.py` |

**Deliverable**: Terminal-based candlestick and indicator charts

### Phase 5: LLM Integration (Week 6)

| Task | Description | Files |
|------|-------------|-------|
| **5.1** | Add conversation history persistence | `orchestrator.ts` |
| **5.2** | Implement loop detection | `tool-orchestrator.ts` |
| **5.3** | Add Turkish language prompts | `prompts.ts` |
| **5.4** | Optimize token usage | `orchestrator.ts` |

**Deliverable**: Production-ready multi-turn agent conversations

---

## Architecture Recommendations

### Recommended Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            USER INTERFACE                                    │
│                    (Next.js / CLI / API)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│         LLM AGENTS              │  │       DIRECT API                │
│  (For complex queries)          │  │  (For dashboards/backtests)     │
├─────────────────────────────────┤  ├─────────────────────────────────┤
│ ┌─────────────────────────────┐ │  │                                 │
│ │     Planning Agent          │ │  │  Python Backend (BIST/)         │
│ │  (Query decomposition)      │ │  │  ├── borsapy_client.py          │
│ └─────────────────────────────┘ │  │  ├── realtime_stream.py         │
│              │                  │  │  ├── portfolio_analytics.py     │
│              ▼                  │  │  └── borsapy_indicators.py      │
│ ┌─────────────────────────────┐ │  │                                 │
│ │     Action Agent            │ │  │  (Direct borsapy library)       │
│ │  (Tool execution)           │ │  │  - Faster for batch ops         │
│ └─────────────────────────────┘ │  │  - No network latency           │
│              │                  │  │  - Full pandas integration      │
│              ▼                  │  │                                 │
│ ┌─────────────────────────────┐ │  └─────────────────────────────────┘
│ │   Validation Agent          │ │
│ │  (Result verification)      │ │
│ └─────────────────────────────┘ │
│              │                  │
│              ▼                  │
│ ┌─────────────────────────────┐ │
│ │     Answer Agent            │ │
│ │  (Response synthesis)       │ │
│ └─────────────────────────────┘ │
└─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   PRIMARY: borsapy library (direct)     FALLBACK: Borsa-MCP (HTTP)          │
│   ├── Quotes & history                   ├── Screener (bypasses SSL)        │
│   ├── Technical indicators               ├── Financial statements           │
│   ├── Index components                   ├── TEFAS funds                    │
│   └── Batch operations                   ├── Crypto markets                 │
│                                          └── US stocks                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tool Routing Strategy

```typescript
// src/lib/agents/tool-router.ts

type ToolSource = "local" | "mcp" | "hybrid";

const TOOL_ROUTING: Record<string, ToolSource> = {
  // Use local borsapy (faster, no latency)
  get_quote: "local",
  get_historical_data: "local",
  get_technical_analysis: "local",
  portfolio_analytics: "local",
  factor_signals: "local",

  // Use MCP (your borsapy has issues)
  screen_securities: "mcp",
  get_financial_statements: "mcp",
  get_financial_ratios: "mcp",
  get_fund_data: "mcp",
  get_crypto_market: "mcp",
  get_sector_comparison: "mcp",

  // Hybrid (try local first, fallback to MCP)
  get_dividends: "hybrid",
  get_news: "hybrid",
};

export async function routeToolCall(
  tool: string,
  params: Record<string, unknown>
): Promise<MCPToolResult> {
  const source = TOOL_ROUTING[tool] || "mcp";

  switch (source) {
    case "local":
      return callLocalPython(tool, params);
    case "mcp":
      return executeMCPTool({ tool, params });
    case "hybrid":
      try {
        const localResult = await callLocalPython(tool, params);
        if (localResult.success && hasData(localResult.data)) {
          return localResult;
        }
      } catch {
        // Fallback to MCP
      }
      return executeMCPTool({ tool, params });
  }
}
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MCP endpoint downtime | Medium | High | Add circuit breaker, local fallbacks |
| Rate limiting on MCP | Low | Medium | Implement request throttling |
| SSL issue persists | High | Medium | MCP fallback already solves this |
| LLM cost increases | Medium | Medium | Token budgets, caching responses |
| Data inconsistency (local vs MCP) | Low | Low | Prefer one source per data type |

---

## Priority Matrix

```
                    IMPACT
              Low        High
         ┌──────────┬──────────┐
    Low  │  Pivot   │  TEFAS   │
         │  Points  │  Funds   │
EFFORT   ├──────────┼──────────┤
         │  Charts  │ Planning │
   High  │  (CLI)   │  Agent   │
         └──────────┴──────────┘

PRIORITY ORDER:
1. 🔴 MCP Fallback for Screener/Financials (High Impact, Low Effort)
2. 🔴 Planning Agent (High Impact, High Effort)
3. 🟡 TEFAS Fund Integration (High Impact, Medium Effort)
4. 🟡 Parallel Task Execution (Medium Impact, Medium Effort)
5. 🟢 Terminal Charts (Low Impact, Low Effort)
6. 🟢 Pivot Points (Low Impact, Low Effort)
```

---

## Summary

### Must Do (This Week)
1. Add MCP fallback for SSL-blocked screener
2. Add MCP fallback for empty financial statements
3. Test MCP endpoint reliability

### Should Do (This Month)
1. Implement Planning Agent from Borsaci architecture
2. Add parallel task execution with dependency tracking
3. Integrate TEFAS fund analyzer
4. Add Validation Agent

### Nice to Have (Future)
1. Terminal candlestick charts via plotext
2. Conversation history persistence
3. Turkish language prompts
4. US stock comparison

### Key Metrics to Track
- **Tool success rate**: Target >95%
- **Average response time**: Target <3s for simple queries
- **Parallel execution speedup**: Target 50-70%
- **Token usage per query**: Target <4000 tokens

---

## Appendix: File References

| File | Purpose | Lines |
|------|---------|-------|
| [borsapy_client.py](BIST/data/Fetcher-Scrapper/borsapy_client.py) | Core borsapy wrapper | 720 |
| [borsapy_indicators.py](BIST/Models/signals/borsapy_indicators.py) | Technical indicators | 1,076 |
| [borsa-mcp-client.ts](../bist-quant-ai/src/lib/agents/borsa-mcp-client.ts) | MCP integration | 828 |
| [tool-orchestrator.ts](../bist-quant-ai/src/lib/agents/tool-orchestrator.ts) | Agent orchestration | 471 |
| [portfolio_analytics.py](BIST/Models/common/portfolio_analytics.py) | Risk metrics | ~150 |
| [realtime_stream.py](BIST/data/Fetcher-Scrapper/realtime_stream.py) | Real-time quotes | ~300 |

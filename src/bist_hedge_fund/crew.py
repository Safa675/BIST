"""
BIST Hedge Fund Crew

A CrewAI-powered hedge fund that analyzes Borsa Istanbul stocks using:
- Market regime detection (Bull/Bear/Stress/Choppy/Recovery)
- Technical analysis (RSI, MACD, Bollinger, SMA, Donchian, Momentum)
- Fundamental analysis (Value composite, Profitability, Balance sheet)
- Risk management (Downside vol targeting, Stop-loss, Inverse vol weighting)
- Portfolio construction (Regime-aware allocation with gold hedge)
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from bist_hedge_fund.tools import (
    calculate_risk_metrics,
    calculate_technical_indicators,
    detect_market_regime,
    fetch_bist_prices,
    fetch_macro_data,
    fetch_xu100_data,
    get_bist_fundamentals,
    get_bist_news,
    screen_bist_stocks,
)


@CrewBase
class BistHedgeFundCrew:
    """BIST Hedge Fund Crew - Multi-agent investment team for Borsa Istanbul"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ── Agents ──────────────────────────────────────────────────────────

    @agent
    def regime_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["regime_analyst"],
            tools=[detect_market_regime, fetch_xu100_data, fetch_macro_data],
            verbose=True,
        )

    @agent
    def technical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["technical_analyst"],
            tools=[calculate_technical_indicators, fetch_bist_prices],
            verbose=True,
        )

    @agent
    def fundamental_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["fundamental_analyst"],
            tools=[get_bist_fundamentals, fetch_bist_prices],
            verbose=True,
        )

    @agent
    def risk_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["risk_manager"],
            tools=[calculate_risk_metrics, fetch_bist_prices, detect_market_regime],
            verbose=True,
        )

    @agent
    def portfolio_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["portfolio_manager"],
            tools=[
                screen_bist_stocks,
                fetch_bist_prices,
                detect_market_regime,
                get_bist_news,
            ],
            verbose=True,
        )

    # ── Tasks ───────────────────────────────────────────────────────────

    @task
    def regime_detection(self) -> Task:
        return Task(config=self.tasks_config["regime_detection"])

    @task
    def technical_analysis(self) -> Task:
        return Task(config=self.tasks_config["technical_analysis"])

    @task
    def fundamental_analysis(self) -> Task:
        return Task(config=self.tasks_config["fundamental_analysis"])

    @task
    def risk_assessment(self) -> Task:
        return Task(config=self.tasks_config["risk_assessment"])

    @task
    def portfolio_recommendation(self) -> Task:
        return Task(config=self.tasks_config["portfolio_recommendation"])

    # ── Crew ────────────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """Assembles the BIST Hedge Fund crew.

        Sequential pipeline:
        1. Regime Detection   → Sets overall equity/gold allocation
        2. Technical Analysis  → RSI, MACD, SMA, Donchian, Momentum signals
        3. Fundamental Analysis → Value, Profitability, Balance sheet
        4. Risk Assessment     → Vol targeting, stop-loss, position sizing
        5. Portfolio Recommendation → Final allocation with specific trades
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

#!/usr/bin/env python3
"""
BIST Hedge Fund - CrewAI Entry Point

This is the main entry point required by CrewAI Enterprise.
It defines the run() function that CrewAI calls to start the crew.

Usage (local):
    cd BIST
    crewai run

Usage (with custom tickers):
    Set BIST_TICKERS environment variable:
    BIST_TICKERS="THYAO,SASA,EREGL,BIMAS,KCHOL" crewai run
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from bist_hedge_fund.crew import BistHedgeFundCrew

# Default BIST stocks to analyze (blue chips + high momentum)
DEFAULT_TICKERS = "THYAO,SASA,EREGL,BIMAS,KCHOL,TUPRS,SAHOL,AKBNK,GARAN,SISE"


def run():
    """Entry point for CrewAI Enterprise and `crewai run`."""
    tickers = os.getenv("BIST_TICKERS", DEFAULT_TICKERS)

    inputs = {"tickers": tickers}

    result = BistHedgeFundCrew().crew().kickoff(inputs=inputs)
    return result


def train(n_iterations: int = 3, model_id: str = "gpt-4o"):
    """Train the crew for improved performance (optional)."""
    inputs = {"tickers": DEFAULT_TICKERS}
    BistHedgeFundCrew().crew().train(
        n_iterations=n_iterations,
        inputs=inputs,
        filename="trained_agents_data.pkl",
    )


if __name__ == "__main__":
    result = run()
    print("\n" + "=" * 70)
    print("  BIST HEDGE FUND - FINAL REPORT")
    print("=" * 70)
    print(result)

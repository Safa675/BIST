"""
XU100 Benchmark Configuration

This configuration runs XU100 through the portfolio engine to compare
regime-aware XU100 vs buy-and-hold XU100.

The portfolio engine will:
- Follow regime signals (exit to gold in Bear/Stress regimes)
- Apply volatility targeting
- Rebalance monthly

This shows how much value the regime filter adds vs passive holding.
"""

SIGNAL_CONFIG = {
    'name': 'xu100',
    'enabled': True,
    'rebalance_frequency': 'monthly',
    'timeline': {
        'start_date': '2014-01-01',  # Match momentum timeline for comparison
        'end_date': '2026-02-03',
    },
    'description': 'XU100 index with regime awareness - benchmark for risk management value',
}

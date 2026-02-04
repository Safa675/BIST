"""
Momentum Factor Configuration

Defines how the momentum signal integrates into the portfolio engine.
"""

SIGNAL_CONFIG = {
    'name': 'momentum',
    'enabled': True,
    'rebalance_frequency': 'monthly',
    'timeline': {
        'start_date': '2014-01-01',  # Technical signals start from 2014
        'end_date': '2026-02-03',    # Today
    },
    'description': '12-1 momentum with downside volatility risk adjustment',
}

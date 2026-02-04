"""
Profitability Factor Configuration

Defines how the profitability signal integrates into the portfolio engine.
"""

SIGNAL_CONFIG = {
    'name': 'profitability',
    'enabled': True,
    'rebalance_frequency': 'quarterly',
    'timeline': {
        'start_date': '2017-01-01',  # Fundamental signals start from 2017
        'end_date': '2026-02-03',    # Today
    },
    'description': 'Profitability factor based on operating income and gross profit relative to total assets',
}

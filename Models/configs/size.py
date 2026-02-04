"""
Size Factor Configuration

Defines how the size signal integrates into the portfolio engine.
"""

SIGNAL_CONFIG = {
    'name': 'size',
    'enabled': True,
    'rebalance_frequency': 'quarterly',
    'timeline': {
        'start_date': '2017-01-01',  # Fundamental signals start from 2017
        'end_date': '2026-02-03',    # Today
    },
    'description': 'Size factor based on market capitalization with liquidity adjustments',
}

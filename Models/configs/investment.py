"""
Investment Factor Configuration

Defines how the investment signal integrates into the portfolio engine.
"""

SIGNAL_CONFIG = {
    'name': 'investment',
    'enabled': True,
    'rebalance_frequency': 'quarterly',
    'timeline': {
        'start_date': '2017-01-01',  # Fundamental signals start from 2017
        'end_date': '2026-02-03',    # Today
    },
    'description': 'Investment factor based on asset growth and capital expenditure patterns',
}

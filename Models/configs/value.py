"""
Value Factor Configuration

Defines how the value signal integrates into the portfolio engine.
"""

SIGNAL_CONFIG = {
    'name': 'value',
    'enabled': True,
    'rebalance_frequency': 'quarterly',
    'timeline': {
        'start_date': '2017-01-01',  # Fundamental signals start from 2017
        'end_date': '2026-02-03',    # Today
    },
    'description': 'Composite value score from E/P, FCF/P, OCF/EV, S/P, EBITDA/EV ratios',
}

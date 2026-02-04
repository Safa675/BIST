"""
SMA Crossover Signal Configuration

10/30 SMA crossover strategy:
- Bullish when 10-day SMA > 30-day SMA
- Bearish when 10-day SMA < 30-day SMA
"""

SIGNAL_CONFIG = {
    'name': 'sma',
    'enabled': True,
    'rebalance_frequency': 'monthly',
    'timeline': {
        'start_date': '2014-01-01',  # Technical signals start from 2014
        'end_date': '2026-02-03',    # Today
    },
    'description': '10/30 SMA crossover - bullish when short MA > long MA',
    'parameters': {
        'short_period': 10,
        'long_period': 30,
    }
}

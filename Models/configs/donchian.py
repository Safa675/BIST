"""
Donchian Channel Breakout Signal Configuration

20-day Donchian Channel strategy:
- Buy when price breaks above 20-day high
- Sell when price breaks below 20-day low
"""

SIGNAL_CONFIG = {
    'name': 'donchian',
    'enabled': True,
    'rebalance_frequency': 'monthly',
    'timeline': {
        'start_date': '2014-01-01',  # Technical signals start from 2014
        'end_date': '2026-02-03',    # Today
    },
    'description': '20-day Donchian Channel breakout strategy',
    'parameters': {
        'lookback': 20,
    }
}

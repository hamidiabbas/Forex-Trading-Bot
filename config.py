# Simple configuration - copy and paste this entire file
SYMBOLS = ['EURUSD', 'GBPUSD', 'XAUUSD']  # Fixed - no more USDJPY

config = {
    'trading': {
        'symbols': SYMBOLS,
        'risk_per_trade': 0.01,
        'max_daily_risk': 0.05
    },
    'mt5': {
        'magic_number': 123456789,
        'max_slippage': 3
    }
}

import logging
logger = logging.getLogger(__name__)
logger.info(f"Config loaded with symbols: {SYMBOLS}")

"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           config.py (Disciplined Crossover Strategy)
 *
 * PURPOSE:
 *
 * This version incorporates psychological discipline rules inspired by
 * Mark Douglas, such as a max daily drawdown and a breakeven stop.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             7.0 (Disciplined)
 *
 ******************************************************************************/
"""

# --- Broker Credentials & Connection Settings ---
MT5_ACCOUNT_NUMBER = 5038274604
MT5_PASSWORD = "G@5iMvHm"  # <-- REPLACE WITH YOUR PASSWORD
MT5_SERVER_NAME = "MetaQuotes-Demo"  # <-- REPLACE WITH YOUR SERVER
MT5_TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe" # <-- VERIFY YOUR PATH

# --- Core Trading Parameters ---
SYMBOLS_TO_TRADE = ["EURUSD", "GBPUSD"]
TIMEFRAMES = { 'EXECUTION': 'H1' }

# --- Global Risk Management Parameters ---
GLOBAL_RISK_PER_TRADE_PERCENTAGE = 1.0
MAX_CONCURRENT_TRADES = 1
MINIMUM_RR_RATIO = 1.5
ATR_SL_MULTIPLIER = 2.0

# --- Crossover Strategy Parameters ---
CROSSOVER_FAST_EMA_PERIOD = 12
CROSSOVER_SLOW_EMA_PERIOD = 26
CROSSOVER_TREND_FILTER_PERIOD = 200

# --- NEW: Psychological Discipline Rules ---
# If true, moves the SL to the entry price after a certain profit is reached.
ENABLE_BREAKEVEN_STOP = True
# The Risk-to-Reward multiple that triggers the breakeven stop (1.0 = 1:1 R:R).
BREAKEVEN_TRIGGER_RR = 1.0

# The maximum percentage of equity the bot is allowed to lose in one day
# before it stops trading for the day. (e.g., 2.0 for 2%)
MAX_DAILY_DRAWDOWN_PERCENT = 2.0

# --- Logging & Diagnostics ---
TRADE_JOURNAL_FILE = "trade_journal.csv"
LOG_LEVEL = "INFO"
ERROR_LOG_FILE = "error_log.txt"

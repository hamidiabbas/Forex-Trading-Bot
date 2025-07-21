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
 * VERSION:             7.1 (Final)
 *
 ******************************************************************************/
"""

# --- Broker Credentials & Connection Settings ---
MT5_ACCOUNT_NUMBER = 5038274604
MT5_PASSWORD = "G@5iMvHm"  # <-- REPLACE WITH YOUR PASSWORD
MT5_SERVER_NAME = "MetaQuotes-Demo"  # <-- REPLACE WITH YOUR SERVER
MT5_TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe" # <-- VERIFY YOUR PATH

# --- ADDED THESE TWO LINES BACK IN ---
CONNECTION_RETRY_ATTEMPTS = 5
CONNECTION_RETRY_DELAY_SECONDS = 10
# -------------------------------------

# --- Core Trading Parameters ---
SYMBOLS_TO_TRADE = ["EURUSD", "GBPUSD"]
TIMEFRAMES = { 'EXECUTION': 'H1' }

# --- Global Risk Management Parameters ---
GLOBAL_RISK_PER_TRADE_PERCENTAGE = 1.0
MAX_CONCURRENT_TRADES = 1
MINIMUM_RR_RATIO = 1.5
ATR_SL_MULTIPLIER = 2.0

# --- Psychological Discipline Rules ---
ENABLE_BREAKEVEN_STOP = True
BREAKEVEN_TRIGGER_RR = 1.0
MAX_DAILY_DRAWDOWN_PERCENT = 2.0

# --- Crossover Strategy Parameters ---
CROSSOVER_FAST_EMA_PERIOD = 12
CROSSOVER_SLOW_EMA_PERIOD = 26
CROSSOVER_TREND_FILTER_PERIOD = 200

# --- Logging & Diagnostics ---
TRADE_JOURNAL_FILE = "trade_journal.csv"
LOG_LEVEL = "INFO"
ERROR_LOG_FILE = "error_log.txt"
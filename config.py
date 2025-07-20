"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           config.py (High-Frequency Crossover Strategy)
 *
 * PURPOSE:
 *
 * This file has been reconfigured to support a new, high-frequency
 * Dual Moving Average Crossover strategy.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             6.0 (Crossover)
 *
 ******************************************************************************/
"""

# --- Broker Credentials & Connection Settings ---
MT5_ACCOUNT_NUMBER = 5038274604
MT5_PASSWORD = "G@5iMvHm"  # <-- REPLACE WITH YOUR PASSWORD
MT5_SERVER_NAME = "MetaQuotes-Demo"  # <-- REPLACE WITH YOUR SERVER
MT5_TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe" # <-- VERIFY YOUR PATH

CONNECTION_RETRY_ATTEMPTS = 5
CONNECTION_RETRY_DELAY_SECONDS = 10

# --- Core Trading Parameters ---
SYMBOLS_TO_TRADE = ["EURUSD", "GBPUSD"] # Reduced symbols for faster testing
TIMEFRAMES = {
    # This strategy only uses the execution timeframe
    'EXECUTION':  'H1'
}

# --- Global Risk Management Parameters ---
GLOBAL_RISK_PER_TRADE_PERCENTAGE = 1.0
MAX_CONCURRENT_TRADES = 1 # Only one trade per symbol at a time
MINIMUM_RR_RATIO = 1.5 # A higher R:R is needed for crossover strategies
ATR_SL_MULTIPLIER = 2.0 # A wider stop to avoid noise

# --- Crossover Strategy Parameters ---
CROSSOVER_FAST_EMA_PERIOD = 12
CROSSOVER_SLOW_EMA_PERIOD = 26
# The filter: only trade in the direction of this long-term trend
CROSSOVER_TREND_FILTER_PERIOD = 200

# --- Logging & Diagnostics ---
TRADE_JOURNAL_FILE = "trade_journal.csv"
LOG_LEVEL = "INFO"
ERROR_LOG_FILE = "error_log.txt"
LOOP_SLEEP_TIMER = 300

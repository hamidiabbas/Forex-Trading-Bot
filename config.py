"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           config.py (Final Aggressive Optimization)
 *
 * PURPOSE:
 *
 * This file serves as the central configuration hub for the entire
 * trading bot. This version uses aggressive parameters to ensure a high
 * trade frequency for analysis.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             5.3 (Aggressive)
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
SYMBOLS_TO_TRADE = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]
TIMEFRAMES = {
    'STRUCTURAL': 'D1',
    'POSITIONAL': 'H4',
    'EXECUTION':  'H1'
}

# --- Global Risk Management Parameters ---
GLOBAL_RISK_PER_TRADE_PERCENTAGE = 1.0
MAX_CONCURRENT_TRADES = 3
# --- OPTIMIZATION: Lowered R:R Ratio ---
MINIMUM_RR_RATIO = 1.2 # Lowered from 1.5
ATR_SL_MULTIPLIER = 1.5

# --- Confluence Strategy Parameters ---
# --- OPTIMIZATION: Lowered score to dramatically increase trade frequency ---
MINIMUM_CONFLUENCE_SCORE = 2 # Lowered from 3 to 2

# --- Indicator Periods (for Confluence calculation) ---
EMA_PERIODS = [10, 20, 50, 100, 200]
SMA_PERIODS = [10, 20, 50, 100, 200]
RSI_PERIOD = 14
STOCH_K = 14
STOCH_D = 3
CCI_PERIOD = 20
ADX_PERIOD = 14
MOMENTUM_PERIOD = 10
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# --- Logging & Diagnostics ---
TRADE_JOURNAL_FILE = "trade_journal.csv"
LOG_LEVEL = "INFO"
ERROR_LOG_FILE = "error_log.txt"
LOOP_SLEEP_TIMER = 300
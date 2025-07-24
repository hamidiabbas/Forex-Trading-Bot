"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           config.py (Secure & Ultimate Robust Path)
 *
 * PURPOSE:
 *
 * This version uses the most robust method to find the .env file and adds
 * debugging prints to diagnose loading issues.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 23, 2025
 *
 * VERSION:             26.2 (Ultimate Robust Path)
 *
 ******************************************************************************/
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# --- THIS IS THE ULTIMATE ROBUST SOLUTION ---
# 1. Get the directory where this config.py file is located.
#    __file__ is a special variable that holds the path to the current script.
config_dir = Path(__file__).resolve().parent

# 2. Join that directory path with the name of your .env file.
#    This creates an absolute path to your .env file.
env_path = config_dir / ".env"

# 3. Add debugging prints to see what's happening.
print("--- Debugging .env Loading ---")
print(f"Looking for .env file at: {env_path}")

# 4. Load the .env file from the explicit path.
#    The load_dotenv() function returns True if it found and loaded the file.
was_loaded = load_dotenv(dotenv_path=env_path)

if was_loaded:
    print(".env file was found and loaded successfully.")
else:
    print("WARNING: .env file was NOT found at the specified path.")
print("----------------------------")


def get_secret(variable_name: str) -> str:
    """
    Gets a secret from environment variables. Raises an error if not found.
    """
    value = os.getenv(variable_name)
    if not value:
        raise RuntimeError(f"Missing required secret: '{variable_name}'. Please check your .env file.")
    return value

# --- Broker Credentials & Connection Settings (Loaded securely) ---
MT5_ACCOUNT_NUMBER = int(get_secret("MT5_ACCOUNT_NUMBER"))
MT5_PASSWORD = get_secret("MT5_PASSWORD")
MT5_SERVER_NAME = get_secret("MT5_SERVER_NAME")
MT5_TERMINAL_PATH = get_secret("MT5_TERMINAL_PATH")

# --- The rest of the file remains the same ---
CONNECTION_RETRY_ATTEMPTS = 5
CONNECTION_RETRY_DELAY_SECONDS = 10

# --- Core Trading Parameters ---
SYMBOLS_TO_TRADE = ["EURUSD", "GBPUSD"]
# NEW: Define roles for each timeframe for the multi-timeframe strategy
TIMEFRAMES = {
    'BIAS': 'H4',
    'POI': 'H1',     # Point of Interest
    'ENTRY': 'M15'
}

# --- Global Risk Management Parameters ---
GLOBAL_RISK_PER_TRADE_PERCENTAGE = 1.0
MAX_CONCURRENT_TRADES = 1
MINIMUM_RR_RATIO = 3.0
ATR_SL_MULTIPLIER = 2.0

# --- Strategy & Regime Filter Parameters ---
ICT_POI_LOOKBACK = 50
RSI_PERIOD = 14
RANGE_RSI_OVERBOUGHT = 70
RANGE_RSI_OVERSOLD = 30
BBANDS_PERIOD = 20
BBANDS_STD = 2.0
BBANDS_SQUEEZE_PERIOD = 100
ADX_PERIOD = 14
ADX_THRESHOLD = 22
TREND_EMA_FAST_PERIOD = 20
TREND_EMA_SLOW_PERIOD = 50

# --- Advanced Backtesting Simulation Settings ---
USE_MT5_COMMISSION = True
FALLBACK_COMMISSION_PER_LOT = 0.0
MIN_SPREAD_PIPS = 0.5
MAX_SPREAD_PIPS = 2.5
MAX_SLIPPAGE_PIPS = 0.5

# --- Logging & Diagnostics ---
TRADE_JOURNAL_FILE = "trade_journal.csv"
LOG_LEVEL = "INFO"
ERROR_LOG_FILE = "error_log.txt"
OPTIMIZATION_PARAMS = {
    # Existing Parameters
    'ADX_PERIOD': range(14, 25, 4), # Test 14, 18, 22
    'TREND_EMA_FAST_PERIOD': [10, 20],
    'TREND_EMA_SLOW_PERIOD': [40, 50],
    'BBANDS_PERIOD': [20, 30],

    # --- NEW PARAMETERS TO TEST ---
    # RSI levels for the Ranging strategy filter
    'RANGE_RSI_OVERSOLD': [25, 30],
    'RANGE_RSI_OVERBOUGHT': [70, 75],

    # MACD Signal period for the Trending strategy filter
    'TREND_MACD_SIGNAL_PERIOD': [9, 12],
}
"""/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           config.py
 *
 * PURPOSE:
 *
 * This file serves as the central configuration hub for the entire
 * trading bot. It is designed to be the single point of modification
 * for all user-adjustable parameters, from broker credentials to
 * complex strategy settings. By externalizing these variables from the
 * core application logic, we enable seamless backtesting, optimization,
 * and strategy adjustments without touching the production code. This
 * promotes stability, maintainability, and a clear separation of
 * concerns, which is paramount in a mission-critical financial
 * application.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             4.0
 *
 ******************************************************************************/"""

# -----------------------------------------------------------------------------
# --- Broker Credentials & Connection Settings
# -----------------------------------------------------------------------------
# These settings are essential for establishing a connection to the
# MetaTrader 5 trading terminal.
#
# IMPORTANT: For security best practices, it is highly recommended to use
# environment variables or a secure vault solution (like HashiCorp Vault)
# to store sensitive credentials in a production environment, rather than
# hardcoding them directly in this file.
# -----------------------------------------------------------------------------

# Your MetaTrader 5 account number.
MT5_ACCOUNT_NUMBER = 5038274604

# The password for your MT5 account.
MT5_PASSWORD = "G@5iMvHm"

# The server name of your broker (e.g., 'MetaQuotes-Demo').
MT5_SERVER_NAME = "MetaQuotes-Demo"

# The full file path to the MT5 terminal executable (terminal64.exe).
# This is required for the Python `MetaTrader5` library to initialize.
# Example for Windows: r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# The number of times the bot will attempt to reconnect to the MT5 server
# if the initial connection or a subsequent check fails.
CONNECTION_RETRY_ATTEMPTS = 5

# The delay in seconds between each connection retry attempt. This prevents
# overwhelming the server with rapid, successive connection requests.
CONNECTION_RETRY_DELAY_SECONDS = 10


# -----------------------------------------------------------------------------
# --- Core Trading Parameters
# -----------------------------------------------------------------------------
# These parameters define the fundamental operational scope of the bot,
# such as which instruments to trade and the timeframes to analyze.
# -----------------------------------------------------------------------------

# A list of currency pair strings that the bot is authorized to trade.
# The bot will iterate through this list in its main loop.
# Example: ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD", "NZDUSD"]
SYMBOLS_TO_TRADE = [
    "EURUSD",
    "GBPUSD",
    "AUDUSD",
    "USDJPY"
]

# A dictionary defining the specific timeframes used for the Multi-Timeframe
# Analysis (MTA). This structured approach allows the bot to make decisions
# based on a comprehensive market view.
#   - STRUCTURAL: The highest timeframe, used to determine the long-term
#                 market structure and dominant trend.
#   - POSITIONAL: The intermediate timeframe, used for identifying the current
#                 market regime (e.g., trending, ranging) and trade setups.
#   - EXECUTION:  The lowest timeframe, used for fine-tuning entry and exit
#                 points and confirming signals with candlestick patterns.
TIMEFRAMES = {
    'STRUCTURAL': 'D1',      # Daily
    'POSITIONAL': 'H4',      # 4-Hour
    'EXECUTION':  'H1'       # 1-Hour
}


# -----------------------------------------------------------------------------
# --- Global Risk Management Parameters
# -----------------------------------------------------------------------------
# This section is the cornerstone of the bot's capital preservation directive.
# These parameters enforce strict, non-negotiable risk limits on all trading
# activities, ensuring that no single trade or series of trades can
# catastrophically impact the account equity.
# -----------------------------------------------------------------------------

# The maximum percentage of the total account equity to be risked on any
# single trade. This is the most critical risk parameter.
# A value of 1.0 means the bot will risk 1% of the account equity.
GLOBAL_RISK_PER_TRADE_PERCENTAGE = 1.0

# The maximum number of trades the bot is allowed to have open simultaneously
# across all symbols. This acts as a global cap on total market exposure.
MAX_CONCURRENT_TRADES = 3

# The minimum acceptable risk-to-reward (R:R) ratio required for a trade
# setup to be considered valid. A value of 2.0 means the potential profit
# of a trade must be at least twice the potential loss.
MINIMUM_RR_RATIO = 2.0

# A multiplier for the Average True Range (ATR) indicator, used to calculate
# the Stop Loss distance. A larger value places the SL further from the entry
# price, giving the trade more room to move in volatile conditions, but it
# also results in a smaller position size to maintain the same risk percentage.
ATR_SL_MULTIPLIER = 1.5


# -----------------------------------------------------------------------------
# --- Strategy-Specific Parameters
# -----------------------------------------------------------------------------
# These parameters allow for the fine-tuning of the individual trading
# strategies that the bot can deploy. This modularity enables the operator
# to optimize or disable specific strategies based on performance analysis.
# -----------------------------------------------------------------------------

# A list of strings containing the names of the strategies that are currently
# active. The bot will only evaluate signals from strategies in this list.
# Available options: "TREND", "RANGE", "BREAKOUT"
ACTIVE_STRATEGIES = ["TREND", "RANGE"]

# The period (number of candles) for the primary moving average used in the
# trend-following strategy to determine the trend direction.
TREND_STRATEGY_MA_PERIOD = 50

# The RSI level below which the market is considered "oversold" in the
# range-trading strategy.
RANGE_STRATEGY_RSI_OVERSOLD = 30

# The RSI level above which the market is considered "overbought" in the
# range-trading strategy.
RANGE_STRATEGY_RSI_OVERBOUGHT = 70

# A factor used in the breakout strategy. To be considered a valid breakout,
# the ATR of the breakout candle must be this much larger than the average
# ATR of the preceding consolidation period.
BREAKOUT_STRATEGY_ATR_SPIKE_FACTOR = 1.5


# -----------------------------------------------------------------------------
# --- Trade Management Parameters
# -----------------------------------------------------------------------------
# These settings control how the bot manages trades after they have been
# opened. This includes advanced features like moving the stop loss to
# breakeven or trailing the stop loss to lock in profits.
# -----------------------------------------------------------------------------

# If True, the bot will automatically move the Stop Loss to the entry price
# once the trade reaches a certain profit level, eliminating the risk of loss.
ENABLE_BREAKEVEN_STOP = True

# The risk-to-reward multiple at which the breakeven stop is triggered.
# A value of 1.0 means the stop will be moved to breakeven when the trade
# is in profit by an amount equal to the initial risk (the distance to the SL).
BREAKEVEN_TRIGGER_RR = 1.0

# If True, the bot will enable a dynamic trailing stop loss after the
# breakeven point has been reached. The trailing stop will follow the price
# as it moves in a favorable direction, locking in profits.
ENABLE_TRAILING_STOP = True

# The ATR multiplier used to calculate the distance of the trailing stop from
# the current price. A smaller value will result in a tighter trail, while a

# larger value will give the price more room to fluctuate.
TRAILING_STOP_ATR_MULTIPLIER = 2.0


# -----------------------------------------------------------------------------
# --- Logging & Diagnostics
# -----------------------------------------------------------------------------
# These parameters configure the bot's logging behavior, which is crucial
# for debugging, performance analysis, and maintaining a transparent audit
# trail of all actions taken.
# -----------------------------------------------------------------------------

# The filename for the CSV file where a detailed record of every trade
# (both opened and closed) will be stored.
TRADE_JOURNAL_FILE = "trade_journal.csv"

# The verbosity level for the console and log file output.
#   - "DEBUG":   Most detailed logs, useful for development and debugging.
#   - "INFO":    Standard operational logs (e.g., connections, trades).
#   - "WARNING": Logs unexpected but non-critical events.
#   - "ERROR":   Logs only when errors occur that prevent normal operation.
LOG_LEVEL = "INFO"

# The name of the file to which error logs will be written.
ERROR_LOG_FILE = "error_log.txt"

# The frequency, in seconds, at which the main trading loop will run.
# This determines how often the bot checks for new signals and manages
# open positions.
LOOP_SLEEP_TIMER = 300 # 5 minutes
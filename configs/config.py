# MetaTrader 5 Configuration
MT5_LOGIN = 5038274604
MT5_PASSWORD = "G@5iMvHm"
MT5_SERVER = "MetaQuotes-Demo"
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Trading Configuration
SYMBOLS_TO_TRADE = ['EURUSD', 'GBPUSD', 'XAUUSD']
RISK_PER_TRADE = 1.0                    # Risk 1% per trade
MAX_POSITION_SIZE = 10.0                # Maximum 10% of account per position
MAX_DAILY_RISK = 5.0                    # Maximum 5% daily risk
MAX_PORTFOLIO_RISK = 10.0               # Maximum 10% total portfolio risk
MIN_RISK_REWARD_RATIO = 1.5             # Minimum 1.5:1 risk/reward ratio

# Position Management
MAX_OPEN_POSITIONS = 5                  # Maximum 5 open positions
MAX_CORRELATION_EXPOSURE = 0.3          # Maximum 30% correlation exposure

# Technical Analysis Settings
ATR_STOP_LOSS_MULTIPLIER = 2.0         # Stop loss at 2x ATR
ATR_TAKE_PROFIT_MULTIPLIER = 3.0       # Take profit at 3x ATR

# Strategy Settings
TREND_EMA_FAST_PERIOD = 20
TREND_EMA_SLOW_PERIOD = 50
RSI_PERIOD = 14
BBANDS_PERIOD = 20
BBANDS_STD = 2.0
RANGE_RSI_OVERBOUGHT = 70
RANGE_RSI_OVERSOLD = 30

# Market Intelligence Settings
TREND_THRESHOLD = 0.7                   # Trend strength threshold
VOLATILITY_THRESHOLD = 0.02             # Volatility regime threshold
MOMENTUM_THRESHOLD = 0.5                # Momentum regime threshold

# Analysis Periods
TREND_ANALYSIS_PERIOD = 50              # Periods for trend analysis
VOLATILITY_ANALYSIS_PERIOD = 20         # Periods for volatility analysis
MOMENTUM_ANALYSIS_PERIOD = 14           # Periods for momentum analysis

# Data Handler Settings
MAX_BARS_PER_REQUEST = 5000             # Maximum bars per data request
DATA_TIMEOUT = 30                       # Data request timeout in seconds

# Execution Manager Settings
MAGIC_NUMBER = 123456789                # Unique identifier for this bot's trades
MAX_SLIPPAGE = 3                        # Maximum allowed slippage in points
EXECUTION_TIMEOUT = 30                  # Order execution timeout in seconds
MAX_CONNECTION_ATTEMPTS = 3             # Maximum MT5 connection attempts

# Bot Control Settings
MIN_ANALYSIS_INTERVAL = 30              # Minimum seconds between analysis
MAIN_LOOP_INTERVAL = 10                 # Main loop sleep interval
CLOSE_POSITIONS_ON_SHUTDOWN = False     # Whether to close all positions on shutdown

# Notification Settings
EMAIL_NOTIFICATIONS_ENABLED = True
SLACK_NOTIFICATIONS_ENABLED = True
DISCORD_NOTIFICATIONS_ENABLED = False
TELEGRAM_NOTIFICATIONS_ENABLED = False
CONSOLE_NOTIFICATIONS_ENABLED = True

# configs/config.py - Complete MT5 Trading Bot Configuration
"""
Complete configuration file for Enhanced Forex Trading Bot
All required parameters for MT5 connection and RL training
"""

# ===== REQUIRED MT5 CONNECTION PARAMETERS =====
# These are MANDATORY for the system to work

# MetaTrader 5 Account Credentials
MT5_LOGIN = 5038800515  # Replace with your actual MT5 account number
MT5_PASSWORD = "@6JhQaZo"  # Replace with your actual MT5 password
MT5_SERVER = "MetaQuotes-Demo"  # Replace with your actual MT5 server name

# MetaTrader 5 Installation Path
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"  # Standard installation path
MT5_TIMEOUT = 10000  # Connection timeout in milliseconds

# ===== TRADING SYMBOLS CONFIGURATION =====
# Primary symbols for trading and analysis
SYMBOLS_TO_TRADE = [
    'EURUSD',    # Euro vs US Dollar
    'GBPUSD',    # British Pound vs US Dollar  
    'USDJPY',    # US Dollar vs Japanese Yen
    'AUDUSD',    # Australian Dollar vs US Dollar
    'USDCAD',    # US Dollar vs Canadian Dollar
    'USDCHF',    # US Dollar vs Swiss Franc
    'XAUUSD',    # Gold vs US Dollar
    'XAGUSD',    # Silver vs US Dollar
]

# ===== TRADING PARAMETERS =====
PRIMARY_TIMEFRAME = 'H1'  # Primary timeframe for analysis
RISK_PERCENTAGE = 0.02    # Risk 2% per trade
MAX_SPREAD = 3           # Maximum spread in pips
SLIPPAGE_PIPS = 1        # Allowed slippage in pips
MAGIC_NUMBER = 123456789 # Unique identifier for bot trades

# ===== REINFORCEMENT LEARNING CONFIGURATION =====
# RL Training Parameters
RL_TRAINING_ENABLED = True
RL_SYMBOLS = ['EURUSD', 'GBPUSD', 'XAUUSD']  # Symbols for RL training
RL_TIMEFRAME = 'H1'
RL_LOOKBACK_PERIOD = 1000  # Number of historical bars for training

# RL Model Configuration
RL_MODEL_CONFIG = {
    'algorithm': 'SAC',  # SAC, PPO, or A2C
    'learning_rate': 3e-4,
    'batch_size': 256,
    'buffer_size': 50000,
    'training_steps': 100000,
    'evaluation_frequency': 5000,
    'save_frequency': 10000
}

# RL Environment Settings
RL_ENV_CONFIG = {
    'initial_balance': 100000.0,
    'transaction_cost': 0.0001,  # 0.01%
    'max_position_size': 1.0,
    'leverage': 1.0,
    'max_drawdown_limit': 0.3,  # 30%
    'reward_scaling': 1.0
}

# ===== ADVANCED FEATURES CONFIGURATION =====
# Alternative Data Settings
ENABLE_SENTIMENT_ANALYSIS = True
ENABLE_NEWS_MONITORING = True  
ENABLE_ECONOMIC_CALENDAR = True

# NewsAPI Configuration (optional - get free key from newsapi.org)
NEWSAPI_KEY = '30d51dc0e59e4264b8baddf4173e2d06'  # Add your NewsAPI key here if available

# News Sources for Sentiment Analysis
NEWS_SOURCES = [
    'bloomberg', 'reuters', 'cnbc', 'financial-times',
    'wall-street-journal', 'marketwatch'
]

# Sentiment Analysis Thresholds
SENTIMENT_THRESHOLD_HIGH = 0.3
SENTIMENT_THRESHOLD_LOW = -0.3

# ===== TECHNICAL ANALYSIS CONFIGURATION =====
# Moving Average Periods
MA_PERIODS = [5, 10, 20, 50, 100, 200]
EMA_PERIODS = [8, 12, 21, 26, 50]

# Oscillator Settings
RSI_PERIODS = [7, 14, 21]
STOCH_PERIODS = [(14, 3, 3), (5, 3, 3)]
MACD_SETTINGS = [(12, 26, 9), (5, 35, 5)]

# Bollinger Bands Settings
BB_PERIODS = [20, 50]
BB_DEVIATIONS = [2.0, 2.5]

# ===== RISK MANAGEMENT CONFIGURATION =====
# Position Sizing
MAX_CONCURRENT_TRADES = 5
MAX_DAILY_TRADES = 20
MAX_WEEKLY_TRADES = 50

# Risk Controls
MAX_DAILY_LOSS = 0.05      # 5% max daily loss
MAX_WEEKLY_LOSS = 0.15     # 15% max weekly loss
MAX_MONTHLY_LOSS = 0.25    # 25% max monthly loss

# Stop Loss and Take Profit
DEFAULT_STOP_LOSS = 50     # pips
DEFAULT_TAKE_PROFIT = 100  # pips
TRAILING_STOP = True
TRAILING_STOP_DISTANCE = 30  # pips

# ===== TRADING HOURS CONFIGURATION =====
# Trading session control
ENABLE_NEWS_FILTER = True
TRADING_HOURS = {
    'start': 0,    # 00:00 UTC
    'end': 24      # 24:00 UTC (always on)
}

# Session-specific settings
ASIAN_SESSION = {'start': 0, 'end': 9}      # 00:00-09:00 UTC
EUROPEAN_SESSION = {'start': 8, 'end': 17}  # 08:00-17:00 UTC  
AMERICAN_SESSION = {'start': 13, 'end': 22} # 13:00-22:00 UTC

# ===== DATA AND CACHING CONFIGURATION =====
# Data Management
DATA_CACHE_HOURS = 24      # Cache data for 24 hours
MAX_CACHE_ITEMS = 1000     # Maximum cached items
AUTO_CLEANUP_CACHE = True  # Automatic cache cleanup

# Historical Data Settings
HISTORICAL_DATA_DAYS = 365  # Days of historical data to fetch
MIN_DATA_POINTS = 1000      # Minimum data points for analysis

# ===== LOGGING AND MONITORING =====
# Logging Configuration
LOG_LEVEL = 'INFO'         # DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = True
LOG_TO_CONSOLE = True
LOG_ROTATION = True
MAX_LOG_SIZE_MB = 50

# Performance Monitoring  
ENABLE_PERFORMANCE_MONITORING = True
PERFORMANCE_LOG_INTERVAL = 3600  # seconds
MEMORY_THRESHOLD_MB = 1000

# ===== NOTIFICATION CONFIGURATION =====
# Alert Settings
ENABLE_ALERTS = True
ENABLE_EMAIL_ALERTS = False
ENABLE_TELEGRAM_ALERTS = False

# Email Configuration (if enabled)
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587
EMAIL_USERNAME = ""        # Your email
EMAIL_PASSWORD = ""        # Your email password
EMAIL_RECIPIENTS = []      # List of recipient emails

# Telegram Configuration (if enabled)
TELEGRAM_BOT_TOKEN = ""    # Your Telegram bot token
TELEGRAM_CHAT_ID = ""      # Your Telegram chat ID

# ===== PATTERN RECOGNITION CONFIGURATION =====
# Chart Pattern Settings
ENABLE_PATTERN_RECOGNITION = True
PATTERN_SENSITIVITY = 0.7   # Pattern matching sensitivity
MIN_PATTERN_BARS = 10      # Minimum bars for pattern recognition

# Market Regime Detection
REGIME_WINDOW = 50         # Window for regime detection
VOLATILITY_THRESHOLD = 0.02 # Volatility threshold for regime classification

# ===== MACHINE LEARNING FEATURES =====
# Feature Engineering
ENABLE_ADVANCED_FEATURES = True
ENABLE_ML_FEATURES = True
ENABLE_STATISTICAL_FEATURES = True

# Feature Selection
MAX_FEATURES = 200         # Maximum number of features
FEATURE_SELECTION_METHOD = 'variance'  # variance, correlation, mutual_info

# ===== BACKTESTING CONFIGURATION =====
# Backtesting Settings
BACKTEST_START_DATE = '2021-01-01'
BACKTEST_END_DATE = '2024-01-01'
BACKTEST_INITIAL_BALANCE = 10000.0
BACKTEST_COMMISSION = 0.0001  # 0.01%

# Walk-Forward Analysis
ENABLE_WALK_FORWARD = True
WALK_FORWARD_PERIODS = 12   # Number of periods
WALK_FORWARD_STEP = 30      # Days per step

# ===== OPTIMIZATION CONFIGURATION =====
# Hyperparameter Optimization
ENABLE_OPTIMIZATION = True
OPTIMIZATION_TRIALS = 100   # Number of optimization trials
OPTIMIZATION_TIMEOUT = 7200 # Optimization timeout in seconds

# Optuna Settings
OPTUNA_STORAGE = None      # Database URL for distributed optimization
OPTUNA_STUDY_NAME = "forex_bot_optimization"

# ===== DEVELOPMENT AND TESTING =====
# Development Mode
DEVELOPMENT_MODE = True    # Enable development features
PAPER_TRADING = True       # Use paper trading for testing
SAVE_DEBUG_DATA = True     # Save debug information

# Testing Configuration
RUN_UNIT_TESTS = True
RUN_INTEGRATION_TESTS = True
TEST_DATA_PATH = "data/test/"

# ===== SYSTEM CONFIGURATION =====
# Performance Settings
MAX_CPU_CORES = 4          # Maximum CPU cores to use
MAX_MEMORY_GB = 8          # Maximum memory usage in GB
ENABLE_GPU = False         # Enable GPU acceleration if available

# File Paths
DATA_PATH = "data/"
MODELS_PATH = "models/"
LOGS_PATH = "logs/"
RESULTS_PATH = "results/"
CACHE_PATH = "cache/"

# ===== VALIDATION AND SAFETY =====
# Input Validation
VALIDATE_INPUTS = True
STRICT_VALIDATION = True

# Safety Checks
ENABLE_SAFETY_CHECKS = True
MAX_POSITION_VALUE = 50000  # Maximum position value in account currency
EMERGENCY_STOP_LOSS = 0.10  # 10% emergency stop loss

# ===== EXPORT CONFIGURATION FOR VERIFICATION =====
def get_config_summary():
    """Return configuration summary for verification"""
    return {
        'mt5_login': MT5_LOGIN,
        'mt5_server': MT5_SERVER,
        'symbols_count': len(SYMBOLS_TO_TRADE),
        'symbols': SYMBOLS_TO_TRADE,
        'rl_enabled': RL_TRAINING_ENABLED,
        'sentiment_enabled': ENABLE_SENTIMENT_ANALYSIS,
        'risk_percentage': RISK_PERCENTAGE,
        'max_concurrent_trades': MAX_CONCURRENT_TRADES
    }

def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    # Check required MT5 parameters
    if not MT5_LOGIN or MT5_LOGIN == 5038274604:
        errors.append("MT5_LOGIN needs to be set to your actual account number")
    
    if not MT5_PASSWORD or MT5_PASSWORD == "yourpassword":
        errors.append("MT5_PASSWORD needs to be set to your actual password")
    
    if not MT5_SERVER or MT5_SERVER == "MetaQuotes-Demo":
        errors.append("MT5_SERVER should be set to your actual server name")
    
    if not SYMBOLS_TO_TRADE:
        errors.append("SYMBOLS_TO_TRADE cannot be empty")
    
    return errors

# Print configuration status when imported
if __name__ == "__main__":
    print("✅ Forex Trading Bot Configuration Loaded")
    print(f"📊 Configuration Summary: {get_config_summary()}")
    
    validation_errors = validate_config()
    if validation_errors:
        print("⚠️ Configuration Warnings:")
        for error in validation_errors:
            print(f"   - {error}")
    else:
        print("✅ Configuration validation passed")

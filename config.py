# config.py - Complete Enhanced Trading Bot Configuration
"""
Complete configuration file for Enhanced Forex Trading Bot with RL Integration
All required parameters for MT5 connection, RL training, and advanced features
"""

import os
from datetime import datetime

# ===== REQUIRED MT5 CONNECTION PARAMETERS =====
MT5_LOGIN = 5038880271
MT5_PASSWORD = "_qWw7aGm"
MT5_SERVER = "MetaQuotes-Demo"
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_TIMEOUT = 100000

# ===== TRADING SYMBOLS CONFIGURATION =====
SYMBOLS_TO_TRADE = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 
    'USDCAD', 'USDCHF', 'XAUUSD', 'XAGUSD'
]

# ===== TRADING PARAMETERS =====
PRIMARY_TIMEFRAME = 'H1'
RISK_PERCENTAGE = 0.02
RISK_PER_TRADE = 0.02
MAX_SPREAD = 30
SLIPPAGE_PIPS = 1
MAGIC_NUMBER = 123456789

# ===== REINFORCEMENT LEARNING CONFIGURATION =====
RL_AVAILABLE = True
RL_TRAINING_ENABLED = True
RL_SYMBOLS = ['EURUSD', 'GBPUSD', 'XAUUSD']
RL_TIMEFRAME = 'H1'
RL_LOOKBACK_PERIOD = 1000

# RL Model Paths
SAC_MODEL_PATH = "models/sac_EURUSD_final.zip"
A2C_MODEL_PATH = "models/rl_EURUSD_final_fixed.zip"
BEST_SAC_PATH = "./best_sac_model_EURUSD_best_model.zip"
BEST_A2C_PATH = "./best_model_EURUSD_best_model.zip"

# RL Model Configuration
RL_MODEL_CONFIG = {
    'algorithm': 'SAC',
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
    'transaction_cost': 0.0001,
    'max_position_size': 1.0,
    'leverage': 1.0,
    'max_drawdown_limit': 0.3,
    'reward_scaling': 1.0
}

# ===== ACCOUNT CONFIGURATION =====
ACCOUNT_BALANCE = 100000
LEVERAGE = 100
MAX_RISK_PER_TRADE = 0.02
MAX_DRAWDOWN = 0.1
MAX_TOTAL_EXPOSURE = 0.1

# ===== TIMING CONFIGURATION =====
MAIN_LOOP_INTERVAL = 10
MIN_ANALYSIS_INTERVAL = 30
PERFORMANCE_UPDATE_INTERVAL = 300

# ===== ADVANCED FEATURES CONFIGURATION =====
ENABLE_SENTIMENT_ANALYSIS = True
ENABLE_NEWS_MONITORING = True
ENABLE_ECONOMIC_CALENDAR = True

# NewsAPI Configuration
NEWSAPI_KEY = '30d51dc0e59e4264b8baddf4173e2d06'
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
MAX_CONCURRENT_TRADES = 5
MAX_DAILY_TRADES = 20
MAX_WEEKLY_TRADES = 50

# Risk Controls
MAX_DAILY_LOSS = 0.05
MAX_WEEKLY_LOSS = 0.15
MAX_MONTHLY_LOSS = 0.25

# Stop Loss and Take Profit
DEFAULT_STOP_LOSS = 50
DEFAULT_TAKE_PROFIT = 100
TRAILING_STOP = True
TRAILING_STOP_DISTANCE = 30
STOP_LOSS_ATR_MULTIPLIER = 2.0
TAKE_PROFIT_ATR_MULTIPLIER = 3.0

# ===== TRADING HOURS CONFIGURATION =====
ENABLE_NEWS_FILTER = True
TRADING_HOURS = {'start': 0, 'end': 24}

# Session-specific settings
ASIAN_SESSION = {'start': 0, 'end': 9}
EUROPEAN_SESSION = {'start': 8, 'end': 17}
AMERICAN_SESSION = {'start': 13, 'end': 22}

# ===== DATA AND CACHING CONFIGURATION =====
DATA_CACHE_HOURS = 24
MAX_CACHE_ITEMS = 1000
AUTO_CLEANUP_CACHE = True
HISTORICAL_DATA_DAYS = 365
MIN_DATA_POINTS = 1000

# Data Source Configuration
DATA_SOURCE = 'MT5'
TIMEFRAMES = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1']

# ===== LOGGING AND MONITORING =====
LOG_LEVEL = 'INFO'
LOG_TO_FILE = True
LOG_TO_CONSOLE = True
LOG_ROTATION = True
MAX_LOG_SIZE_MB = 50

# Performance Monitoring
ENABLE_PERFORMANCE_MONITORING = True
PERFORMANCE_LOG_INTERVAL = 3600
MEMORY_THRESHOLD_MB = 1000

# ===== NOTIFICATION CONFIGURATION =====
NOTIFICATIONS_ENABLED = True
ENABLE_ALERTS = True
ENABLE_EMAIL_ALERTS = False
ENABLE_TELEGRAM_ALERTS = False

# Email Configuration
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587
EMAIL_USERNAME = ""
EMAIL_PASSWORD = ""
EMAIL_RECIPIENTS = []

# Telegram Configuration
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# ===== PATTERN RECOGNITION CONFIGURATION =====
ENABLE_PATTERN_RECOGNITION = True
PATTERN_SENSITIVITY = 0.7
MIN_PATTERN_BARS = 10

# Market Regime Detection
REGIME_WINDOW = 50
VOLATILITY_THRESHOLD = 0.02

# ===== MACHINE LEARNING FEATURES =====
ENABLE_ADVANCED_FEATURES = True
ENABLE_ML_FEATURES = True
ENABLE_STATISTICAL_FEATURES = True

# Feature Selection
MAX_FEATURES = 200
FEATURE_SELECTION_METHOD = 'variance'

# ===== BACKTESTING CONFIGURATION =====
BACKTEST_START_DATE = '2021-01-01'
BACKTEST_END_DATE = '2024-01-01'
BACKTEST_INITIAL_BALANCE = 100000.0
BACKTEST_COMMISSION = 0.0001

# Walk-Forward Analysis
ENABLE_WALK_FORWARD = True
WALK_FORWARD_PERIODS = 12
WALK_FORWARD_STEP = 30

# ===== OPTIMIZATION CONFIGURATION =====
ENABLE_OPTIMIZATION = True
OPTIMIZATION_TRIALS = 100
OPTIMIZATION_TIMEOUT = 7200

# Optuna Settings
OPTUNA_STORAGE = None
OPTUNA_STUDY_NAME = "forex_bot_optimization"

# ===== DEVELOPMENT AND TESTING =====
DEVELOPMENT_MODE = True
PAPER_TRADING = True
SAVE_DEBUG_DATA = True

# Testing Configuration
RUN_UNIT_TESTS = True
RUN_INTEGRATION_TESTS = True
TEST_DATA_PATH = "data/test/"

# ===== SYSTEM CONFIGURATION =====
# Performance Settings
MAX_CPU_CORES = 4
MAX_MEMORY_GB = 8
ENABLE_GPU = False

# File Paths
DATA_PATH = "data/"
MODELS_PATH = "models/"
LOGS_PATH = "logs/"
RESULTS_PATH = "results/"
CACHE_PATH = "cache/"

# System Settings
CLOSE_POSITIONS_ON_SHUTDOWN = True

# ===== VALIDATION AND SAFETY =====
VALIDATE_INPUTS = True
STRICT_VALIDATION = True

# Safety Checks
ENABLE_SAFETY_CHECKS = True
MAX_POSITION_VALUE = 50000
EMERGENCY_STOP_LOSS = 0.10

# ===== STRATEGY SPECIFIC CONFIGURATIONS =====
# Trend Following Strategy
TREND_EMA_FAST_PERIOD = 12
TREND_EMA_SLOW_PERIOD = 26
TREND_RSI_PERIOD = 14

# Mean Reversion Strategy
RANGE_RSI_OVERBOUGHT = 70
RANGE_RSI_OVERSOLD = 30
BBANDS_PERIOD = 20
BBANDS_STD = 2.0

# Divergence Strategy
DIVERGENCE_LOOKBACK = 20
DIVERGENCE_MIN_BARS = 5

# ===== CLASS-BASED CONFIGURATION =====
class Config:
    """Configuration class for easy access to all parameters"""
    
    def __init__(self):
        # Import all module-level variables
        import sys
        current_module = sys.modules[__name__]
        
        for name in dir(current_module):
            if not name.startswith('_') and not callable(getattr(current_module, name)):
                setattr(self, name, getattr(current_module, name))
    
    def get_config_summary(self):
        """Return configuration summary for verification"""
        return {
            'mt5_login': self.MT5_LOGIN,
            'mt5_server': self.MT5_SERVER,
            'symbols_count': len(self.SYMBOLS_TO_TRADE),
            'symbols': self.SYMBOLS_TO_TRADE,
            'rl_enabled': self.RL_TRAINING_ENABLED,
            'sentiment_enabled': self.ENABLE_SENTIMENT_ANALYSIS,
            'risk_percentage': self.RISK_PERCENTAGE,
            'max_concurrent_trades': self.MAX_CONCURRENT_TRADES
        }
    
    def validate_config(self):
        """Validate configuration parameters"""
        errors = []
        
        # Check required MT5 parameters
        if not self.MT5_LOGIN:
            errors.append("5038274604")
        
        if not self.MT5_PASSWORD:
            errors.append("G@5iMvHm")
        
        if not self.MT5_SERVER:
            errors.append("MetaQuotes-Demo")
        
        if not self.SYMBOLS_TO_TRADE:
            errors.append("EURUSD', 'GBPUSD', 'XAUUSD")
        
        return errors
    
    def __getattr__(self, name):
        """Fallback for missing attributes"""
        defaults = {
            'LOG_LEVEL': 'INFO',
            'ENABLE_LIVE_TRADING': False,
            'MAX_SLIPPAGE': 3
        }
        return defaults.get(name, None)

# Create global config instance
config = Config()

# ===== UTILITY FUNCTIONS =====
def get_config_summary():
    """Return configuration summary for verification"""
    return config.get_config_summary()

def validate_config():
    """Validate configuration parameters"""
    return config.validate_config()

def print_config_status():
    """Print configuration status"""
    print("✅ Enhanced Forex Trading Bot Configuration Loaded")
    print(f"📊 Configuration Summary: {get_config_summary()}")
    
    validation_errors = validate_config()
    if validation_errors:
        print("⚠️ Configuration Warnings:")
        for error in validation_errors:
            print(f"   - {error}")
    else:
        print("✅ Configuration validation passed")

# Print configuration status when imported
if __name__ == "__main__":
    print_config_status()

# Enhanced Sentiment Configuration
SENTIMENT_ENABLED = True
SENTIMENT_UPDATE_INTERVAL = 300
SENTIMENT_THRESHOLD = 0.1
SENTIMENT_HIDDEN_DIM = 768
SENTIMENT_NUM_HEADS = 12
SENTIMENT_DROPOUT = 0.1
NEWS_SOURCES = []
SOCIAL_SOURCES = []

# Strategy Manager Configuration
TREND_EMA_FAST_PERIOD = 12
TREND_EMA_SLOW_PERIOD = 26
RSI_PERIOD = 14
BBANDS_PERIOD = 20
BBANDS_STD = 2.0
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RANGE_RSI_OVERBOUGHT = 75
RANGE_RSI_OVERSOLD = 25

"""
Enhanced Trading Bot Configuration with XAUUSD Integration
Complete configuration for professional trading system
"""

import os
from datetime import datetime
from config import config  # ✅ Import the new configuration

# Use the config throughout your main file
symbols = config.get('trading.symbols')  # ['EURUSD', 'GBPUSD', 'XAUUSD']

class TradingConfig:
    """
    Comprehensive trading configuration with XAUUSD optimization
    """
    
    def __init__(self):
        self.config = {
            # ✅ UPDATED: Core Trading Configuration with XAUUSD
            'trading': {
                'symbols': ['EURUSD', 'GBPUSD', 'XAUUSD'],  # ✅ REPLACED USDJPY with XAUUSD
                'risk_per_trade': 0.01,  # 1% risk per trade
                'max_daily_risk': 0.05,  # 5% maximum daily risk
                'max_positions_per_symbol': 1,  # One position per symbol
                'max_total_positions': 3,  # Maximum total open positions
                'enable_traditional_signals': True,
                'enable_rl_signals': True,
                'min_confidence_threshold': 0.6,  # Minimum confidence for signal execution
                'risk_reward_ratio': 3,  # Target 1:1.5 risk/reward
                'max_spread_pips': {  # Maximum spread limits per symbol
                    'EURUSD': 10,
                    'GBPUSD': 10,
                    'XAUUSD': 50  # ✅ Gold-specific spread limit
                }
            },
            
            # ✅ UPDATED: MetaTrader 5 Configuration with XAUUSD optimization
            'mt5': {
                'magic_number': 123456789,
                'max_slippage': 30,  # Default slippage in points
                'max_connection_attempts': 10,
                'connection_timeout': 30,
                'symbol_specific_slippage': {  # ✅ Symbol-specific slippage settings
                    'EURUSD': 10,
                    'GBPUSD': 10,
                    'XAUUSD': 30,  # ✅ Optimized for Gold
                    'USDJPY': 5,  # Legacy (not used)
                    'GBPJPY': 4,
                    'EURJPY': 4,
                    'AUDJPY': 4
                }
            },
            
            # ✅ ENHANCED: Symbol-Specific Parameters
            'symbol_params': {
                'EURUSD': {
                    'min_volume': 0.01,
                    'max_volume': 10.0,
                    'volume_step': 0.01,
                    'typical_spread': 1.5,
                    'trading_sessions': ['london', 'newyork'],
                    'volatility_adjustment': 1.0
                },
                'GBPUSD': {
                    'min_volume': 0.01,
                    'max_volume': 10.0,
                    'volume_step': 0.01,
                    'typical_spread': 2.0,
                    'trading_sessions': ['london', 'newyork'],
                    'volatility_adjustment': 1.1
                },
                'XAUUSD': {  # ✅ NEW: Gold-specific parameters
                    'min_volume': 0.01,
                    'max_volume': 5.0,  # Lower max volume for Gold
                    'volume_step': 0.01,
                    'typical_spread': 3.0,
                    'trading_sessions': ['london', 'newyork', 'asian'],  # Gold trades 24/5
                    'volatility_adjustment': 1.2,  # Higher volatility adjustment
                    'point_value': 100,  # Gold point value
                    'contract_size': 100  # Gold contract size
                }
            },
            
            # Execution Configuration
            'execution': {
                'enable_partial_fills': True,
                'max_retry_attempts': 10,
                'retry_delay_seconds': 1,
                'execution_timeout': 30,
                'enable_slippage_protection': True,
                'max_execution_delay': 5.0  # Maximum execution delay in seconds
            },
            
            # ✅ ENHANCED: Risk Management with XAUUSD considerations
            'risk_management': {
                'use_dynamic_position_sizing': True,
                'max_risk_per_trade': 0.01,  # 1%
                'max_daily_risk': 0.05,  # 5%
                'max_weekly_risk': 0.15,  # 15%
                'stop_loss_multiplier': 2.0,
                'take_profit_multiplier': 3.0,
                'trailing_stop_enabled': False,
                'correlation_limit': 0.7,  # Maximum correlation between positions
                'symbol_risk_weights': {  # Risk weighting per symbol
                    'EURUSD': 1.0,
                    'GBPUSD': 1.1,
                    'XAUUSD': 1.3  # ✅ Higher risk weight for Gold (more volatile)
                }
            },
            
            # RL Model Configuration
            'rl_model': {
                'model_type': 'A2C',
                'model_path': './best_model_EURUSD/best_model.zip',
                'observation_size': 32,
                'enable_model_validation': True,
                'prediction_confidence_threshold': 0.6,
                'model_update_interval': 24  # Hours
            },
            
            # Market Intelligence Configuration
            'market_intelligence': {
                'trend_threshold': 0.7,
                'volatility_threshold': 0.02,
                'momentum_threshold': 0.5,
                'trend_analysis_period': 50,
                'volatility_analysis_period': 20,
                'momentum_analysis_period': 14,
                'enable_regime_detection': True,
                'regime_smoothing_factor': 3
            },
            
            # Strategy Configuration
            'strategy': {
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'bb_threshold': 0.95,
                'macd_signal_threshold': 0.001,
                'adx_trend_threshold': 25,
                'enable_confluence_trading': True,
                'min_signal_confluence': 2  # Minimum number of confirming indicators
            },
            
            # Data Handling Configuration
            'data_handler': {
                'enable_data_caching': True,
                'cache_size_limit': 1000,
                'data_validation_enabled': True,
                'missing_data_tolerance': 0.05,  # 5% missing data tolerance
                'timeframe_priorities': ['M1', 'M5', 'M15', 'H1']
            },
            
            # Notification Configuration
            'notifications': {
                'enable_console_notifications': True,
                'enable_email_notifications': False,
                'enable_telegram_notifications': False,
                'notification_levels': ['ERROR', 'WARNING', 'INFO'],
                'trade_notification_enabled': True,
                'performance_notification_interval': 3600  # Seconds
            },
            
            # Performance Monitoring
            'performance_monitor': {
                'enable_monitoring': True,
                'update_interval_seconds': 300,
                'retention_days': 30,
                'enable_trade_analytics': True,
                'enable_drawdown_monitoring': True,
                'max_drawdown_threshold': 0.1  # 10%
            },
            
            # Logging Configuration
            'logging': {
                'log_level': 'INFO',
                'enable_file_logging': True,
                'log_file_path': 'logs/trading_bot.log',
                'max_log_file_size': 10485760,  # 10MB
                'backup_count': 5,
                'enable_trade_logging': True,
                'enable_performance_logging': True
            },
            
            # ✅ NEW: Session-Based Trading Configuration
            'trading_sessions': {
                'asian': {
                    'start_hour': 0,
                    'end_hour': 9,
                    'timezone': 'Asia/Tokyo',
                    'enabled_symbols': ['XAUUSD'], #'USDJPY']  # Gold and JPY pairs active
                },
                'london': {
                    'start_hour': 8,
                    'end_hour': 17,
                    'timezone': 'Europe/London',
                    'enabled_symbols': ['EURUSD', 'GBPUSD', 'XAUUSD']  # All symbols active
                },
                'newyork': {
                    'start_hour': 13,
                    'end_hour': 22,
                    'timezone': 'America/New_York',
                    'enabled_symbols': ['EURUSD', 'GBPUSD', 'XAUUSD']  # All symbols active
                }
            },
            
            # Emergency and Safety Configuration
            'emergency': {
                'enable_emergency_stop': True,
                'max_consecutive_losses': 5,
                'emergency_close_threshold': 0.05,  # 5% account loss
                'enable_weekend_close': True,
                'market_hours_check': True
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation (e.g., 'trading.symbols')"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, key: str, value):
        """Update configuration value using dot notation"""
        keys = key.split('.')
        config_section = self.config
        
        for k in keys[:-1]:
            if k not in config_section:
                config_section[k] = {}
            config_section = config_section[k]
        
        config_section[keys[-1]] = value
    
    def get_symbol_config(self, symbol: str) -> dict:
        """Get symbol-specific configuration"""
        return self.config.get('symbol_params', {}).get(symbol, {})
    
    def get_slippage_for_symbol(self, symbol: str) -> int:
        """Get symbol-specific slippage setting"""
        return self.config.get('mt5', {}).get('symbol_specific_slippage', {}).get(
            symbol, self.config.get('mt5', {}).get('max_slippage', 3)
        )
    
    def is_symbol_tradeable(self, symbol: str, session: str = None) -> bool:
        """Check if symbol is tradeable in current/specified session"""
        if symbol not in self.config.get('trading', {}).get('symbols', []):
            return False
        
        if session:
            session_config = self.config.get('trading_sessions', {}).get(session, {})
            enabled_symbols = session_config.get('enabled_symbols', [])
            return symbol in enabled_symbols
        
        return True

# Create global configuration instance
config = TradingConfig()

# ✅ VALIDATION: Ensure XAUUSD is properly configured
if 'XAUUSD' not in config.get('trading.symbols', []):
    print("WARNING: XAUUSD not found in trading symbols!")
else:
    print("✅ XAUUSD successfully configured in trading symbols")

# Export for easy importing
__all__ = ['config', 'TradingConfig']

# config.py - Complete MT5 Configuration
"""
Complete MT5 Configuration for Forex Trading System
"""

class Config:
    def __init__(self):
        # MetaTrader 5 Configuration
        self.MT5_LOGIN = 5038274604  # Replace with your MT5 login
        self.MT5_PASSWORD = 'G@5iMvHm'  # Replace with your MT5 password
        self.MT5_SERVER = 'MetaQuotes-Demo'  # Replace with your broker's server
        self.MT5_PATH = r'C:\Program Files\MetaTrader 5\terminal64.exe'
        
        # Data Handler Settings
        self.MAX_BARS_PER_REQUEST = 10000
        self.DATA_TIMEOUT = 30
        
        # Trading Configuration
        self.SYMBOLS_TO_TRADE = ['EURUSD', 'GBPUSD', 'XAUUSD']
        self.RISK_PER_TRADE = 0.01  # 1% risk per trade
        self.MAX_POSITION_SIZE = 0.10  # Maximum 10% of account per position
        
        # Strategy Settings
        self.TREND_EMA_FAST_PERIOD = 20
        self.TREND_EMA_SLOW_PERIOD = 50
        self.RSI_PERIOD = 14
        self.RSI_OVERBOUGHT = 70
        self.RSI_OVERSOLD = 30

config = Config()

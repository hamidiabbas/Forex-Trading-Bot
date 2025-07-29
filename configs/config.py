"""
Complete Trading Bot Configuration with XAUUSD Requote Protection
Optimized for Gold trading and requote elimination
"""

class TradingConfig:
    """
    Enhanced trading configuration with XAUUSD requote protection
    """
    
    def __init__(self):
        self.config = {
            # ✅ ENHANCED: Core Trading Configuration with XAUUSD optimization
            'trading': {
                'symbols': ['EURUSD', 'GBPUSD', 'XAUUSD'],  # ✅ FIXED: Gold instead of USDJPY
                'risk_per_trade': 0.01,  # 1% risk per trade
                'max_daily_risk': 0.05,  # 5% maximum daily risk
                'max_positions_per_symbol': 1,  # One position per symbol
                'max_total_positions': 3,  # Maximum total open positions
                'enable_traditional_signals': True,
                'enable_rl_signals': True,
                'min_confidence_threshold': 0.6,  # Minimum confidence for signal execution
                'risk_reward_ratio': 1.5,  # Target 1:1.5 risk/reward
                'max_spread_pips': {  # Maximum spread limits per symbol
                    'EURUSD': 3,
                    'GBPUSD': 4,
                    'XAUUSD': 8  # ✅ Higher spread tolerance for Gold
                },
                # ✅ NEW: Symbol trading sessions
                'symbol_sessions': {
                    'EURUSD': ['london', 'newyork'],
                    'GBPUSD': ['london', 'newyork'],
                    'XAUUSD': ['asian', 'london', 'newyork']  # Gold trades 24/5
                }
            },
            
            # ✅ ENHANCED: MetaTrader 5 Configuration with XAUUSD requote protection
            'mt5': {
                'magic_number': 123456789,
                'max_slippage': 3,  # Default slippage in points
                'max_connection_attempts': 3,
                'connection_timeout': 30,
                # ✅ CRITICAL: Symbol-specific execution parameters for requote protection
                'symbol_execution': {
                    'XAUUSD': {
                        'max_slippage': 12,  # Much higher initial slippage for Gold
                        'max_retry_attempts': 8,  # More retries for Gold
                        'retry_delay': 0.3,  # Faster retries (300ms)
                        'max_slippage_escalation': 25,  # Can escalate up to 25 points
                        'requote_escalation_aggressive': True  # Aggressive escalation on requotes
                    },
                    'USDJPY': {  # Legacy support (not actively used)
                        'max_slippage': 8,
                        'max_retry_attempts': 6,
                        'retry_delay': 0.5,
                        'max_slippage_escalation': 15,
                        'requote_escalation_aggressive': True
                    },
                    'GBPJPY': {
                        'max_slippage': 6,
                        'max_retry_attempts': 5,
                        'retry_delay': 0.5,
                        'max_slippage_escalation': 12,
                        'requote_escalation_aggressive': False
                    },
                    'EURJPY': {
                        'max_slippage': 6,
                        'max_retry_attempts': 5,
                        'retry_delay': 0.5,
                        'max_slippage_escalation': 12,
                        'requote_escalation_aggressive': False
                    },
                    'EURUSD': {
                        'max_slippage': 3,
                        'max_retry_attempts': 3,
                        'retry_delay': 1.0,
                        'max_slippage_escalation': 8,
                        'requote_escalation_aggressive': False
                    },
                    'GBPUSD': {
                        'max_slippage': 3,
                        'max_retry_attempts': 3,
                        'retry_delay': 1.0,
                        'max_slippage_escalation': 8,
                        'requote_escalation_aggressive': False
                    }
                }
            },
            
            # ✅ ENHANCED: Symbol-Specific Parameters with XAUUSD focus
            'symbol_params': {
                'EURUSD': {
                    'min_volume': 0.01,
                    'max_volume': 10.0,
                    'volume_step': 0.01,
                    'typical_spread': 1.5,
                    'volatility_adjustment': 1.0,
                    'point_value': 10,  # $10 per pip for standard lot
                    'contract_size': 100000
                },
                'GBPUSD': {
                    'min_volume': 0.01,
                    'max_volume': 10.0,
                    'volume_step': 0.01,
                    'typical_spread': 2.0,
                    'volatility_adjustment': 1.1,
                    'point_value': 10,  # $10 per pip for standard lot
                    'contract_size': 100000
                },
                'XAUUSD': {  # ✅ ENHANCED: Gold-specific parameters for requote protection
                    'min_volume': 0.01,
                    'max_volume': 5.0,  # Lower max volume for Gold (more volatile)
                    'volume_step': 0.01,
                    'typical_spread': 4.0,  # Wider typical spread
                    'volatility_adjustment': 1.5,  # Higher volatility adjustment
                    'point_value': 100,  # $100 per point for standard lot (100oz)
                    'contract_size': 100,  # 100oz contracts
                    'tick_size': 0.01,  # Minimum price movement
                    'tick_value': 1.0,  # Value of minimum price movement
                    'execution_risk_level': 'HIGH',  # High execution risk
                    'preferred_sessions': ['london', 'newyork'],  # Best execution sessions
                    'avoid_news_minutes': 30  # Avoid trading 30min before/after major news
                }
            },
            
            # ✅ ENHANCED: Execution Configuration with requote protection
            'execution': {
                'enable_partial_fills': True,
                'execution_timeout': 30,
                'enable_slippage_protection': True,
                'max_execution_delay': 5.0,  # Maximum execution delay in seconds
                'enable_requote_protection': True,  # ✅ NEW: Enable requote protection
                'requote_retry_fast_mode': True,  # ✅ NEW: Fast retry mode for requotes
                'log_execution_details': True,  # ✅ NEW: Detailed execution logging
                # ✅ NEW: Market condition adjustments
                'market_volatility_adjustments': {
                    'high_volatility_threshold': 0.02,  # 2% daily range
                    'high_volatility_slippage_multiplier': 1.5,
                    'news_event_slippage_multiplier': 2.0
                }
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
                    'XAUUSD': 1.4  # ✅ Higher risk weight for Gold (more volatile + requote risk)
                },
                # ✅ NEW: Execution risk adjustments
                'execution_risk_adjustments': {
                    'high_execution_risk_symbols': ['XAUUSD'],
                    'high_risk_position_size_reduction': 0.8,  # Reduce position size by 20%
                    'high_risk_confidence_threshold': 0.7  # Higher confidence required
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
                'performance_notification_interval': 3600,  # Seconds
                'requote_notification_enabled': True  # ✅ NEW: Notify on excessive requotes
            },
            
            # Performance Monitoring
            'performance_monitor': {
                'enable_monitoring': True,
                'update_interval_seconds': 300,
                'retention_days': 30,
                'enable_trade_analytics': True,
                'enable_drawdown_monitoring': True,
                'max_drawdown_threshold': 0.1,  # 10%
                'enable_execution_analytics': True,  # ✅ NEW: Monitor execution performance
                'requote_monitoring_enabled': True  # ✅ NEW: Monitor requote rates
            },
            
            # Logging Configuration
            'logging': {
                'log_level': 'INFO',
                'enable_file_logging': True,
                'log_file_path': 'logs/trading_bot.log',
                'max_log_file_size': 10485760,  # 10MB
                'backup_count': 5,
                'enable_trade_logging': True,
                'enable_performance_logging': True,
                'enable_execution_logging': True  # ✅ NEW: Detailed execution logging
            },
            
            # ✅ ENHANCED: Session-Based Trading Configuration
            'trading_sessions': {
                'asian': {
                    'start_hour': 0,
                    'end_hour': 9,
                    'timezone': 'Asia/Tokyo',
                    'enabled_symbols': ['XAUUSD'],  # Gold active in Asian session
                    'execution_quality': 'MEDIUM',
                    'typical_volatility': 'MEDIUM'
                },
                'london': {
                    'start_hour': 8,
                    'end_hour': 17,
                    'timezone': 'Europe/London',
                    'enabled_symbols': ['EURUSD', 'GBPUSD', 'XAUUSD'],  # All symbols active
                    'execution_quality': 'HIGH',  # Best execution quality
                    'typical_volatility': 'HIGH'
                },
                'newyork': {
                    'start_hour': 13,
                    'end_hour': 22,
                    'timezone': 'America/New_York',
                    'enabled_symbols': ['EURUSD', 'GBPUSD', 'XAUUSD'],  # All symbols active
                    'execution_quality': 'HIGH',  # Best execution quality
                    'typical_volatility': 'HIGH'
                },
                'overlap_london_ny': {
                    'start_hour': 13,
                    'end_hour': 17,
                    'timezone': 'America/New_York',
                    'enabled_symbols': ['EURUSD', 'GBPUSD', 'XAUUSD'],
                    'execution_quality': 'EXCELLENT',  # Best possible execution
                    'typical_volatility': 'VERY_HIGH'
                }
            },
            
            # ✅ NEW: Market Condition Detection
            'market_conditions': {
                'volatility_thresholds': {
                    'low': 0.01,    # 1% daily range
                    'medium': 0.02, # 2% daily range
                    'high': 0.03,   # 3% daily range
                    'extreme': 0.05 # 5% daily range
                },
                'news_impact_symbols': {
                    'USD_NEWS': ['EURUSD', 'GBPUSD', 'XAUUSD'],
                    'EUR_NEWS': ['EURUSD'],
                    'GBP_NEWS': ['GBPUSD'],
                    'GOLD_NEWS': ['XAUUSD']
                },
                'high_impact_news_avoid_minutes': 30  # Avoid trading 30min around news
            },
            
            # Emergency and Safety Configuration
            'emergency': {
                'enable_emergency_stop': True,
                'max_consecutive_losses': 5,
                'emergency_close_threshold': 0.05,  # 5% account loss
                'enable_weekend_close': True,
                'market_hours_check': True,
                'max_requote_failures_per_hour': 10,  # ✅ NEW: Auto-pause on excessive requotes
                'execution_failure_threshold': 0.3  # ✅ NEW: 30% failure rate triggers pause
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
    
    def get_execution_params(self, symbol: str) -> dict:
        """Get symbol-specific execution parameters"""
        return self.config.get('mt5', {}).get('symbol_execution', {}).get(symbol, {})
    
    def get_slippage_for_symbol(self, symbol: str) -> int:
        """Get symbol-specific slippage setting"""
        exec_params = self.get_execution_params(symbol)
        return exec_params.get('max_slippage', self.config.get('mt5', {}).get('max_slippage', 3))
    
    def is_symbol_tradeable(self, symbol: str, session: str = None) -> bool:
        """Check if symbol is tradeable in current/specified session"""
        if symbol not in self.config.get('trading', {}).get('symbols', []):
            return False
        
        if session:
            session_config = self.config.get('trading_sessions', {}).get(session, {})
            enabled_symbols = session_config.get('enabled_symbols', [])
            return symbol in enabled_symbols
        
        return True
    
    def should_avoid_trading_due_to_execution_risk(self, symbol: str, recent_failures: int) -> bool:
        """Check if trading should be avoided due to execution risk"""
        if symbol == 'XAUUSD':
            max_failures = self.config.get('emergency', {}).get('max_requote_failures_per_hour', 10)
            return recent_failures >= max_failures
        return False

# Create global configuration instance
config = TradingConfig()

# ✅ VALIDATION: Ensure XAUUSD is properly configured
symbols = config.get('trading.symbols', [])
if 'XAUUSD' not in symbols:
    print("❌ WARNING: XAUUSD not found in trading symbols!")
else:
    print("✅ XAUUSD successfully configured in trading symbols")

if 'USDJPY' in symbols:
    print("⚠️ WARNING: USDJPY still found in symbols - should be removed")
else:
    print("✅ USDJPY successfully removed from trading symbols")

# Validate XAUUSD execution parameters
xau_params = config.get_execution_params('XAUUSD')
if xau_params:
    print(f"✅ XAUUSD execution parameters configured:")
    print(f"   Max Slippage: {xau_params.get('max_slippage', 'N/A')} points")
    print(f"   Max Retries: {xau_params.get('max_retry_attempts', 'N/A')}")
    print(f"   Retry Delay: {xau_params.get('retry_delay', 'N/A')}s")
else:
    print("❌ WARNING: XAUUSD execution parameters not configured!")

# Export for easy importing
__all__ = ['config', 'TradingConfig']

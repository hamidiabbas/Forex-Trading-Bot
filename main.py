"""
Professional Forex Trading Bot with Enhanced RL Integration
Production-grade implementation with SAC/A2C support - COMPLETE FIXED VERSION
"""

import os
import sys
import logging
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from collections import Counter
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

# Enhanced RL Integration imports
RL_AVAILABLE = False
try:
    from stable_baselines3 import SAC, A2C
    import torch
    RL_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not available. RL features disabled.")

# Enhanced Core Module Imports - Fixed
try:
    import config
    from enhanced_datahandler import EnhancedDataHandler
    from enhanced_marketintelligence import EnhancedMarketIntelligence  
    from enhanced_tradingenvironment import EnhancedTradingEnvironment
    from enhanced_train_rl_model import EnhancedRLModelManager
    from enhanced_feature_engineering import FeatureManager
    from dynamic_kelly_position_sizing import KellyPositionManager
    from enhanced_sentiment_integration import SentimentManager
    from strategymanager import StrategyManager
    from riskmanager import RiskManager
    from executionmanager import ExecutionManager
    from notificationmanager import NotificationManager
except ImportError as e:
    logging.error(f"Missing enhanced module: {e}")
    print(f"ERROR: Missing enhanced module: {e}")
    print("\nRequired enhanced files:")
    print("- enhanced_datahandler.py")
    print("- enhanced_marketintelligence.py") 
    print("- enhanced_tradingenvironment.py")
    print("- enhanced_train_rl_model.py")
    print("- enhanced_feature_engineering.py")
    print("- dynamic_kelly_position_sizing.py")
    print("- enhanced_sentiment_integration.py")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Enhanced logging configuration
def setup_enhanced_logging():
    """Setup comprehensive logging system"""
    try:
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(
            log_dir / 'enhanced_trading_bot.log', 
            encoding='utf-8', 
            mode='a'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Configure root logger
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Suppress noisy loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
        return True
        
    except Exception as e:
        print(f"Logging setup failed: {e}")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return False

# Initialize enhanced logging
setup_enhanced_logging()

class EnhancedTradingBot:
    """
    Professional Trading Bot with Complete Enhanced Integration - COMPLETE FIXED VERSION
    Features: SAC/A2C RL, Kelly Position Sizing, Sentiment Analysis, Advanced Risk Management
    """
    
    def __init__(self, config_path: str = 'config.py'):
        """Initialize the enhanced trading bot"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 60)
        self.logger.info("INITIALIZING ENHANCED FOREX TRADING BOT")
        self.logger.info("=" * 60)
        
        # Core configuration - Keep reference but pass module directly to components
        self.config = config
        self.start_time = time.time()
        
        # Control variables
        self.stop_event = threading.Event()
        self.status = "Initializing Enhanced Systems"
        self.last_analysis_time = {}
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.current_positions = []
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        
        # Enhanced Components Initialization
        try:
            self._initialize_enhanced_components()
            self.logger.info("All enhanced components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced components: {e}")
            raise
        
        # RL Model Integration
        self._initialize_rl_system()
        
        # Final system checks
        self._run_system_diagnostics()
        
        self.logger.info("Enhanced Trading Bot initialization completed successfully")
        self.logger.info(f"RL Status: {'Enabled' if self.rl_enabled else 'Disabled'}")
        self.logger.info(f"RL Model: {self.rl_model_type if self.rl_model_type else 'None'}")

    def _initialize_enhanced_components(self):
        """Initialize all enhanced trading components - COMPLETE FIXED VERSION"""
        
        # Enhanced Feature Management
        self.feature_manager = FeatureManager()
        self.logger.info("✓ FeatureManager initialized")
        
        # ✅ CRITICAL FIX: Enhanced Data Handler with direct config module passing
        self.datahandler = EnhancedDataHandler(config)  # Pass config module directly
        self.logger.info("✓ EnhancedDataHandler initialized")
        
        # ✅ FIXED: Enhanced Market Intelligence with correct parameters
        self.marketintelligence = EnhancedMarketIntelligence(config, user_config=None)
        self.logger.info("✓ EnhancedMarketIntelligence initialized")
        
        # Kelly Position Sizing - Use config module directly
        self.kelly_manager = KellyPositionManager(config)
        self.logger.info("✓ KellyPositionManager initialized")
        
        # Enhanced Sentiment Analysis - Use config module directly
        self.sentiment_manager = SentimentManager(config)
        self.logger.info("✓ SentimentManager initialized")
        
        # Traditional components (enhanced compatibility) - All use config module directly
        self.strategymanager = StrategyManager(config, self.marketintelligence)
        self.riskmanager = RiskManager(config, self.kelly_manager)
        self.executionmanager = ExecutionManager(config)
        self.notificationmanager = NotificationManager(config)
        
        self.logger.info("✓ Traditional components initialized with enhanced compatibility")

    def _initialize_rl_system(self):
        """Initialize the enhanced RL system with SAC/A2C support - COMPLETE FIXED VERSION"""
        
        # RL Model Manager
        self.rl_model_manager = None
        self.rl_enabled = False
        self.rl_model_type = None
        self.rl_model = None
        
        # RL Performance Tracking
        self.rl_signal_count = 0
        self.rl_successful_trades = 0
        self.rl_failed_trades = 0
        self.rl_performance_history = []
        
        if not RL_AVAILABLE:
            self.logger.warning("RL libraries not available - RL features disabled")
            return
        
        try:
            # ✅ CRITICAL FIX: Initialize Enhanced RL Model Manager with direct config module
            self.rl_model_manager = EnhancedRLModelManager(config)
            
            # Load best available model (SAC preferred, then A2C)
            model_loaded = self.rl_model_manager.load_best_model()
            
            if model_loaded:
                self.rl_enabled = True
                self.rl_model_type = self.rl_model_manager.get_model_type()
                self.rl_model = self.rl_model_manager.get_model()
                self.logger.info(f"✓ Enhanced RL System initialized: {self.rl_model_type}")
            else:
                self.logger.warning("No RL models found - traditional strategies only")
                
        except Exception as e:
            self.logger.error(f"RL system initialization failed: {e}")
            self.rl_enabled = False

    def _run_system_diagnostics(self):
        """Run comprehensive system diagnostics - COMPLETE FIXED VERSION"""
        self.logger.info("Running Enhanced System Diagnostics...")
        
        diagnostics = {
            'mt5_connection': getattr(self.datahandler, 'connected', False),
            'feature_manager': hasattr(self.feature_manager, 'validate_features'),
            'rl_system': self.rl_enabled,
            'sentiment_system': hasattr(self.sentiment_manager, 'is_available'),
            'kelly_sizing': hasattr(self.kelly_manager, 'is_configured'),
        }
        
        self.logger.info("System Diagnostics Results:")
        for component, status in diagnostics.items():
            status_symbol = "✓" if status else "✗"
            self.logger.info(f"  {status_symbol} {component}: {'OK' if status else 'FAILED'}")
        
        if self.rl_enabled and self.rl_model_manager:
            try:
                rl_diagnostics = self.rl_model_manager.get_diagnostics()
                self.logger.info("RL System Details:")
                for key, value in rl_diagnostics.items():
                    self.logger.info(f"  • {key}: {value}")
            except:
                self.logger.info("RL System: Basic initialization successful")

    def analyze_market_enhanced(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Enhanced market analysis with RL, sentiment, and regime detection - COMPLETE FIXED VERSION"""
        try:
            # Get multi-timeframe data
            data_dict = self.datahandler.get_data(symbol, 'M15', 200)
            if data_dict is None or len(data_dict) < 50:
                self.logger.warning(f"No data available for {symbol}")
                return None
            
            # Enhanced feature engineering
            try:
                features_df = self.feature_manager.engineer_features(data_dict)
                if features_df is None or len(features_df) < 50:
                    self.logger.warning(f"Insufficient feature data for {symbol}")
                    return None
            except Exception as e:
                self.logger.warning(f"Feature engineering failed for {symbol}: {e}")
                return None
            
            # Market regime identification
            try:
                regime = self.marketintelligence.identify_regime(data_dict)
            except Exception as e:
                self.logger.warning(f"Regime detection failed for {symbol}: {e}")
                regime = 'normal'
            
            # Sentiment analysis
            try:
                sentiment_score = self.sentiment_manager.get_market_sentiment(symbol)
            except Exception as e:
                self.logger.warning(f"Sentiment analysis failed for {symbol}: {e}")
                sentiment_score = 0.0
            
            # RL Signal Generation (Highest Priority)
            rl_signal = None
            if self.rl_enabled and self.rl_model_manager:
                try:
                    rl_signal = self.rl_model_manager.generate_signal(symbol, data_dict)
                    if rl_signal:
                        self.rl_signal_count += 1
                        self.logger.info(f"RL Signal Generated: {rl_signal['direction']} {symbol}")
                except Exception as e:
                    self.logger.warning(f"RL signal generation failed for {symbol}: {e}")
            
            # Traditional signals (backup/confirmation)
            try:
                traditional_signal = self.strategymanager.evaluate_signals(symbol, data_dict, regime)
            except Exception as e:
                self.logger.warning(f"Traditional signal generation failed for {symbol}: {e}")
                traditional_signal = None
            
            # Signal prioritization and fusion
            final_signal = self._fuse_signals(rl_signal, traditional_signal, sentiment_score)
            
            if final_signal:
                # Enhance signal with additional context
                final_signal.update({
                    'regime': regime,
                    'sentiment_score': sentiment_score,
                    'current_price': data_dict['Close'].iloc[-1] if 'Close' in data_dict.columns else 0,
                    'volume': data_dict.get('Volume', pd.Series([240])).iloc[-1] if 'Volume' in data_dict.columns else 240,
                    'volatility': 0.02,  # Default volatility
                    'analysis_time': datetime.now(),
                    'data_quality': len(data_dict),
                    'feature_count': len(features_df.columns) if features_df is not None else 0
                })
                
                self.logger.info(f"Enhanced signal generated for {symbol}: "
                               f"{final_signal['strategy']} - {final_signal['direction']}")
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Enhanced market analysis failed for {symbol}: {e}")
            return None

    def _fuse_signals(self, rl_signal, traditional_signal, sentiment_score):
        """Advanced signal fusion with confidence weighting - COMPLETE FIXED VERSION"""
        
        # RL signal has highest priority if available and confident
        if rl_signal and rl_signal.get('confidence', 0) > 0.7:
            return rl_signal
        
        # Consider sentiment confirmation
        if rl_signal and traditional_signal:
            rl_direction = rl_signal.get('direction', '')
            trad_direction = traditional_signal.get('direction', '')
            
            # If both agree and sentiment confirms, boost confidence
            if rl_direction == trad_direction:
                sentiment_confirms = (
                    (sentiment_score > 0.1 and rl_direction == 'BUY') or
                    (sentiment_score < -0.1 and rl_direction == 'SELL')
                )
                
                if sentiment_confirms:
                    rl_signal['confidence'] = min(0.95, rl_signal.get('confidence', 0.8) + 0.15)
                    rl_signal['strategy'] = f"RL+Traditional+Sentiment"
                    return rl_signal
        
        # Fallback priorities
        if rl_signal:
            return rl_signal
        elif traditional_signal:
            return traditional_signal
        
        return None

    def process_signal_enhanced(self, signal: Dict[str, Any]) -> bool:
        """Enhanced signal processing with Kelly sizing and risk management - COMPLETE FIXED VERSION"""
        if not signal:
            return False
        
        try:
            symbol = signal['symbol']
            signal_source = 'RL' if 'RL' in signal.get('strategy', '') else 'Traditional'
            
            # Enhanced Kelly position sizing
            try:
                kelly_params = self.kelly_manager.calculate_position_size(
                    symbol=symbol,
                    confidence=signal.get('confidence', 0.5),
                    expected_return=0.02,
                    risk_level=0.01,
                    account_balance=self.executionmanager.get_account_balance(),
                    market_regime=signal.get('regime', 'normal')
                )
            except Exception as e:
                self.logger.warning(f"Kelly sizing failed for {symbol}: {e}")
                kelly_params = {'position_size': 0.01}
            
            if not kelly_params or kelly_params.get('position_size', 0) <= 0:
                self.logger.warning(f"Kelly sizing rejected {signal_source} signal for {symbol}")
                if signal_source == 'RL':
                    self.rl_failed_trades += 1
                return False
            
            # Enhanced risk management validation
            try:
                risk_params = self.riskmanager.calculate_enhanced_risk(
                    symbol=symbol,
                    direction=signal['direction'],
                    entry_price=signal.get('entry_price', 0),
                    stop_loss=signal.get('stop_loss', 0),
                    take_profit=signal.get('take_profit', 0),
                    confidence=signal.get('confidence', 0.5),
                    strategy=signal.get('strategy', 'Unknown')
                )
            except Exception as e:
                self.logger.warning(f"Risk management failed for {symbol}: {e}")
                risk_params = None
            
            if not risk_params:
                self.logger.warning(f"Risk management rejected {signal_source} signal for {symbol}")
                if signal_source == 'RL':
                    self.rl_failed_trades += 1
                return False
            
            # Execute the trade
            try:
                execution_result = self.executionmanager.execute_trade(signal, risk_params)
            except Exception as e:
                self.logger.warning(f"Trade execution failed for {symbol}: {e}")
                execution_result = None
            
            if execution_result and execution_result.get('success'):
                self.total_trades += 1
                self.current_positions.append({
                    'symbol': symbol,
                    'direction': signal['direction'],
                    'strategy': signal['strategy'],
                    'entry_time': datetime.now(),
                    'entry_price': signal.get('entry_price'),
                    'position_size': risk_params.get('position_size'),
                    'stop_loss': risk_params.get('stop_loss'),
                    'take_profit': risk_params.get('take_profit')
                })
                
                # Track RL performance
                if signal_source == 'RL':
                    self.logger.info(f"RL trade executed: {self.rl_model_type} - {symbol}")
                
                # Enhanced notifications
                try:
                    self.notificationmanager.send_trade_notification(signal, risk_params, execution_result)
                except Exception as e:
                    self.logger.warning(f"Notification failed: {e}")
                
                return True
            else:
                self.logger.warning(f"Trade execution failed for {symbol}")
                if signal_source == 'RL':
                    self.rl_failed_trades += 1
                return False
                
        except Exception as e:
            self.logger.error(f"Enhanced signal processing failed: {e}")
            return False

    def update_performance_metrics_enhanced(self):
        """Update comprehensive performance metrics with RL tracking - COMPLETE FIXED VERSION"""
        try:
            # Get current account status
            current_equity = self.executionmanager.get_account_equity()
            current_balance = self.executionmanager.get_account_balance()
            
            # Calculate performance metrics
            win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
            uptime_hours = (time.time() - self.start_time) / 3600
            
            # RL specific metrics
            rl_metrics = self._calculate_rl_metrics()
            
            # Log comprehensive performance report
            self.logger.info("=" * 70)
            self.logger.info("ENHANCED PERFORMANCE UPDATE")
            self.logger.info("=" * 70)
            self.logger.info("Account Status:")
            self.logger.info(f"  Current Equity: ${current_equity:,.2f}")
            self.logger.info(f"  Current Balance: ${current_balance:,.2f}")
            self.logger.info(f"  Total Trades: {self.total_trades}")
            self.logger.info(f"  Overall Win Rate: {win_rate:.1f}%")
            self.logger.info(f"  Active Positions: {len(self.current_positions)}")
            self.logger.info(f"  Uptime: {uptime_hours:.1f} hours")
            
            self.logger.info("RL Performance:")
            for key, value in rl_metrics.items():
                self.logger.info(f"  {key}: {value}")
            
            self.logger.info("Enhanced Systems Status:")
            self.logger.info(f"  Feature Manager: {'Active' if self.feature_manager else 'Inactive'}")
            self.logger.info(f"  Sentiment Analysis: {'Active' if hasattr(self.sentiment_manager, 'is_available') else 'Inactive'}")
            self.logger.info(f"  Kelly Sizing: {'Active' if hasattr(self.kelly_manager, 'is_configured') else 'Inactive'}")
            self.logger.info("=" * 70)
            
        except Exception as e:
            self.logger.error(f"Error updating enhanced performance metrics: {e}")

    def _calculate_rl_metrics(self) -> Dict[str, Any]:
        """Calculate detailed RL performance metrics - COMPLETE FIXED VERSION"""
        if not self.rl_enabled:
            return {'status': 'RL Disabled'}
        
        rl_success_rate = 0
        rl_win_rate = 0
        if self.rl_signal_count > 0:
            rl_success_rate = ((self.rl_signal_count - self.rl_failed_trades) / self.rl_signal_count) * 100
            rl_win_rate = (self.rl_successful_trades / max(1, self.rl_signal_count)) * 100
        
        return {
            'Model Type': self.rl_model_type or 'None',
            'Status': 'Active' if self.rl_enabled else 'Inactive',
            'Signals Generated': self.rl_signal_count,
            'Success Rate': f"{rl_success_rate:.1f}%",
            'Win Rate': f"{rl_win_rate:.1f}%",
            'Failed Executions': self.rl_failed_trades,
            'Recent Performance': f"{np.mean(self.rl_performance_history[-10:]):.2f}" if len(self.rl_performance_history) > 0 else 'N/A'
        }

    def run_enhanced(self):
        """Enhanced main trading loop with comprehensive monitoring - COMPLETE FIXED VERSION"""
        self.logger.info("=" * 70)
        self.logger.info("STARTING ENHANCED TRADING BOT SESSION")
        self.logger.info("=" * 70)
        
        # Connect to data sources
        try:
            if hasattr(self.datahandler, 'connect'):
                connected = self.datahandler.connect()
            else:
                connected = getattr(self.datahandler, 'connected', True)
                
            if not connected:
                self.logger.error("Failed to connect to enhanced data handler")
                return
        except Exception as e:
            self.logger.warning(f"Data handler connection check failed: {e}")
        
        # Display enhanced startup summary
        self._display_startup_summary()
        
        # Main enhanced trading loop
        last_performance_update = time.time()
        last_system_check = time.time()
        loop_count = 0
        
        try:
            while not self.stop_event.is_set():
                loop_start_time = time.time()
                loop_count += 1
                
                # Update status
                active_positions = len(self.current_positions)
                self.status = f"Enhanced Loop {loop_count} - RL: {self.rl_model_type or 'Off'} - Positions: {active_positions}"
                
                try:
                    # Analyze each symbol with enhanced methods
                    symbols = getattr(self.config, 'SYMBOLS_TO_TRADE', ['EURUSD', 'GBPUSD', 'XAUUSD'])
                    for symbol in symbols:
                        try:
                            # Check analysis interval
                            if symbol in self.last_analysis_time:
                                time_since_last = time.time() - self.last_analysis_time[symbol]
                                min_interval = getattr(self.config, 'MIN_ANALYSIS_INTERVAL', 30)
                                if time_since_last < min_interval:
                                    continue
                            
                            # Enhanced market analysis
                            signal = self.analyze_market_enhanced(symbol)
                            
                            # Enhanced signal processing
                            if signal:
                                success = self.process_signal_enhanced(signal)
                                if success and 'RL' in signal.get('strategy', ''):
                                    # Track RL performance
                                    pass  # Will be updated when trade closes
                            
                            # Update analysis timing
                            self.last_analysis_time[symbol] = time.time()
                            
                        except Exception as symbol_error:
                            self.logger.error(f"Error processing {symbol}: {symbol_error}")
                            continue
                    
                    # Enhanced position management
                    try:
                        self.executionmanager.manage_positions()
                    except Exception as e:
                        self.logger.warning(f"Position management failed: {e}")
                    
                    # Performance metrics update (every 5 minutes)
                    performance_interval = getattr(self.config, 'PERFORMANCE_UPDATE_INTERVAL', 300)
                    if time.time() - last_performance_update >= performance_interval:
                        self.update_performance_metrics_enhanced()
                        last_performance_update = time.time()
                    
                    # System health check (every 10 minutes)
                    if time.time() - last_system_check >= 600:
                        self._run_system_health_check()
                        last_system_check = time.time()
                    
                    # Loop control
                    loop_duration = time.time() - loop_start_time
                    sleep_time = max(0, getattr(self.config, 'MAIN_LOOP_INTERVAL', 10) - loop_duration)
                    time.sleep(sleep_time)
                    
                except Exception as loop_error:
                    self.logger.error(f"Error in enhanced main loop iteration: {loop_error}")
                    time.sleep(5)  # Brief pause before continuing
                    
        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
        except Exception as e:
            self.logger.error(f"Critical error in enhanced main loop: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.shutdown_enhanced()

    def _display_startup_summary(self):
        """Display comprehensive startup summary - COMPLETE FIXED VERSION"""
        symbols = getattr(self.config, 'SYMBOLS_TO_TRADE', ['EURUSD', 'GBPUSD', 'XAUUSD'])
        risk_per_trade = getattr(self.config, 'RISK_PER_TRADE', 1.0)
        
        self.logger.info("Enhanced Trading Configuration:")
        self.logger.info(f"  RL Integration: {'Enabled' if self.rl_enabled else 'Disabled'}")
        self.logger.info(f"  RL Model: {self.rl_model_type if self.rl_model_type else 'None'}")
        self.logger.info(f"  Symbols: {symbols}")
        self.logger.info(f"  Risk per Trade: {risk_per_trade}%")
        self.logger.info(f"  Kelly Sizing: {'Enabled' if hasattr(self.kelly_manager, 'is_configured') else 'Disabled'}")
        self.logger.info(f"  Sentiment Analysis: {'Enabled' if hasattr(self.sentiment_manager, 'is_available') else 'Disabled'}")
        self.logger.info(f"  Python Version: {sys.version}")
        self.logger.info(f"  RL Libraries: {'Available' if RL_AVAILABLE else 'Not Available'}")
        self.logger.info("=" * 70)

    def _run_system_health_check(self):
        """Run periodic system health checks - COMPLETE FIXED VERSION"""
        try:
            health_status = {
                'mt5_connection': getattr(self.datahandler, 'connected', False),
                'rl_model_loaded': self.rl_enabled and self.rl_model is not None,
                'active_positions': len(self.current_positions),
                'memory_usage': self._get_memory_usage(),
                'last_signal_time': max(self.last_analysis_time.values()) if self.last_analysis_time else 0
            }
            
            # Log health status
            self.logger.info("System Health Check:")
            for component, status in health_status.items():
                self.logger.info(f"  • {component}: {status}")
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

    def _get_memory_usage(self) -> str:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            return f"{memory_mb:.1f} MB"
        except ImportError:
            return "N/A"

    def shutdown_enhanced(self):
        """Enhanced graceful shutdown - COMPLETE FIXED VERSION"""
        self.logger.info("Shutting down Enhanced Trading Bot...")
        
        try:
            # Set stop event
            self.stop_event.set()
            
            # Close positions if configured
            if getattr(self.config, 'CLOSE_POSITIONS_ON_SHUTDOWN', False):
                self.logger.info("Closing all positions before shutdown...")
                try:
                    self.executionmanager.close_all_positions()
                except Exception as e:
                    self.logger.warning(f"Failed to close positions: {e}")
            
            # Disconnect from data sources
            try:
                if hasattr(self.datahandler, 'disconnect'):
                    self.datahandler.disconnect()
            except Exception as e:
                self.logger.warning(f"Data handler disconnect failed: {e}")
            
            # Final performance report
            self.update_performance_metrics_enhanced()
            
            # Save RL model state if applicable
            if self.rl_enabled and self.rl_model_manager:
                try:
                    if hasattr(self.rl_model_manager, 'save_performance_state'):
                        self.rl_model_manager.save_performance_state()
                except Exception as e:
                    self.logger.warning(f"RL state save failed: {e}")
            
            # Enhanced shutdown notification
            shutdown_stats = {
                'total_trades': self.total_trades,
                'rl_signals': self.rl_signal_count,
                'rl_model': self.rl_model_type or 'None',
                'uptime_hours': f"{(time.time() - self.start_time) / 3600:.1f}",
                'final_status': 'Clean Shutdown'
            }
            
            shutdown_message = "\n".join([f"{k}: {v}" for k, v in shutdown_stats.items()])
            
            try:
                self.notificationmanager.send_system_notification(
                    "Enhanced Trading Bot Shutdown", 
                    shutdown_message
                )
            except:
                pass  # Don't fail shutdown on notification error
            
            self.logger.info("Enhanced Trading Bot shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during enhanced shutdown: {e}")

    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced status - COMPLETE FIXED VERSION"""
        rl_performance = 0
        if self.rl_signal_count > 0:
            rl_performance = (self.rl_successful_trades / self.rl_signal_count) * 100
        
        return {
            'status': self.status,
            'uptime_hours': (time.time() - self.start_time) / 3600,
            'rl_enabled': self.rl_enabled,
            'rl_model_type': self.rl_model_type,
            'rl_signals_generated': self.rl_signal_count,
            'rl_success_rate': rl_performance,
            'total_trades': self.total_trades,
            'win_rate': (self.winning_trades / max(1, self.total_trades)) * 100,
            'active_positions': len(self.current_positions),
            'enhanced_features': {
                'kelly_sizing': hasattr(self.kelly_manager, 'is_configured'),
                'sentiment_analysis': hasattr(self.sentiment_manager, 'is_available'),
                'feature_engineering': True,
                'regime_detection': True
            }
        }

def main():
    """Enhanced main entry point - COMPLETE FIXED VERSION"""
    try:
        # Initialize the enhanced trading bot
        bot = EnhancedTradingBot()
        
        # Start the enhanced main loop
        bot.run_enhanced()
        
    except Exception as e:
        logging.error(f"Critical error in enhanced main: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

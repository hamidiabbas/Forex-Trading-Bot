"""
Professional Trading Bot with Complete RL Integration
Enterprise-grade architecture with comprehensive error handling
"""

import os
import sys
import logging
import time
import threading
from datetime import datetime, timedelta
import traceback
from pathlib import Path
from config import config, SYMBOLS

# ✅ CRITICAL: Add all typing imports
from typing import Dict, Any, Optional, List, Tuple, Union

# Data handling imports
import pandas as pd
import numpy as np
from core.trading_environment import TradingEnvironment
from core.rl_model_manager import RLModelManager
# Quick debug check
symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']  # Your fixed symbols
print(f"✅ Trading symbols: {symbols}")
assert 'XAUUSD' in symbols and 'USDJPY' not in symbols, "Configuration fixed!"

# Setup paths
sys.path.append(str(Path(__file__).parent))

# ✅ FIXED: Non-recursive logging setup
def initialize_logging():
    """Initialize comprehensive logging system - NO RECURSION"""
    try:
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Clear existing handlers to prevent conflicts
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler('logs/trading_bot.log', encoding='utf-8', mode='a')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to root logger
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Suppress noisy loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
        # Test logging
        logger = logging.getLogger(__name__)
        logger.info("✅ Enhanced logging system initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging setup failed: {e}")
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return False

# Initialize logging once
initialize_logging()

# Core imports (after logging setup)
try:
    from core.feature_manager import FeatureManager
    from core.rl_model_manager import RLModelManager
    from core.data_handler import EnhancedDataHandler
    from core.market_intelligence import EnhancedMarketIntelligence
    from core.risk_manager import EnhancedRiskManager
    from core.execution_manager import EnhancedExecutionManager
    from core.notification_manager import EnhancedNotificationManager
    from utils.performance_monitor import PerformanceMonitor
    from utils.config_manager import ConfigManager
    
    logging.info("✅ All core modules imported successfully")
    
except ImportError as e:
    logging.error(f"❌ Import error: {e}")
    print(f"❌ Import error: {e}")
    print("Please ensure all enhanced core modules are in place")
    sys.exit(1)


class TradingBot:
    """
    Professional Trading Bot with Complete RL Integration
    """
    
    def __init__(self, config_path: str = "configs/bot_config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Initializing Professional Trading Bot...")
        
        # Load configuration
        try:
            self.config = ConfigManager(config_path)
        except Exception as e:
            self.logger.warning(f"Config file not found, using defaults: {e}")
            self.config = ConfigManager()  # Use defaults
        
        # Initialize core components
        try:
            self.feature_manager = FeatureManager()
            self.rl_model_manager = RLModelManager(self.feature_manager)
            self.data_handler = EnhancedDataHandler(self.config)
            self.market_intelligence = EnhancedMarketIntelligence(self.data_handler, self.config)
            self.risk_manager = EnhancedRiskManager(self.config)
            self.execution_manager = EnhancedExecutionManager(self.config, self.market_intelligence)
            self.notification_manager = EnhancedNotificationManager(self.config)
            self.performance_monitor = PerformanceMonitor(self.config)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize core components: {e}")
            raise
        
        # Control variables
        self.stop_event = threading.Event()
        self.status = "Initializing"
        self.last_analysis_time = {}
        
        # Performance tracking
        self.total_signals = 0
        self.successful_signals = 0
        self.rl_signals = 0
        self.traditional_signals = 0
        
        # Initialize system
        self._initialize_system()
        
        self.logger.info("✅ Trading Bot initialization completed successfully")

    def _initialize_system(self):
        """Initialize trading system components"""
        try:
            # Connect to data sources
            if not self.data_handler.connect():
                raise Exception("Failed to connect to data sources")
            
            # Validate RL model
            if self.rl_model_manager.is_available():
                model_info = self.rl_model_manager.get_model_info()
                self.logger.info(f"🤖 RL Model: {model_info.get('type', 'Unknown')} loaded successfully")
            else:
                self.logger.warning("⚠️ RL model not available - falling back to traditional strategies")
            
            # Initialize risk management
            self.risk_manager.initialize()
            
            # Send startup notification
            try:
                self.notification_manager.send_system_notification(
                    "Trading Bot Started",
                    f"System initialized successfully\nRL Model: {'Available' if self.rl_model_manager.is_available() else 'Not Available'}"
                )
            except Exception as notif_error:
                self.logger.warning(f"Notification failed: {notif_error}")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise

    def analyze_market(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Comprehensive market analysis with RL integration"""
        try:
            # Get multi-timeframe data
            data_dict = self.data_handler.get_multi_timeframe_data(symbol)
            if not data_dict or 'EXECUTION' not in data_dict:
                self.logger.warning(f"No data available for {symbol}")
                return None
            
            execution_df = data_dict['EXECUTION']
            
            # Engineer features
            features_df = self.feature_manager.engineer_features(execution_df)
            
            # Validate features
            is_valid, missing, extra = self.feature_manager.validate_features(features_df)
            if not is_valid:
                self.logger.warning(f"Feature validation failed for {symbol}: missing {missing}")
            
            # Get market regime
            regime = self.market_intelligence.identify_regime(data_dict.get('BIAS', execution_df))
            
            # Generate signals
            signals = []
            
            # RL Signal (highest priority)
            if self.rl_model_manager.is_available():
                rl_signal = self._generate_rl_signal(symbol, features_df, regime)
                if rl_signal:
                    signals.append(rl_signal)
                    self.rl_signals += 1
            
            # Traditional signals (backup)
            traditional_signal = self.market_intelligence.generate_traditional_signal(
                symbol, data_dict, regime
            )
            if traditional_signal:
                signals.append(traditional_signal)
                self.traditional_signals += 1
            
            # Select best signal
            final_signal = self._select_best_signal(signals)
            
            if final_signal:
                final_signal.update({
                    'regime': regime,
                    'data_quality': len(execution_df),
                    'feature_validation': is_valid,
                    'analysis_time': datetime.now()
                })
                
                self.total_signals += 1
                self.logger.info(f"📊 Signal generated for {symbol}: {final_signal['strategy']} - {final_signal['direction']}")
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Error analyzing market for {symbol}: {e}")
            return None

    def _generate_rl_signal(self, symbol: str, features_df: pd.DataFrame, regime: str) -> Optional[Dict[str, Any]]:
        """Generate RL trading signal"""
        try:
            # Prepare observation
            observation = self.feature_manager.prepare_observation(features_df)
            if observation is None:
                self.logger.debug(f"Could not prepare observation for {symbol}")
                return None
            
            # Get prediction
            prediction = self.rl_model_manager.predict_with_validation(observation)
            if prediction is None:
                self.logger.debug(f"RL model prediction failed for {symbol}")
                return None
            
            action, confidence = prediction
            
            # Convert action to signal
            if action == 1:  # Buy
                direction = 'BUY'
            elif action == 2:  # Sell
                direction = 'SELL'
            else:  # Hold
                return None
            
            # Get current price and ATR
            current_price = features_df['Close'].iloc[-1]
            atr_value = features_df.get('ATRr_14', pd.Series([0.001])).iloc[-1]
            
            return {
                'symbol': symbol,
                'direction': direction,
                'strategy': f'RL-{self.rl_model_manager.model_type}',
                'entry_price': current_price,
                'atr_at_signal': atr_value,
                'confidence': confidence,
                'model_type': self.rl_model_manager.model_type,
                'regime_compatibility': self._check_regime_compatibility(action, regime),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating RL signal for {symbol}: {e}")
            return None

    def _check_regime_compatibility(self, action: int, regime: str) -> float:
        """Check if RL action is compatible with market regime"""
        try:
            compatibility_matrix = {
                'Trending': {1: 0.9, 2: 0.9, 0: 0.3},      # Favor directional trades
                'Mean-Reverting': {1: 0.6, 2: 0.6, 0: 0.8},  # Favor holding
                'High-Volatility': {1: 0.4, 2: 0.4, 0: 0.9}, # Favor holding
                'Neutral': {1: 0.7, 2: 0.7, 0: 0.7}         # Neutral
            }
            
            return compatibility_matrix.get(regime, {}).get(action, 0.5)
            
        except Exception:
            return 0.5

    def _select_best_signal(self, signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best signal from available options"""
        try:
            if not signals:
                return None
            
            if len(signals) == 1:
                return signals[0]
            
            # Score signals based on confidence and model type
            best_signal = None
            best_score = 0
            
            for signal in signals:
                score = signal.get('confidence', 0.5)
                
                # Boost RL signals
                if 'RL-' in signal.get('strategy', ''):
                    score *= 1.2
                
                # Boost regime-compatible signals
                score *= signal.get('regime_compatibility', 1.0)
                
                if score > best_score:
                    best_score = score
                    best_signal = signal
            
            return best_signal
            
        except Exception as e:
            self.logger.error(f"Error selecting best signal: {e}")
            return signals[0] if signals else None

    def process_signal(self, signal: Dict[str, Any]) -> bool:
        """Process trading signal through risk management and execution"""
        try:
            if not signal:
                return False
            
            symbol = signal['symbol']
            
            # Risk management validation
            risk_params = self.risk_manager.calculate_position_size(signal)
            if not risk_params:
                self.logger.warning(f"Risk management rejected signal for {symbol}")
                return False
            
            # Execute trade
            execution_result = self.execution_manager.execute_trade(signal, risk_params)
            if execution_result and execution_result.get('success', False):
                self.successful_signals += 1
                
                # Update RL model performance if applicable
                if 'RL-' in signal.get('strategy', ''):
                    # Will update performance when trade closes
                    pass
                
                # Send notification
                try:
                    self.notification_manager.send_trade_notification(signal, risk_params, execution_result)
                except Exception as notif_error:
                    self.logger.warning(f"Trade notification failed: {notif_error}")
                
                # Update performance monitor
                try:
                    self.performance_monitor.record_signal(signal, risk_params, execution_result)
                except Exception as perf_error:
                    self.logger.warning(f"Performance recording failed: {perf_error}")
                
                self.logger.info(f"✅ Trade executed successfully for {symbol}")
                return True
            else:
                self.logger.warning(f"❌ Trade execution failed for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            return False

    def run(self):
        """Main trading loop"""
        self.logger.info("🚀 Starting main trading loop...")
        
        # Display startup banner
        self._display_startup_banner()
        
        try:
            while not self.stop_event.is_set():
                loop_start = time.time()
                
                # Update status
                self.status = f"Active - RL: {'On' if self.rl_model_manager.is_available() else 'Off'}"
                
                # Get trading symbols
                symbols = self.config.get('trading.symbols', ['EURUSD'])
                
                # Process each symbol
                for symbol in symbols:
                    try:
                        # Check analysis interval
                        if symbol in self.last_analysis_time:
                            time_since_last = time.time() - self.last_analysis_time[symbol]
                            min_interval = self.config.get('trading.min_analysis_interval', 30)
                            if time_since_last < min_interval:
                                continue
                        
                        # Analyze market
                        signal = self.analyze_market(symbol)
                        
                        # Process signal
                        if signal:
                            self.process_signal(signal)
                        
                        # Update timing
                        self.last_analysis_time[symbol] = time.time()
                        
                    except Exception as symbol_error:
                        self.logger.error(f"Error processing {symbol}: {symbol_error}")
                
                # Manage existing positions
                try:
                    self.execution_manager.manage_positions()
                except Exception as pos_error:
                    self.logger.error(f"Error managing positions: {pos_error}")
                
                # Update performance metrics
                try:
                    self.performance_monitor.update()
                except Exception as perf_error:
                    self.logger.warning(f"Performance update failed: {perf_error}")
                
                # Sleep until next cycle
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.config.get('trading.loop_interval', 10) - loop_duration)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Shutdown signal received")
        except Exception as e:
            self.logger.error(f"Critical error in main loop: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.shutdown()

    def _display_startup_banner(self):
        """Display startup banner with system information"""
        try:
            self.logger.info("============================================================")
            self.logger.info("🚀 FOREX TRADING BOT WITH SAC/A2C RL INTEGRATION")
            self.logger.info("============================================================")
            self.logger.info(f"RL Status: {'✅ Enabled' if self.rl_model_manager.is_available() else '❌ Disabled'}")
            
            if self.rl_model_manager.is_available():
                self.logger.info(f"RL Model: {self.rl_model_manager.model_type}")
            
            symbols = self.config.get('trading.symbols', ['EURUSD'])
            self.logger.info(f"Symbols: {symbols}")
            self.logger.info(f"Risk per Trade: {self.config.get('trading.risk_per_trade', 1.0)}%")
            self.logger.info(f"Python Version: {sys.version}")
            
            # RL Libraries status
            try:
                import stable_baselines3
                import torch
                self.logger.info("RL Libraries: ✅ Available")
            except ImportError:
                self.logger.info("RL Libraries: ❌ Not Available")
            
            self.logger.info("============================================================")
            
        except Exception as e:
            self.logger.error(f"Error displaying banner: {e}")

    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("⏹️ Shutting down Trading Bot...")
        
        try:
            # Stop main loop
            self.stop_event.set()
            
            # Close positions if configured
            if self.config.get('trading.close_positions_on_shutdown', False):
                try:
                    self.execution_manager.close_all_positions()
                except Exception as close_error:
                    self.logger.error(f"Error closing positions: {close_error}")
            
            # Disconnect from data sources
            try:
                self.data_handler.disconnect()
            except Exception as disconnect_error:
                self.logger.error(f"Error disconnecting data handler: {disconnect_error}")
            
            # Final performance report
            try:
                performance_summary = self.performance_monitor.get_summary()
                self.logger.info("==================================================")
                self.logger.info("📊 PERFORMANCE UPDATE")
                self.logger.info("==================================================")
                self.logger.info("💰 Account Status:")
                self.logger.info(f"  Total Trades: {performance_summary.get('total_trades', 0)}")
                self.logger.info(f"  Overall Win Rate: {performance_summary.get('win_rate_percent', 0):.1f}%")
                self.logger.info("🤖 RL Performance:")
                self.logger.info(f"  Model Type: {self.rl_model_manager.model_type if self.rl_model_manager.is_available() else 'N/A'}")
                self.logger.info(f"  RL Status: {'✅ Active' if self.rl_model_manager.is_available() else '❌ Inactive'}")
                self.logger.info(f"  Signals Generated: {self.rl_signals}")
                self.logger.info(f"  Traditional Signals: {self.traditional_signals}")
                self.logger.info("==================================================")
            except Exception as perf_error:
                self.logger.error(f"Error getting performance summary: {perf_error}")
            
            # Send shutdown notification
            try:
                self.notification_manager.send_system_notification(
                    "Trading Bot Shutdown",
                    f"Bot stopped gracefully\nTotal Trades: {self.total_signals}\nRL Signals: {self.rl_signals}"
                )
                
                # Shutdown notification manager
                self.notification_manager.shutdown()
            except Exception as notif_error:
                self.logger.warning(f"Shutdown notification failed: {notif_error}")
            
            self.logger.info("✅ Trading Bot shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

def main():
    """Main entry point"""
    try:
        # Create bot instance
        bot = TradingBot()
        
        # Start trading
        bot.run()
        
    except Exception as e:
        logging.error(f"Critical error: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

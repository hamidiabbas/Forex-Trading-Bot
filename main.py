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
import codecs

# Configure logging - FIXED Unicode-safe logging
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tradingbot.log', encoding='utf-8'),
        logging.StreamHandler(),
    ],
    force=True
)

# RL Integration imports
try:
    from stable_baselines3 import SAC, A2C
    import torch
    RL_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not available. RL features disabled.")
    RL_AVAILABLE = False

# Your existing imports
import config
from datahandler import DataHandler
from marketintelligence import MarketIntelligence
from strategymanager import StrategyManager
from riskmanager import RiskManager
from executionmanager import ExecutionManager
from notificationmanager import NotificationManager

# Load environment variables
load_dotenv()

class TradingBot:
    def __init__(self):
        """Initialize the trading bot with SAC/A2C RL integration"""
        logging.info("Initializing Trading Bot with RL Integration...")
        
        # Core components
        self.datahandler = DataHandler(config)
        self.marketintelligence = MarketIntelligence(self.datahandler, config)
        self.strategymanager = StrategyManager(config, self.marketintelligence)
        self.riskmanager = RiskManager(config)
        self.executionmanager = ExecutionManager(config, self.marketintelligence)
        self.notificationmanager = NotificationManager(config)
        
        # RL Integration components
        self.rl_model = None
        self.rl_model_type = None
        self.rl_enabled = RL_AVAILABLE
        self.rl_signal_count = 0
        self.rl_successful_trades = 0
        self.rl_failed_trades = 0
        
        # RL Model paths (SAC gets priority)
        self.sac_model_path = "model/sac_EURUSD_final.zip"
        self.a2c_model_path = "model/rl_EURUSD_final_fixed.zip"
        self.best_sac_path = "./best/sac_model_EURUSD_best_model.zip"
        self.best_a2c_path = "./best/model_EURUSD_best_model.zip"
        
        # Control variables
        self.stop_event = threading.Event()
        self.status = "Initializing"
        self.last_analysis_time = {}
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.current_positions = {}
        
        # RL Performance tracking
        self.rl_performance_history = []
        self.traditional_performance_history = []
        
        # Initialize RL model
        self.load_rl_model()
        
        logging.info("Trading Bot initialization completed successfully")
        logging.info(f"RL Status: {'Enabled' if self.rl_enabled else 'Disabled'}")
        logging.info(f"RL Model Type: {self.rl_model_type if self.rl_model_type else 'None'}")

    def load_rl_model(self):
        """Load the trained SAC or A2C model with priority to SAC"""
        if not RL_AVAILABLE:
            logging.warning("stable-baselines3 not available. RL features disabled.")
            self.rl_enabled = False
            return
            
        model_loaded = False
        
        # Try to load SAC model first (priority)
        for model_path, model_name in [
            (self.best_sac_path, "Best SAC"),
            (self.sac_model_path, "Final SAC"),
            (self.best_a2c_path, "Best A2C"),
            (self.a2c_model_path, "Final A2C")
        ]:
            if os.path.exists(model_path):
                try:
                    if "sac" in model_path.lower():
                        self.rl_model = SAC.load(model_path)
                        self.rl_model_type = "SAC"
                    else:
                        self.rl_model = A2C.load(model_path)
                        self.rl_model_type = "A2C"
                    
                    logging.info(f"{model_name} model loaded successfully from {model_path}")
                    model_loaded = True
                    break
                except Exception as e:
                    logging.warning(f"Failed to load {model_name} from {model_path}: {e}")
                    continue
        
        if not model_loaded:
            logging.warning("No RL model could be loaded. Available models:")
            for path in [self.sac_model_path, self.a2c_model_path, self.best_sac_path, self.best_a2c_path]:
                exists = "✓" if os.path.exists(path) else "✗"
                logging.warning(f"{exists} {path}")
            self.rl_enabled = False
        else:
            self.rl_enabled = True
            # Test model inference
            self.test_rl_model()

    def test_rl_model(self):
        """Test RL model inference"""
        if self.rl_model is None:
            return
            
        try:
            # Create a dummy observation for testing
            dummy_obs = np.random.random(30).astype(np.float32)  # Adjust size as needed
            action, _ = self.rl_model.predict(dummy_obs, deterministic=True)
            logging.info(f"RL model test successful. Sample action: {action}")
        except Exception as e:
            logging.error(f"RL model test failed: {e}")
            self.rl_enabled = False

    def get_rl_signal(self, symbol, datadict):
        """Get trading signal from RL model (SAC or A2C)"""
        if not self.rl_enabled or self.rl_model is None:
            return None
            
        try:
            df = datadict.get("EXECUTION")
            if df is None or len(df) < 20:
                return None
                
            # Prepare observation (same as training)
            obs = self.prepare_rl_observation(df)
            if obs is None:
                return None
                
            # Get prediction from RL model
            start_time = time.time()
            action, _ = self.rl_model.predict(obs, deterministic=True)
            inference_time = time.time() - start_time
            
            # Convert action to trading signal
            current_price = df["Close"].iloc[-1]
            atr_value = df.get("ATRr_14", pd.Series([0.001])).iloc[-1]
            
            # Calculate confidence based on model type and recent performance
            confidence = self.calculate_signal_confidence(action, obs)
            
            signal = None
            if action == 1:  # Buy
                self.rl_signal_count += 1
                signal = {
                    "symbol": symbol,
                    "direction": "BUY",
                    "strategy": f"RL-{self.rl_model_type}",
                    "entry_price": current_price,
                    "atr_at_signal": atr_value,
                    "confidence": confidence,
                    "timestamp": datetime.now(),
                    "inference_time": inference_time,
                    "model_type": self.rl_model_type
                }
            elif action == 2:  # Sell
                self.rl_signal_count += 1
                signal = {
                    "symbol": symbol,
                    "direction": "SELL",
                    "strategy": f"RL-{self.rl_model_type}",
                    "entry_price": current_price,
                    "atr_at_signal": atr_value,
                    "confidence": confidence,
                    "timestamp": datetime.now(),
                    "inference_time": inference_time,
                    "model_type": self.rl_model_type
                }
                
            if signal:
                logging.info(f"{self.rl_model_type} Signal: {signal['direction']} {symbol} (Confidence: {confidence:.2f})")
                
            return signal
            
        except Exception as e:
            logging.error(f"Error getting RL signal for {symbol}: {e}")
            return None

    def prepare_rl_observation(self, df):
        """Prepare observation for RL model (same as training)"""
        try:
            # Use last 30 technical indicators as observation
            features = ["Close", "RSI_14", "MACD_12_26_9", "ATRr_14", "EMA_20", "EMA_50"]
            
            if not all(col in df.columns for col in features):
                logging.warning("Missing required features for RL observation")
                return None
                
            obs = []
            for feature in features:
                values = df[feature].tail(5).values  # Last 5 values per feature
                obs.extend(values)
                
            return np.array(obs, dtype=np.float32)
            
        except Exception as e:
            logging.error(f"Error preparing RL observation: {e}")
            return None

    def calculate_signal_confidence(self, action, obs):
        """Calculate signal confidence based on model type and recent performance"""
        try:
            base_confidence = 0.75  # Base confidence level
            
            # Adjust based on recent RL performance
            if len(self.rl_performance_history) > 0:
                recent_performance = np.mean(self.rl_performance_history[-10:])
                performance_adjustment = (recent_performance - 0.5) * 0.2
                base_confidence += performance_adjustment
                
            # Ensure confidence is within reasonable bounds
            return max(0.5, min(0.60, base_confidence))
            
        except Exception as e:
            logging.error(f"Error calculating signal confidence: {e}")
            return 0.75

    def analyze_market(self, symbol):
        """Comprehensive market analysis including RL signals"""
        try:
            # Get multi-timeframe data
            datadict = self.datahandler.get_multitimeframe_data(symbol)
            if not datadict:
                logging.warning(f"No data available for {symbol}")
                return None
                
            # Get market regime
            regime = self.marketintelligence.identify_regime(datadict["BIAS"])
            
            # Get RL signal (highest priority)
            rl_signal = self.get_rl_signal(symbol, datadict)
            
            # Get traditional signals as backup
            traditional_signal = self.strategymanager.evaluate_signals(
                symbol, datadict, len(datadict["EXECUTION"]) - 1, regime
            )
            
            # Signal prioritization: RL first, then traditional
            final_signal = rl_signal if rl_signal else traditional_signal
            
            if final_signal:
                # Add market context to signal
                final_signal.update({
                    "regime": regime,
                    "current_price": datadict["EXECUTION"]["Close"].iloc[-1],
                    "volume": datadict["EXECUTION"].get("Volume", pd.Series([370])).iloc[-1],
                    "analysis_time": datetime.now(),
                    "data_quality": len(datadict["EXECUTION"]),
                    "signal_source": "RL" if rl_signal else "Traditional"
                })
                
                logging.info(f"{final_signal['signal_source']} signal for {symbol}: {final_signal['strategy']} - {final_signal['direction']} (Regime: {regime})")
                
            return final_signal
            
        except Exception as e:
            logging.error(f"Error analyzing market for {symbol}: {e}")
            logging.error(traceback.format_exc())
            return None

    def process_signal(self, signal):
        """Process trading signal through risk management and execution"""
        if not signal:
            return False
            
        try:
            symbol = signal["symbol"]
            signal_source = "RL" if "RL-" in signal.get("strategy", "") else "Traditional"
            
            # Calculate position size and risk parameters
            risk_params = self.riskmanager.calculate_position_size(signal)
            if not risk_params:
                logging.warning(f"Risk management rejected {signal_source} signal for {symbol}")
                if signal_source == "RL":
                    self.rl_failed_trades += 1
                return False
                
            # Execute the trade
            execution_result = self.executionmanager.execute_trade(signal, risk_params)
            
            if execution_result:
                self.total_trades += 1
                
                # Track RL vs Traditional performance separately
                if signal_source == "RL":
                    logging.info(f"{signal.get('model_type', 'RL')} trade executed for {symbol}")
                else:
                    logging.info(f"Traditional trade executed for {symbol}")
                    
                # Send notification
                self.notificationmanager.send_trade_notification(signal, risk_params, execution_result)
                return True
            else:
                logging.warning(f"Trade execution failed for {symbol}")
                if signal_source == "RL":
                    self.rl_failed_trades += 1
                return False
                
        except Exception as e:
            logging.error(f"Error processing signal: {e}")
            return False

    def run(self):
        """Main trading loop with SAC/A2C RL integration"""
        logging.info("Starting Trading Bot main loop with RL integration...")
        
        # Connect to data sources
        if not self.datahandler.connect():
            logging.error("Failed to connect to data handler")
            return
            
        # Display startup summary
        logging.info("=" * 60)
        logging.info("FOREX TRADING BOT - LIVE TRADING SESSION")
        logging.info("=" * 60)
        logging.info(f"RL Integration: {'Enabled' if self.rl_enabled else 'Disabled'}")
        logging.info(f"RL Model: {self.rl_model_type if self.rl_model_type else 'None'}")
        logging.info(f"Symbols: {config.SYMBOLS_TO_TRADE}")
        logging.info(f"Risk per Trade: {getattr(config, 'RISK_PER_TRADE', 1.0)}%")
        logging.info("=" * 60)
        
        # Main trading loop
        last_performance_update = time.time()
        loop_count = 0
        
        try:
            while not self.stop_event.is_set():
                loop_start_time = time.time()
                loop_count += 1
                self.status = f"Active - Loop {loop_count} - RL {'On' if self.rl_enabled else 'Off'}"
                
                try:
                    # Analyze each symbol
                    for symbol in config.SYMBOLS_TO_TRADE:
                        try:
                            # Check if enough time has passed since last analysis
                            if symbol in self.last_analysis_time:
                                time_since_last = time.time() - self.last_analysis_time[symbol]
                                if time_since_last < getattr(config, 'MIN_ANALYSIS_INTERVAL', 30):
                                    continue
                                    
                            # Perform market analysis
                            signal = self.analyze_market(symbol)
                            
                            # Process signal if generated
                            if signal:
                                success = self.process_signal(signal)
                                
                                # Track RL performance
                                if "RL-" in signal.get("strategy", "") and success:
                                    # Will update actual performance when trade closes
                                    pass
                                    
                            # Update last analysis time
                            self.last_analysis_time[symbol] = time.time()
                            
                        except Exception as symbol_error:
                            logging.error(f"Error processing {symbol}: {symbol_error}")
                            continue
                            
                    # Update performance metrics every 5 minutes
                    if time.time() - last_performance_update > 300:
                        self.update_performance_metrics()
                        last_performance_update = time.time()
                        
                    # Manage existing positions
                    self.executionmanager.manage_positions()
                    
                    # Update status
                    active_positions = len(self.current_positions)
                    self.status = f"Running - RL {self.rl_model_type if self.rl_enabled else 'Off'} - Positions: {active_positions}"
                    
                    # Sleep to control loop frequency
                    loop_duration = time.time() - loop_start_time
                    sleep_time = max(0, getattr(config, 'MAIN_LOOP_INTERVAL', 10) - loop_duration)
                    time.sleep(sleep_time)
                    
                except Exception as loop_error:
                    logging.error(f"Error in main loop iteration: {loop_error}")
                    logging.error(traceback.format_exc())
                    time.sleep(60)  # Wait before retrying
                    
        except KeyboardInterrupt:
            logging.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logging.error(f"Critical error in main loop: {e}")
            logging.error(traceback.format_exc())
        finally:
            self.shutdown()

    def update_performance_metrics(self):
        """Update and log comprehensive performance metrics"""
        try:
            # Get current positions and P&L
            current_equity = self.executionmanager.get_account_equity()
            current_balance = self.executionmanager.get_account_balance()
            
            # Calculate overall win rate
            win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
            
            # RL specific metrics
            rl_win_rate = 0
            rl_success_rate = 0
            if self.rl_signal_count > 0:
                rl_win_rate = (self.rl_successful_trades / self.rl_signal_count) * 100
                rl_success_rate = ((self.rl_signal_count - self.rl_failed_trades) / self.rl_signal_count) * 100
                
            # Log comprehensive performance
            logging.info("=" * 50)
            logging.info("PERFORMANCE UPDATE")
            logging.info("=" * 50)
            logging.info(f"Account Status:")
            logging.info(f"  Current Equity: ${current_equity:,.2f}")
            logging.info(f"  Current Balance: ${current_balance:,.2f}")
            logging.info(f"  Total Trades: {self.total_trades}")
            logging.info(f"  Overall Win Rate: {win_rate:.1f}%")
            logging.info(f"RL Performance:")
            logging.info(f"  Model Type: {self.rl_model_type if self.rl_model_type else 'None'}")
            logging.info(f"  RL Status: {'Active' if self.rl_enabled else 'Inactive'}")
            logging.info(f"  Signals Generated: {self.rl_signal_count}")
            logging.info(f"  Signal Success Rate: {rl_success_rate:.1f}%")
            logging.info(f"  RL Win Rate: {rl_win_rate:.1f}%")
            logging.info(f"  Failed Executions: {self.rl_failed_trades}")
            
            # Recent performance trend
            if len(self.rl_performance_history) > 0:
                recent_rl_perf = np.mean(self.rl_performance_history[-5:])
                logging.info(f"  Recent RL Performance: {recent_rl_perf:.3f}")
            
            logging.info("=" * 50)
            
        except Exception as e:
            logging.error(f"Error updating performance metrics: {e}")

    def shutdown(self):
        """Graceful shutdown of the trading bot"""
        logging.info("Shutting down Trading Bot...")
        
        try:
            # Set stop event
            self.stop_event.set()
            
            # Close all positions if configured to do so
            if hasattr(config, 'CLOSE_POSITIONS_ON_SHUTDOWN') and config.CLOSE_POSITIONS_ON_SHUTDOWN:
                logging.info("Closing all positions before shutdown...")
                self.executionmanager.close_all_positions()
                
            # Disconnect from data sources
            self.datahandler.disconnect()
            
            # Final performance report
            self.update_performance_metrics()
            
            # Send shutdown notification
            shutdown_message = f"Trading Bot stopped successfully\n"
            shutdown_message += f"Total Trades: {self.total_trades}\n"
            shutdown_message += f"RL Signals: {self.rl_signal_count}\n"
            shutdown_message += f"RL Model: {self.rl_model_type if self.rl_model_type else 'None'}"
            
            self.notificationmanager.send_system_notification("Trading Bot Shutdown", shutdown_message)
            
            logging.info("Trading Bot shutdown completed successfully")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")

def main():
    """Main entry point"""
    try:
        # Initialize and start the trading bot
        bot = TradingBot()
        bot.start_time = time.time()
        
        # Log startup information
        logging.info("=" * 60)
        logging.info("FOREX TRADING BOT WITH SAC/A2C RL INTEGRATION")
        logging.info("=" * 60)
        logging.info(f"RL Status: {'Enabled' if bot.rl_enabled else 'Disabled'}")
        logging.info(f"RL Model: {bot.rl_model_type if bot.rl_model_type else 'None'}")
        logging.info(f"Symbols: {config.SYMBOLS_TO_TRADE}")
        logging.info(f"Risk per Trade: {getattr(config, 'RISK_PER_TRADE', 1.0)}%")
        logging.info(f"Python Version: {sys.version}")
        logging.info(f"RL Libraries Available: {'Yes' if RL_AVAILABLE else 'No'}")
        logging.info("=" * 60)
        
        # Start the main trading loop
        bot.run()
        
    except Exception as e:
        logging.error(f"Critical error in main: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

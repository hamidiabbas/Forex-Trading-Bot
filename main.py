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

# RL Integration imports
from stable_baselines3 import A2C
import torch

# Your existing imports
import config
from data_handler import DataHandler
from market_intelligence import MarketIntelligence
from strategy_manager import StrategyManager
from risk_manager import RiskManager
from execution_manager import ExecutionManager
from notification_manager import NotificationManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class TradingBot:
    def __init__(self):
        """Initialize the trading bot with all components including RL integration"""
        logging.info("Initializing Trading Bot with RL Integration...")
        
        # Core components
        self.data_handler = DataHandler(config)
        self.market_intelligence = MarketIntelligence(self.data_handler, config)
        self.strategy_manager = StrategyManager(config, self.market_intelligence)
        self.risk_manager = RiskManager(config)
        self.execution_manager = ExecutionManager(config)
        self.notification_manager = NotificationManager(config)
        
        # RL Integration components
        self.rl_model = None
        self.rl_model_path = "model_rl_EURUSD_final_fixed.zip"
        self.rl_enabled = True
        self.rl_signal_count = 0
        self.rl_successful_trades = 0
        
        # Control variables
        self.stop_event = threading.Event()
        self.status = "Initializing"
        self.last_analysis_time = {}
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.current_positions = {}
        
        # Initialize RL model
        self._load_rl_model()
        
        logging.info("Trading Bot initialization completed successfully")

    def _load_rl_model(self):
        """Load the trained RL model"""
        try:
            if os.path.exists(self.rl_model_path):
                self.rl_model = A2C.load(self.rl_model_path)
                logging.info(f"RL model loaded successfully from {self.rl_model_path}")
                self.rl_enabled = True
            else:
                logging.warning(f"RL model file not found at {self.rl_model_path}")
                self.rl_enabled = False
        except Exception as e:
            logging.error(f"Error loading RL model: {e}")
            self.rl_model = None
            self.rl_enabled = False

    def _prepare_rl_observation(self, df):
        """Prepare observation for RL model (same preprocessing as training)"""
        try:
            if df is None or len(df) < 20:
                return None
            
            # Remove non-numeric columns (same as training environment)
            numeric_df = df.select_dtypes(include=[np.number])
            
            # Get the latest data point
            latest_data = numeric_df.iloc[-1].copy()
            
            # Add same additional features as training
            try:
                if len(df) >= 20:
                    price_change = df['Close'].pct_change().iloc[-1]
                    volatility = df['Close'].pct_change().rolling(20).std().iloc[-1]
                else:
                    price_change = 0
                    volatility = 0
                    
                if len(df) >= 10:
                    momentum = df['Close'].pct_change(periods=10).iloc[-1]
                else:
                    momentum = 0
                
                # Add calculated features to observation
                latest_data['price_change'] = price_change if not pd.isna(price_change) else 0
                latest_data['volatility'] = volatility if not pd.isna(volatility) else 0
                latest_data['momentum'] = momentum if not pd.isna(momentum) else 0
                
            except Exception as feature_error:
                logging.warning(f"Error calculating additional features: {feature_error}")
                # Use defaults if calculation fails
                latest_data['price_change'] = 0
                latest_data['volatility'] = 0
                latest_data['momentum'] = 0
            
            # Add position info (in live trading, we assume no current RL position for simplicity)
            # In a more advanced implementation, you'd track actual RL positions
            position_info = np.array([0.0, 0.0, 0.0])  # position, entry_price_norm, unrealized_pnl_norm
            
            # Combine all features
            obs_values = latest_data.fillna(0).values.astype(np.float32)
            obs = np.concatenate([obs_values, position_info]).astype(np.float32)
            
            # Validate observation shape
            if len(obs) == 0 or np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                logging.warning("Invalid observation detected, using fallback")
                return None
            
            return obs
            
        except Exception as e:
            logging.error(f"Error preparing RL observation: {e}")
            return None

    def get_rl_signal(self, symbol, data_dict):
        """Get trading signal from RL model"""
        if not self.rl_enabled or self.rl_model is None:
            return None
            
        try:
            df = data_dict.get('EXECUTION')
            if df is None or len(df) < 20:
                return None
            
            obs = self._prepare_rl_observation(df)
            if obs is None:
                return None
            
            # Get prediction from RL model
            action, _states = self.rl_model.predict(obs, deterministic=True)
            
            # Convert action to trading signal
            current_price = df['Close'].iloc[-1]
            atr_value = df.get('ATRr_14', pd.Series([0.001])).iloc[-1]
            
            if action == 1:  # Buy signal
                self.rl_signal_count += 1
                return {
                    'symbol': symbol,
                    'direction': 'BUY',
                    'strategy': 'RL-Agent',
                    'entry_price': current_price,
                    'atr_at_signal': atr_value,
                    'confidence': 0.8,  # You could extract this from model if available
                    'timestamp': datetime.now()
                }
            elif action == 2:  # Sell signal
                self.rl_signal_count += 1
                return {
                    'symbol': symbol,
                    'direction': 'SELL',
                    'strategy': 'RL-Agent',
                    'entry_price': current_price,
                    'atr_at_signal': atr_value,
                    'confidence': 0.8,
                    'timestamp': datetime.now()
                }
            
            # Action 0 = Hold, return None
            return None
            
        except Exception as e:
            logging.error(f"Error getting RL signal for {symbol}: {e}")
            return None

    def analyze_market(self, symbol):
        """Comprehensive market analysis including RL signals"""
        try:
            # Get multi-timeframe data
            data_dict = self.data_handler.get_multi_timeframe_data(symbol)
            if not data_dict:
                logging.warning(f"No data available for {symbol}")
                return None
            
            # Get market regime
            regime = self.market_intelligence.identify_regime(data_dict['BIAS'])
            
            # Get RL signal (highest priority)
            rl_signal = self.get_rl_signal(symbol, data_dict)
            
            # Get traditional signals as backup
            traditional_signal = self.strategy_manager.evaluate_signals(
                symbol, data_dict, len(data_dict['EXECUTION']) - 1, regime
            )
            
            # Signal prioritization: RL first, then traditional
            final_signal = rl_signal if rl_signal else traditional_signal
            
            if final_signal:
                # Add market context to signal
                final_signal.update({
                    'regime': regime,
                    'current_price': data_dict['EXECUTION']['Close'].iloc[-1],
                    'volume': data_dict['EXECUTION'].get('Volume', pd.Series([0])).iloc[-1],
                    'analysis_time': datetime.now()
                })
                
                logging.info(f"Signal generated for {symbol}: {final_signal['strategy']} - {final_signal['direction']}")
                
            return final_signal
            
        except Exception as e:
            logging.error(f"Error analyzing market for {symbol}: {e}")
            return None

    def process_signal(self, signal):
        """Process trading signal through risk management and execution"""
        if not signal:
            return False
            
        try:
            symbol = signal['symbol']
            
            # Calculate position size and risk parameters
            risk_params = self.risk_manager.calculate_position_size(signal)
            if not risk_params:
                logging.warning(f"Risk management rejected signal for {symbol}")
                return False
            
            # Execute the trade
            execution_result = self.execution_manager.execute_trade(signal, risk_params)
            if execution_result:
                self.total_trades += 1
                
                # Track RL performance separately
                if signal.get('strategy') == 'RL-Agent':
                    logging.info(f"RL trade executed successfully for {symbol}")
                
                # Send notification
                self.notification_manager.send_trade_notification(signal, risk_params, execution_result)
                
                return True
            else:
                logging.warning(f"Trade execution failed for {symbol}")
                return False
                
        except Exception as e:
            logging.error(f"Error processing signal: {e}")
            return False

    def update_performance_metrics(self):
        """Update and log performance metrics"""
        try:
            # Get current positions and P&L
            current_equity = self.execution_manager.get_account_equity()
            current_balance = self.execution_manager.get_account_balance()
            
            # Calculate win rate
            win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
            
            # RL specific metrics
            rl_win_rate = 0
            if self.rl_signal_count > 0:
                rl_win_rate = (self.rl_successful_trades / self.rl_signal_count) * 100
            
            # Log performance
            logging.info(f"Performance Update:")
            logging.info(f"  Total Trades: {self.total_trades}")
            logging.info(f"  Win Rate: {win_rate:.1f}%")
            logging.info(f"  Current Equity: ${current_equity:,.2f}")
            logging.info(f"  Current Balance: ${current_balance:,.2f}")
            logging.info(f"  RL Signals Generated: {self.rl_signal_count}")
            logging.info(f"  RL Win Rate: {rl_win_rate:.1f}%")
            logging.info(f"  RL Model Status: {'Active' if self.rl_enabled else 'Inactive'}")
            
        except Exception as e:
            logging.error(f"Error updating performance metrics: {e}")

    def run(self):
        """Main trading loop with RL integration"""
        logging.info("Starting Trading Bot main loop...")
        
        # Connect to data sources
        if not self.data_handler.connect():
            logging.error("Failed to connect to data handler")
            return
        
        # Main trading loop
        last_performance_update = time.time()
        
        try:
            while not self.stop_event.is_set():
                loop_start_time = time.time()
                self.status = "Analyzing markets with RL integration..."
                
                try:
                    # Analyze each symbol
                    for symbol in config.SYMBOLS_TO_TRADE:
                        try:
                            # Check if enough time has passed since last analysis
                            if symbol in self.last_analysis_time:
                                time_since_last = time.time() - self.last_analysis_time[symbol]
                                if time_since_last < config.MIN_ANALYSIS_INTERVAL:
                                    continue
                            
                            # Perform market analysis
                            signal = self.analyze_market(symbol)
                            
                            # Process signal if generated
                            if signal:
                                self.process_signal(signal)
                            
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
                    self.execution_manager.manage_positions()
                    
                    # Update status
                    self.status = "Running - RL Integration Active" if self.rl_enabled else "Running - RL Disabled"
                    
                    # Sleep to control loop frequency
                    loop_duration = time.time() - loop_start_time
                    sleep_time = max(0, config.MAIN_LOOP_INTERVAL - loop_duration)
                    time.sleep(sleep_time)
                    
                except Exception as loop_error:
                    logging.error(f"Error in main loop iteration: {loop_error}")
                    time.sleep(60)  # Wait before retrying
                    
        except KeyboardInterrupt:
            logging.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logging.error(f"Critical error in main loop: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown of the trading bot"""
        logging.info("Shutting down Trading Bot...")
        
        try:
            # Set stop event
            self.stop_event.set()
            
            # Close all positions if configured to do so
            if hasattr(config, 'CLOSE_POSITIONS_ON_SHUTDOWN') and config.CLOSE_POSITIONS_ON_SHUTDOWN:
                self.execution_manager.close_all_positions()
            
            # Disconnect from data sources
            self.data_handler.disconnect()
            
            # Final performance report
            self.update_performance_metrics()
            
            # Send shutdown notification
            self.notification_manager.send_system_notification(
                "Trading Bot Shutdown",
                f"Bot stopped successfully after {self.total_trades} total trades"
            )
            
            logging.info("Trading Bot shutdown completed successfully")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")

    def get_status(self):
        """Get current bot status"""
        return {
            'status': self.status,
            'rl_enabled': self.rl_enabled,
            'rl_model_loaded': self.rl_model is not None,
            'total_trades': self.total_trades,
            'rl_signals_generated': self.rl_signal_count,
            'win_rate': (self.winning_trades / max(1, self.total_trades)) * 100,
            'uptime': time.time() - getattr(self, 'start_time', time.time())
        }

def main():
    """Main entry point"""
    try:
        # Initialize and start the trading bot
        bot = TradingBot()
        bot.start_time = time.time()
        
        # Log startup information
        logging.info("="*50)
        logging.info("FOREX TRADING BOT WITH RL INTEGRATION")
        logging.info("="*50)
        logging.info(f"RL Model: {'Loaded' if bot.rl_enabled else 'Not Available'}")
        logging.info(f"Symbols to trade: {config.SYMBOLS_TO_TRADE}")
        logging.info(f"Risk per trade: {config.RISK_PER_TRADE}%")
        logging.info("="*50)
        
        # Start the main trading loop
        bot.run()
        
    except Exception as e:
        logging.error(f"Critical error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

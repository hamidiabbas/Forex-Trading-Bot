"""
/******************************************************************************
 *
 * FILE NAME:           main.py (with Human Interface)
 *
 * PURPOSE:
 *
 * This version integrates the human_interface module, allowing for real-time
 * interactive control of the bot.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 24, 2025
 *
 * VERSION:             37.0
 *
 ******************************************************************************/
"""

import MetaTrader5 as mt5
import time
import logging
import threading
from stable_baselines3 import A2C
import numpy as np
# Import all the other modules
import config
from data_handler import DataHandler
from market_intelligence import MarketIntelligence
from risk_manager import RiskManager
from strategy_manager import StrategyManager
from execution_manager import ExecutionManager
from human_interface import BotInterface # Import the new interface

# ... (logging configuration is unchanged) ...

class TradingBot:
    def __init__(self):
        logging.info("Initializing bot components...")
        self.data_handler = DataHandler(config)
        self.market_intelligence = MarketIntelligence(self.data_handler, config)
        self.risk_manager = RiskManager(self.data_handler, config)
        self.strategy_manager = StrategyManager(config, self.market_intelligence)
        self.execution_manager = ExecutionManager(self.data_handler, config)
        self.strategy_manager = StrategyManager(config, self.market_intelligence)
        self.execution_manager = ExecutionManager(self.data_handler, config, self.market_intelligence)
        # --- NEW: Event for graceful shutdown ---
        self.stop_event = threading.Event()
        self.status = "Initializing"

    def run(self):
        """ The main trading loop. """
        logging.info("Starting the trading bot...")
        if not self.data_handler.connect():
            logging.error("CRITICAL: Failed to connect. The bot cannot start.")
            return
            
        # --- NEW: Start the human interface thread ---
        self.interface = BotInterface(self)
        self.interface.start()

        # The main loop now checks the stop_event
        while not self.stop_event.is_set():
            try:
                # ... (the core trading logic loop is the same as before) ...
                self.status = "Analyzing markets..."
                # ...
                
                self.status = "Managing open positions..."
                # ...
                
                self.status = f"Cycle complete. Waiting..."
                time.sleep(60) # Configurable sleep timer

            except KeyboardInterrupt:
                self.stop() # Also handle Ctrl+C gracefully
                break
            except Exception as e:
                logging.exception(f"An unexpected error occurred: {e}")
                time.sleep(60)

        logging.info("Bot has been shut down.")
        self.data_handler.disconnect()

    # --- NEW METHODS for the interface to call ---
    def stop(self):
        """ Sets the event to stop the main loop. """
        self.stop_event.set()

    def show_status(self):
        """ Prints the current bot status and account info. """
        print(f"\n--- Bot Status ---")
        print(f"Current Activity: {self.status}")
        account_info = self.data_handler.get_account_info()
        if account_info:
            print(f"Account Equity: ${account_info['equity']:,.2f}")
            print(f"Account Balance: ${account_info['balance']:,.2f}")
            print(f"Margin Level: {account_info['margin_level']:.2f}%")
        else:
            print("Could not retrieve account info.")
        print("--------------------\n")

    def show_open_positions(self):
        """ Prints a list of open positions. """
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            print("\nNo open positions.\n")
            return
        
        print("\n--- Open Positions ---")
        for pos in positions:
            print(f"  Ticket: {pos.ticket}, Symbol: {pos.symbol}, Lots: {pos.volume}, P/L: ${pos.profit:,.2f}")
        print("----------------------\n")


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
class TradingBot:
    def __init__(self):
        # ... existing initialization ...
        
        # FIXED: Add RL model initialization
        self.rl_model = None
        self.rl_model_path = "model_rl_EURUSD_final_fixed.zip"
        self._load_rl_model()
    
    def _load_rl_model(self):
        """Load the trained RL model"""
        try:
            self.rl_model = A2C.load(self.rl_model_path)
            logging.info("RL model loaded successfully")
        except Exception as e:
            logging.warning(f"Could not load RL model: {e}")
            self.rl_model = None
    
    def _prepare_rl_observation(self, df):
        """Prepare observation for RL model (same as training)"""
        try:
            # Get the latest data point
            latest_data = df.iloc[-1]
            
            # Create base observation (same features as training)
            features_to_drop = ['Open', 'High', 'Low', 'Volume', 'hurst', 
                               'fib_0.236', 'fib_0.382', 'fib_0.500', 'fib_0.618']
            
            obs_data = latest_data.drop(labels=features_to_drop, errors='ignore')
            
            # Add same additional features as training
            if len(df) >= 20:
                price_change = df['Close'].pct_change().iloc[-1]
                volatility = df['Close'].pct_change().rolling(20).std().iloc[-1]
            else:
                price_change = 0
                volatility = 0
                
            if len(df) >= 10:
                momentum_series = df['Close'].rolling(10).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
                momentum = momentum_series.iloc[-1]
            else:
                momentum = 0
            
            # Add calculated features
            obs_data = obs_data.append(pd.Series({
                'price_change': price_change,
                'volatility': volatility,
                'momentum': momentum
            }))
            
            # Add position info (assuming no current position for simplicity)
            position_info = np.array([0.0, 0.0, 0.0])  # position, entry_price_norm, unrealized_pnl_norm
            
            # Combine and handle NaN
            obs = np.concatenate([obs_data.fillna(0).values, position_info]).astype(np.float32)
            
            return obs
            
        except Exception as e:
            logging.error(f"Error preparing RL observation: {e}")
            return None
    
    def get_rl_signal(self, symbol, data_dict):
        """FIXED: Get trading signal from RL model"""
        if self.rl_model is None:
            return None
            
        try:
            df = data_dict.get('EXECUTION')
            if df is None or len(df) < 20:
                return None
            
            obs = self._prepare_rl_observation(df)
            if obs is None:
                return None
            
            # Get prediction
            action, _ = self.rl_model.predict(obs, deterministic=True)
            
            # Convert action to signal
            if action == 1:  # Buy signal
                return {
                    'symbol': symbol,
                    'direction': 'BUY',
                    'strategy': 'RL-Agent',
                    'entry_price': df['Close'].iloc[-1],
                    'atr_at_signal': df.get('ATRr_14', pd.Series([0.001])).iloc[-1]
                }
            elif action == 2:  # Sell signal
                return {
                    'symbol': symbol,
                    'direction': 'SELL',
                    'strategy': 'RL-Agent',
                    'entry_price': df['Close'].iloc[-1],
                    'atr_at_signal': df.get('ATRr_14', pd.Series([0.001])).iloc[-1]
                }
            
            return None  # Hold signal
            
        except Exception as e:
            logging.error(f"Error getting RL signal: {e}")
            return None

    def run(self):
        """Modified main loop to include RL signals"""
        # ... existing code until the main trading loop ...
        
        while not self.stop_event.is_set():
            try:
                self.status = "Analyzing markets with RL..."
                
                for symbol in config.SYMBOLS_TO_TRADE:
                    # Get multi-timeframe data
                    data_dict = self.data_handler.get_multi_timeframe_data(symbol)
                    if not data_dict:
                        continue
                    
                    # FIXED: Include RL signal in decision making
                    rl_signal = self.get_rl_signal(symbol, data_dict)
                    
                    # Get traditional signals
                    regime = self.market_intelligence.identify_regime(data_dict['BIAS'])
                    traditional_signal = self.strategy_manager.evaluate_signals(
                        symbol, data_dict, len(data_dict['EXECUTION']) - 1, regime
                    )
                    
                    # Combine signals (RL takes priority if available)
                    final_signal = rl_signal if rl_signal else traditional_signal
                    
                    if final_signal:
                        logging.info(f"Trading signal from {final_signal.get('strategy', 'Unknown')}: "
                                   f"{final_signal['direction']} {symbol}")
                        
                        # Process signal through risk management and execution
                        risk_params = self.risk_manager.calculate_position_size(final_signal)
                        if risk_params:
                            self.execution_manager.execute_trade(final_signal, risk_params)
                
                # ... rest of existing main loop ...
                
            except Exception as e:
                logging.exception(f"Error in main loop: {e}")
                time.sleep(60)
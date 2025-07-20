"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           main.py
 *
 * PURPOSE:
 *
 * This is the main entry point of the application. It orchestrates all
 * the other modules, creating a cohesive and functioning trading bot.
 * It is responsible for initializing all the components, running the
 * main trading loop, and gracefully shutting down the application.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             4.0
 *
 ******************************************************************************/
"""

import MetaTrader5 as mt5 # Import mt5 here to access its methods
import time
import logging
import threading

# Import all the other modules
import config
from data_handler import DataHandler
from market_intelligence import MarketIntelligence
from risk_manager import RiskManager
from strategy_manager import StrategyManager
from execution_manager import ExecutionManager
from performance_analyzer import PerformanceAnalyzer

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                    handlers=[
                        logging.FileHandler(config.ERROR_LOG_FILE),
                        logging.StreamHandler()
                    ])

class TradingBot:
    """
    The main trading bot class that encapsulates the entire application logic.
    """

    def __init__(self):
        """
        Initializes all the necessary components (modules) of the trading bot.
        """
        logging.info("Initializing bot components...")
        self.data_handler = DataHandler(config)
        # --- CHANGE: Pass the 'config' object to MarketIntelligence ---
        self.market_intelligence = MarketIntelligence(self.data_handler, config)
        self.risk_manager = RiskManager(self.data_handler, config)
        self.strategy_manager = StrategyManager(config)
        self.execution_manager = ExecutionManager(self.data_handler, config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        self.all_dataframes = {}

    def run(self):
        """
        The main trading loop.
        """
        logging.info("Starting the trading bot...")
        if not self.data_handler.connect():
            logging.error("CRITICAL: Failed to connect to MetaTrader 5. The bot cannot start. Exiting.")
            return

        while True:
            try:
                self.data_handler.check_connection()
                if not self.data_handler.connection_status:
                    logging.warning("Connection lost. Waiting for the next cycle to retry.")
                    time.sleep(config.LOOP_SLEEP_TIMER)
                    continue

                account_info = self.data_handler.get_account_info()
                if account_info:
                    logging.info(f"Account Equity: {account_info['equity']:.2f} | Balance: {account_info['balance']:.2f} | Margin Level: {account_info['margin_level']:.2f}%")
                else:
                    logging.warning("Could not retrieve account info in this cycle.")
                    time.sleep(config.LOOP_SLEEP_TIMER)
                    continue

                for symbol in config.SYMBOLS_TO_TRADE:
                    logging.info(f"--- Analyzing {symbol} ---")

                    analyzed_data_mta = self.market_intelligence.analyze_all_timeframes(symbol)
                    if not analyzed_data_mta:
                        logging.warning(f"Could not retrieve or analyze data for {symbol}. Skipping.")
                        continue
                    
                    self.all_dataframes[symbol] = analyzed_data_mta.get(config.TIMEFRAMES['POSITIONAL'])

                    positional_df = analyzed_data_mta.get(config.TIMEFRAMES['POSITIONAL'])
                    if positional_df is None or positional_df.empty:
                        logging.warning(f"Positional data for {symbol} is missing. Skipping.")
                        continue
                    
                    regime = self.market_intelligence.determine_market_regime(positional_df)
                    logging.info(f"Market regime for {symbol}: {regime}")

                    signal = self.strategy_manager.evaluate_signals(symbol, analyzed_data_mta, regime)

                    if signal:
                        logging.info(f"Signal found for {symbol}: {signal['direction']} ({signal['strategy']})")
                        
                        support_resistance = self.market_intelligence.identify_support_resistance(positional_df)
                        sl_price, tp_prices, sl_pips = self.risk_manager.calculate_sl_tp(signal, signal['entry_price'], signal['atr_at_signal'], support_resistance)
                        lot_size = self.risk_manager.calculate_position_size(account_info['equity'], sl_pips, symbol)

                        if lot_size > 0 and len(mt5.positions_get(symbol=symbol)) == 0:
                            logging.info(f"Preparing to execute trade: {lot_size} lots of {symbol}, SL={sl_price:.5f}, TP={tp_prices[0]:.5f}")
                            trade_thread = threading.Thread(target=self.execution_manager.execute_trade,
                                                            args=(signal, lot_size, sl_price, tp_prices))
                            trade_thread.start()
                        else:
                            logging.warning(f"Trade for {symbol} aborted. Invalid lot size ({lot_size}) or existing position.")

                self.execution_manager.manage_open_positions()

                performance_metrics = self.performance_analyzer.analyze()
                if performance_metrics:
                    logging.info(f"Performance Metrics: {performance_metrics}")

                logging.info(f"--- Cycle complete. Waiting for {config.LOOP_SLEEP_TIMER} seconds. ---")
                time.sleep(config.LOOP_SLEEP_TIMER)

            except KeyboardInterrupt:
                logging.info("Shutdown signal received. Disconnecting and closing the bot.")
                self.data_handler.disconnect()
                break
            except Exception as e:
                logging.exception(f"An unexpected error occurred in the main loop: {e}")
                time.sleep(config.LOOP_SLEEP_TIMER)

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
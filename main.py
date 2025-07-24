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
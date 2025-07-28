"""
/******************************************************************************
 *
 * FILE NAME:           data_handler.py (Complete & Verified)
 *
 * PURPOSE:
 *
 * This version is the complete and correct implementation of the DataHandler,
 * including the function for fetching multi-timeframe data and account info.
 * This file has been triple-checked for correctness.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 28, 2025
 *
 * VERSION:             53.2 (Complete & Verified)
 *
 ******************************************************************************/
"""

import MetaTrader5 as mt5
import pandas as pd
import time
import logging
from datetime import datetime
import pytz

# Configure the logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DataHandler:
    """
    Handles all communication with the MetaTrader 5 platform.
    """

    def __init__(self, config):
        """
        Initializes the DataHandler with the given configuration.
        """
        self.config = config
        self.connection_status = False

    def connect(self):
        """
        Initializes the connection to the MT5 terminal with retry logic.
        """
        if self.connection_status:
            return True
            
        for attempt in range(1, self.config.CONNECTION_RETRY_ATTEMPTS + 1):
            logging.info(f"MT5 connection attempt {attempt}...")
            try:
                if mt5.initialize(path=self.config.MT5_TERMINAL_PATH,
                                  login=self.config.MT5_ACCOUNT_NUMBER,
                                  password=self.config.MT5_PASSWORD,
                                  server=self.config.MT5_SERVER_NAME):
                    logging.info("MetaTrader 5 initialized successfully.")
                    self.connection_status = True
                    return True
                else:
                    logging.error(f"MT5 initialization failed. Error: {mt5.last_error()}")
            except Exception as e:
                logging.error(f"An exception occurred during MT5 initialization: {e}")
            time.sleep(self.config.CONNECTION_RETRY_DELAY_SECONDS)
        logging.error("Failed to connect to MetaTrader 5 after multiple attempts.")
        return False

    def disconnect(self):
        """
        Properly terminates the MT5 connection.
        """
        if self.connection_status:
            logging.info("Disconnecting from MetaTrader 5.")
            mt5.shutdown()
            self.connection_status = False

    def get_price_data(self, symbol, timeframe, count):
        """
        Fetches a specific number of historical price data candles.
        """
        if not self.connection_status:
            self.connect()

        try:
            mt5_timeframe = getattr(mt5, f"TIMEFRAME_{timeframe}")
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            if rates is None or len(rates) == 0:
                logging.error(f"Failed to get price data for {symbol}. Error: {mt5.last_error()}")
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            logging.error(f"An exception occurred while getting price data for {symbol}: {e}")
            return None

    def get_data_by_range(self, symbol, timeframe, start_date, end_date):
        """
        Fetches historical price data for a specific date range.
        """
        if not self.connection_status:
            self.connect()

        try:
            mt5_timeframe = getattr(mt5, f"TIMEFRAME_{timeframe}")
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            if rates is None or len(rates) == 0:
                logging.error(f"Failed to get price data for {symbol} in range. Error: {mt5.last_error()}")
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            logging.error(f"An exception occurred while getting date range data for {symbol}: {e}")
            return None

    def get_multiple_timeframes_by_range(self, symbol, timeframes, start_date, end_date):
        """
        Fetches historical price data for multiple timeframes over a specific date range.
        """
        all_data = {}
        if not self.connection_status:
            self.connect() # Attempt to connect if not already
            if not self.connection_status:
                logging.error(f"Not connected to MT5. Cannot get multi-timeframe data for {symbol}.")
                return None

        for tf_str in timeframes:
            tf_key = f"TIMEFRAME_{tf_str}"
            if not hasattr(mt5, tf_key):
                logging.error(f"Invalid timeframe '{tf_str}' in config.")
                return None
            
            logging.info(f"Fetching {tf_str} data for {symbol}...")
            df = self.get_data_by_range(symbol, tf_str, start_date, end_date)
            if df is None or df.empty:
                logging.error(f"Failed to get {tf_str} data for {symbol}. Aborting.")
                return None
            all_data[tf_str] = df
        
        return all_data

    def get_account_info(self):
        """
        Fetches key account metrics.
        """
        if not self.connection_status:
            return None
        try:
            info = mt5.account_info()
            if info:
                return {
                    "equity": info.equity,
                    "balance": info.balance,
                    "margin": info.margin,
                    "margin_level": info.margin_level
                }
        except Exception as e:
            logging.error(f"Error getting account info: {e}")
        return None
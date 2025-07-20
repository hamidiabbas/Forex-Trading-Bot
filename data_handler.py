"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           data_handler.py
 *
 * PURPOSE:
 *
 * This module is the sole gateway for all interactions with the
 * MetaTrader 5 (MT5) trading platform. It encapsulates all the
 * complexities of the MT5 API, providing a clean, robust, and
 * fault-tolerant interface for the rest of the application. Its
 * responsibilities include managing the connection lifecycle, fetching
 * all market and account data, and performing initial data validation.
 * This abstraction is critical for creating a modular and testable
 * system, ensuring that any changes to the broker's API or connection
 * logic are isolated to this single module.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             4.0
 *
 ******************************************************************************/
"""

import MetaTrader5 as mt5
import pandas as pd
import time
import logging
from datetime import datetime, timedelta


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

        Args:
            config: The configuration object with all necessary parameters.
        """
        self.config = config
        self.connection_status = False

    def connect(self):
        """
        Initializes the connection to the MT5 terminal with retry logic.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
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
        self.connection_status = False
        return False

    def disconnect(self):
        """
        Properly terminates the MT5 connection.
        """
        logging.info("Disconnecting from MetaTrader 5.")
        mt5.shutdown()
        self.connection_status = False

    def check_connection(self):
        """
        Periodically verifies the connection status and attempts to reconnect if lost.
        """
        if not self.connection_status:
            logging.warning("Connection to MT5 lost. Attempting to reconnect...")
            self.connect()
        else:
            if mt5.terminal_info() is None:
                logging.warning("Connection to MT5 lost. Attempting to reconnect...")
                self.connection_status = False
                self.connect()

    def get_account_info(self):
        """
        Retrieves key account metrics.

        Returns:
            dict: A dictionary with account information or None if it fails.
        """
        if not self.connection_status:
            logging.error("Not connected to MT5. Cannot get account info.")
            return None

        try:
            account_info = mt5.account_info()
            if account_info is not None:
                return {
                    "equity": account_info.equity,
                    "balance": account_info.balance,
                    "margin": account_info.margin,
                    "margin_level": account_info.margin_level
                }
            else:
                logging.error(f"Failed to get account info. Error: {mt5.last_error()}")
                return None
        except Exception as e:
            logging.error(f"An exception occurred while getting account info: {e}")
            return None

    def get_price_data(self, symbol, timeframe, count):
        """
        Fetches historical price data and performs validation.

        Args:
            symbol (str): The currency pair to fetch data for.
            timeframe (str): The timeframe of the data (e.g., 'H1').
            count (int): The number of candles to fetch.

        Returns:
            pd.DataFrame: A validated pandas DataFrame with price data, or None if it fails.
        """
        if not self.connection_status:
            logging.error(f"Not connected to MT5. Cannot get price data for {symbol}.")
            return None

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
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            if self._validate_price_data(df, symbol):
                return df
            else:
                return None

        except Exception as e:
            logging.error(f"An exception occurred while getting price data for {symbol}: {e}")
            return None

    def _validate_price_data(self, df, symbol):
        """
        Performs sanity checks on the price data.

        Args:
            df (pd.DataFrame): The price data DataFrame.
            symbol (str): The symbol for which the data is being validated.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        if df.empty:
            logging.warning(f"Price data for {symbol} is empty.")
            return False

        if (df['High'] < df['Low']).any():
            logging.warning(f"Corrupted data for {symbol}: High price is lower than Low price.")
            return False

        if (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
            logging.warning(f"Zero or negative value candles found for {symbol}.")
            return False

        # Calculate the difference between consecutive timestamps
        time_diff = df.index.to_series().diff().dropna()
        
        # Check for gaps on weekdays. The weekday check must be aligned with the time_diff series.
        # We slice the weekday check with [1:] to match the length of time_diff (which is len(df)-1).
        weekday_check = df.index.dayofweek.isin([0,1,2,3,4])[1:]

        # --- THIS IS THE CORRECTED LINE ---
        if ((time_diff > timedelta(hours=24)) & weekday_check).any():
            logging.warning(f"Potential data gap found for {symbol}.")

        return True
    # Add this new method to your data_handler.py file
    
    def get_data_by_range(self, symbol, timeframe, start_date, end_date):
        """
        Fetches historical price data for a specific date range.

        Args:
            symbol (str): The currency pair to fetch data for.
            timeframe (str): The timeframe of the data (e.g., 'H1').
            start_date (datetime): The start date of the period.
            end_date (datetime): The end date of the period.

        Returns:
            pd.DataFrame: A validated pandas DataFrame with price data, or None if fails.
        """
        if not self.connection_status:
            logging.error(f"Not connected to MT5. Cannot get price data for {symbol}.")
            return None

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
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            if self._validate_price_data(df, symbol):
                return df
            else:
                return None

        except Exception as e:
            logging.error(f"An exception occurred while getting date range data for {symbol}: {e}")
            return None
# enhanced_datahandler.py - Production Ready DataHandler
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import pickle
import hashlib
import threading
import queue
from pathlib import Path
import time
import os

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedDataHandler:
    """Enhanced Data Handler with caching, validation, and multiple sources"""
    
    def __init__(self, user_config):
        self.user_config = user_config
        self.connected = False
        self.cache_dir = Path('cache/market_data')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache = {}
        self.connection_lock = threading.Lock()
        
        # MT5 Configuration
        if MT5_AVAILABLE:
            self.mt5_login = getattr(user_config, 'MT5_LOGIN', 5038274604)
            self.mt5_password = getattr(user_config, 'MT5_PASSWORD', 'yourpassword')
            self.mt5_server = getattr(user_config, 'MT5_SERVER', 'MetaQuotes-Demo')
            self.mt5_path = getattr(user_config, 'MT5_PATH', r'C:\\Program Files\\MetaTrader 5\\terminal64.exe')
            self.mt5_timeout = getattr(user_config, 'MT5_TIMEOUT', 10000)
            
            # Timeframe mappings
            self.timeframes = {
                'M1': mt5.TIMEFRAME_M1, 'M2': mt5.TIMEFRAME_M2, 'M3': mt5.TIMEFRAME_M3,
                'M4': mt5.TIMEFRAME_M4, 'M5': mt5.TIMEFRAME_M5, 'M6': mt5.TIMEFRAME_M6,
                'M10': mt5.TIMEFRAME_M10, 'M12': mt5.TIMEFRAME_M12, 'M15': mt5.TIMEFRAME_M15,
                'M20': mt5.TIMEFRAME_M20, 'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1, 'H2': mt5.TIMEFRAME_H2, 'H3': mt5.TIMEFRAME_H3,
                'H4': mt5.TIMEFRAME_H4, 'H6': mt5.TIMEFRAME_H6, 'H8': mt5.TIMEFRAME_H8,
                'H12': mt5.TIMEFRAME_H12, 'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1,
                'MN1': mt5.TIMEFRAME_MN1
            }
        
        # Data source priority
        self.data_sources = []
        if MT5_AVAILABLE:
            self.data_sources.append('MT5')
        if YFINANCE_AVAILABLE:
            self.data_sources.append('YFINANCE')
        
        # Cache settings
        self.cache_hours = getattr(user_config, 'DATA_CACHE_HOURS', 24)
        self.enable_cache = True
        
        logger.info(f"Enhanced DataHandler initialized")
        logger.info(f"   Available sources: {', '.join(self.data_sources)}")
    
    def connect(self) -> bool:
        """Connect to MT5"""
        try:
            if not MT5_AVAILABLE:
                logger.error("MT5 not available")
                return False
            
            if not mt5.initialize(path=self.mt5_path):
                error_code = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error_code}")
                return False
            
            if not mt5.login(self.mt5_login, password=self.mt5_password, server=self.mt5_server):
                error_code = mt5.last_error()
                logger.error(f"MT5 login failed: {error_code}")
                return False
            
            self.connected = True
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"✅ Connected to MT5 - Account: {account_info.login}")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def get_historical_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical data with enhanced error handling"""
        try:
            if not self.connected and not self.connect():
                logger.error("Failed to connect to data source")
                return pd.DataFrame()
            
            # Get data from MT5
            mt5_timeframe = self.timeframes.get(timeframe, mt5.TIMEFRAME_H1)
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Rename columns
            df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            # Clean data
            df = self.validate_and_clean_data(df, symbol)
            
            logger.info(f"✅ Retrieved {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def validate_and_clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Comprehensive data validation and cleaning"""
        try:
            if data.empty:
                return data
            
            # Remove invalid OHLC data
            invalid_mask = (
                (data['High'] < data['Low']) |
                (data['High'] < data['Open']) |
                (data['High'] < data['Close']) |
                (data['Low'] > data['Open']) |
                (data['Low'] > data['Close']) |
                (data[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
            )
            
            if invalid_mask.any():
                logger.warning(f"Removed {invalid_mask.sum()} invalid bars for {symbol}")
                data = data[~invalid_mask]
            
            # Handle missing values
            data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            if 'Volume' not in data.columns:
                data['Volume'] = 1000000
            
            data['Volume'] = data['Volume'].fillna(1000000)
            
            return data
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return data
    
    def disconnect(self):
        """Disconnect from MT5"""
        try:
            if self.connected and MT5_AVAILABLE:
                mt5.shutdown()
                self.connected = False
                logger.info("Disconnected from MT5")
        except Exception as e:
            logger.error(f"Disconnect error: {e}")

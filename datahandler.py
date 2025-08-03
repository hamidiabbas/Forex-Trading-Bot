# datahandler.py - Complete MT5 Integration
"""
Professional DataHandler with MetaTrader 5 Integration
Direct connection to MT5 terminal for reliable historical data
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Any
import time

logger = logging.getLogger(__name__)

class DataHandler:
    """Professional MT5 Data Handler for Forex Trading"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.connected = False
        self.connection_attempts = 0
        
        # MT5 Configuration
        self.mt5_login = getattr(config, 'MT5_LOGIN', 5038274604)
        self.mt5_password = getattr(config, 'MT5_PASSWORD', 'yourpassword')
        self.mt5_server = getattr(config, 'MT5_SERVER', 'MetaQuotes-Demo')
        self.mt5_path = getattr(config, 'MT5_PATH', r'C:\Program Files\MetaTrader 5\terminal64.exe')
        
        # Data settings
        self.max_bars_per_request = getattr(config, 'MAX_BARS_PER_REQUEST', 10000)
        self.data_timeout = getattr(config, 'DATA_TIMEOUT', 30)
        
        # Timeframe mappings
        self.timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1
        }
        
        # Multi-timeframe structure for analysis
        self.analysis_timeframes = {
            'EXECUTION': 'H1',    # Main execution timeframe
            'SIGNAL': 'H4',       # Signal confirmation
            'BIAS': 'D1',         # Market bias/trend
            'CONTEXT': 'W1'       # Long-term context
        }
        
        logger.info("MT5 DataHandler initialized")
    
    def connect(self) -> bool:
        """Establish connection to MetaTrader 5"""
        try:
            self.connection_attempts += 1
            logger.info(f"MT5 connection attempt {self.connection_attempts}...")
            
            # Initialize MT5 connection
            if not mt5.initialize(path=self.mt5_path):
                error_code = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error_code}")
                return False
            
            # Login to account
            if not mt5.login(self.mt5_login, password=self.mt5_password, server=self.mt5_server):
                error_code = mt5.last_error()
                logger.error(f"MT5 login failed: {error_code}")
                return False
            
            # Verify connection with account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account information")
                return False
            
            self.connected = True
            logger.info(f"✅ Connected to MT5 successfully!")
            logger.info(f"Account: {account_info.login}")
            logger.info(f"Server: {account_info.server}")
            logger.info(f"Balance: ${account_info.balance:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from MetaTrader 5"""
        try:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                logger.info("Disconnected from MetaTrader 5")
        except Exception as e:
            logger.error(f"Error disconnecting from MT5: {e}")
    
    def get_data_by_range(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get historical data for a specific date range - Perfect for backtesting"""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            # Convert timeframe string to MT5 constant
            if timeframe not in self.timeframes:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            mt5_timeframe = self.timeframes[timeframe]
            
            logger.info(f"Fetching {symbol} {timeframe} data from {start_date.date()} to {end_date.date()}")
            
            # Get rates for the specified range
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data for {symbol} {timeframe} from {start_date} to {end_date}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Rename columns to standard format
            df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            # Clean data
            if 'spread' in df.columns:
                df.drop('spread', axis=1, inplace=True)
            
            # Ensure proper data types
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            # Remove any NaN rows
            df.dropna(inplace=True)
            
            if df.empty:
                logger.warning(f"Empty DataFrame after processing for {symbol} {timeframe}")
                return None
            
            logger.info(f"✅ Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol} {timeframe}: {e}")
            return None
    
    def get_data(self, symbol: str, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """Get recent data for a specific symbol and timeframe"""
        if not self.connected:
            return None
        
        try:
            # Convert timeframe string to MT5 constant
            if timeframe not in self.timeframes:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            mt5_timeframe = self.timeframes[timeframe]
            
            # Get rates from MT5
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, min(bars, self.max_bars_per_request))
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No rates data for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Rename columns to standard format
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low', 
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            # Remove spread column if exists
            if 'spread' in df.columns:
                df.drop('spread', axis=1, inplace=True)
            
            # Ensure proper data types
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            # Remove any NaN rows
            df.dropna(inplace=True)
            
            if df.empty:
                logger.warning(f"Empty DataFrame after processing for {symbol} {timeframe}")
                return None
            
            logger.debug(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol} {timeframe}: {e}")
            return None
    
    def get_multi_timeframe_data(self, symbol: str, bars: int = 500) -> Optional[Dict[str, pd.DataFrame]]:
        """Get multi-timeframe data for comprehensive market analysis"""
        if not self.connected:
            logger.warning("Not connected to MT5, attempting to connect...")
            if not self.connect():
                return None
        
        try:
            data_dict = {}
            
            # Get data for each analysis timeframe
            for analysis_type, timeframe_str in self.analysis_timeframes.items():
                try:
                    # Get raw OHLCV data
                    df = self.get_data(symbol, timeframe_str, bars)
                    if df is not None and not df.empty:
                        # Add basic technical indicators for analysis
                        df_analyzed = self.add_basic_indicators(df)
                        data_dict[analysis_type] = df_analyzed
                        logger.debug(f"Retrieved {len(df_analyzed)} bars for {symbol} {analysis_type} ({timeframe_str})")
                    else:
                        logger.warning(f"No data retrieved for {symbol} {analysis_type} ({timeframe_str})")
                except Exception as tf_error:
                    logger.error(f"Error getting {analysis_type} data for {symbol}: {tf_error}")
                    continue
            
            # Ensure we have at least execution timeframe data
            if 'EXECUTION' not in data_dict:
                logger.error(f"Failed to get execution timeframe data for {symbol}")
                return None
            
            # Fill missing timeframes with execution data if needed
            for analysis_type in self.analysis_timeframes.keys():
                if analysis_type not in data_dict:
                    logger.warning(f"Using EXECUTION data for missing {analysis_type} timeframe")
                    data_dict[analysis_type] = data_dict['EXECUTION'].copy()
            
            logger.info(f"✅ Multi-timeframe data retrieved for {symbol}: {list(data_dict.keys())}")
            return data_dict
            
        except Exception as e:
            logger.error(f"Error getting multi-timeframe data for {symbol}: {e}")
            return None
    
    def add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to the data"""
        try:
            df_enhanced = df.copy()
            
            # Basic moving averages
            df_enhanced['SMA_20'] = df_enhanced['Close'].rolling(20).mean()
            df_enhanced['SMA_50'] = df_enhanced['Close'].rolling(50).mean()
            df_enhanced['EMA_20'] = df_enhanced['Close'].ewm(span=20).mean()
            df_enhanced['EMA_50'] = df_enhanced['Close'].ewm(span=50).mean()
            
            # RSI
            delta = df_enhanced['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df_enhanced['RSI_14'] = 100 - (100 / (1 + rs))
            
            # ATR
            high_low = df_enhanced['High'] - df_enhanced['Low']
            high_close = np.abs(df_enhanced['High'] - df_enhanced['Close'].shift())
            low_close = np.abs(df_enhanced['Low'] - df_enhanced['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df_enhanced['ATR_14'] = true_range.rolling(14).mean()
            df_enhanced['ATRr_14'] = df_enhanced['ATR_14'] / df_enhanced['Close']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2.0
            sma = df_enhanced['Close'].rolling(bb_period).mean()
            std = df_enhanced['Close'].rolling(bb_period).std()
            df_enhanced['BBU_20_2.0'] = sma + (std * bb_std)
            df_enhanced['BBL_20_2.0'] = sma - (std * bb_std)
            df_enhanced['BBM_20_2.0'] = sma
            
            # MACD
            ema12 = df_enhanced['Close'].ewm(span=12).mean()
            ema26 = df_enhanced['Close'].ewm(span=26).mean()
            df_enhanced['MACD_12_26_9'] = ema12 - ema26
            df_enhanced['MACDs_12_26_9'] = df_enhanced['MACD_12_26_9'].ewm(span=9).mean()
            
            # Price action features
            df_enhanced['High_Low_Ratio'] = df_enhanced['High'] / df_enhanced['Low']
            df_enhanced['Close_Open_Ratio'] = df_enhanced['Close'] / df_enhanced['Open']
            df_enhanced['Price_Range'] = (df_enhanced['High'] - df_enhanced['Low']) / df_enhanced['Close']
            
            # Volume features
            df_enhanced['Volume_SMA'] = df_enhanced['Volume'].rolling(20).mean()
            df_enhanced['Volume_Ratio'] = df_enhanced['Volume'] / df_enhanced['Volume_SMA']
            
            # Volatility and momentum features
            df_enhanced['Returns'] = df_enhanced['Close'].pct_change()
            df_enhanced['Volatility_20'] = df_enhanced['Returns'].rolling(20).std()
            df_enhanced['Momentum_10'] = df_enhanced['Close'].pct_change(10)
            
            # Clean NaN values
            df_enhanced = df_enhanced.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            logger.debug(f"Added basic indicators. Shape: {df_enhanced.shape}")
            return df_enhanced
            
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return df
    
    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current price information for a symbol"""
        try:
            if not self.connected:
                return None
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return {
                'bid': float(tick.bid),
                'ask': float(tick.ask),
                'spread': float(tick.ask - tick.bid),
                'time': datetime.fromtimestamp(tick.time)
            }
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information"""
        try:
            if not self.connected:
                return None
            
            info = mt5.symbol_info(symbol)
            if info is None:
                return None
            
            return {
                'symbol': info.name,
                'digits': info.digits,
                'point': info.point,
                'spread': info.spread,
                'contract_size': info.trade_contract_size,
                'volume_min': info.volume_min,
                'volume_max': info.volume_max,
                'volume_step': info.volume_step
            }
            
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test MT5 connection"""
        try:
            if not self.connected:
                return self.connect()
            
            # Test with a simple query
            account_info = mt5.account_info()
            return account_info is not None
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

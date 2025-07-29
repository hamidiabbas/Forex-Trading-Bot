"""
Enhanced Data Handler with Professional MT5 Integration
Provides caching, validation, and multi-timeframe data management
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import pytz
from typing import Dict, Optional, Union, List, Any
import time
from pathlib import Path
import pickle
import threading

class EnhancedDataHandler:
    """
    Professional data handler with caching and advanced features
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connected = False
        self.connection_lock = threading.Lock()
        
        # Configuration
        self.max_bars = config.get('data.max_bars_per_request', 10000)
        self.timeout = config.get('data.timeout', 30)
        self.cache_enabled = config.get('data.enable_cache', True)
        self.cache_duration = config.get('data.cache_duration_seconds', 60)
        
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
        
        # Multi-timeframe structure
        self.analysis_timeframes = {
            'EXECUTION': 'H1',
            'SIGNAL': 'H4', 
            'BIAS': 'D1',
            'CONTEXT': 'W1'
        }
        
        # Data cache
        self.data_cache = {}
        self.cache_timestamps = {}
        
        # Performance tracking
        self.requests_made = 0
        self.cache_hits = 0
        self.connection_attempts = 0
        
        self.logger.info("EnhancedDataHandler initialized")

    def connect(self) -> bool:
        """Enhanced MT5 connection with retry logic"""
        with self.connection_lock:
            try:
                self.connection_attempts += 1
                self.logger.info(f"MT5 connection attempt {self.connection_attempts}...")
                
                if not mt5.initialize():
                    error_code = mt5.last_error()
                    self.logger.error(f"MT5 initialization failed: {error_code}")
                    return False
                
                # Test connection
                account_info = mt5.account_info()
                if account_info is None:
                    self.logger.error("Failed to get account information")
                    return False
                
                self.connected = True
                self.logger.info("✅ Enhanced MT5 connection established successfully")
                self.logger.info(f"Account: {account_info.login}")
                self.logger.info(f"Balance: ${account_info.balance:.2f}")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Enhanced MT5 connection error: {e}")
                return False

    def disconnect(self) -> None:
        """Enhanced disconnect with cleanup"""
        try:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                
                # Log performance statistics
                total_requests = self.requests_made
                cache_hit_rate = (self.cache_hits / max(1, total_requests)) * 100
                
                self.logger.info("Disconnecting from MetaTrader 5")
                self.logger.info(f"Session Statistics:")
                self.logger.info(f"  Total Requests: {total_requests}")
                self.logger.info(f"  Cache Hits: {self.cache_hits}")
                self.logger.info(f"  Cache Hit Rate: {cache_hit_rate:.1f}%")
                
                # Clear cache
                self.data_cache.clear()
                self.cache_timestamps.clear()
                
        except Exception as e:
            self.logger.error(f"Error during enhanced disconnect: {e}")

    def get_multi_timeframe_data(self, symbol: str, bars: int = 500) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Enhanced multi-timeframe data retrieval with caching
        """
        try:
            if not self.connected:
                if not self.connect():
                    return None
            
            cache_key = f"{symbol}_multi_{bars}"
            
            # Check cache first
            if self._is_cache_valid(cache_key):
                self.cache_hits += 1
                self.logger.debug(f"Cache hit for {symbol} multi-timeframe data")
                return self.data_cache[cache_key]
            
            self.requests_made += 1
            data_dict = {}
            
            # Get data for each timeframe
            for analysis_type, timeframe_str in self.analysis_timeframes.items():
                try:
                    df = self.get_data(symbol, timeframe_str, bars)
                    if df is not None and not df.empty:
                        # Add basic indicators
                        df_enhanced = self._add_enhanced_indicators(df)
                        data_dict[analysis_type] = df_enhanced
                        
                        self.logger.debug(f"Retrieved {len(df_enhanced)} bars for {symbol} {analysis_type}")
                    else:
                        self.logger.warning(f"No data for {symbol} {analysis_type}")
                        
                except Exception as tf_error:
                    self.logger.error(f"Error getting {analysis_type} data: {tf_error}")
                    continue
            
            # Validate we have minimum required data
            if 'EXECUTION' not in data_dict:
                self.logger.error(f"Failed to get execution timeframe data for {symbol}")
                return None
            
            # Fill missing timeframes
            for analysis_type in self.analysis_timeframes.keys():
                if analysis_type not in data_dict:
                    self.logger.warning(f"Using EXECUTION data for missing {analysis_type}")
                    data_dict[analysis_type] = data_dict['EXECUTION'].copy()
            
            # Cache the result
            if self.cache_enabled:
                self.data_cache[cache_key] = data_dict
                self.cache_timestamps[cache_key] = time.time()
            
            self.logger.debug(f"Multi-timeframe data retrieved for {symbol}: {list(data_dict.keys())}")
            return data_dict
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced multi-timeframe data for {symbol}: {e}")
            return None

    def get_data(self, symbol: str, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """Enhanced single timeframe data retrieval"""
        try:
            if not self.connected:
                return None
            
            # Check cache
            cache_key = f"{symbol}_{timeframe}_{bars}"
            if self._is_cache_valid(cache_key):
                self.cache_hits += 1
                return self.data_cache[cache_key]
            
            self.requests_made += 1
            
            # Validate timeframe
            if timeframe not in self.timeframes:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            mt5_timeframe = self.timeframes[timeframe]
            
            # Get rates with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, min(bars, self.max_bars))
                    if rates is not None and len(rates) > 0:
                        break
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    else:
                        return None
            
            if rates is None or len(rates) == 0:
                self.logger.warning(f"No rates data for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Standardize column names
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
            
            # Ensure numeric types
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            # Remove NaN rows
            df.dropna(inplace=True)
            
            if df.empty:
                self.logger.warning(f"Empty DataFrame after processing for {symbol} {timeframe}")
                return None
            
            # Cache result
            if self.cache_enabled:
                self.data_cache[cache_key] = df
                self.cache_timestamps[cache_key] = time.time()
            
            self.logger.debug(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced data for {symbol} {timeframe}: {e}")
            return None

    def get_data_by_range(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Enhanced historical data retrieval"""
        try:
            if not self.connected:
                if not self.connect():
                    return None
            
            if timeframe not in self.timeframes:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            mt5_timeframe = self.timeframes[timeframe]
            
            # Get historical rates
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                self.logger.warning(f"No historical data for {symbol} {timeframe}")
                return None
            
            # Process same as get_data
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close', 
                'tick_volume': 'Volume'
            }, inplace=True)
            
            if 'spread' in df.columns:
                df.drop('spread', axis=1, inplace=True)
            
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            df.dropna(inplace=True)
            
            self.logger.info(f"Retrieved {len(df)} historical bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None

    def _add_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced technical indicators"""
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
            rs = gain / loss
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
            df_enhanced['MACDh_12_26_9'] = df_enhanced['MACD_12_26_9'] - df_enhanced['MACDs_12_26_9']
            
            # Forward fill NaN values
            df_enhanced = df_enhanced.ffill().fillna(0)
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"Error adding enhanced indicators: {e}")
            return df

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if not self.cache_enabled:
            return False
        
        if cache_key not in self.data_cache:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        age = time.time() - self.cache_timestamps[cache_key]
        return age < self.cache_duration

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Enhanced current price retrieval"""
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
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Enhanced symbol information"""
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
                'volume_step': info.volume_step,
                'currency_base': info.currency_base,
                'currency_profit': info.currency_profit,
                'currency_margin': info.currency_margin
            }
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    def is_market_open(self, symbol: str) -> bool:
        """Enhanced market status check"""
        try:
            if not self.connected:
                return False
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False
            
            # Check if recent tick (within 2 minutes)
            current_time = datetime.now().timestamp()
            return (current_time - tick.time) < 120
            
        except Exception as e:
            self.logger.error(f"Error checking market status for {symbol}: {e}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get data handler performance statistics"""
        total_requests = max(1, self.requests_made)
        cache_hit_rate = (self.cache_hits / total_requests) * 100
        
        return {
            'connected': self.connected,
            'total_requests': self.requests_made,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'cached_items': len(self.data_cache),
            'connection_attempts': self.connection_attempts,
            'cache_enabled': self.cache_enabled
        }

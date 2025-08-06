"""
Enhanced Data Handler with Advanced Features
Complete implementation for ForexBot live trading and backtesting
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import pytz
from typing import Dict, Optional, Union, List, Tuple
import time
import threading
from concurrent.futures import ThreadPoolExecutor


class DataHandler:
    """
    Basic DataHandler for backward compatibility
    """
    
    def __init__(self, config):
        """Initialize basic DataHandler"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connected = False
        
        # MT5 Timeframe mappings
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
        
        self.logger.info("Basic DataHandler initialized")

    def connect(self) -> bool:
        """Connect to MetaTrader 5"""
        try:
            if not mt5.initialize():
                return False
            self.connected = True
            return True
        except:
            return False

    def disconnect(self) -> None:
        """Disconnect from MetaTrader 5"""
        try:
            if self.connected:
                mt5.shutdown()
                self.connected = False
        except:
            pass

    def get_data_by_range(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get historical data by date range"""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            mt5_timeframe = self.timeframes.get(timeframe, mt5.TIMEFRAME_H1)
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'tick_volume': 'Volume'
            }, inplace=True)
            
            if 'spread' in df.columns:
                df.drop('spread', axis=1, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting data: {e}")
            return None


class EnhancedDataHandler:
    """
    Enhanced DataHandler with advanced features for live trading and backtesting
    """
    
    def __init__(self, config):
        """Initialize Enhanced DataHandler with advanced features"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connected = False
        self.connection_attempts = 0
        self.connection_lock = threading.Lock()
        
        # Enhanced configuration
        self.max_bars = getattr(config, 'MAX_BARS_PER_REQUEST', 100000)
        self.timeout = getattr(config, 'DATA_TIMEOUT', 30)
        self.retry_attempts = getattr(config, 'CONNECTION_RETRY_ATTEMPTS', 3)
        self.retry_delay = getattr(config, 'CONNECTION_RETRY_DELAY', 2)
        
        # MT5 Timeframe mappings
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
            'EXECUTION': getattr(config, 'EXECUTION_TIMEFRAME', 'H1'),
            'SIGNAL': getattr(config, 'SIGNAL_TIMEFRAME', 'H4'),
            'BIAS': getattr(config, 'BIAS_TIMEFRAME', 'D1'),
            'CONTEXT': getattr(config, 'CONTEXT_TIMEFRAME', 'W1')
        }
        
        # Data caching
        self.data_cache = {}
        self.cache_expiry = getattr(config, 'DATA_CACHE_EXPIRY_MINUTES', 5)
        
        # Performance monitoring
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
        
        self.logger.info("✅ Enhanced DataHandler initialized successfully")

    def connect(self) -> bool:
        """Enhanced connection with retry logic and monitoring"""
        with self.connection_lock:
            if self.connected:
                return True
                
            for attempt in range(self.retry_attempts):
                try:
                    self.connection_attempts += 1
                    self.logger.info(f"🔌 MT5 connection attempt #{self.connection_attempts} (retry {attempt + 1})")
                    
                    if not mt5.initialize():
                        error_code = mt5.last_error()
                        self.logger.error(f"❌ MT5 initialization failed: {error_code}")
                        if attempt < self.retry_attempts - 1:
                            time.sleep(self.retry_delay)
                        continue
                    
                    # Verify connection
                    account_info = mt5.account_info()
                    if account_info is None:
                        self.logger.error("❌ Failed to get account information")
                        mt5.shutdown()
                        if attempt < self.retry_attempts - 1:
                            time.sleep(self.retry_delay)
                        continue
                    
                    self.connected = True
                    self.logger.info(f"✅ MetaTrader 5 connected successfully (Account: {account_info.login})")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"❌ MT5 connection error (attempt {attempt + 1}): {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
            
            self.error_count += 1
            return False

    def disconnect(self) -> None:
        """Enhanced disconnection with cleanup"""
        with self.connection_lock:
            try:
                if self.connected:
                    mt5.shutdown()
                    self.connected = False
                    self.logger.info("🔌 Disconnected from MetaTrader 5")
                    
                # Clear cache on disconnect
                self.data_cache.clear()
                
            except Exception as e:
                self.logger.error(f"❌ Error during disconnect: {e}")
                self.connected = False

    def get_current_data(self, symbol: str, timeframe: str = 'H1', bars: int = 500) -> Optional[pd.DataFrame]:
        """Get current market data with caching"""
        cache_key = f"{symbol}_{timeframe}_{bars}_current"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"📁 Using cached data for {symbol} {timeframe}")
            return self.data_cache[cache_key]['data'].copy()
        
        # Fetch fresh data
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            self.request_count += 1
            self.last_request_time = datetime.now()
            
            mt5_timeframe = self.timeframes.get(timeframe, mt5.TIMEFRAME_H1)
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, min(bars, self.max_bars))
            
            if rates is None or len(rates) == 0:
                self.error_count += 1
                self.logger.warning(f"⚠️ No data returned for {symbol} {timeframe}")
                return None
            
            df = self._process_raw_data(rates)
            if df is not None:
                # Cache the data
                self.data_cache[cache_key] = {
                    'data': df.copy(),
                    'timestamp': datetime.now()
                }
                
                self.logger.debug(f"📊 Retrieved {len(df)} bars for {symbol} {timeframe}")
            
            return df
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Error getting current data for {symbol} {timeframe}: {e}")
            return None

    def get_data_by_range(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Enhanced historical data retrieval with caching"""
        cache_key = f"{symbol}_{timeframe}_{start_date.isoformat()}_{end_date.isoformat()}"
        
        # Check cache for historical data (longer cache time)
        if cache_key in self.data_cache:
            cache_age = (datetime.now() - self.data_cache[cache_key]['timestamp']).total_seconds() / 60
            if cache_age < 60:  # Cache historical data for 1 hour
                self.logger.debug(f"📁 Using cached historical data for {symbol} {timeframe}")
                return self.data_cache[cache_key]['data'].copy()
        
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            self.request_count += 1
            self.last_request_time = datetime.now()
            
            mt5_timeframe = self.timeframes.get(timeframe, mt5.TIMEFRAME_H1)
            self.logger.info(f"📈 Retrieving {symbol} {timeframe} data from {start_date} to {end_date}")
            
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                self.error_count += 1
                error_info = mt5.last_error()
                self.logger.warning(f"⚠️ No historical data for {symbol} {timeframe}. MT5 Error: {error_info}")
                return None
            
            df = self._process_raw_data(rates)
            if df is not None:
                # Cache historical data
                self.data_cache[cache_key] = {
                    'data': df.copy(),
                    'timestamp': datetime.now()
                }
                
                self.logger.info(f"✅ Retrieved {len(df)} historical bars for {symbol} {timeframe}")
            
            return df
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Error getting historical data for {symbol} {timeframe}: {e}")
            return None

    def get_multi_timeframe_data(self, symbol: str, bars: int = 500) -> Optional[Dict[str, pd.DataFrame]]:
        """Enhanced multi-timeframe data retrieval with parallel processing"""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            self.logger.debug(f"📊 Getting multi-timeframe data for {symbol}")
            
            # Use ThreadPoolExecutor for parallel data retrieval
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_timeframe = {}
                
                for analysis_type, timeframe_str in self.analysis_timeframes.items():
                    future = executor.submit(self.get_current_data, symbol, timeframe_str, bars)
                    future_to_timeframe[future] = analysis_type
                
                data_dict = {}
                for future in future_to_timeframe:
                    analysis_type = future_to_timeframe[future]
                    try:
                        df = future.result(timeout=self.timeout)
                        if df is not None:
                            # Add technical indicators
                            df_analyzed = self._add_enhanced_indicators(df)
                            data_dict[analysis_type] = df_analyzed
                    except Exception as e:
                        self.logger.error(f"❌ Error getting {analysis_type} data: {e}")
            
            # Ensure we have at least execution timeframe data
            if 'EXECUTION' not in data_dict:
                self.logger.error("❌ Failed to get execution timeframe data")
                return None
            
            # Fill missing timeframes with execution data
            for analysis_type in self.analysis_timeframes.keys():
                if analysis_type not in data_dict:
                    self.logger.warning(f"⚠️ Using EXECUTION data for missing {analysis_type} timeframe")
                    data_dict[analysis_type] = data_dict['EXECUTION'].copy()
            
            self.logger.info(f"✅ Multi-timeframe data retrieved for {symbol}")
            return data_dict
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Error getting multi-timeframe data: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Enhanced current price retrieval with validation"""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.warning(f"⚠️ No tick data for {symbol}")
                return None
            
            # Use bid price for consistency
            return tick.bid
            
        except Exception as e:
            self.logger.error(f"❌ Error getting current price for {symbol}: {e}")
            return None

    def validate_symbol(self, symbol: str) -> bool:
        """Enhanced symbol validation with detailed info"""
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.warning(f"⚠️ Symbol {symbol} not found")
                return False
            
            if not symbol_info.visible:
                self.logger.warning(f"⚠️ Symbol {symbol} not visible in Market Watch")
                return False
            
            self.logger.debug(f"✅ Symbol {symbol} validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating symbol {symbol}: {e}")
            return False

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Enhanced symbol information retrieval"""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            
            info_dict = {
                'symbol': symbol_info.name,
                'description': symbol_info.description,
                'digits': symbol_info.digits,
                'point': symbol_info.point,
                'spread': symbol_info.spread,
                'min_lot': symbol_info.volume_min,
                'max_lot': symbol_info.volume_max,
                'lot_step': symbol_info.volume_step,
                'contract_size': symbol_info.trade_contract_size,
                'currency_base': symbol_info.currency_base,
                'currency_profit': symbol_info.currency_profit,
                'currency_margin': symbol_info.currency_margin,
                'visible': symbol_info.visible,
                'trade_mode': symbol_info.trade_mode,
                'margin_initial': symbol_info.margin_initial,
                'margin_maintenance': symbol_info.margin_maintenance
            }
            
            return info_dict
            
        except Exception as e:
            self.logger.error(f"❌ Error getting symbol info for {symbol}: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, Union[int, float, str]]:
        """Get data handler performance statistics"""
        uptime = (datetime.now() - self.last_request_time).total_seconds() / 60 if self.last_request_time else 0
        
        return {
            'connected': self.connected,
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate': (self.error_count / max(1, self.request_count)) * 100,
            'connection_attempts': self.connection_attempts,
            'cache_size': len(self.data_cache),
            'uptime_minutes': uptime,
            'last_request': self.last_request_time.isoformat() if self.last_request_time else 'None'
        }

    def _process_raw_data(self, rates) -> Optional[pd.DataFrame]:
        """Enhanced data processing with validation"""
        try:
            if rates is None or len(rates) == 0:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time and set as index
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
            
            # Remove MT5 specific columns
            columns_to_drop = ['spread', 'real_volume']
            for col in columns_to_drop:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)
            
            # Ensure proper data types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Validate OHLC data
            df = self._validate_ohlc_data(df)
            
            # Remove any rows with NaN values
            df.dropna(inplace=True)
            
            if df.empty:
                self.logger.warning("⚠️ DataFrame empty after processing")
                return None
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error processing raw data: {e}")
            return None

    def _validate_ohlc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC data integrity"""
        try:
            if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                return df
            
            # Check for valid OHLC relationships
            valid_mask = (
                (df['High'] >= df['Low']) &
                (df['High'] >= df['Open']) &
                (df['High'] >= df['Close']) &
                (df['Low'] <= df['Open']) &
                (df['Low'] <= df['Close']) &
                (df['Open'] > 0) &
                (df['High'] > 0) &
                (df['Low'] > 0) &
                (df['Close'] > 0)
            )
            
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                self.logger.warning(f"⚠️ Removed {invalid_count} invalid OHLC rows")
                df = df[valid_mask]
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error validating OHLC data: {e}")
            return df

    def _add_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced technical indicators"""
        try:
            df_enhanced = df.copy()
            
            # Enhanced moving averages
            periods = [10, 20, 50, 100]
            for period in periods:
                if len(df_enhanced) >= period:
                    df_enhanced[f'SMA_{period}'] = df_enhanced['Close'].rolling(period).mean()
                    df_enhanced[f'EMA_{period}'] = df_enhanced['Close'].ewm(span=period).mean()
            
            # Enhanced RSI with multiple periods
            rsi_periods = [14, 21]
            for period in rsi_periods:
                df_enhanced[f'RSI_{period}'] = self._calculate_rsi(df_enhanced['Close'], period)
            
            # Enhanced MACD
            ema_12 = df_enhanced['Close'].ewm(span=12).mean()
            ema_26 = df_enhanced['Close'].ewm(span=26).mean()
            df_enhanced['MACD_12_26_9'] = ema_12 - ema_26
            df_enhanced['MACDs_12_26_9'] = df_enhanced['MACD_12_26_9'].ewm(span=9).mean()
            df_enhanced['MACDh_12_26_9'] = df_enhanced['MACD_12_26_9'] - df_enhanced['MACDs_12_26_9']
            
            # Enhanced Bollinger Bands
            for period in [20, 50]:
                bb_middle = df_enhanced['Close'].rolling(period).mean()
                bb_std = df_enhanced['Close'].rolling(period).std()
                df_enhanced[f'BBM_{period}_2.0'] = bb_middle
                df_enhanced[f'BBU_{period}_2.0'] = bb_middle + (bb_std * 2)
                df_enhanced[f'BBL_{period}_2.0'] = bb_middle - (bb_std * 2)
            
            # Enhanced ATR
            atr_periods = [14, 21]
            for period in atr_periods:
                atr = self._calculate_atr(df_enhanced, period)
                df_enhanced[f'ATR_{period}'] = atr
                df_enhanced[f'ATRr_{period}'] = atr / df_enhanced['Close']
            
            # ADX and Directional Movement
            df_enhanced = self._add_adx_indicators(df_enhanced)
            
            # Stochastic Oscillator
            df_enhanced = self._add_stochastic_indicators(df_enhanced)
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"❌ Error adding enhanced indicators: {e}")
            return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with enhanced smoothing"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            
            # Avoid division by zero
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Fill NaN with neutral RSI
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating RSI: {e}")
            return pd.Series(50, index=prices.index)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR with enhanced validation"""
        try:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(period).mean()
            
            return atr.fillna(0.001)  # Fill NaN with small positive value
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating ATR: {e}")
            return pd.Series(0.001, index=df.index)

    def _add_adx_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ADX and Directional Movement indicators"""
        try:
            # Calculate True Range
            tr = self._calculate_atr(df, 1)
            
            # Calculate directional movement
            dm_plus = np.where(
                (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                np.maximum(df['High'] - df['High'].shift(1), 0),
                0
            )
            dm_minus = np.where(
                (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                np.maximum(df['Low'].shift(1) - df['Low'], 0),
                0
            )
            
            # Smooth the values
            period = 14
            alpha = 1.0 / period
            
            tr_smooth = pd.Series(tr).ewm(alpha=alpha).mean()
            dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=alpha).mean()
            dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=alpha).mean()
            
            # Calculate DI+ and DI-
            df['DMP_14'] = 100 * (dm_plus_smooth / tr_smooth)
            df['DMN_14'] = 100 * (dm_minus_smooth / tr_smooth)
            
            # Calculate DX and ADX
            di_sum = df['DMP_14'] + df['DMN_14']
            dx = 100 * np.abs(df['DMP_14'] - df['DMN_14']) / di_sum.replace(0, np.nan)
            df['ADX_14'] = dx.ewm(alpha=alpha).mean()
            
            # Fill NaN values
            df['DMP_14'] = df['DMP_14'].fillna(20)
            df['DMN_14'] = df['DMN_14'].fillna(20)
            df['ADX_14'] = df['ADX_14'].fillna(20)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error adding ADX indicators: {e}")
            # Add default values if calculation fails
            df['ADX_14'] = 20.0
            df['DMP_14'] = 20.0
            df['DMN_14'] = 20.0
            return df

    def _add_stochastic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        try:
            period = 14
            k_smooth = 3
            d_smooth = 3
            
            # Calculate %K
            low_min = df['Low'].rolling(period).min()
            high_max = df['High'].rolling(period).max()
            
            k_raw = 100 * (df['Close'] - low_min) / (high_max - low_min)
            df[f'STOCHk_{period}_{k_smooth}_{d_smooth}'] = k_raw.rolling(k_smooth).mean()
            df[f'STOCHd_{period}_{k_smooth}_{d_smooth}'] = df[f'STOCHk_{period}_{k_smooth}_{d_smooth}'].rolling(d_smooth).mean()
            
            # Fill NaN values
            df[f'STOCHk_{period}_{k_smooth}_{d_smooth}'] = df[f'STOCHk_{period}_{k_smooth}_{d_smooth}'].fillna(50)
            df[f'STOCHd_{period}_{k_smooth}_{d_smooth}'] = df[f'STOCHd_{period}_{k_smooth}_{d_smooth}'].fillna(50)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error adding Stochastic indicators: {e}")
            # Add default values if calculation fails
            df[f'STOCHk_14_3_3'] = 50.0
            df[f'STOCHd_14_3_3'] = 50.0
            return df

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.data_cache:
            return False
        
        cache_age = (datetime.now() - self.data_cache[cache_key]['timestamp']).total_seconds() / 60
        return cache_age < self.cache_expiry

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.data_cache.clear()
        self.logger.info("🗑️ Data cache cleared")

    def __del__(self):
        """Cleanup when DataHandler is destroyed"""
        self.disconnect()

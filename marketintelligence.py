# marketintelligence.py - Complete Enhanced Market Intelligence
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

class EnhancedMarketIntelligence:
    """
    Enhanced Market Intelligence with updated constructor to fix initialization error
    """
    
    def __init__(self, config, datahandler=None, additional_param=None):
        """
        Initialize Enhanced Market Intelligence
        
        Args:
            config: Configuration object or dictionary (primary parameter)
            datahandler: Optional data handler (for backwards compatibility)
            additional_param: Optional additional parameter (for future extensibility)
        """
        self.config = config
        self.datahandler = datahandler  # Optional parameter for backwards compatibility
        self.additional_param = additional_param  # Optional additional parameter
        self.lock = threading.Lock()
        
        # Market intelligence parameters from config
        self.trend_threshold = self._get_config_value('market_intelligence.trend_threshold', 0.7)
        self.volatility_threshold = self._get_config_value('market_intelligence.volatility_threshold', 0.02)
        self.momentum_threshold = self._get_config_value('market_intelligence.momentum_threshold', 0.5)
        
        # Analysis periods from config
        self.trend_period = self._get_config_value('market_intelligence.trend_analysis_period', 50)
        self.volatility_period = self._get_config_value('market_intelligence.volatility_analysis_period', 20)
        self.momentum_period = self._get_config_value('market_intelligence.momentum_analysis_period', 14)
        
        # Strategy parameters from config
        strategy_config = self._get_config_value('strategy', {})
        self.rsi_overbought = strategy_config.get('rsi_overbought', 70)
        self.rsi_oversold = strategy_config.get('rsi_oversold', 30)
        self.bb_threshold = strategy_config.get('bb_threshold', 0.95)
        self.macd_threshold = strategy_config.get('macd_signal_threshold', 0.001)
        
        # Data cache and performance tracking
        self.data_cache = {}
        self.cache_expiry = 300  # 5 minutes
        self.regime_history = []
        self.signal_history = []
        self.pattern_cache = {}
        
        logger.info("Enhanced Market Intelligence initialized successfully")
        logger.info(f"Trend Threshold: {self.trend_threshold}")
        logger.info(f"Analysis Periods: T={self.trend_period}, V={self.volatility_period}, M={self.momentum_period}")
    
    def _get_config_value(self, key: str, default=None):
        """Safely get configuration values with dot notation support"""
        try:
            if isinstance(self.config, dict):
                keys = key.split('.')
                value = self.config
                for k in keys:
                    value = value.get(k, {})
                return value if value != {} else default
            else:
                # If config object has a get method
                return getattr(self.config, 'get', lambda x, d: default)(key, default)
        except Exception:
            return default
    
    def get_market_data(self, symbol: str, timeframe: str = 'M15', count: int = 200) -> Optional[Dict[str, Any]]:
        """Get comprehensive market data with caching"""
        try:
            with self.lock:
                cache_key = f"{symbol}_{timeframe}_{count}"
                
                # Check cache
                if cache_key in self.data_cache:
                    cached_data, timestamp = self.data_cache[cache_key]
                    if (datetime.now() - timestamp).seconds < self.cache_expiry:
                        return cached_data
                
                # Get fresh data (simplified for demo)
                market_data = self._fetch_market_data(symbol, timeframe, count)
                if market_data:
                    # Cache the data
                    self.data_cache[cache_key] = (market_data, datetime.now())
                    
                    # Clean old cache entries
                    self._clean_cache()
                
                return market_data
                
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _fetch_market_data(self, symbol: str, timeframe: str, count: int) -> Optional[Dict[str, Any]]:
        """Fetch market data from data source or generate synthetic data"""
        try:
            # Try to get real data if datahandler is available
            if self.datahandler and hasattr(self.datahandler, 'get_data'):
                real_data = self.datahandler.get_data(symbol, timeframe, count)
                if real_data is not None and len(real_data) > 0:
                    analyzed_data = self.analyze_data(real_data)
                    return self._create_market_data_dict(analyzed_data, symbol)
            
            # Fallback to synthetic data for demo
            synthetic_data = self._generate_synthetic_data(symbol, count)
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return self._generate_synthetic_data(symbol, count)
    
    def _generate_synthetic_data(self, symbol: str, count: int) -> Dict[str, Any]:
        """Generate synthetic market data for testing"""
        try:
            # Base prices for different symbols
            base_prices = {
                'EURUSD': 1.1000,
                'GBPUSD': 1.3000, 
                'XAUUSD': 2000.0,
                'USDJPY': 148.0
            }
            
            base_price = base_prices.get(symbol, 1.1000)
            
            # Generate price series with realistic patterns
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
            returns = np.random.normal(0.0001, 0.01, count)
            dates = pd.date_range(start=datetime.now() - pd.Timedelta(hours=count), periods=count, freq='15min')
            
            price = base_price
            ohlc_data = []
            
            for i in range(count):
                open_price = price
                return_val = returns[i]
                close_price = open_price * (1 + return_val)
                
                # Generate high/low with some logic
                volatility = abs(return_val) * 2
                high_price = max(open_price, close_price) * (1 + volatility)
                low_price = min(open_price, close_price) * (1 - volatility)
                
                ohlc_data.append({
                    'time': dates[i],
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'tick_volume': np.random.randint(50, 200)
                })
                
                price = close_price
            
            df = pd.DataFrame(ohlc_data)
            df_with_indicators = self.analyze_data(df)
            return self._create_market_data_dict(df_with_indicators, symbol)
            
        except Exception as e:
            logger.error(f"Error generating synthetic data for {symbol}: {e}")
            return None
    
    def _create_market_data_dict(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Create standardized market data dictionary"""
        try:
            if df is None or len(df) == 0:
                return None
            
            current = df.iloc[-1]
            
            return {
                'EXECUTION': df,  # Full dataframe for analysis
                'symbol': symbol,
                'current_price': float(current.get('close', 0)),
                'timestamp': current.get('time', datetime.now()),
                
                # Technical indicators
                'rsi': float(current.get('RSI_14', 50)),
                'macd': float(current.get('MACD_12_26_9', 0)),
                'macd_signal': float(current.get('MACDs_12_26_9', 0)),
                'bb_upper': float(current.get('BB_upper', current.get('close', 0))),
                'bb_lower': float(current.get('BB_lower', current.get('close', 0))),
                'bb_middle': float(current.get('BB_middle', current.get('close', 0))),
                'atr': float(current.get('ATR_14', 0.001)),
                'ema_20': float(current.get('EMA_20', current.get('close', 0))),
                'ema_50': float(current.get('EMA_50', current.get('close', 0))),
                
                # Price action
                'high_low_ratio': float(current.get('High_Low_Ratio', 1.0)),
                'close_open_ratio': float(current.get('Close_Open_Ratio', 1.0)),
                'price_range': float(current.get('Price_Range', 0.01)),
                
                # Volume and volatility
                'volume_ratio': float(current.get('Volume_Ratio', 1.0)),
                'volatility': float(current.get('Volatility_20', 0.01)),
                'momentum': float(current.get('Momentum_10', 0.0)),
            }
            
        except Exception as e:
            logger.error(f"Error creating market data dict: {e}")
            return None
    
    def _clean_cache(self):
        """Clean expired cache entries"""
        try:
            current_time = datetime.now()
            expired_keys = []
            
            for key, (data, timestamp) in self.data_cache.items():
                if (current_time - timestamp).seconds > self.cache_expiry:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.data_cache[key]
                
        except Exception:
            pass  # Silent cleanup
    
    def generate_traditional_signal(self, symbol: str, data_dict: Dict) -> Optional[Dict[str, Any]]:
        """Generate traditional technical analysis signal"""
        try:
            execution_df = data_dict.get('EXECUTION')
            if execution_df is None or len(execution_df) < 50:
                return None
            
            # Get current market data
            current_price = execution_df['close'].iloc[-1]
            atr_value = execution_df.get('ATR_14', pd.Series([0.001])).iloc[-1]
            
            # Get regime
            regime = self.identify_regime(execution_df)
            
            # Generate signal based on regime
            signal = None
            if regime == 'trending':
                signal = self._generate_trend_following_signal(execution_df)
            elif regime in ['ranging', 'normal']:
                signal = self._generate_mean_reverting_signal(execution_df)  
            elif regime == 'high_volatility':
                signal = self._generate_volatility_breakout_signal(execution_df)
            else:
                signal = self._generate_neutral_signal(execution_df)
            
            if signal:
                # Enhance signal with additional information
                signal.update({
                    'symbol': symbol,
                    'entry_price': current_price,
                    'atr_at_signal': atr_value,
                    'strategy': f'Traditional-{regime}',
                    'confidence': self._calculate_traditional_confidence(signal, execution_df, regime),
                    'timestamp': datetime.now(),
                    'regime': regime
                })
                
                # Add technical confirmations
                signal['confirmations'] = self._get_signal_confirmations(execution_df, signal['direction'])
                
                # Track signal history
                self.signal_history.append(signal.copy())
                if len(self.signal_history) > 100:
                    self.signal_history = self.signal_history[-100:]
                
                logger.debug(f"Traditional signal generated: {signal['direction']} ({signal['reason']})")
                return signal
                
        except Exception as e:
            logger.error(f"Error generating traditional signal: {e}")
            return None
    
    def analyze_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze market data and add technical indicators"""
        try:
            df = data.copy()
            required_cols = ['open', 'high', 'low', 'close']
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return df
            
            # Technical indicators
            df['RSI_14'] = self._calculate_rsi(df['close'])
            df['MACD_12_26_9'], df['MACDs_12_26_9'] = self._calculate_macd(df['close'])
            df['BB_upper'], df['BB_lower'], df['BB_middle'] = self._calculate_bollinger_bands(df['close'])
            df['ATR_14'] = self._calculate_atr(df)
            df['EMA_20'] = df['close'].ewm(span=20).mean()
            df['EMA_50'] = df['close'].ewm(span=50).mean()
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['SMA_30'] = df['close'].rolling(window=30).mean()
            
            # Price action features
            df['High_Low_Ratio'] = df['high'] / df['low']
            df['Close_Open_Ratio'] = df['close'] / df['open']
            df['Price_Range'] = (df['high'] - df['low']) / df['close']
            
            # Volume features
            if 'tick_volume' in df.columns:
                df['Volume'] = df['tick_volume']
            elif 'volume' in df.columns:
                df['Volume'] = df['volume']  
            else:
                df['Volume'] = 1000000
                
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Volatility features
            df['Returns'] = df['close'].pct_change()
            df['Volatility_20'] = df['Returns'].rolling(20).std()
            df['Momentum_10'] = df['close'].pct_change(10)
            
            # Clean NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            logger.debug(f"Analysis complete: {len(df.columns)} features generated")
            return df
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return data
    
    def identify_regime(self, data: pd.DataFrame) -> str:
        """Identify current market regime"""
        try:
            if len(data) < 50:
                return 'insufficient_data'
            
            recent_data = data.tail(50)
            if 'Returns' in recent_data.columns:
                returns = recent_data['Returns'].dropna()
            else:
                returns = recent_data['close'].pct_change().dropna()
            
            if len(returns) == 0:
                return 'unknown'
                
            trend_strength = abs(returns.mean()) / (returns.std() + 1e-10)
            volatility = returns.std()
            
            if trend_strength > 0.15 and volatility < 0.025:
                return 'strong_trending'
            elif trend_strength > 0.08:
                return 'trending'
            elif volatility > 0.03:
                return 'high_volatility'
            elif trend_strength < 0.05 and volatility < 0.015:
                return 'ranging'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"Error identifying regime: {e}")
            return 'unknown'
    
    def _generate_trend_following_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate trend following signals"""
        try:
            signals = []
            
            # EMA Crossover Strategy
            if 'EMA_20' in df.columns and 'EMA_50' in df.columns and len(df) >= 3:
                ema20_current = df['EMA_20'].iloc[-1]
                ema50_current = df['EMA_50'].iloc[-1]
                ema20_prev = df['EMA_20'].iloc[-2]
                ema50_prev = df['EMA_50'].iloc[-2]
                
                # Bullish crossover
                if ema20_prev <= ema50_prev and ema20_current > ema50_current:
                    signals.append({
                        'direction': 'BUY',
                        'reason': 'EMA20 cross above EMA50',
                        'strength': 0.8
                    })
                # Bearish crossover
                elif ema20_prev >= ema50_prev and ema20_current < ema50_current:
                    signals.append({
                        'direction': 'SELL', 
                        'reason': 'EMA20 cross below EMA50',
                        'strength': 0.8
                    })
            
            # MACD Trend Signal
            if all(col in df.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9']) and len(df) >= 3:
                macd_current = df['MACD_12_26_9'].iloc[-1]
                signal_current = df['MACDs_12_26_9'].iloc[-1] 
                macd_prev = df['MACD_12_26_9'].iloc[-2]
                signal_prev = df['MACDs_12_26_9'].iloc[-2]
                
                # Bullish MACD crossover
                if macd_prev <= signal_prev and macd_current > signal_current and macd_current > 0:
                    signals.append({
                        'direction': 'BUY',
                        'reason': 'MACD bullish crossover',
                        'strength': 0.7
                    })
                # Bearish MACD crossover
                elif macd_prev >= signal_prev and macd_current < signal_current and macd_current < 0:
                    signals.append({
                        'direction': 'SELL',
                        'reason': 'MACD bearish crossover', 
                        'strength': 0.7
                    })
            
            # Return strongest signal
            if signals:
                return max(signals, key=lambda x: x['strength'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating trend following signal: {e}")
            return None
    
    def _generate_mean_reverting_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate mean reverting signals"""
        try:
            # RSI Oversold/Overbought
            if 'RSI_14' in df.columns:
                rsi = df['RSI_14'].iloc[-1]
                
                if rsi < self.rsi_oversold:
                    return {
                        'direction': 'BUY',
                        'reason': f'RSI oversold ({rsi:.1f})',
                        'strength': 0.8
                    }
                elif rsi > self.rsi_overbought:
                    return {
                        'direction': 'SELL', 
                        'reason': f'RSI overbought ({rsi:.1f})',
                        'strength': 0.8
                    }
            
            # Bollinger Bands mean reversion
            if all(col in df.columns for col in ['close', 'BB_upper', 'BB_lower']):
                current_price = df['close'].iloc[-1]
                bb_upper = df['BB_upper'].iloc[-1]
                bb_lower = df['BB_lower'].iloc[-1]
                
                if current_price <= bb_lower:
                    return {
                        'direction': 'BUY',
                        'reason': 'Price at lower Bollinger Band',
                        'strength': 0.7
                    }
                elif current_price >= bb_upper:
                    return {
                        'direction': 'SELL',
                        'reason': 'Price at upper Bollinger Band',
                        'strength': 0.7
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating mean reverting signal: {e}")
            return None
    
    def _generate_volatility_breakout_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate volatility breakout signals"""
        try:
            if 'close' in df.columns and len(df) >= 20:
                current_price = df['close'].iloc[-1]
                high_20 = df['high'].rolling(20).max().iloc[-1]
                low_20 = df['low'].rolling(20).min().iloc[-1]
                
                if current_price > high_20 * 1.001:  # Small buffer
                    return {
                        'direction': 'BUY',
                        'reason': '20-period high breakout',
                        'strength': 0.75
                    }
                elif current_price < low_20 * 0.999:  # Small buffer
                    return {
                        'direction': 'SELL',
                        'reason': '20-period low breakdown', 
                        'strength': 0.75
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating volatility breakout signal: {e}")
            return None
    
    def _generate_neutral_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate neutral/conservative signals"""
        try:
            # Only generate signals with high confidence in neutral conditions
            if all(col in df.columns for col in ['RSI_14', 'MACD_12_26_9', 'close']):
                rsi = df['RSI_14'].iloc[-1]
                macd = df['MACD_12_26_9'].iloc[-1]
                
                # Very conservative signals
                if rsi < 25 and macd > 0:
                    return {
                        'direction': 'BUY',
                        'reason': 'Very oversold with MACD positive',
                        'strength': 0.6
                    }
                elif rsi > 75 and macd < 0:
                    return {
                        'direction': 'SELL',
                        'reason': 'Very overbought with MACD negative',
                        'strength': 0.6
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating neutral signal: {e}")
            return None
    
    def _calculate_traditional_confidence(self, signal: Dict, df: pd.DataFrame, regime: str) -> float:
        """Calculate confidence score for traditional signals"""
        try:
            base_confidence = signal.get('strength', 0.5)
            
            # Regime adjustment
            regime_multipliers = {
                'trending': 1.2,
                'strong_trending': 1.3,
                'ranging': 0.9,
                'high_volatility': 0.8,
                'normal': 1.0,
                'unknown': 0.7
            }
            
            confidence = base_confidence * regime_multipliers.get(regime, 1.0)
            return min(confidence, 1.0)
            
        except Exception:
            return 0.5
    
    def _get_signal_confirmations(self, df: pd.DataFrame, direction: str) -> List[str]:
        """Get technical confirmations for signal"""
        try:
            confirmations = []
            
            if 'RSI_14' in df.columns:
                rsi = df['RSI_14'].iloc[-1]
                if direction == 'BUY' and rsi < 50:
                    confirmations.append('RSI_support')
                elif direction == 'SELL' and rsi > 50:
                    confirmations.append('RSI_resistance')
            
            if all(col in df.columns for col in ['close', 'EMA_20']):
                price = df['close'].iloc[-1]
                ema20 = df['EMA_20'].iloc[-1]
                
                if direction == 'BUY' and price > ema20:
                    confirmations.append('Above_EMA20')
                elif direction == 'SELL' and price < ema20:
                    confirmations.append('Below_EMA20')
            
            return confirmations
            
        except Exception:
            return []
    
    def shutdown(self):
        """Clean shutdown of market intelligence"""
        try:
            logger.info("Shutting down Enhanced Market Intelligence...")
            
            # Clear caches
            self.data_cache.clear()
            self.regime_history.clear()
            self.signal_history.clear()
            self.pattern_cache.clear()
            
            logger.info("Enhanced Market Intelligence shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during market intelligence shutdown: {e}")
    
    # Technical indicator calculation methods (same as before)
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd.fillna(0), macd_signal.fillna(0)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> tuple:
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band.fillna(prices), lower_band.fillna(prices), sma.fillna(prices)
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        high, low, close = data['high'], data['low'], data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window).mean()
        return atr.fillna(0.001)

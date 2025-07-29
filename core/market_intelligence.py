"""
Enhanced Market Intelligence without TA-Lib Dependencies
Complete implementation with custom technical indicators
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

class EnhancedMarketIntelligence:
    """
    Complete Market Intelligence system without TA-Lib dependencies
    """
    
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        
        # Market intelligence parameters
        self.trend_threshold = config.get('market_intelligence.trend_threshold', 0.7)
        self.volatility_threshold = config.get('market_intelligence.volatility_threshold', 0.02)
        self.momentum_threshold = config.get('market_intelligence.momentum_threshold', 0.5)
        
        # Analysis periods
        self.trend_period = config.get('market_intelligence.trend_analysis_period', 50)
        self.volatility_period = config.get('market_intelligence.volatility_analysis_period', 20)
        self.momentum_period = config.get('market_intelligence.momentum_analysis_period', 14)
        
        # Strategy parameters
        self.strategy_config = config.get('strategy', {})
        self.rsi_overbought = self.strategy_config.get('rsi_overbought', 70)
        self.rsi_oversold = self.strategy_config.get('rsi_oversold', 30)
        self.bb_threshold = self.strategy_config.get('bb_threshold', 0.95)
        self.macd_threshold = self.strategy_config.get('macd_signal_threshold', 0.001)
        
        # Data cache
        self.data_cache = {}
        self.cache_expiry = 300  # 5 minutes
        
        # Initialize MT5 connection
        self._initialize_mt5()
        
        logger.info("✅ Enhanced Market Intelligence initialized (No TA-Lib)")
        logger.info(f"   Trend Threshold: {self.trend_threshold}")
        logger.info(f"   Analysis Periods: T{self.trend_period}, V{self.volatility_period}, M{self.momentum_period}")
    
    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection for market intelligence"""
        try:
            if not mt5.initialize():
                logger.warning("MT5 not available for market intelligence - using synthetic data mode")
                return False
            
            logger.info("✅ MT5 connected for market intelligence")
            return True
            
        except Exception as e:
            logger.warning(f"MT5 initialization failed: {e} - using synthetic data mode")
            return False
    
    def get_market_data(self, symbol: str, timeframe: str = 'M15', count: int = 200) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive market data with caching
        """
        try:
            with self.lock:
                cache_key = f"{symbol}_{timeframe}_{count}"
                
                # Check cache
                if cache_key in self.data_cache:
                    cached_data, timestamp = self.data_cache[cache_key]
                    if (datetime.now() - timestamp).seconds < self.cache_expiry:
                        return cached_data
                
                # Get fresh data
                market_data = self._fetch_market_data(symbol, timeframe, count)
                
                if market_data:
                    # Cache the data
                    self.data_cache[cache_key] = (market_data, datetime.now())
                    
                    # Clean old cache entries
                    self._clean_cache()
                    
                    return market_data
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _fetch_market_data(self, symbol: str, timeframe: str, count: int) -> Optional[Dict[str, Any]]:
        """Fetch market data from MT5 or generate synthetic data"""
        try:
            # Try to get real data from MT5
            if mt5.initialize():
                mt5_timeframe = self._get_mt5_timeframe(timeframe)
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
                
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    # Add technical indicators
                    df_with_indicators = self._add_custom_technical_indicators(df)
                    
                    # Create market data dictionary
                    market_data = self._create_market_data_dict(df_with_indicators, symbol)
                    
                    logger.debug(f"✅ Real market data loaded for {symbol}: {len(df)} bars")
                    return market_data
            
            # Fallback to synthetic data
            logger.debug(f"Using synthetic data for {symbol}")
            return self._generate_synthetic_market_data(symbol, count)
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return self._generate_synthetic_market_data(symbol, count)
    
    def _get_mt5_timeframe(self, timeframe: str):
        """Convert timeframe string to MT5 constant"""
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        return timeframe_map.get(timeframe, mt5.TIMEFRAME_M15)
    
    def _generate_synthetic_market_data(self, symbol: str, count: int) -> Dict[str, Any]:
        """Generate synthetic market data for testing"""
        try:
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
            
            # Base price for different symbols
            base_prices = {
                'EURUSD': 1.1000,
                'GBPUSD': 1.3000,
                'XAUUSD': 2000.0,
                'USDJPY': 148.0
            }
            base_price = base_prices.get(symbol, 1.1000)
            
            # Generate price series
            returns = np.random.normal(0.0001, 0.01, count)
            dates = pd.date_range(start=datetime.now() - timedelta(hours=count), periods=count, freq='15min')
            
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
            df_with_indicators = self._add_custom_technical_indicators(df)
            
            return self._create_market_data_dict(df_with_indicators, symbol)
            
        except Exception as e:
            logger.error(f"Error generating synthetic data for {symbol}: {e}")
            return None
    
    def _add_custom_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators using custom implementations"""
        try:
            # Ensure we have enough data
            if len(df) < 50:
                logger.warning("Insufficient data for technical indicators")
                return df
            
            # Price arrays
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            open_prices = df['open'].values
            volume = df.get('tick_volume', pd.Series([100] * len(df))).values
            
            # ✅ CUSTOM IMPLEMENTATIONS (No TA-Lib)
            
            # 1. Simple Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # 2. Exponential Moving Averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # 3. RSI (Custom Implementation)
            df['rsi'] = self._calculate_rsi(df['close'])
            
            # 4. MACD (Custom Implementation)
            macd_line, macd_signal, macd_histogram = self._calculate_macd(df['close'])
            df['macd'] = macd_line
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_histogram
            
            # 5. Bollinger Bands (Custom Implementation)
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # 6. ATR (Custom Implementation)
            df['atr'] = self._calculate_atr(df)
            
            # 7. Stochastic Oscillator (Custom Implementation)
            stoch_k, stoch_d = self._calculate_stochastic(df)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # 8. Williams %R (Custom Implementation)
            df['williams_r'] = self._calculate_williams_r(df)
            
            # 9. CCI (Custom Implementation)
            df['cci'] = self._calculate_cci(df)
            
            # 10. ADX (Custom Implementation)
            df['adx'] = self._calculate_adx(df)
            
            # 11. Momentum (Custom Implementation)
            df['momentum'] = self._calculate_momentum(df['close'])
            
            # 12. Volume indicators
            if 'tick_volume' in df.columns:
                df['volume_sma'] = df['tick_volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
            else:
                df['volume_ratio'] = 1.0
            
            # 13. Custom indicators
            df['price_change_pct'] = df['close'].pct_change()
            df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            
            # 14. Trend indicators
            df['sma_trend'] = np.where(df['close'] > df['sma_20'], 1, -1)
            df['ema_crossover'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)
            
            # 15. Support/Resistance levels
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            
            # Fill NaN values
            df = df.fillna(method='bfill').fillna(0)
            
            logger.debug(f"✅ Custom technical indicators added: {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error adding custom technical indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Custom RSI calculation"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Custom MACD calculation"""
        try:
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd_line = exp1 - exp2
            macd_signal = macd_line.ewm(span=signal).mean()
            macd_histogram = macd_line - macd_signal
            return macd_line, macd_signal, macd_histogram
        except Exception:
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros, zeros
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Custom Bollinger Bands calculation"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, sma, lower_band
        except Exception:
            sma = prices.rolling(window=period).mean()
            return sma, sma, sma
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Custom ATR calculation"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr
        except Exception:
            return pd.Series([0.01] * len(df), index=df.index)
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Custom Stochastic Oscillator calculation"""
        try:
            lowest_low = df['low'].rolling(window=k_period).min()
            highest_high = df['high'].rolling(window=k_period).max()
            
            k_percent = ((df['close'] - lowest_low) / (highest_high - lowest_low)) * 100
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return k_percent, d_percent
        except Exception:
            default_series = pd.Series([50] * len(df), index=df.index)
            return default_series, default_series
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Custom Williams %R calculation"""
        try:
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            
            williams_r = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
            return williams_r
        except Exception:
            return pd.Series([-50] * len(df), index=df.index)
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Custom CCI calculation"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(
                lambda x: np.abs(x - x.mean()).mean()
            )
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            return cci
        except Exception:
            return pd.Series([0] * len(df), index=df.index)
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Custom ADX calculation (simplified)"""
        try:
            # Calculate directional movement
            high_diff = df['high'].diff()
            low_diff = df['low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            # True Range
            tr = self._calculate_atr(df, 1)  # Single period TR
            
            # Directional Indicators
            plus_di = (pd.Series(plus_dm).rolling(window=period).mean() / tr.rolling(window=period).mean()) * 100
            minus_di = (pd.Series(minus_dm).rolling(window=period).mean() / tr.rolling(window=period).mean()) * 100
            
            # ADX calculation
            dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            adx = dx.rolling(window=period).mean()
            
            return adx
        except Exception:
            return pd.Series([25] * len(df), index=df.index)
    
    def _calculate_momentum(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """Custom Momentum calculation"""
        try:
            momentum = prices.diff(period)
            return momentum
        except Exception:
            return pd.Series([0] * len(prices), index=prices.index)
    
    def _create_market_data_dict(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Create comprehensive market data dictionary"""
        try:
            if len(df) == 0:
                return None
            
            # Current (latest) values
            current = df.iloc[-1]
            
            # Price data
            market_data = {
                'symbol': symbol,
                'timestamp': current['time'] if 'time' in current else datetime.now(),
                'open': float(current['open']),
                'high': float(current['high']),
                'low': float(current['low']),
                'close': float(current['close']),
                'current_price': float(current['close']),
                'tick_volume': int(current.get('tick_volume', 100)),
                
                # Price changes
                'price_change_pct': float(current.get('price_change_pct', 0)),
                'daily_range_pct': float(current.get('daily_range_pct', 0)),
                'body_size': float(current.get('body_size', 0)),
                
                # Technical indicators
                'sma_20': float(current.get('sma_20', current['close'])),
                'sma_50': float(current.get('sma_50', current['close'])),
                'ema_12': float(current.get('ema_12', current['close'])),
                'ema_26': float(current.get('ema_26', current['close'])),
                
                'rsi': float(current.get('rsi', 50)),
                'macd': float(current.get('macd', 0)),
                'macd_signal': float(current.get('macd_signal', 0)),
                'macd_histogram': float(current.get('macd_histogram', 0)),
                
                'bb_upper': float(current.get('bb_upper', current['close'])),
                'bb_middle': float(current.get('bb_middle', current['close'])),
                'bb_lower': float(current.get('bb_lower', current['close'])),
                
                'atr': float(current.get('atr', current['close'] * 0.01)),
                'stoch_k': float(current.get('stoch_k', 50)),
                'stoch_d': float(current.get('stoch_d', 50)),
                'williams_r': float(current.get('williams_r', -50)),
                'cci': float(current.get('cci', 0)),
                'adx': float(current.get('adx', 25)),
                'momentum': float(current.get('momentum', 0)),
                
                # Volume
                'volume_ratio': float(current.get('volume_ratio', 1.0)),
                
                # Derived indicators
                'sma20_distance': (current['close'] - current.get('sma_20', current['close'])) / current['close'],
                'sma50_distance': (current['close'] - current.get('sma_50', current['close'])) / current['close'],
                'bb_position': self._calculate_bb_position(current),
                'trend_strength': float(current.get('adx', 25)) / 100,
                'volatility': float(current.get('atr', current['close'] * 0.01)) / current['close'],
                
                # Support/Resistance
                'resistance': float(current.get('resistance', current['high'])),
                'support': float(current.get('support', current['low'])),
                'distance_to_resistance': (current.get('resistance', current['high']) - current['close']) / current['close'],
                'distance_to_support': (current['close'] - current.get('support', current['low'])) / current['close'],
                
                # Market regime
                'trend_direction': self._determine_trend_direction(df),
                'market_phase': self._determine_market_phase(current),
                'volatility_regime': self._determine_volatility_regime(df),
                
                # Raw dataframe for advanced analysis
                'dataframe': df
            }
            
            # Add symbol-specific adjustments
            if symbol == 'XAUUSD':
                market_data['contract_size'] = 100
                market_data['tick_size'] = 0.01
                market_data['tick_value'] = 1.0
            else:
                market_data['contract_size'] = 100000
                market_data['tick_size'] = 0.0001 if 'JPY' not in symbol else 0.01
                market_data['tick_value'] = 10 if 'JPY' not in symbol else 1
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error creating market data dict: {e}")
            return None
    
    def _calculate_bb_position(self, current_data) -> float:
        """Calculate position within Bollinger Bands (0-1)"""
        try:
            bb_upper = current_data.get('bb_upper', current_data['close'])
            bb_lower = current_data.get('bb_lower', current_data['close'])
            price = current_data['close']
            
            if bb_upper > bb_lower:
                return (price - bb_lower) / (bb_upper - bb_lower)
            return 0.5
            
        except Exception:
            return 0.5
    
    def _determine_trend_direction(self, df: pd.DataFrame) -> str:
        """Determine overall trend direction"""
        try:
            if len(df) < 20:
                return 'NEUTRAL'
            
            recent_data = df.tail(20)
            
            # Multiple trend indicators
            sma_trend = 1 if recent_data['close'].iloc[-1] > recent_data.get('sma_20', recent_data['close']).iloc[-1] else -1
            ema_trend = 1 if recent_data.get('ema_12', recent_data['close']).iloc[-1] > recent_data.get('ema_26', recent_data['close']).iloc[-1] else -1
            price_trend = 1 if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else -1
            
            trend_score = sma_trend + ema_trend + price_trend
            
            if trend_score >= 2:
                return 'BULLISH'
            elif trend_score <= -2:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception:
            return 'NEUTRAL'
    
    def _determine_market_phase(self, current_data) -> str:
        """Determine current market phase"""
        try:
            rsi = current_data.get('rsi', 50)
            bb_position = self._calculate_bb_position(current_data)
            adx = current_data.get('adx', 25)
            
            if adx > 25:  # Trending market
                if rsi > 70 or bb_position > 0.8:
                    return 'OVERBOUGHT_TREND'
                elif rsi < 30 or bb_position < 0.2:
                    return 'OVERSOLD_TREND'
                else:
                    return 'TRENDING'
            else:  # Ranging market
                if rsi > 60 or bb_position > 0.7:
                    return 'OVERBOUGHT_RANGE'
                elif rsi < 40 or bb_position < 0.3:
                    return 'OVERSOLD_RANGE'
                else:
                    return 'RANGING'
                    
        except Exception:
            return 'UNKNOWN'
    
    def _determine_volatility_regime(self, df: pd.DataFrame) -> str:
        """Determine volatility regime"""
        try:
            if len(df) < 20:
                return 'NORMAL'
            
            recent_atr = df.tail(20)['atr'].mean() if 'atr' in df.columns else 0
            historical_atr = df['atr'].mean() if 'atr' in df.columns else 0
            
            if historical_atr > 0:
                volatility_ratio = recent_atr / historical_atr
                
                if volatility_ratio > 1.5:
                    return 'HIGH'
                elif volatility_ratio < 0.7:
                    return 'LOW'
                else:
                    return 'NORMAL'
            
            return 'NORMAL'
            
        except Exception:
            return 'NORMAL'
    
    def generate_traditional_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate traditional technical analysis signal
        """
        try:
            if not market_data:
                return None
            
            signals = []
            
            # RSI-based signals
            rsi_signal = self._generate_rsi_signal(market_data)
            if rsi_signal:
                signals.append(rsi_signal)
            
            # Bollinger Bands signals
            bb_signal = self._generate_bb_signal(market_data)
            if bb_signal:
                signals.append(bb_signal)
            
            # MACD signals
            macd_signal = self._generate_macd_signal(market_data)
            if macd_signal:
                signals.append(macd_signal)
            
            # Moving Average signals
            ma_signal = self._generate_ma_signal(market_data)
            if ma_signal:
                signals.append(ma_signal)
            
            # Combine signals
            if signals:
                combined_signal = self._combine_traditional_signals(signals, symbol, market_data)
                return combined_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating traditional signal for {symbol}: {e}")
            return None
    
    def _generate_rsi_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Generate RSI-based signal"""
        try:
            rsi = market_data.get('rsi', 50)
            
            if rsi < self.rsi_oversold:
                return {
                    'type': 'RSI',
                    'direction': 'BUY',
                    'strength': min(1.0, (self.rsi_oversold - rsi) / 20),
                    'reason': f'RSI oversold: {rsi:.1f}'
                }
            elif rsi > self.rsi_overbought:
                return {
                    'type': 'RSI',
                    'direction': 'SELL',
                    'strength': min(1.0, (rsi - self.rsi_overbought) / 20),
                    'reason': f'RSI overbought: {rsi:.1f}'
                }
            
            return None
            
        except Exception:
            return None
    
    def _generate_bb_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Generate Bollinger Bands signal"""
        try:
            bb_position = market_data.get('bb_position', 0.5)
            current_price = market_data.get('current_price', 0)
            bb_upper = market_data.get('bb_upper', current_price)
            bb_lower = market_data.get('bb_lower', current_price)
            
            if bb_position < 0.1:  # Near lower band
                return {
                    'type': 'BOLLINGER',
                    'direction': 'BUY',
                    'strength': 1.0 - bb_position * 10,
                    'reason': f'Price near BB lower band: {current_price:.5f} vs {bb_lower:.5f}'
                }
            elif bb_position > 0.9:  # Near upper band
                return {
                    'type': 'BOLLINGER',
                    'direction': 'SELL',
                    'strength': (bb_position - 0.9) * 10,
                    'reason': f'Price near BB upper band: {current_price:.5f} vs {bb_upper:.5f}'
                }
            
            return None
            
        except Exception:
            return None
    
    def _generate_macd_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Generate MACD signal"""
        try:
            macd = market_data.get('macd', 0)
            macd_signal = market_data.get('macd_signal', 0)
            macd_histogram = market_data.get('macd_histogram', 0)
            
            # MACD line crossing signal line
            if macd > macd_signal and macd_histogram > 0:
                strength = min(1.0, abs(macd_histogram) / self.macd_threshold)
                if strength > 0.3:
                    return {
                        'type': 'MACD',
                        'direction': 'BUY',
                        'strength': strength,
                        'reason': f'MACD bullish crossover: {macd:.6f} > {macd_signal:.6f}'
                    }
            elif macd < macd_signal and macd_histogram < 0:
                strength = min(1.0, abs(macd_histogram) / self.macd_threshold)
                if strength > 0.3:
                    return {
                        'type': 'MACD',
                        'direction': 'SELL',
                        'strength': strength,
                        'reason': f'MACD bearish crossover: {macd:.6f} < {macd_signal:.6f}'
                    }
            
            return None
            
        except Exception:
            return None
    
    def _generate_ma_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Generate Moving Average signal"""
        try:
            current_price = market_data.get('current_price', 0)
            sma_20 = market_data.get('sma_20', current_price)
            sma_50 = market_data.get('sma_50', current_price)
            ema_12 = market_data.get('ema_12', current_price)
            ema_26 = market_data.get('ema_26', current_price)
            
            signals = 0
            
            # Price above/below MAs
            if current_price > sma_20 > sma_50:
                signals += 1
            elif current_price < sma_20 < sma_50:
                signals -= 1
            
            # EMA crossover
            if ema_12 > ema_26:
                signals += 1
            elif ema_12 < ema_26:
                signals -= 1
            
            if signals >= 2:
                return {
                    'type': 'MOVING_AVERAGE',
                    'direction': 'BUY',
                    'strength': 0.7,
                    'reason': f'Bullish MA alignment: Price {current_price:.5f} > SMA20 {sma_20:.5f} > SMA50 {sma_50:.5f}'
                }
            elif signals <= -2:
                return {
                    'type': 'MOVING_AVERAGE',
                    'direction': 'SELL',
                    'strength': 0.7,
                    'reason': f'Bearish MA alignment: Price {current_price:.5f} < SMA20 {sma_20:.5f} < SMA50 {sma_50:.5f}'
                }
            
            return None
            
        except Exception:
            return None
    
    def _combine_traditional_signals(self, signals: List[Dict], symbol: str, 
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple traditional signals"""
        try:
            buy_signals = [s for s in signals if s['direction'] == 'BUY']
            sell_signals = [s for s in signals if s['direction'] == 'SELL']
            
            # Calculate signal strength
            buy_strength = sum(s['strength'] for s in buy_signals)
            sell_strength = sum(s['strength'] for s in sell_signals)
            
            # Determine final signal
            if buy_strength > sell_strength and buy_strength > 0.5:
                direction = 'BUY'
                confidence = min(0.9, buy_strength / len(signals))
                reasons = [s['reason'] for s in buy_signals]
            elif sell_strength > buy_strength and sell_strength > 0.5:
                direction = 'SELL'
                confidence = min(0.9, sell_strength / len(signals))
                reasons = [s['reason'] for s in sell_signals]
            else:
                return None  # No clear signal
            
            # Calculate entry, stop loss, and take profit
            current_price = market_data.get('current_price', 0)
            atr = market_data.get('atr', current_price * 0.01)
            
            if direction == 'BUY':
                entry_price = current_price
                stop_loss = current_price - (atr * 2.0)
                take_profit = current_price + (atr * 3.0)
            else:
                entry_price = current_price
                stop_loss = current_price + (atr * 2.0)
                take_profit = current_price - (atr * 3.0)
            
            # Create signal
            signal = {
                'symbol': symbol,
                'direction': direction,
                'confidence': confidence,
                'strategy': 'Traditional-Multi-Indicator',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now(),
                'reasons': reasons,
                'signal_types': [s['type'] for s in signals if s['direction'] == direction],
                'market_phase': market_data.get('market_phase', 'UNKNOWN'),
                'trend_direction': market_data.get('trend_direction', 'NEUTRAL'),
                'volatility_regime': market_data.get('volatility_regime', 'NORMAL')
            }
            
            logger.debug(f"Traditional signal: {symbol} {direction} (confidence: {confidence:.2f})")
            return signal
            
        except Exception as e:
            logger.error(f"Error combining traditional signals: {e}")
            return None
    
    def _clean_cache(self) -> None:
        """Clean expired cache entries"""
        try:
            current_time = datetime.now()
            expired_keys = []
            
            for key, (data, timestamp) in self.data_cache.items():
                if (current_time - timestamp).seconds > self.cache_expiry:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.data_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get market intelligence summary"""
        try:
            symbols = self.config.get_trading_symbols()
            summary = {
                'timestamp': datetime.now(),
                'symbols_monitored': len(symbols),
                'cache_entries': len(self.data_cache),
                'mt5_connected': mt5.initialize() if mt5 else False,
                'custom_indicators': True,  # Using custom implementations
                'talib_dependency': False,  # No TA-Lib required
                'analysis_parameters': {
                    'trend_threshold': self.trend_threshold,
                    'volatility_threshold': self.volatility_threshold,
                    'momentum_threshold': self.momentum_threshold
                }
            }
            
            # Get current market state for each symbol
            market_states = {}
            for symbol in symbols[:3]:  # Limit to avoid too many calls
                try:
                    market_data = self.get_market_data(symbol)
                    if market_data:
                        market_states[symbol] = {
                            'price': market_data.get('current_price', 0),
                            'trend': market_data.get('trend_direction', 'NEUTRAL'),
                            'phase': market_data.get('market_phase', 'UNKNOWN'),
                            'volatility': market_data.get('volatility_regime', 'NORMAL'),
                            'rsi': market_data.get('rsi', 50),
                            'atr': market_data.get('atr', 0)
                        }
                except Exception:
                    continue
            
            summary['market_states'] = market_states
            return summary
            
        except Exception as e:
            logger.error(f"Error getting market summary: {e}")
            return {'error': str(e)}
    
    def shutdown(self) -> None:
        """Clean shutdown of market intelligence"""
        try:
            logger.info("📊 Market Intelligence Final Summary:")
            summary = self.get_market_summary()
            logger.info(f"   Symbols Monitored: {summary.get('symbols_monitored', 0)}")
            logger.info(f"   Cache Entries: {summary.get('cache_entries', 0)}")
            logger.info(f"   Custom Indicators: ✅ No TA-Lib Dependencies")
            
            # Clear cache
            self.data_cache.clear()
            
            # Shutdown MT5 connection
            try:
                mt5.shutdown()
            except Exception:
                pass
            
            logger.info("✅ Market Intelligence shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during market intelligence shutdown: {e}")

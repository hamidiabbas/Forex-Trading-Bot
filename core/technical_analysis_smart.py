# core/technical_analysis_smart.py
"""
Smart Technical Analysis with automatic TA-Lib fallback
PRODUCTION READY - Works with or without TA-Lib
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Smart TA-Lib import with fallback
TALIB_AVAILABLE = False
try:
    import talib
    TALIB_AVAILABLE = True
    print("✅ TA-Lib successfully imported - Using optimized calculations")
except ImportError:
    print("⚠️ TA-Lib not available - Using pandas fallback calculations")
    talib = None

class TrendDirection(Enum):
    STRONG_BULLISH = 2
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    STRONG_BEARISH = -2

class SignalStrength(Enum):
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1

@dataclass
class TechnicalSignal:
    direction: TrendDirection
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: str
    timestamp: datetime
    indicators_used: List[str]
    signal_details: Dict

class ComprehensiveTechnicalAnalyzer:
    """
    Production-ready Technical Analysis that works with or without TA-Lib
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.indicator_cache = {}
        self.using_talib = TALIB_AVAILABLE
        
        if self.using_talib:
            self.logger.info("✅ Technical Analysis initialized with TA-Lib (optimized)")
        else:
            self.logger.info("⚠️ Technical Analysis initialized with pandas fallback")
    
    def analyze_symbol(self, 
                      symbol: str, 
                      timeframe: str, 
                      price_data: pd.DataFrame,
                      lookback_periods: int = 200) -> Dict:
        """
        Complete technical analysis - works with or without TA-Lib
        """
        
        if len(price_data) < 50:
            self.logger.error(f"Insufficient data for analysis: {len(price_data)}")
            return None
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in price_data.columns for col in required_columns):
            self.logger.error(f"Missing required columns. Have: {price_data.columns.tolist()}")
            return None
        
        try:
            # Extract and validate price arrays
            open_prices = pd.to_numeric(price_data['Open'], errors='coerce').fillna(method='ffill').values
            high_prices = pd.to_numeric(price_data['High'], errors='coerce').fillna(method='ffill').values
            low_prices = pd.to_numeric(price_data['Low'], errors='coerce').fillna(method='ffill').values
            close_prices = pd.to_numeric(price_data['Close'], errors='coerce').fillna(method='ffill').values
            
            # Validate data integrity
            if len(close_prices) == 0 or np.all(np.isnan(close_prices)):
                self.logger.error("Invalid price data - all NaN values")
                return None
            
            # Analysis structure
            analysis = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': price_data.index[-1] if len(price_data.index) > 0 else datetime.now(),
                'current_price': close_prices[-1],
                'indicators': {},
                'signals': {},
                'market_structure': {},
                'risk_metrics': {},
                'using_talib': self.using_talib
            }
            
            # Calculate all indicators
            analysis['indicators']['moving_averages'] = self._calculate_moving_averages(close_prices)
            analysis['indicators']['momentum'] = self._calculate_momentum_indicators(high_prices, low_prices, close_prices)
            analysis['indicators']['volatility'] = self._calculate_volatility_indicators(high_prices, low_prices, close_prices)
            
            # Generate signals
            analysis['signals']['trend'] = self._analyze_trend(analysis['indicators'])
            analysis['signals']['momentum'] = self._analyze_momentum(analysis['indicators'])
            analysis['signals']['reversal'] = self._detect_reversal_signals(analysis['indicators'])
            
            # Calculate risk metrics and support/resistance
            analysis['risk_metrics'] = self._calculate_risk_metrics(high_prices, low_prices, close_prices)
            analysis['market_structure']['sr_levels'] = self._calculate_support_resistance(high_prices, low_prices, close_prices)
            
            # Generate final trading signal
            analysis['trading_signal'] = self._generate_trading_signal(analysis)
            
            self.logger.debug(f"Technical analysis completed for {symbol} using {'TA-Lib' if self.using_talib else 'pandas'}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis for {symbol}: {e}")
            return None
    
    def _calculate_moving_averages(self, close_prices: np.ndarray) -> Dict:
        """Calculate moving averages with smart method selection"""
        try:
            if self.using_talib:
                # TA-Lib optimized calculations
                return {
                    'sma_8': talib.SMA(close_prices, timeperiod=8),
                    'sma_21': talib.SMA(close_prices, timeperiod=21),
                    'sma_50': talib.SMA(close_prices, timeperiod=50),
                    'sma_200': talib.SMA(close_prices, timeperiod=200),
                    'ema_8': talib.EMA(close_prices, timeperiod=8),
                    'ema_21': talib.EMA(close_prices, timeperiod=21),
                    'ema_50': talib.EMA(close_prices, timeperiod=50),
                    'ema_200': talib.EMA(close_prices, timeperiod=200),
                }
            else:
                # Pandas fallback calculations
                close_series = pd.Series(close_prices)
                return {
                    'sma_8': close_series.rolling(window=8, min_periods=1).mean().values,
                    'sma_21': close_series.rolling(window=21, min_periods=1).mean().values,
                    'sma_50': close_series.rolling(window=50, min_periods=1).mean().values,
                    'sma_200': close_series.rolling(window=200, min_periods=1).mean().values,
                    'ema_8': close_series.ewm(span=8, min_periods=1).mean().values,
                    'ema_21': close_series.ewm(span=21, min_periods=1).mean().values,
                    'ema_50': close_series.ewm(span=50, min_periods=1).mean().values,
                    'ema_200': close_series.ewm(span=200, min_periods=1).mean().values,
                }
        except Exception as e:
            self.logger.error(f"Error calculating moving averages: {e}")
            return {}
    
    def _calculate_momentum_indicators(self, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray) -> Dict:
        """Calculate momentum indicators with smart method selection"""
        try:
            if self.using_talib:
                # TA-Lib optimized
                momentum = {
                    'rsi_14': talib.RSI(close_prices, timeperiod=14),
                    'rsi_21': talib.RSI(close_prices, timeperiod=21),
                    'stoch_k': None, 'stoch_d': None,
                    'williams_r': talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14),
                    'macd_line': None, 'macd_signal': None, 'macd_histogram': None,
                }
                
                # MACD
                macd_line, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
                momentum.update({
                    'macd_line': macd_line,
                    'macd_signal': macd_signal,
                    'macd_histogram': macd_hist
                })
                
                # Stochastic
                slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, 
                                         fastk_period=14, slowk_period=3, slowd_period=3)
                momentum.update({
                    'stoch_k': slowk,
                    'stoch_d': slowd
                })
                
            else:
                # Pandas fallback
                close_series = pd.Series(close_prices)
                high_series = pd.Series(high_prices)
                low_series = pd.Series(low_prices)
                
                momentum = {
                    'rsi_14': self._calculate_rsi_fallback(close_series, 14),
                    'rsi_21': self._calculate_rsi_fallback(close_series, 21),
                    'williams_r': self._calculate_williams_r_fallback(high_series, low_series, close_series, 14),
                    'macd_line': None, 'macd_signal': None, 'macd_histogram': None,
                    'stoch_k': None, 'stoch_d': None,
                }
                
                # MACD fallback
                macd_line, macd_signal, macd_hist = self._calculate_macd_fallback(close_series)
                momentum.update({
                    'macd_line': macd_line,
                    'macd_signal': macd_signal,
                    'macd_histogram': macd_hist
                })
                
                # Stochastic fallback
                stoch_k = self._calculate_stochastic_k_fallback(high_series, low_series, close_series, 14)
                momentum.update({
                    'stoch_k': stoch_k,
                    'stoch_d': pd.Series(stoch_k).rolling(window=3, min_periods=1).mean().values
                })
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {e}")
            return {}
    
    def _calculate_volatility_indicators(self, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray) -> Dict:
        """Calculate volatility indicators with smart method selection"""
        try:
            if self.using_talib:
                # TA-Lib optimized
                volatility = {
                    'atr_14': talib.ATR(high_prices, low_prices, close_prices, timeperiod=14),
                    'atr_21': talib.ATR(high_prices, low_prices, close_prices, timeperiod=21),
                    'bb_upper': None, 'bb_middle': None, 'bb_lower': None,
                }
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
                volatility.update({
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'bb_lower': bb_lower
                })
                
            else:
                # Pandas fallback
                close_series = pd.Series(close_prices)
                high_series = pd.Series(high_prices)
                low_series = pd.Series(low_prices)
                
                # ATR calculation
                hl = high_series - low_series
                hc = np.abs(high_series - close_series.shift())
                lc = np.abs(low_series - close_series.shift())
                true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
                
                volatility = {
                    'atr_14': true_range.rolling(window=14, min_periods=1).mean().values,
                    'atr_21': true_range.rolling(window=21, min_periods=1).mean().values,
                    'bb_upper': None, 'bb_middle': None, 'bb_lower': None,
                }
                
                # Bollinger Bands fallback
                bb_middle = close_series.rolling(window=20, min_periods=1).mean()
                bb_std = close_series.rolling(window=20, min_periods=1).std()
                volatility.update({
                    'bb_upper': (bb_middle + (bb_std * 2)).values,
                    'bb_middle': bb_middle.values,
                    'bb_lower': (bb_middle - (bb_std * 2)).values
                })
            
            # Volatility regime classification (same for both methods)
            atr_14 = volatility['atr_14']
            if len(atr_14) > 20:
                recent_atr = atr_14[-20:]
                current_atr = atr_14[-1]
                avg_atr = np.nanmean(recent_atr[~np.isnan(recent_atr)])
                
                if current_atr > avg_atr * 1.5:
                    volatility['volatility_regime'] = 'high'
                elif current_atr < avg_atr * 0.7:
                    volatility['volatility_regime'] = 'low'
                else:
                    volatility['volatility_regime'] = 'normal'
            else:
                volatility['volatility_regime'] = 'normal'
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {e}")
            return {}
    
    # Fallback calculation methods
    def _calculate_rsi_fallback(self, close_series: pd.Series, period: int = 14) -> np.ndarray:
        """RSI calculation using pandas"""
        try:
            delta = close_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50).values
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return np.full(len(close_series), 50.0)
    
    def _calculate_macd_fallback(self, close_series: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD calculation using pandas"""
        try:
            ema_fast = close_series.ewm(span=12, min_periods=1).mean()
            ema_slow = close_series.ewm(span=26, min_periods=1).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=9, min_periods=1).mean()
            macd_histogram = macd_line - macd_signal
            return macd_line.values, macd_signal.values, macd_histogram.values
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            zeros = np.zeros(len(close_series))
            return zeros, zeros, zeros
    
    def _calculate_stochastic_k_fallback(self, high_series: pd.Series, low_series: pd.Series, 
                                        close_series: pd.Series, period: int = 14) -> np.ndarray:
        """Stochastic %K calculation using pandas"""
        try:
            lowest_low = low_series.rolling(window=period, min_periods=1).min()
            highest_high = high_series.rolling(window=period, min_periods=1).max()
            k_percent = 100 * (close_series - lowest_low) / (highest_high - lowest_low)
            return k_percent.fillna(50).values
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic %K: {e}")
            return np.full(len(close_series), 50.0)
    
    def _calculate_williams_r_fallback(self, high_series: pd.Series, low_series: pd.Series, 
                                      close_series: pd.Series, period: int = 14) -> np.ndarray:
        """Williams %R calculation using pandas"""
        try:
            highest_high = high_series.rolling(window=period, min_periods=1).max()
            lowest_low = low_series.rolling(window=period, min_periods=1).min()
            williams_r = -100 * (highest_high - close_series) / (highest_high - lowest_low)
            return williams_r.fillna(-50).values
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {e}")
            return np.full(len(close_series), -50.0)
    
    def _analyze_trend(self, indicators: Dict) -> Dict:
        """Analyze trend from indicators"""
        try:
            mas = indicators.get('moving_averages', {})
            momentum = indicators.get('momentum', {})
            
            trend_signals = []
            
            # SMA trend analysis
            sma_21 = mas.get('sma_21', [])
            sma_50 = mas.get('sma_50', [])
            
            if len(sma_21) > 0 and len(sma_50) > 0:
                if not np.isnan(sma_21[-1]) and not np.isnan(sma_50[-1]):
                    if sma_21[-1] > sma_50[-1]:
                        trend_signals.append(1)
                    else:
                        trend_signals.append(-1)
            
            # MACD trend analysis
            macd_line = momentum.get('macd_line', [])
            macd_signal = momentum.get('macd_signal', [])
            
            if len(macd_line) > 0 and len(macd_signal) > 0:
                if not np.isnan(macd_line[-1]) and not np.isnan(macd_signal[-1]):
                    if macd_line[-1] > macd_signal[-1]:
                        trend_signals.append(1)
                    else:
                        trend_signals.append(-1)
            
            # Calculate overall trend
            avg_signal = np.mean(trend_signals) if trend_signals else 0
            
            if avg_signal > 0.5:
                direction = TrendDirection.BULLISH
                strength = SignalStrength.STRONG
            elif avg_signal < -0.5:
                direction = TrendDirection.BEARISH
                strength = SignalStrength.STRONG
            else:
                direction = TrendDirection.NEUTRAL
                strength = SignalStrength.WEAK
            
            return {
                'direction': direction,
                'strength': strength,
                'score': avg_signal,
                'confidence': min(abs(avg_signal), 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {e}")
            return {
                'direction': TrendDirection.NEUTRAL,
                'strength': SignalStrength.WEAK,
                'score': 0.0,
                'confidence': 0.0
            }
    
    def _analyze_momentum(self, indicators: Dict) -> Dict:
        """Analyze momentum indicators"""
        try:
            momentum = indicators.get('momentum', {})
            momentum_signals = []
            
            # RSI analysis
            rsi_14 = momentum.get('rsi_14', [])
            if len(rsi_14) > 0 and not np.isnan(rsi_14[-1]):
                rsi_value = rsi_14[-1]
                if 30 < rsi_value < 70:  # Normal range
                    if rsi_value > 50:
                        momentum_signals.append(0.5)
                    else:
                        momentum_signals.append(-0.5)
                elif rsi_value >= 70:  # Overbought
                    momentum_signals.append(-0.3)
                elif rsi_value <= 30:  # Oversold
                    momentum_signals.append(0.3)
            
            avg_momentum = np.mean(momentum_signals) if momentum_signals else 0
            
            return {
                'score': avg_momentum,
                'confidence': min(abs(avg_momentum), 1.0),
                'signals': momentum_signals
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'signals': []}
    
    def _detect_reversal_signals(self, indicators: Dict) -> Dict:
        """Detect potential reversal signals"""
        try:
            momentum = indicators.get('momentum', {})
            reversal_probability = 0.0
            reversal_signals = []
            
            # RSI extremes
            rsi_14 = momentum.get('rsi_14', [])
            if len(rsi_14) > 0 and not np.isnan(rsi_14[-1]):
                rsi_value = rsi_14[-1]
                if rsi_value > 80:
                    reversal_signals.append('rsi_overbought')
                    reversal_probability += 0.4
                elif rsi_value < 20:
                    reversal_signals.append('rsi_oversold')
                    reversal_probability += 0.4
            
            return {
                'probability': min(reversal_probability, 1.0),
                'signals': reversal_signals
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting reversal signals: {e}")
            return {'probability': 0.0, 'signals': []}
    
    def _calculate_support_resistance(self, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray) -> Dict:
        """Calculate support and resistance levels"""
        try:
            high_series = pd.Series(high_prices)
            low_series = pd.Series(low_prices)
            
            # Use recent data for S/R calculation
            recent_period = min(50, len(high_prices))
            recent_highs = high_series[-recent_period:].rolling(window=5, min_periods=1).max().dropna().unique()
            recent_lows = low_series[-recent_period:].rolling(window=5, min_periods=1).min().dropna().unique()
            
            current_price = close_prices[-1]
            
            # Find nearest levels
            resistance_levels = recent_highs[recent_highs > current_price]
            support_levels = recent_lows[recent_lows < current_price]
            
            nearest_resistance = resistance_levels.min() if len(resistance_levels) > 0 else None
            nearest_support = support_levels.max() if len(support_levels) > 0 else None
            
            return {
                'resistance_levels': recent_highs.tolist(),
                'support_levels': recent_lows.tolist(),
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return {
                'resistance_levels': [],
                'support_levels': [],
                'nearest_resistance': None,
                'nearest_support': None
            }
    
    def _calculate_risk_metrics(self, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray) -> Dict:
        """Calculate risk metrics"""
        try:
            close_series = pd.Series(close_prices)
            high_series = pd.Series(high_prices)
            low_series = pd.Series(low_prices)
            
            # ATR calculation
            hl = high_series - low_series
            hc = np.abs(high_series - close_series.shift())
            lc = np.abs(low_series - close_series.shift())
            
            true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            atr_14 = true_range.rolling(window=14, min_periods=1).mean().iloc[-1]
            
            # Volatility
            returns = close_series.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.01
            
            return {
                'atr_14': atr_14 if not np.isnan(atr_14) else 0.001,
                'volatility': volatility if not np.isnan(volatility) else 0.01,
                'current_price': close_prices[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {
                'atr_14': 0.001,
                'volatility': 0.01,
                'current_price': close_prices[-1] if len(close_prices) > 0 else 1.0
            }
    
    def _generate_trading_signal(self, analysis: Dict) -> TechnicalSignal:
        """Generate comprehensive trading signal"""
        try:
            trend = analysis['signals']['trend']
            momentum_sig = analysis['signals']['momentum']
            reversal = analysis['signals']['reversal']
            risk_metrics = analysis['risk_metrics']
            
            current_price = analysis['current_price']
            
            # Combined signal calculation
            trend_score = trend['score']
            momentum_score = momentum_sig.get('score', 0)
            reversal_probability = reversal.get('probability', 0)
            
            signal_score = (trend_score * 0.6) + (momentum_score * 0.3) - (reversal_probability * 0.1)
            
            # Determine direction and strength
            if signal_score > 0.4:
                direction = TrendDirection.BULLISH if signal_score < 0.7 else TrendDirection.STRONG_BULLISH
            elif signal_score < -0.4:
                direction = TrendDirection.BEARISH if signal_score > -0.7 else TrendDirection.STRONG_BEARISH
            else:
                direction = TrendDirection.NEUTRAL
            
            strength_value = abs(signal_score)
            if strength_value > 0.8:
                strength = SignalStrength.VERY_STRONG
            elif strength_value > 0.6:
                strength = SignalStrength.STRONG
            elif strength_value > 0.4:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Calculate stop loss and take profit using ATR
            atr = risk_metrics.get('atr_14', 0.001)
            
            if direction.value > 0:  # Bullish
                stop_loss = current_price - (atr * 2.0)
                take_profit = current_price + (atr * 3.0)
            elif direction.value < 0:  # Bearish
                stop_loss = current_price + (atr * 2.0)
                take_profit = current_price - (atr * 3.0)
            else:  # Neutral
                stop_loss = current_price
                take_profit = current_price
            
            return TechnicalSignal(
                direction=direction,
                strength=strength,
                confidence=min(abs(signal_score), 1.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timeframe=analysis['timeframe'],
                timestamp=analysis['timestamp'],
                indicators_used=['sma', 'ema', 'rsi', 'macd', 'stochastic', 'atr'],
                signal_details={
                    'signal_score': signal_score,
                    'trend_score': trend_score,
                    'momentum_score': momentum_score,
                    'reversal_probability': reversal_probability,
                    'atr_used': atr,
                    'using_talib': self.using_talib
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return TechnicalSignal(
                direction=TrendDirection.NEUTRAL,
                strength=SignalStrength.VERY_WEAK,
                confidence=0.0,
                entry_price=analysis.get('current_price', 0),
                stop_loss=analysis.get('current_price', 0),
                take_profit=analysis.get('current_price', 0),
                timeframe=analysis.get('timeframe', 'Unknown'),
                timestamp=datetime.now(),
                indicators_used=[],
                signal_details={}
            )

# core/technical_analysis_no_talib.py
"""
Complete Technical Analysis Engine WITHOUT TA-Lib dependency
Drop-in replacement for when TA-Lib is not available
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

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
    Complete technical analysis engine without TA-Lib dependency
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.indicator_cache = {}
        self.price_data_cache = {}
        
    def analyze_symbol(self, 
                      symbol: str, 
                      timeframe: str, 
                      price_data: pd.DataFrame,
                      lookback_periods: int = 200) -> Dict:
        """
        Complete technical analysis for a symbol without TA-Lib
        """
        
        if len(price_data) < lookback_periods:
            self.logger.warning(f"Insufficient data: need {lookback_periods}, got {len(price_data)}")
            lookback_periods = len(price_data)
        
        if lookback_periods < 50:
            self.logger.error(f"Insufficient data for analysis: {lookback_periods}")
            return None
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in price_data.columns for col in required_columns):
            self.logger.error(f"Missing required columns. Have: {price_data.columns.tolist()}")
            return None
        
        # Extract price arrays
        try:
            open_prices = price_data['Open'].values.astype(float)
            high_prices = price_data['High'].values.astype(float)
            low_prices = price_data['Low'].values.astype(float)
            close_prices = price_data['Close'].values.astype(float)
            volume = price_data.get('Volume', pd.Series(np.ones(len(price_data)))).values.astype(float)
        except Exception as e:
            self.logger.error(f"Error extracting price data: {e}")
            return None
        
        # Complete technical analysis
        analysis = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': price_data.index[-1] if not price_data.index.empty else datetime.now(),
            'current_price': close_prices[-1],
            'indicators': {},
            'signals': {},
            'market_structure': {},
            'risk_metrics': {}
        }
        
        try:
            # 1. MOVING AVERAGES
            analysis['indicators']['moving_averages'] = self._calculate_moving_averages(close_prices)
            
            # 2. MOMENTUM INDICATORS
            analysis['indicators']['momentum'] = self._calculate_momentum_indicators(
                high_prices, low_prices, close_prices
            )
            
            # 3. VOLATILITY INDICATORS
            analysis['indicators']['volatility'] = self._calculate_volatility_indicators(
                high_prices, low_prices, close_prices
            )
            
            # 4. SUPPORT/RESISTANCE LEVELS
            analysis['market_structure']['sr_levels'] = self._calculate_support_resistance(
                high_prices, low_prices, close_prices
            )
            
            # 5. TREND ANALYSIS
            analysis['signals']['trend'] = self._analyze_trend(analysis['indicators'])
            
            # 6. MOMENTUM ANALYSIS
            analysis['signals']['momentum'] = self._analyze_momentum(analysis['indicators'])
            
            # 7. REVERSAL SIGNALS
            analysis['signals']['reversal'] = self._detect_reversal_signals(
                analysis['indicators'], close_prices
            )
            
            # 8. RISK METRICS
            analysis['risk_metrics'] = self._calculate_risk_metrics(
                high_prices, low_prices, close_prices
            )
            
            # 9. GENERATE TRADING SIGNAL
            analysis['trading_signal'] = self._generate_trading_signal(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            return None
    
    def _calculate_moving_averages(self, close_prices: np.ndarray) -> Dict:
        """Calculate moving averages without TA-Lib"""
        try:
            close_series = pd.Series(close_prices)
            
            mas = {
                # Simple Moving Averages
                'sma_8': close_series.rolling(window=8).mean().values,
                'sma_21': close_series.rolling(window=21).mean().values,
                'sma_50': close_series.rolling(window=50).mean().values,
                'sma_100': close_series.rolling(window=100).mean().values,
                'sma_200': close_series.rolling(window=200).mean().values,
                
                # Exponential Moving Averages
                'ema_8': close_series.ewm(span=8).mean().values,
                'ema_21': close_series.ewm(span=21).mean().values,
                'ema_50': close_series.ewm(span=50).mean().values,
                'ema_100': close_series.ewm(span=100).mean().values,
                'ema_200': close_series.ewm(span=200).mean().values,
            }
            
            # Calculate MA relationships
            current_price = close_prices[-1]
            mas['ma_alignment'] = self._calculate_ma_alignment(mas, current_price)
            
            return mas
            
        except Exception as e:
            self.logger.error(f"Error calculating moving averages: {e}")
            return {}
    
    def _calculate_momentum_indicators(self, high_prices: np.ndarray, 
                                     low_prices: np.ndarray, 
                                     close_prices: np.ndarray) -> Dict:
        """Calculate momentum indicators without TA-Lib"""
        try:
            close_series = pd.Series(close_prices)
            high_series = pd.Series(high_prices)
            low_series = pd.Series(low_prices)
            
            momentum = {
                # RSI (Relative Strength Index)
                'rsi_14': self._calculate_rsi(close_series, 14),
                'rsi_21': self._calculate_rsi(close_series, 21),
                
                # MACD (Moving Average Convergence Divergence)
                'macd_line': None,
                'macd_signal': None,
                'macd_histogram': None,
                
                # Stochastic Oscillator
                'stoch_k': self._calculate_stochastic_k(high_series, low_series, close_series, 14),
                'stoch_d': None,
                
                # Williams %R
                'williams_r': self._calculate_williams_r(high_series, low_series, close_series, 14),
                
                # Rate of Change
                'roc': close_series.pct_change(periods=10).values * 100,
                
                # Momentum
                'momentum': (close_series / close_series.shift(10) - 1).values * 100,
            }
            
            # Calculate MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(close_series)
            momentum['macd_line'] = macd_line
            momentum['macd_signal'] = macd_signal
            momentum['macd_histogram'] = macd_histogram
            
            # Calculate Stochastic %D
            if momentum['stoch_k'] is not None:
                stoch_k_series = pd.Series(momentum['stoch_k'])
                momentum['stoch_d'] = stoch_k_series.rolling(window=3).mean().values
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {e}")
            return {}
    
    def _calculate_volatility_indicators(self, high_prices: np.ndarray,
                                       low_prices: np.ndarray, 
                                       close_prices: np.ndarray) -> Dict:
        """Calculate volatility indicators without TA-Lib"""
        try:
            close_series = pd.Series(close_prices)
            high_series = pd.Series(high_prices)
            low_series = pd.Series(low_prices)
            
            # True Range calculation
            hl = high_series - low_series
            hc = np.abs(high_series - close_series.shift())
            lc = np.abs(low_series - close_series.shift())
            
            true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            
            volatility = {
                # Average True Range
                'atr_14': true_range.rolling(window=14).mean().values,
                'atr_21': true_range.rolling(window=21).mean().values,
                
                # Bollinger Bands
                'bb_upper': None,
                'bb_middle': None,
                'bb_lower': None,
                
                # Standard Deviation
                'stddev': close_series.rolling(window=20).std().values,
                
                # True Range
                'true_range': true_range.values,
            }
            
            # Calculate Bollinger Bands
            bb_middle = close_series.rolling(window=20).mean()
            bb_std = close_series.rolling(window=20).std()
            
            volatility['bb_upper'] = (bb_middle + (bb_std * 2)).values
            volatility['bb_middle'] = bb_middle.values
            volatility['bb_lower'] = (bb_middle - (bb_std * 2)).values
            
            # Volatility regime
            recent_atr = volatility['atr_14'][-20:]
            current_atr = volatility['atr_14'][-1]
            avg_atr = np.nanmean(recent_atr[~np.isnan(recent_atr)])
            
            if current_atr > avg_atr * 1.5:
                volatility['volatility_regime'] = 'high'
            elif current_atr < avg_atr * 0.7:
                volatility['volatility_regime'] = 'low'
            else:
                volatility['volatility_regime'] = 'normal'
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {e}")
            return {}
    
    def _calculate_rsi(self, close_series: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate RSI without TA-Lib"""
        try:
            delta = close_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.values
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return np.full(len(close_series), 50.0)
    
    def _calculate_macd(self, close_series: pd.Series, 
                       fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple:
        """Calculate MACD without TA-Lib"""
        try:
            ema_fast = close_series.ewm(span=fast_period).mean()
            ema_slow = close_series.ewm(span=slow_period).mean()
            
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal_period).mean()
            macd_histogram = macd_line - macd_signal
            
            return macd_line.values, macd_signal.values, macd_histogram.values
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return np.zeros(len(close_series)), np.zeros(len(close_series)), np.zeros(len(close_series))
    
    def _calculate_stochastic_k(self, high_series: pd.Series, low_series: pd.Series, 
                               close_series: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate Stochastic %K without TA-Lib"""
        try:
            lowest_low = low_series.rolling(window=period).min()
            highest_high = high_series.rolling(window=period).max()
            
            k_percent = 100 * (close_series - lowest_low) / (highest_high - lowest_low)
            
            return k_percent.values
            
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic %K: {e}")
            return np.full(len(close_series), 50.0)
    
    def _calculate_williams_r(self, high_series: pd.Series, low_series: pd.Series, 
                             close_series: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate Williams %R without TA-Lib"""
        try:
            highest_high = high_series.rolling(window=period).max()
            lowest_low = low_series.rolling(window=period).min()
            
            williams_r = -100 * (highest_high - close_series) / (highest_high - lowest_low)
            
            return williams_r.values
            
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {e}")
            return np.full(len(close_series), -50.0)
    
    def _calculate_support_resistance(self, high_prices: np.ndarray,
                                    low_prices: np.ndarray,
                                    close_prices: np.ndarray) -> Dict:
        """Calculate support and resistance levels"""
        try:
            # Find recent swing highs and lows
            high_series = pd.Series(high_prices)
            low_series = pd.Series(low_prices)
            
            # Use last 50 periods for S/R calculation
            recent_period = min(50, len(high_prices))
            recent_highs = high_series[-recent_period:].rolling(window=5).max().dropna().unique()
            recent_lows = low_series[-recent_period:].rolling(window=5).min().dropna().unique()
            
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
    
    def _calculate_ma_alignment(self, mas: Dict, current_price: float) -> str:
        """Calculate moving average alignment"""
        try:
            sma_20 = mas.get('sma_21', [np.nan])[-1]
            sma_50 = mas.get('sma_50', [np.nan])[-1]
            
            if np.isnan(sma_20) or np.isnan(sma_50):
                return 'insufficient_data'
            
            if current_price > sma_20 > sma_50:
                return 'bullish_aligned'
            elif current_price < sma_20 < sma_50:
                return 'bearish_aligned'
            else:
                return 'mixed'
                
        except Exception as e:
            self.logger.error(f"Error calculating MA alignment: {e}")
            return 'unknown'
    
    def _analyze_trend(self, indicators: Dict) -> Dict:
        """Analyze trend based on indicators"""
        try:
            mas = indicators.get('moving_averages', {})
            momentum = indicators.get('momentum', {})
            
            trend_signals = []
            
            # Moving average trend
            ma_alignment = mas.get('ma_alignment', 'unknown')
            if ma_alignment == 'bullish_aligned':
                trend_signals.append(1)
            elif ma_alignment == 'bearish_aligned':
                trend_signals.append(-1)
            else:
                trend_signals.append(0)
            
            # MACD trend
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
            
            # RSI momentum
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
            
            # Calculate overall momentum
            avg_momentum = np.mean(momentum_signals) if momentum_signals else 0
            
            return {
                'score': avg_momentum,
                'confidence': min(abs(avg_momentum), 1.0),
                'signals': momentum_signals
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'signals': []}
    
    def _detect_reversal_signals(self, indicators: Dict, close_prices: np.ndarray) -> Dict:
        """Detect potential reversal signals"""
        try:
            momentum = indicators.get('momentum', {})
            volatility = indicators.get('volatility', {})
            
            reversal_probability = 0.0
            reversal_signals = []
            
            # RSI extremes
            rsi_14 = momentum.get('rsi_14', [])
            if len(rsi_14) > 0 and not np.isnan(rsi_14[-1]):
                rsi_value = rsi_14[-1]
                if rsi_value > 80:
                    reversal_signals.append('rsi_overbought')
                    reversal_probability += 0.3
                elif rsi_value < 20:
                    reversal_signals.append('rsi_oversold')
                    reversal_probability += 0.3
            
            # Bollinger Band extremes
            bb_upper = volatility.get('bb_upper', [])
            bb_lower = volatility.get('bb_lower', [])
            
            if (len(bb_upper) > 0 and len(bb_lower) > 0 and 
                not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1])):
                current_price = close_prices[-1]
                
                if current_price > bb_upper[-1]:
                    reversal_signals.append('bb_upper_break')
                    reversal_probability += 0.2
                elif current_price < bb_lower[-1]:
                    reversal_signals.append('bb_lower_break')
                    reversal_probability += 0.2
            
            return {
                'probability': min(reversal_probability, 1.0),
                'signals': reversal_signals
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting reversal signals: {e}")
            return {'probability': 0.0, 'signals': []}
    
    def _calculate_risk_metrics(self, high_prices: np.ndarray,
                               low_prices: np.ndarray,
                               close_prices: np.ndarray) -> Dict:
        """Calculate risk metrics"""
        try:
            close_series = pd.Series(close_prices)
            
            # ATR for position sizing
            high_series = pd.Series(high_prices)
            low_series = pd.Series(low_prices)
            
            hl = high_series - low_series
            hc = np.abs(high_series - close_series.shift())
            lc = np.abs(low_series - close_series.shift())
            
            true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            atr_14 = true_range.rolling(window=14).mean().iloc[-1]
            
            # Volatility
            returns = close_series.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
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
            
            # Combined signal score
            trend_score = trend['score']
            momentum_score = momentum_sig.get('score', 0)
            reversal_probability = reversal.get('probability', 0)
            
            signal_score = (trend_score * 0.6) + (momentum_score * 0.3) - (reversal_probability * 0.1)
            
            # Determine direction
            if signal_score > 0.4:
                direction = TrendDirection.BULLISH if signal_score < 0.7 else TrendDirection.STRONG_BULLISH
            elif signal_score < -0.4:
                direction = TrendDirection.BEARISH if signal_score > -0.7 else TrendDirection.STRONG_BEARISH
            else:
                direction = TrendDirection.NEUTRAL
            
            # Determine strength
            strength_value = abs(signal_score)
            if strength_value > 0.8:
                strength = SignalStrength.VERY_STRONG
            elif strength_value > 0.6:
                strength = SignalStrength.STRONG
            elif strength_value > 0.4:
                strength = SignalStrength.MODERATE
            elif strength_value > 0.2:
                strength = SignalStrength.WEAK
            else:
                strength = SignalStrength.VERY_WEAK
            
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
                    'atr_used': atr
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

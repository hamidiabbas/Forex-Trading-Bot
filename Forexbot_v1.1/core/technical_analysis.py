# core/technical_analysis.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import talib
from dataclasses import dataclass
from enum import Enum
import logging

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
    timestamp: pd.Timestamp
    indicators_used: List[str]
    signal_details: Dict

class ComprehensiveTechnicalAnalyzer:
    """
    Complete technical analysis engine with all major indicators
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
        Complete technical analysis for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., 'H1', 'H4', 'D1')
            price_data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            lookback_periods: Number of periods to analyze
            
        Returns:
            Complete analysis dictionary with all indicators and signals
        """
        
        if len(price_data) < lookback_periods:
            raise ValueError(f"Insufficient data: need {lookback_periods}, got {len(price_data)}")
        
        # Cache key for this analysis
        cache_key = f"{symbol}_{timeframe}_{len(price_data)}"
        
        # Extract price arrays
        open_prices = price_data['open'].values
        high_prices = price_data['high'].values
        low_prices = price_data['low'].values
        close_prices = price_data['close'].values
        volume = price_data.get('volume', pd.Series(np.ones(len(price_data)))).values
        
        # Complete technical analysis
        analysis = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': price_data.index[-1],
            'current_price': close_prices[-1],
            'indicators': {},
            'signals': {},
            'market_structure': {},
            'risk_metrics': {}
        }
        
        # 1. MOVING AVERAGES
        analysis['indicators']['moving_averages'] = self._calculate_moving_averages(close_prices)
        
        # 2. MOMENTUM INDICATORS
        analysis['indicators']['momentum'] = self._calculate_momentum_indicators(
            high_prices, low_prices, close_prices, volume
        )
        
        # 3. VOLATILITY INDICATORS
        analysis['indicators']['volatility'] = self._calculate_volatility_indicators(
            high_prices, low_prices, close_prices
        )
        
        # 4. VOLUME INDICATORS
        analysis['indicators']['volume'] = self._calculate_volume_indicators(
            high_prices, low_prices, close_prices, volume
        )
        
        # 5. SUPPORT/RESISTANCE LEVELS
        analysis['market_structure']['sr_levels'] = self._calculate_support_resistance(
            high_prices, low_prices, close_prices
        )
        
        # 6. CHART PATTERNS
        analysis['market_structure']['patterns'] = self._detect_chart_patterns(
            open_prices, high_prices, low_prices, close_prices
        )
        
        # 7. TREND ANALYSIS
        analysis['signals']['trend'] = self._analyze_trend(analysis['indicators'])
        
        # 8. MOMENTUM ANALYSIS
        analysis['signals']['momentum'] = self._analyze_momentum(analysis['indicators'])
        
        # 9. REVERSAL SIGNALS
        analysis['signals']['reversal'] = self._detect_reversal_signals(
            analysis['indicators'], analysis['market_structure']
        )
        
        # 10. RISK METRICS
        analysis['risk_metrics'] = self._calculate_risk_metrics(
            high_prices, low_prices, close_prices
        )
        
        # 11. GENERATE TRADING SIGNAL
        analysis['trading_signal'] = self._generate_trading_signal(analysis)
        
        return analysis
    
    def _calculate_moving_averages(self, close_prices: np.ndarray) -> Dict:
        """Calculate all moving average indicators"""
        
        try:
            mas = {
                # Simple Moving Averages
                'sma_8': talib.SMA(close_prices, timeperiod=8),
                'sma_21': talib.SMA(close_prices, timeperiod=21),
                'sma_50': talib.SMA(close_prices, timeperiod=50),
                'sma_100': talib.SMA(close_prices, timeperiod=100),
                'sma_200': talib.SMA(close_prices, timeperiod=200),
                
                # Exponential Moving Averages
                'ema_8': talib.EMA(close_prices, timeperiod=8),
                'ema_21': talib.EMA(close_prices, timeperiod=21),
                'ema_50': talib.EMA(close_prices, timeperiod=50),
                'ema_100': talib.EMA(close_prices, timeperiod=100),
                'ema_200': talib.EMA(close_prices, timeperiod=200),
                
                # Weighted Moving Average
                'wma_21': talib.WMA(close_prices, timeperiod=21),
                
                # Triple Exponential Moving Average (TEMA)
                'tema_21': talib.TEMA(close_prices, timeperiod=21),
                
                # Kaufman Adaptive Moving Average
                'kama_21': talib.KAMA(close_prices, timeperiod=21),
            }
            
            # Calculate MA relationships
            current_price = close_prices[-1]
            mas['ma_alignment'] = self._calculate_ma_alignment(mas, current_price)
            mas['ma_slopes'] = self._calculate_ma_slopes(mas)
            mas['ma_distances'] = self._calculate_ma_distances(mas, current_price)
            
            return mas
            
        except Exception as e:
            self.logger.error(f"Error calculating moving averages: {e}")
            return {}
    
    def _calculate_momentum_indicators(self, high_prices: np.ndarray, 
                                     low_prices: np.ndarray, 
                                     close_prices: np.ndarray,
                                     volume: np.ndarray) -> Dict:
        """Calculate all momentum indicators"""
        
        try:
            momentum = {
                # RSI (Relative Strength Index)
                'rsi_14': talib.RSI(close_prices, timeperiod=14),
                'rsi_21': talib.RSI(close_prices, timeperiod=21),
                
                # MACD (Moving Average Convergence Divergence)
                'macd_line': None,
                'macd_signal': None,
                'macd_histogram': None,
                
                # Stochastic Oscillator
                'stoch_k': None,
                'stoch_d': None,
                
                # Williams %R
                'williams_r': talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14),
                
                # Commodity Channel Index
                'cci': talib.CCI(high_prices, low_prices, close_prices, timeperiod=14),
                
                # Average Directional Index
                'adx': talib.ADX(high_prices, low_prices, close_prices, timeperiod=14),
                'plus_di': talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14),
                'minus_di': talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14),
                
                # Money Flow Index (requires volume)
                'mfi': talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14),
                
                # Rate of Change
                'roc': talib.ROC(close_prices, timeperiod=10),
                
                # Momentum
                'momentum': talib.MOM(close_prices, timeperiod=10),
            }
            
            # Calculate MACD
            macd_line, macd_signal, macd_histogram = talib.MACD(close_prices, 
                                                               fastperiod=12, 
                                                               slowperiod=26, 
                                                               signalperiod=9)
            momentum['macd_line'] = macd_line
            momentum['macd_signal'] = macd_signal
            momentum['macd_histogram'] = macd_histogram
            
            # Calculate Stochastic
            stoch_k, stoch_d = talib.STOCH(high_prices, low_prices, close_prices,
                                          fastk_period=14, slowk_period=3, 
                                          slowk_matype=0, slowd_period=3, slowd_matype=0)
            momentum['stoch_k'] = stoch_k
            momentum['stoch_d'] = stoch_d
            
            # Add momentum analysis
            momentum['rsi_analysis'] = self._analyze_rsi(momentum['rsi_14'])
            momentum['macd_analysis'] = self._analyze_macd(momentum)
            momentum['stoch_analysis'] = self._analyze_stochastic(momentum)
            momentum['adx_analysis'] = self._analyze_adx(momentum)
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {e}")
            return {}
    
    def _calculate_volatility_indicators(self, high_prices: np.ndarray,
                                       low_prices: np.ndarray, 
                                       close_prices: np.ndarray) -> Dict:
        """Calculate volatility indicators"""
        
        try:
            volatility = {
                # Average True Range
                'atr_14': talib.ATR(high_prices, low_prices, close_prices, timeperiod=14),
                'atr_21': talib.ATR(high_prices, low_prices, close_prices, timeperiod=21),
                
                # Bollinger Bands
                'bb_upper': None,
                'bb_middle': None,
                'bb_lower': None,
                
                # Standard Deviation
                'stddev': talib.STDDEV(close_prices, timeperiod=20, nbdev=1),
                
                # True Range
                'true_range': talib.TRANGE(high_prices, low_prices, close_prices),
                
                # Normalized ATR (ATR / Price ratio)
                'natr': talib.NATR(high_prices, low_prices, close_prices, timeperiod=14),
            }
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, 
                                                        timeperiod=20, 
                                                        nbdevup=2, 
                                                        nbdevdn=2, 
                                                        matype=0)
            volatility['bb_upper'] = bb_upper
            volatility['bb_middle'] = bb_middle
            volatility['bb_lower'] = bb_lower
            
            # Add volatility analysis
            volatility['bb_analysis'] = self._analyze_bollinger_bands(volatility, close_prices[-1])
            volatility['atr_analysis'] = self._analyze_atr(volatility, close_prices)
            volatility['volatility_regime'] = self._determine_volatility_regime(volatility)
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {e}")
            return {}
    
    def _calculate_volume_indicators(self, high_prices: np.ndarray,
                                   low_prices: np.ndarray,
                                   close_prices: np.ndarray,
                                   volume: np.ndarray) -> Dict:
        """Calculate volume-based indicators"""
        
        try:
            if len(volume) == 0 or np.all(volume == 1):
                # No real volume data
                return {
                    'volume_available': False,
                    'obv': np.zeros_like(close_prices),
                    'ad_line': np.zeros_like(close_prices),
                    'volume_ma': np.ones_like(close_prices),
                }
            
            volume_indicators = {
                'volume_available': True,
                
                # On-Balance Volume
                'obv': talib.OBV(close_prices, volume),
                
                # Accumulation/Distribution Line
                'ad_line': talib.AD(high_prices, low_prices, close_prices, volume),
                
                # Chaikin A/D Oscillator
                'ad_oscillator': talib.ADOSC(high_prices, low_prices, close_prices, volume,
                                            fastperiod=3, slowperiod=10),
                
                # Volume Moving Average
                'volume_ma': talib.SMA(volume, timeperiod=20),
                
                # Volume Rate of Change
                'volume_roc': talib.ROC(volume, timeperiod=10),
            }
            
            # Volume analysis
            volume_indicators['volume_analysis'] = self._analyze_volume(volume_indicators, volume)
            
            return volume_indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {e}")
            return {'volume_available': False}
    
    def _calculate_support_resistance(self, high_prices: np.ndarray,
                                    low_prices: np.ndarray,
                                    close_prices: np.ndarray) -> Dict:
        """Calculate support and resistance levels"""
        
        try:
            current_price = close_prices[-1]
            
            # Pivot Points (Traditional)
            pivot_points = self._calculate_pivot_points(high_prices, low_prices, close_prices)
            
            # Fibonacci Retracements
            fib_levels = self._calculate_fibonacci_levels(high_prices, low_prices, close_prices)
            
            # Dynamic Support/Resistance using swing highs and lows
            swing_levels = self._calculate_swing_levels(high_prices, low_prices, close_prices)
            
            # Psychological levels (round numbers)
            psychological_levels = self._calculate_psychological_levels(current_price)
            
            return {
                'pivot_points': pivot_points,
                'fibonacci_levels': fib_levels,
                'swing_levels': swing_levels,
                'psychological_levels': psychological_levels,
                'nearest_support': self._find_nearest_support(current_price, pivot_points, fib_levels, swing_levels),
                'nearest_resistance': self._find_nearest_resistance(current_price, pivot_points, fib_levels, swing_levels),
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return {}
    
    def _detect_chart_patterns(self, open_prices: np.ndarray,
                             high_prices: np.ndarray, 
                             low_prices: np.ndarray,
                             close_prices: np.ndarray) -> Dict:
        """Detect chart patterns"""
        
        try:
            patterns = {
                # Candlestick Patterns
                'doji': talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices),
                'hammer': talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices),
                'hanging_man': talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices),
                'engulfing_bullish': talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices),
                'morning_star': talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices),
                'evening_star': talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices),
                'shooting_star': talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices),
                'dragonfly_doji': talib.CDLDRAGONFLYDOJI(open_prices, high_prices, low_prices, close_prices),
                'gravestone_doji': talib.CDLGRAVESTONEDOJI(open_prices, high_prices, low_prices, close_prices),
                
                # Price Action Patterns
                'higher_highs': self._detect_higher_highs(high_prices),
                'higher_lows': self._detect_higher_lows(low_prices),
                'lower_highs': self._detect_lower_highs(high_prices),
                'lower_lows': self._detect_lower_lows(low_prices),
                
                # Divergence Patterns
                'bullish_divergence': self._detect_bullish_divergence(high_prices, low_prices, close_prices),
                'bearish_divergence': self._detect_bearish_divergence(high_prices, low_prices, close_prices),
            }
            
            # Pattern significance analysis
            patterns['pattern_analysis'] = self._analyze_patterns(patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting chart patterns: {e}")
            return {}
    
    def _analyze_trend(self, indicators: Dict) -> Dict:
        """Comprehensive trend analysis"""
        
        try:
            mas = indicators.get('moving_averages', {})
            momentum = indicators.get('momentum', {})
            
            trend_signals = []
            trend_weights = []
            
            # 1. Moving Average Trend (Weight: 0.30)
            ma_trend = self._analyze_ma_trend(mas)
            trend_signals.append(ma_trend['direction_score'])
            trend_weights.append(0.30)
            
            # 2. ADX Trend Strength (Weight: 0.25)
            adx_trend = self._analyze_adx_trend(momentum)
            trend_signals.append(adx_trend['strength_score'])
            trend_weights.append(0.25)
            
            # 3. MACD Trend (Weight: 0.20)
            macd_trend = self._analyze_macd_trend(momentum)
            trend_signals.append(macd_trend['direction_score'])
            trend_weights.append(0.20)
            
            # 4. Price Action Trend (Weight: 0.15)
            price_trend = self._analyze_price_action_trend(indicators)
            trend_signals.append(price_trend['direction_score'])
            trend_weights.append(0.15)
            
            # 5. Momentum Trend (Weight: 0.10)
            momentum_trend = self._analyze_momentum_trend(momentum)
            trend_signals.append(momentum_trend['direction_score'])
            trend_weights.append(0.10)
            
            # Calculate weighted trend score
            weighted_trend_score = sum(signal * weight for signal, weight in zip(trend_signals, trend_weights))
            
            # Determine trend direction and strength
            if weighted_trend_score > 0.6:
                direction = TrendDirection.STRONG_BULLISH
                strength = SignalStrength.VERY_STRONG
            elif weighted_trend_score > 0.3:
                direction = TrendDirection.BULLISH
                strength = SignalStrength.STRONG
            elif weighted_trend_score > -0.3:
                direction = TrendDirection.NEUTRAL
                strength = SignalStrength.WEAK
            elif weighted_trend_score > -0.6:
                direction = TrendDirection.BEARISH
                strength = SignalStrength.STRONG
            else:
                direction = TrendDirection.STRONG_BEARISH
                strength = SignalStrength.VERY_STRONG
            
            return {
                'direction': direction,
                'strength': strength,
                'score': weighted_trend_score,
                'confidence': min(abs(weighted_trend_score), 1.0),
                'components': {
                    'ma_trend': ma_trend,
                    'adx_trend': adx_trend,
                    'macd_trend': macd_trend,
                    'price_trend': price_trend,
                    'momentum_trend': momentum_trend,
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {e}")
            return {'direction': TrendDirection.NEUTRAL, 'strength': SignalStrength.WEAK, 'score': 0.0}
    
    def _generate_trading_signal(self, analysis: Dict) -> TechnicalSignal:
        """Generate comprehensive trading signal from analysis"""
        
        try:
            trend = analysis['signals']['trend']
            momentum_sig = analysis['signals']['momentum']
            reversal = analysis['signals']['reversal']
            sr_levels = analysis['market_structure']['sr_levels']
            risk_metrics = analysis['risk_metrics']
            
            current_price = analysis['current_price']
            
            # Determine signal direction
            trend_score = trend['score']
            momentum_score = momentum_sig.get('score', 0)
            reversal_probability = reversal.get('probability', 0)
            
            # Combined signal score
            signal_score = (trend_score * 0.5) + (momentum_score * 0.3) - (reversal_probability * 0.2)
            
            # Determine direction
            if signal_score > 0.4:
                direction = TrendDirection.BULLISH if signal_score < 0.7 else TrendDirection.STRONG_BULLISH
                signal_direction = 'long'
            elif signal_score < -0.4:
                direction = TrendDirection.BEARISH if signal_score > -0.7 else TrendDirection.STRONG_BEARISH
                signal_direction = 'short'
            else:
                direction = TrendDirection.NEUTRAL
                signal_direction = 'none'
            
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
            
            # Calculate stop loss and take profit
            atr = risk_metrics.get('atr_14', 0.001)
            
            if signal_direction == 'long':
                stop_loss = current_price - (atr * 2.0)
                take_profit = current_price + (atr * 3.0)
            elif signal_direction == 'short':
                stop_loss = current_price + (atr * 2.0)
                take_profit = current_price - (atr * 3.0)
            else:
                stop_loss = current_price
                take_profit = current_price
            
            # Adjust based on support/resistance
            nearest_support = sr_levels.get('nearest_support', current_price)
            nearest_resistance = sr_levels.get('nearest_resistance', current_price)
            
            if signal_direction == 'long' and nearest_support:
                stop_loss = max(stop_loss, nearest_support - (atr * 0.5))
            elif signal_direction == 'short' and nearest_resistance:
                stop_loss = min(stop_loss, nearest_resistance + (atr * 0.5))
            
            return TechnicalSignal(
                direction=direction,
                strength=strength,
                confidence=min(abs(signal_score), 1.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timeframe=analysis['timeframe'],
                timestamp=analysis['timestamp'],
                indicators_used=['trend', 'momentum', 'reversal', 'support_resistance', 'volatility'],
                signal_details={
                    'signal_score': signal_score,
                    'trend_score': trend_score,
                    'momentum_score': momentum_score,
                    'reversal_probability': reversal_probability,
                    'atr_used': atr,
                    'nearest_support': nearest_support,
                    'nearest_resistance': nearest_resistance,
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
                timestamp=pd.Timestamp.now(),
                indicators_used=[],
                signal_details={}
            )

    # Additional helper methods would continue here...
    # (Due to length constraints, I'm showing the structure)
    
    def _analyze_rsi(self, rsi_values: np.ndarray) -> Dict:
        """Analyze RSI indicator"""
        if len(rsi_values) < 2 or np.isnan(rsi_values[-1]):
            return {'signal': 'neutral', 'level': 'normal', 'divergence': False}
        
        current_rsi = rsi_values[-1]
        previous_rsi = rsi_values[-2]
        
        # RSI levels
        if current_rsi > 70:
            level = 'overbought'
        elif current_rsi < 30:
            level = 'oversold'
        else:
            level = 'normal'
        
        # RSI direction
        if current_rsi > previous_rsi + 2:
            signal = 'bullish'
        elif current_rsi < previous_rsi - 2:
            signal = 'bearish'
        else:
            signal = 'neutral'
            
        return {
            'signal': signal,
            'level': level,
            'value': current_rsi,
            'change': current_rsi - previous_rsi,
            'divergence': self._detect_rsi_divergence(rsi_values)
        }

    def _analyze_macd(self, momentum: Dict) -> Dict:
        """Analyze MACD indicator"""
        macd_line = momentum.get('macd_line', np.array([]))
        macd_signal = momentum.get('macd_signal', np.array([]))
        macd_histogram = momentum.get('macd_histogram', np.array([]))
        
        if len(macd_line) < 2:
            return {'signal': 'neutral', 'crossover': False, 'histogram_direction': 'flat'}
        
        current_macd = macd_line[-1]
        current_signal = macd_signal[-1]
        current_hist = macd_histogram[-1]
        previous_hist = macd_histogram[-2]
        
        # MACD signal
        if current_macd > current_signal:
            signal = 'bullish'
        else:
            signal = 'bearish'
            
        # Histogram direction
        if current_hist > previous_hist:
            hist_direction = 'increasing'
        elif current_hist < previous_hist:
            hist_direction = 'decreasing'
        else:
            hist_direction = 'flat'
            
        # Check for crossovers
        previous_macd = macd_line[-2]
        previous_signal_line = macd_signal[-2]
        
        bullish_crossover = (previous_macd <= previous_signal_line and current_macd > current_signal)
        bearish_crossover = (previous_macd >= previous_signal_line and current_macd < current_signal)
        
        return {
            'signal': signal,
            'histogram_direction': hist_direction,
            'bullish_crossover': bullish_crossover,
            'bearish_crossover': bearish_crossover,
            'macd_value': current_macd,
            'signal_value': current_signal,
            'histogram_value': current_hist
        }

# Continue with additional helper methods...

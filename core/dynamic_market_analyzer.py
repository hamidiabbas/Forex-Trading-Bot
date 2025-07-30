# core/dynamic_market_analyzer.py
import numpy as np
from datetime import datetime, timedelta
from enum import Enum

class MarketRegime(Enum):
    STRONG_TREND = "strong_trend"
    WEAK_TREND = "weak_trend" 
    RANGING = "ranging"
    REVERSAL = "reversal"
    BREAKOUT = "breakout"

class TrendStrength(Enum):
    VERY_STRONG = 4
    STRONG = 3
    MODERATE = 2
    WEAK = 1
    NONE = 0

class DynamicMarketAnalyzer:
    def __init__(self, symbol_manager, data_manager):
        self.symbol_manager = symbol_manager
        self.data_manager = data_manager
        self.trend_history = {}
        self.regime_history = {}
        
    def analyze_current_market_state(self, symbol, timeframes=['M15', 'H1', 'H4']):
        """Comprehensive multi-timeframe market analysis"""
        
        market_state = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'regime': None,
            'trend_strength': None,
            'reversal_probability': 0.0,
            'momentum_score': 0.0,
            'volatility_level': 0.0,
            'support_resistance': {},
            'action_required': False
        }
        
        # Multi-timeframe analysis
        timeframe_scores = {}
        for tf in timeframes:
            tf_analysis = self._analyze_timeframe(symbol, tf)
            timeframe_scores[tf] = tf_analysis
            
        # Consensus analysis
        market_state.update(self._calculate_consensus(timeframe_scores))
        
        # Store historical data
        self._update_history(symbol, market_state)
        
        return market_state
    
    def _analyze_timeframe(self, symbol, timeframe):
        """Detailed single timeframe analysis"""
        
        # Get price data
        prices = self.data_manager.get_ohlcv(symbol, timeframe, 100)
        
        if len(prices) < 50:
            return None
            
        # Calculate technical indicators
        analysis = {
            'trend_direction': self._calculate_trend_direction(prices),
            'trend_strength': self._calculate_trend_strength(prices),
            'momentum': self._calculate_momentum(prices),
            'volatility': self._calculate_volatility(prices),
            'support_resistance': self._find_support_resistance(prices),
            'reversal_signals': self._detect_reversal_signals(prices)
        }
        
        return analysis
    
    def _calculate_trend_direction(self, prices):
        """Calculate trend direction using multiple methods"""
        close_prices = prices['close'].values
        
        # EMA trend
        ema_20 = self._ema(close_prices, 20)
        ema_50 = self._ema(close_prices, 50)
        
        # ADX for trend strength
        adx = self._calculate_adx(prices)
        
        # Price action trend
        recent_highs = prices['high'].rolling(10).max()
        recent_lows = prices['low'].rolling(10).min()
        
        trend_signals = []
        
        # EMA crossover
        if ema_20[-1] > ema_50[-1]:
            trend_signals.append(1)
        else:
            trend_signals.append(-1)
            
        # ADX confirmation
        if adx[-1] > 25:  # Strong trend
            if close_prices[-1] > close_prices[-5]:
                trend_signals.append(1)
            else:
                trend_signals.append(-1)
        else:
            trend_signals.append(0)  # Weak trend
            
        # Price action
        if recent_highs.iloc[-1] > recent_highs.iloc[-5]:
            trend_signals.append(1)
        elif recent_lows.iloc[-1] < recent_lows.iloc[-5]:
            trend_signals.append(-1)
        else:
            trend_signals.append(0)
            
        # Consensus
        avg_signal = np.mean(trend_signals)
        
        if avg_signal > 0.5:
            return "bullish"
        elif avg_signal < -0.5:
            return "bearish"
        else:
            return "neutral"
    
    def _detect_reversal_signals(self, prices):
        """Advanced reversal detection using multiple confirmations"""
        
        reversal_probability = 0.0
        reversal_signals = []
        
        # 1. Divergence detection
        divergence_score = self._detect_divergence(prices)
        reversal_signals.append(divergence_score)
        
        # 2. Support/Resistance breaks
        sr_break_score = self._detect_sr_breaks(prices)
        reversal_signals.append(sr_break_score)
        
        # 3. Volume confirmation
        volume_score = self._analyze_volume_pattern(prices)
        reversal_signals.append(volume_score)
        
        # 4. Candlestick patterns
        pattern_score = self._detect_reversal_patterns(prices)
        reversal_signals.append(pattern_score)
        
        # 5. Moving average breaks
        ma_break_score = self._detect_ma_breaks(prices)
        reversal_signals.append(ma_break_score)
        
        # Calculate weighted probability
        weights = [0.25, 0.20, 0.15, 0.20, 0.20]  # Adjust based on effectiveness
        reversal_probability = sum(score * weight for score, weight in zip(reversal_signals, weights))
        
        return {
            'probability': max(0, min(1, reversal_probability)),
            'signals': {
                'divergence': reversal_signals[0],
                'sr_break': reversal_signals[1],
                'volume': reversal_signals[2],
                'patterns': reversal_signals[3],
                'ma_break': reversal_signals[4]
            }
        }

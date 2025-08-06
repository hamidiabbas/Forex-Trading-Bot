"""
Enhanced Strategy Manager - COMPLETE PRODUCTION VERSION
Maintains all existing advanced strategies while adding missing methods for trading bot integration
Handles multiple trading strategies with regime-based selection and RL compatibility
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    ✅ ENHANCED: Comprehensive strategy manager for forex trading
    Maintains all existing advanced strategies while adding missing methods for bot integration
    """
    
    def __init__(self, config, market_intelligence):
        self.config = config
        self.market_intelligence = market_intelligence
        self.logger = logging.getLogger(__name__)
        
        # Strategy configuration with enhanced parameters
        self.trend_fast_period = getattr(config, 'TREND_EMA_FAST_PERIOD', 12)
        self.trend_slow_period = getattr(config, 'TREND_EMA_SLOW_PERIOD', 26)
        self.rsi_period = getattr(config, 'RSI_PERIOD', 14)
        self.bb_period = getattr(config, 'BBANDS_PERIOD', 20)
        self.bb_std = getattr(config, 'BBANDS_STD', 2.0)
        
        # Risk management parameters
        self.rsi_overbought = getattr(config, 'RSI_OVERBOUGHT', 70)
        self.rsi_oversold = getattr(config, 'RSI_OVERSOLD', 30)
        self.range_rsi_overbought = getattr(config, 'RANGE_RSI_OVERBOUGHT', 75)
        self.range_rsi_oversold = getattr(config, 'RANGE_RSI_OVERSOLD', 25)
        
        # Enhanced strategy parameters
        self.strategy_params = {
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'profit_ratio': 1.5,
            'confidence_threshold': 0.3,
            'lookback_periods': [5, 10, 20],
            'momentum_threshold': 0.01,
            'volatility_threshold': 0.02
        }
        
        # Performance tracking
        self.signal_history = []
        self.strategy_performance = {}
        self.regime_performance = {}
        
        # Strategy weights based on regime
        self.regime_strategy_weights = {
            'trending': {'trend_following': 0.4, 'momentum': 0.3, 'breakout': 0.2, 'mean_reversion': 0.1},
            'ranging': {'mean_reversion': 0.4, 'support_resistance': 0.3, 'oscillator': 0.2, 'neutral': 0.1},
            'high_volatility': {'breakout': 0.4, 'momentum': 0.3, 'volatility': 0.2, 'trend_following': 0.1},
            'low_volatility': {'mean_reversion': 0.3, 'neutral': 0.3, 'oscillator': 0.2, 'support_resistance': 0.2},
            'normal': {'trend_following': 0.25, 'mean_reversion': 0.25, 'momentum': 0.25, 'neutral': 0.25}
        }
        
        self.logger.info("StrategyManager initialized successfully")

    def evaluate_signals(self, symbol: str, data: Union[pd.DataFrame, Dict], regime: str = 'normal') -> Optional[Dict[str, Any]]:
        """
        ✅ FIXED: Master signal evaluation method with correct signature
        Compatible with both new calling pattern and existing advanced strategies
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            data: DataFrame or dictionary containing market data
            regime: Current market regime
            
        Returns:
            Signal dictionary if valid setup found, None otherwise
        """
        try:
            # Handle different input data types
            df = self._extract_dataframe(data)
            if df is None or len(df) < 50:
                logger.warning(f"Insufficient data for signal evaluation: {symbol}")
                return None
            
            # Get current index (last available data point)
            current_index = len(df) - 1
            
            # Ensure we have enough historical data
            min_required = max(self.trend_slow_period, self.rsi_period, self.bb_period, 50)
            if current_index < min_required:
                logger.warning(f"Not enough data for {symbol}: {current_index} < {min_required}")
                return None
            
            # Add technical indicators if not present
            df = self._ensure_technical_indicators(df)
            
            logger.debug(f"Evaluating signals for {symbol} in {regime} regime")
            
            # Strategy selection based on regime with enhanced logic
            signal = None
            regime_normalized = self._normalize_regime(regime)
            
            # Get regime-specific strategy weights
            strategy_weights = self.regime_strategy_weights.get(regime_normalized, self.regime_strategy_weights['normal'])
            
            # Try strategies in order of preference based on regime
            for strategy_name, weight in sorted(strategy_weights.items(), key=lambda x: x[1], reverse=True):
                if weight > 0.1:  # Only try strategies with significant weight
                    signal = self._evaluate_regime_strategy(symbol, df, current_index, regime_normalized, strategy_name)
                    if signal:
                        signal['regime_weight'] = weight
                        break
            
            # Fallback to legacy strategy evaluation if no signal
            if not signal:
                signal = self._evaluate_legacy_strategies(symbol, df, current_index, regime)
            
            # Enhance signal with regime context
            if signal:
                signal['regime'] = regime
                signal['timestamp'] = datetime.now()
                signal['data_quality'] = len(df)
                signal['regime_normalized'] = regime_normalized
                
                # Track signal for performance analysis
                self._track_signal(signal, symbol)
                
                logger.info(f"✅ {signal['strategy']} signal generated for {symbol}: {signal['direction']} (confidence: {signal['confidence']:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error evaluating signals for {symbol}: {e}")
            return None

    def _extract_dataframe(self, data: Union[pd.DataFrame, Dict]) -> Optional[pd.DataFrame]:
        """Extract DataFrame from various input types"""
        try:
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, dict):
                # Try different keys in order of preference
                for key in ['EXECUTION', 'M15', 'BIAS', 'H1', 'execution', 'm15']:
                    if key in data and isinstance(data[key], pd.DataFrame):
                        return data[key]
                # If no standard keys, get first DataFrame
                for value in data.values():
                    if isinstance(value, pd.DataFrame) and len(value) > 0:
                        return value
            return None
        except Exception as e:
            logger.error(f"Error extracting DataFrame: {e}")
            return None

    def _normalize_regime(self, regime: str) -> str:
        """Normalize regime names to standard categories"""
        regime_lower = regime.lower()
        
        if any(term in regime_lower for term in ['trend', 'bullish', 'bearish', 'momentum']):
            return 'trending'
        elif any(term in regime_lower for term in ['range', 'consolidat', 'sideways']):
            return 'ranging'
        elif any(term in regime_lower for term in ['high_vol', 'volatile', 'breakout']):
            return 'high_volatility'
        elif any(term in regime_lower for term in ['low_vol', 'quiet', 'calm']):
            return 'low_volatility'
        else:
            return 'normal'

    def _ensure_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required technical indicators are present"""
        try:
            # Add basic indicators if missing
            if f'EMA{self.trend_fast_period}' not in df.columns:
                df[f'EMA{self.trend_fast_period}'] = df['Close'].ewm(span=self.trend_fast_period).mean()
            
            if f'EMA{self.trend_slow_period}' not in df.columns:
                df[f'EMA{self.trend_slow_period}'] = df['Close'].ewm(span=self.trend_slow_period).mean()
            
            if f'RSI{self.rsi_period}' not in df.columns:
                df[f'RSI{self.rsi_period}'] = self._calculate_rsi(df['Close'], self.rsi_period)
            
            # MACD
            if 'MACD_12_26_9' not in df.columns:
                macd_line, macd_signal = self._calculate_macd(df['Close'])
                df['MACD_12_26_9'] = macd_line
                df['MACDs_12_26_9'] = macd_signal
            
            # Bollinger Bands
            bb_cols = [f'BBU_{self.bb_period}_{self.bb_std}', f'BBL_{self.bb_period}_{self.bb_std}']
            if not all(col in df.columns for col in bb_cols):
                bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(df['Close'], self.bb_period, self.bb_std)
                df[f'BBU_{self.bb_period}_{self.bb_std}'] = bb_upper
                df[f'BBL_{self.bb_period}_{self.bb_std}'] = bb_lower
                df[f'BBM_{self.bb_period}_{self.bb_std}'] = bb_middle
            
            # ATR
            if 'ATRr_14' not in df.columns:
                df['ATRr_14'] = self._calculate_simple_atr(df, 14)
            
            return df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
        except Exception as e:
            logger.error(f"Error ensuring technical indicators: {e}")
            return df

    def _evaluate_regime_strategy(self, symbol: str, df: pd.DataFrame, i: int, regime: str, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Evaluate specific strategy based on regime"""
        try:
            if strategy_name == 'trend_following':
                return self.evaluate_trend_following_strategy(symbol, df, i)
            elif strategy_name == 'mean_reversion':
                return self.evaluate_mean_reversion_strategy(symbol, df, i)
            elif strategy_name == 'momentum':
                return self._evaluate_momentum_strategy(symbol, df, i)
            elif strategy_name == 'breakout':
                return self.evaluate_volatility_breakout_strategy(symbol, df, i)
            elif strategy_name == 'support_resistance':
                return self._evaluate_support_resistance_strategy(symbol, df, i)
            elif strategy_name == 'oscillator':
                return self._evaluate_oscillator_strategy(symbol, df, i)
            elif strategy_name == 'neutral':
                return self.evaluate_neutral_strategy(symbol, df, i)
            elif strategy_name == 'volatility':
                return self._evaluate_volatility_strategy(symbol, df, i)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error evaluating {strategy_name} strategy: {e}")
            return None

    def _evaluate_legacy_strategies(self, symbol: str, df: pd.DataFrame, i: int, regime: str) -> Optional[Dict[str, Any]]:
        """Fallback to legacy strategy evaluation methods"""
        try:
            # Create datadict for legacy compatibility
            datadict = {'EXECUTION': df}
            
            # Check for divergence first (highest priority)
            if regime.lower() in ['mean-reverting', 'ranging']:
                divergence_signal = self._check_rsi_divergence(df, i)
                if divergence_signal:
                    return self.evaluate_divergence_reversal_strategy(symbol, df, i, divergence_signal)
            
            # Try other legacy strategies
            strategies_to_try = [
                ('trending', self.evaluate_trend_following_strategy),
                ('ranging', self.evaluate_mean_reversion_strategy),
                ('high_volatility', self.evaluate_volatility_breakout_strategy),
                ('normal', self.evaluate_neutral_strategy)
            ]
            
            regime_normalized = self._normalize_regime(regime)
            
            for strategy_regime, strategy_func in strategies_to_try:
                if strategy_regime == regime_normalized or regime_normalized == 'normal':
                    try:
                        signal = strategy_func(symbol, df, i)
                        if signal:
                            return signal
                    except Exception as strategy_error:
                        logger.warning(f"Legacy strategy {strategy_func.__name__} failed: {strategy_error}")
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error in legacy strategy evaluation: {e}")
            return None

    def evaluate_divergence_reversal_strategy(self, symbol: str, df: pd.DataFrame, i: int, divergence_signal: str) -> Optional[Dict[str, Any]]:
        """
        ✅ PRESERVED: Aggressive counter-trend reversal strategy based on RSI divergence
        """
        try:
            logger.debug(f"--- Divergence Signal Detected: {divergence_signal} ---")
            
            if divergence_signal == 'BULLISH':
                return self.generate_signal(symbol, 'BUY', 'Divergence-Reversal', df, i)
            elif divergence_signal == 'BEARISH':
                return self.generate_signal(symbol, 'SELL', 'Divergence-Reversal', df, i)
                
            return None
            
        except Exception as e:
            logger.error(f"Error in divergence reversal strategy: {e}")
            return None
    
    def evaluate_trend_following_strategy(self, symbol: str, df: pd.DataFrame, i: int) -> Optional[Dict[str, Any]]:
        """
        ✅ PRESERVED: Trend following strategy using EMA crossover, MACD, and RSI confluence
        """
        try:
            if i < self.trend_slow_period:
                return None
            
            # Get indicator columns
            fast_ema_col = f'EMA{self.trend_fast_period}'
            slow_ema_col = f'EMA{self.trend_slow_period}'
            rsi_col = f'RSI{self.rsi_period}'
            macd_col = 'MACD_12_26_9'
            macd_signal_col = 'MACDs_12_26_9'
            
            # Check if indicators exist
            required_cols = [fast_ema_col, slow_ema_col, rsi_col, macd_col, macd_signal_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing indicators for trend following: {missing_cols}")
                return None
            
            # Current and previous values
            if i == 0:
                return None
            
            prev_fast_ema = df[fast_ema_col].iloc[i-1]
            prev_slow_ema = df[slow_ema_col].iloc[i-1]
            last_fast_ema = df[fast_ema_col].iloc[i]
            last_slow_ema = df[slow_ema_col].iloc[i]
            last_macd = df[macd_col].iloc[i]
            last_macd_signal = df[macd_signal_col].iloc[i]
            last_rsi = df[rsi_col].iloc[i]
            
            # Bullish crossover
            is_bullish_cross = (prev_fast_ema <= prev_slow_ema and last_fast_ema > last_slow_ema)
            if is_bullish_cross and last_macd > last_macd_signal and last_rsi > 50:
                return self.generate_signal(symbol, 'BUY', 'Confluence-Trend', df, i)
            
            # Bearish crossover
            is_bearish_cross = (prev_fast_ema >= prev_slow_ema and last_fast_ema < last_slow_ema)
            if is_bearish_cross and last_macd < last_macd_signal and last_rsi < 50:
                return self.generate_signal(symbol, 'SELL', 'Confluence-Trend', df, i)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in trend following strategy: {e}")
            return None
    
    def evaluate_mean_reversion_strategy(self, symbol: str, df: pd.DataFrame, i: int) -> Optional[Dict[str, Any]]:
        """
        ✅ PRESERVED: Bollinger Band mean-reversion strategy with RSI filter
        """
        try:
            if i < max(self.bb_period, self.rsi_period):
                return None
            
            # Get indicator columns
            bb_upper_col = f'BBU_{self.bb_period}_{self.bb_std}'
            bb_lower_col = f'BBL_{self.bb_period}_{self.bb_std}'
            rsi_col = f'RSI{self.rsi_period}'
            
            # Check if indicators exist
            required_cols = [bb_upper_col, bb_lower_col, rsi_col]
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing indicators for mean reversion: {[col for col in required_cols if col not in df.columns]}")
                return None
            
            if i == 0:
                return None
            
            # Get values
            last_high = df['High'].iloc[i]
            prev_high = df['High'].iloc[i-1]
            last_low = df['Low'].iloc[i]
            prev_low = df['Low'].iloc[i-1]
            last_rsi = df[rsi_col].iloc[i]
            upper_band = df[bb_upper_col].iloc[i]
            lower_band = df[bb_lower_col].iloc[i]
            prev_upper_band = df[bb_upper_col].iloc[i-1]
            prev_lower_band = df[bb_lower_col].iloc[i-1]
            
            # Sell signal: price touches upper band and RSI overbought
            if (prev_high <= prev_upper_band and last_high >= upper_band and 
                last_rsi >= self.range_rsi_overbought):
                return self.generate_signal(symbol, 'SELL', 'Mean-Reversion', df, i)
            
            # Buy signal: price touches lower band and RSI oversold
            if (prev_low >= prev_lower_band and last_low <= lower_band and 
                last_rsi <= self.range_rsi_oversold):
                return self.generate_signal(symbol, 'BUY', 'Mean-Reversion', df, i)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in mean reversion strategy: {e}")
            return None
    
    def evaluate_volatility_breakout_strategy(self, symbol: str, df: pd.DataFrame, i: int) -> Optional[Dict[str, Any]]:
        """
        ✅ PRESERVED: High volatility breakout strategy using ATR and volume
        """
        try:
            atr_col = 'ATRr_14'  # Assuming 14-period ATR
            
            if atr_col not in df.columns:
                logger.warning(f"ATR column {atr_col} not found for volatility breakout")
                return None
            
            if i == 0:
                return None
            
            # Get current price and ATR
            current_price = df['Close'].iloc[i]
            prev_price = df['Close'].iloc[i-1]
            current_atr = df[atr_col].iloc[i]
            
            # Calculate price change
            price_change = abs(current_price - prev_price)
            
            # Breakout if price change exceeds 1.5 * ATR
            if price_change > (1.5 * current_atr):
                direction = 'BUY' if current_price > prev_price else 'SELL'
                return self.generate_signal(symbol, direction, 'Volatility-Breakout', df, i)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in volatility breakout strategy: {e}")
            return None
    
    def evaluate_neutral_strategy(self, symbol: str, df: pd.DataFrame, i: int) -> Optional[Dict[str, Any]]:
        """
        ✅ PRESERVED: Conservative strategy for neutral market conditions
        """
        try:
            # Use a combination of RSI and MACD for neutral markets
            rsi_col = f'RSI{self.rsi_period}'
            macd_col = 'MACD_12_26_9'
            macd_signal_col = 'MACDs_12_26_9'
            
            required_cols = [rsi_col, macd_col, macd_signal_col]
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing indicators for neutral strategy: {[col for col in required_cols if col not in df.columns]}")
                return None
            
            last_rsi = df[rsi_col].iloc[i]
            last_macd = df[macd_col].iloc[i]
            last_macd_signal = df[macd_signal_col].iloc[i]
            
            # Conservative buy signal
            if last_rsi < 35 and last_macd > last_macd_signal:
                return self.generate_signal(symbol, 'BUY', 'Neutral-Conservative', df, i)
            
            # Conservative sell signal
            if last_rsi > 65 and last_macd < last_macd_signal:
                return self.generate_signal(symbol, 'SELL', 'Neutral-Conservative', df, i)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in neutral strategy: {e}")
            return None

    def _evaluate_momentum_strategy(self, symbol: str, df: pd.DataFrame, i: int) -> Optional[Dict[str, Any]]:
        """
        ✅ NEW: Momentum-based strategy
        """
        try:
            if i < 20:
                return None
            
            # Calculate momentum over multiple timeframes
            momentum_5 = (df['Close'].iloc[i] - df['Close'].iloc[i-5]) / df['Close'].iloc[i-5]
            momentum_10 = (df['Close'].iloc[i] - df['Close'].iloc[i-10]) / df['Close'].iloc[i-10]
            momentum_20 = (df['Close'].iloc[i] - df['Close'].iloc[i-20]) / df['Close'].iloc[i-20]
            
            # RSI confirmation
            rsi_col = f'RSI{self.rsi_period}'
            if rsi_col in df.columns:
                current_rsi = df[rsi_col].iloc[i]
            else:
                current_rsi = 50
            
            # Strong bullish momentum
            if (momentum_5 > 0.01 and momentum_10 > 0.005 and momentum_20 > 0.002 and current_rsi > 45):
                return self.generate_signal(symbol, 'BUY', 'Momentum-Strategy', df, i)
            
            # Strong bearish momentum
            if (momentum_5 < -0.01 and momentum_10 < -0.005 and momentum_20 < -0.002 and current_rsi < 55):
                return self.generate_signal(symbol, 'SELL', 'Momentum-Strategy', df, i)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in momentum strategy: {e}")
            return None

    def _evaluate_support_resistance_strategy(self, symbol: str, df: pd.DataFrame, i: int) -> Optional[Dict[str, Any]]:
        """
        ✅ NEW: Support and resistance strategy
        """
        try:
            if i < 50:
                return None
            
            # Calculate support and resistance levels
            recent_highs = df['High'].iloc[i-20:i].max()
            recent_lows = df['Low'].iloc[i-20:i].min()
            current_price = df['Close'].iloc[i]
            
            # ATR for dynamic levels
            atr = self._calculate_simple_atr(df, 14).iloc[i]
            
            # Buy at support
            if abs(current_price - recent_lows) < atr * 0.5:
                return self.generate_signal(symbol, 'BUY', 'Support-Resistance', df, i)
            
            # Sell at resistance
            if abs(current_price - recent_highs) < atr * 0.5:
                return self.generate_signal(symbol, 'SELL', 'Support-Resistance', df, i)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in support resistance strategy: {e}")
            return None

    def _evaluate_oscillator_strategy(self, symbol: str, df: pd.DataFrame, i: int) -> Optional[Dict[str, Any]]:
        """
        ✅ NEW: Oscillator-based strategy
        """
        try:
            rsi_col = f'RSI{self.rsi_period}'
            if rsi_col not in df.columns:
                return None
            
            current_rsi = df[rsi_col].iloc[i]
            
            # Multiple oscillator confirmation
            signals = 0
            
            # RSI signals
            if current_rsi < 30:
                signals += 1
            elif current_rsi > 70:
                signals -= 1
            
            # MACD confirmation
            if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
                macd = df['MACD_12_26_9'].iloc[i]
                macd_signal = df['MACDs_12_26_9'].iloc[i]
                
                if macd > macd_signal:
                    signals += 0.5
                else:
                    signals -= 0.5
            
            # Generate signal
            if signals >= 1.5:
                return self.generate_signal(symbol, 'BUY', 'Oscillator-Strategy', df, i)
            elif signals <= -1.5:
                return self.generate_signal(symbol, 'SELL', 'Oscillator-Strategy', df, i)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in oscillator strategy: {e}")
            return None

    def _evaluate_volatility_strategy(self, symbol: str, df: pd.DataFrame, i: int) -> Optional[Dict[str, Any]]:
        """
        ✅ NEW: Volatility-based strategy
        """
        try:
            if i < 20:
                return None
            
            # Calculate volatility
            returns = df['Close'].pct_change()
            current_vol = returns.iloc[i-20:i].std()
            historical_vol = returns.iloc[i-50:i-20].std() if i >= 50 else current_vol
            
            # Volatility expansion/contraction
            vol_ratio = current_vol / (historical_vol + 1e-10)
            
            # Price momentum
            momentum = (df['Close'].iloc[i] - df['Close'].iloc[i-10]) / df['Close'].iloc[i-10]
            
            # High volatility with momentum
            if vol_ratio > 1.5 and abs(momentum) > 0.01:
                direction = 'BUY' if momentum > 0 else 'SELL'
                return self.generate_signal(symbol, direction, 'Volatility-Strategy', df, i)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in volatility strategy: {e}")
            return None

    def generate_signal(self, symbol: str, direction: str, strategy: str, df: pd.DataFrame, i: int) -> Dict[str, Any]:
        """
        ✅ PRESERVED: Generate standardized signal dictionary with enhancements
        """
        try:
            current_price = df['Close'].iloc[i]
            
            # Get ATR value with fallback
            if 'ATRr_14' in df.columns:
                atr_value = df['ATRr_14'].iloc[i]
            else:
                atr_value = self._calculate_simple_atr(df, 14).iloc[i]
            
            # Fallback ATR calculation
            if pd.isna(atr_value) or atr_value <= 0:
                atr_value = current_price * 0.001  # 0.1% of current price as fallback
            
            # Calculate stop loss and take profit levels
            atr_multiplier = self.strategy_params['atr_multiplier']
            profit_ratio = self.strategy_params['profit_ratio']
            
            if direction == 'BUY':
                stop_loss = current_price - (atr_value * atr_multiplier)
                take_profit = current_price + (atr_value * atr_multiplier * profit_ratio)
            else:
                stop_loss = current_price + (atr_value * atr_multiplier)
                take_profit = current_price - (atr_value * atr_multiplier * profit_ratio)
            
            # Calculate confidence
            confidence = self._calculate_confidence(strategy, df, i)
            
            signal = {
                'symbol': symbol,
                'direction': direction,
                'strategy': strategy,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr_at_signal': atr_value,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'bar_index': i,
                'risk_reward_ratio': abs(take_profit - current_price) / abs(current_price - stop_loss)
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None

    def _check_rsi_divergence(self, df: pd.DataFrame, i: int) -> Optional[str]:
        """
        ✅ PRESERVED: Check for RSI divergence patterns
        """
        try:
            rsi_col = f'RSI{self.rsi_period}'
            if rsi_col not in df.columns or i < 20:
                return None
            
            # Look for divergence in last 10 bars
            lookback = 10
            if i < lookback:
                return None
            
            recent_prices = df['Close'].iloc[i-lookback:i+1]
            recent_rsi = df[rsi_col].iloc[i-lookback:i+1]
            
            # Simple divergence detection
            price_trend = recent_prices.iloc[-1] - recent_prices.iloc[0]
            rsi_trend = recent_rsi.iloc[-1] - recent_rsi.iloc[0]
            
            # Bullish divergence: price falling but RSI rising
            if price_trend < 0 and rsi_trend > 2:
                return 'BULLISH'
            
            # Bearish divergence: price rising but RSI falling  
            if price_trend > 0 and rsi_trend < -2:
                return 'BEARISH'
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking RSI divergence: {e}")
            return None

    def _calculate_confidence(self, strategy: str, df: pd.DataFrame, i: int) -> float:
        """
        ✅ ENHANCED: Calculate confidence score for the signal based on strategy type and market conditions
        """
        try:
            base_confidence = {
                'Divergence-Reversal': 0.8,
                'Confluence-Trend': 0.75,
                'Mean-Reversion': 0.7,
                'Volatility-Breakout': 0.65,
                'Neutral-Conservative': 0.6,
                'Momentum-Strategy': 0.7,
                'Support-Resistance': 0.65,
                'Oscillator-Strategy': 0.6,
                'Volatility-Strategy': 0.65
            }
            
            confidence = base_confidence.get(strategy, 0.5)
            
            # Adjust based on recent strategy performance
            if strategy in self.strategy_performance:
                recent_performance = self.strategy_performance[strategy]
                if recent_performance > 0.6:
                    confidence += 0.1
                elif recent_performance < 0.4:
                    confidence -= 0.1
            
            # Market condition adjustments
            try:
                # Volume confirmation (if available)
                if 'Volume' in df.columns and i > 0:
                    current_volume = df['Volume'].iloc[i]
                    avg_volume = df['Volume'].iloc[max(0, i-20):i].mean()
                    if current_volume > avg_volume * 1.2:
                        confidence += 0.05
                
                # Volatility adjustment
                if 'ATRr_14' in df.columns:
                    current_atr = df['ATRr_14'].iloc[i]
                    avg_atr = df['ATRr_14'].iloc[max(0, i-20):i].mean()
                    atr_ratio = current_atr / (avg_atr + 1e-10)
                    
                    # Moderate volatility is preferred
                    if 0.8 <= atr_ratio <= 1.2:
                        confidence += 0.05
                    elif atr_ratio > 2.0 or atr_ratio < 0.5:
                        confidence -= 0.1
                        
            except Exception as adj_error:
                logger.debug(f"Confidence adjustment failed: {adj_error}")
            
            return np.clip(confidence, 0.3, 0.95)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    # ✅ NEW METHODS: Required helper methods for RL compatibility
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator manually"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return pd.Series([50] * len(prices), index=prices.index)

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD with signal line"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            return macd.fillna(0), macd_signal.fillna(0)
        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            return pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index)

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper.fillna(prices), lower.fillna(prices), sma.fillna(prices)
        except Exception as e:
            logger.error(f"Bollinger Bands calculation error: {e}")
            return prices, prices, prices

    def _calculate_simple_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate simple ATR for stop loss/take profit"""
        try:
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            # Fill NaN values
            atr = atr.fillna(df['Close'] * 0.01)
            
            return atr
            
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return pd.Series([df['Close'].mean() * 0.01] * len(df), index=df.index)

    def _track_signal(self, signal: Dict[str, Any], symbol: str):
        """
        ✅ PRESERVED: Track signal for performance analysis with enhancements
        """
        try:
            signal_record = {
                'symbol': symbol,
                'strategy': signal['strategy'],
                'direction': signal['direction'],
                'timestamp': signal['timestamp'],
                'confidence': signal['confidence'],
                'regime': signal.get('regime', 'unknown'),
                'entry_price': signal['entry_price']
            }
            
            self.signal_history.append(signal_record)
            
            # Keep only last 200 signals (increased from 100)
            if len(self.signal_history) > 200:
                self.signal_history = self.signal_history[-200:]
                
            # Update regime performance tracking
            regime = signal.get('regime', 'unknown')
            if regime not in self.regime_performance:
                self.regime_performance[regime] = {'signals': 0, 'success': 0}
            self.regime_performance[regime]['signals'] += 1
                
        except Exception as e:
            logger.error(f"Error tracking signal: {e}")
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """
        ✅ ENHANCED: Get performance statistics for all strategies with regime analysis
        """
        try:
            if not self.signal_history:
                return {'status': 'No signals generated yet'}
            
            # Count signals by strategy
            strategy_counts = {}
            regime_counts = {}
            for signal in self.signal_history:
                strategy = signal['strategy']
                regime = signal.get('regime', 'unknown')
                
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Calculate average confidence by strategy
            strategy_confidence = {}
            for strategy in strategy_counts:
                confidences = [s['confidence'] for s in self.signal_history if s['strategy'] == strategy]
                strategy_confidence[strategy] = np.mean(confidences)
            
            # Calculate average confidence by regime
            regime_confidence = {}
            for regime in regime_counts:
                confidences = [s['confidence'] for s in self.signal_history if s.get('regime') == regime]
                regime_confidence[regime] = np.mean(confidences) if confidences else 0.5
            
            return {
                'total_signals': len(self.signal_history),
                'strategy_counts': strategy_counts,
                'strategy_confidence': strategy_confidence,
                'regime_counts': regime_counts,
                'regime_confidence': regime_confidence,
                'regime_performance': self.regime_performance,
                'last_signal': self.signal_history[-1]['timestamp'] if self.signal_history else None,
                'active_strategies': list(strategy_counts.keys()),
                'recent_signals': len([s for s in self.signal_history if (datetime.now() - s['timestamp']).seconds < 3600])
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return {'error': str(e)}
    
    def update_strategy_performance(self, strategy: str, success: bool):
        """
        ✅ PRESERVED: Update strategy performance based on trade outcome
        """
        try:
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = 0.5
            
            # Simple moving average of success rate
            current_rate = self.strategy_performance[strategy]
            new_rate = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
            self.strategy_performance[strategy] = new_rate
            
            logger.debug(f"Updated {strategy} performance: {new_rate:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    def reset_performance_tracking(self):
        """
        ✅ PRESERVED: Reset all performance tracking data
        """
        try:
            self.signal_history.clear()
            self.strategy_performance.clear()
            self.regime_performance.clear()
            logger.info("Performance tracking data reset")
            
        except Exception as e:
            logger.error(f"Error resetting performance tracking: {e}")
    
    def shutdown(self):
        """
        ✅ PRESERVED: Clean shutdown of strategy manager
        """
        try:
            logger.info("Shutting down StrategyManager...")
            
            # Log final performance statistics
            performance = self.get_strategy_performance()
            if 'total_signals' in performance:
                logger.info(f"Total signals generated: {performance['total_signals']}")
                logger.info(f"Active strategies: {performance.get('active_strategies', [])}")
            
            # Clear data
            self.signal_history.clear()
            self.strategy_performance.clear()
            self.regime_performance.clear()
            
            logger.info("StrategyManager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during StrategyManager shutdown: {e}")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive strategy manager diagnostics"""
        return {
            'class': 'StrategyManager',
            'total_strategies': len(self.regime_strategy_weights),
            'signal_history_size': len(self.signal_history),
            'tracked_strategies': list(self.strategy_performance.keys()),
            'tracked_regimes': list(self.regime_performance.keys()),
            'strategy_params': self.strategy_params,
            'regime_strategy_weights': self.regime_strategy_weights,
            'methods_available': [
                'evaluate_signals', 'generate_signal', 'update_strategy_performance',
                'get_strategy_performance', 'reset_performance_tracking'
            ]
        }

# ✅ PRESERVED: Helper functions for strategy validation
def validate_strategy_config(config) -> bool:
    """
    Validate strategy configuration parameters
    """
    required_params = [
        'TREND_EMA_FAST_PERIOD',
        'TREND_EMA_SLOW_PERIOD', 
        'RSI_PERIOD',
        'BBANDS_PERIOD',
        'RSI_OVERBOUGHT',
        'RSI_OVERSOLD'
    ]
    
    for param in required_params:
        if not hasattr(config, param):
            logging.warning(f"Missing strategy config parameter: {param}")
            return False
    
    return True

def create_strategy_manager(config, market_intelligence):
    """
    Factory function to create StrategyManager instance
    """
    return StrategyManager(config, market_intelligence)

"""
Enhanced Market Intelligence with Advanced Pattern Recognition
Complete implementation with OBV overflow fix
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

class EnhancedMarketIntelligence:
    """
    Enhanced market intelligence with advanced analytics and pattern recognition
    """
    
    def __init__(self, data_handler, config):
        self.data_handler = data_handler
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.trend_threshold = config.get('market_intelligence.trend_threshold', 0.7)
        self.volatility_threshold = config.get('market_intelligence.volatility_threshold', 0.02)
        self.momentum_threshold = config.get('market_intelligence.momentum_threshold', 0.5)
        
        # Analysis periods
        self.trend_period = config.get('market_intelligence.trend_analysis_period', 50)
        self.volatility_period = config.get('market_intelligence.volatility_analysis_period', 20)
        self.momentum_period = config.get('market_intelligence.momentum_analysis_period', 14)
        
        # Performance tracking
        self.regime_history = []
        self.signal_history = []
        self.pattern_cache = {}
        
        # Signal generation parameters
        self.rsi_overbought = config.get('strategy.rsi_overbought', 70)
        self.rsi_oversold = config.get('strategy.rsi_oversold', 30)
        self.bb_threshold = config.get('strategy.bb_threshold', 0.95)
        
        self.logger.info("Enhanced MarketIntelligence initialized successfully")

    def identify_regime(self, df: pd.DataFrame) -> str:
        """Enhanced market regime identification with multi-factor analysis"""
        try:
            if df is None or len(df) < 50:
                self.logger.warning("Insufficient data for regime identification")
                return "Neutral"
            
            # Calculate multiple regime indicators
            trend_strength = self._calculate_enhanced_trend_strength(df)
            volatility_regime = self._calculate_enhanced_volatility_regime(df)
            momentum_regime = self._calculate_enhanced_momentum_regime(df)
            volume_regime = self._calculate_volume_regime(df)
            
            # Advanced regime scoring with weighted factors
            regime_scores = {
                'Trending': 0,
                'Mean-Reverting': 0,
                'High-Volatility': 0,
                'Neutral': 0
            }
            
            # Trend-based scoring (40% weight)
            if trend_strength > 0.8:
                regime_scores['Trending'] += 4
            elif trend_strength > 0.6:
                regime_scores['Trending'] += 3
            elif trend_strength > 0.4:
                regime_scores['Trending'] += 1
            elif trend_strength < 0.3:
                regime_scores['Mean-Reverting'] += 3
            elif trend_strength < 0.4:
                regime_scores['Mean-Reverting'] += 2
            
            # Volatility-based scoring (30% weight)
            if volatility_regime > 0.8:
                regime_scores['High-Volatility'] += 3
            elif volatility_regime > 0.6:
                regime_scores['High-Volatility'] += 2
            elif volatility_regime < 0.3:
                regime_scores['Mean-Reverting'] += 2
            elif volatility_regime < 0.4:
                regime_scores['Mean-Reverting'] += 1
            
            # Momentum-based scoring (20% weight)
            if momentum_regime > 0.7:
                regime_scores['Trending'] += 2
            elif momentum_regime > 0.5:
                regime_scores['Trending'] += 1
            elif momentum_regime < 0.3:
                regime_scores['Mean-Reverting'] += 2
            
            # Volume-based scoring (10% weight)
            if volume_regime > 0.6:
                regime_scores['Trending'] += 1
            elif volume_regime < 0.4:
                regime_scores['Mean-Reverting'] += 1
            
            # Determine regime
            max_score = max(regime_scores.values())
            if max_score == 0:
                regime = "Neutral"
            else:
                regime = max(regime_scores, key=regime_scores.get)
            
            # Apply regime smoothing to avoid rapid changes
            regime = self._smooth_regime_changes(regime)
            
            # Track regime history for analysis
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': regime,
                'trend_strength': trend_strength,
                'volatility_regime': volatility_regime,
                'momentum_regime': momentum_regime,
                'volume_regime': volume_regime,
                'scores': regime_scores.copy()
            })
            
            # Keep only last 100 regime readings
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            self.logger.debug(f"Regime: {regime} (T:{trend_strength:.2f}, V:{volatility_regime:.2f}, M:{momentum_regime:.2f})")
            return regime
            
        except Exception as e:
            self.logger.error(f"Error in enhanced regime identification: {e}")
            return "Neutral"

    def _smooth_regime_changes(self, current_regime: str) -> str:
        """Smooth regime changes to avoid whipsaws"""
        try:
            if len(self.regime_history) < 3:
                return current_regime
            
            # Get last 3 regime readings
            recent_regimes = [entry['regime'] for entry in self.regime_history[-3:]]
            
            # If all recent regimes are the same and different from current, require stronger signal
            if len(set(recent_regimes)) == 1 and recent_regimes[0] != current_regime:
                # Require at least 2 consecutive readings of new regime
                if len(self.regime_history) >= 2:
                    if self.regime_history[-1]['regime'] == current_regime:
                        return current_regime
                    else:
                        return recent_regimes[0]  # Keep previous regime
            
            return current_regime
            
        except Exception as e:
            self.logger.error(f"Error smoothing regime changes: {e}")
            return current_regime

    def _calculate_enhanced_trend_strength(self, df: pd.DataFrame) -> float:
        """Enhanced trend strength calculation with multiple indicators"""
        try:
            if 'Close' not in df.columns or len(df) < self.trend_period:
                return 0.5
            
            close_prices = df['Close'].tail(self.trend_period)
            trend_score = 0
            max_conditions = 0
            
            # 1. Multiple moving average analysis
            ma_periods = [10, 20, 50]
            mas = {}
            for period in ma_periods:
                if len(close_prices) >= period:
                    mas[f'MA_{period}'] = close_prices.rolling(period).mean()
            
            # MA alignment scoring
            if len(mas) >= 3:
                latest_price = close_prices.iloc[-1]
                ma_values = [mas[f'MA_{p}'].iloc[-1] for p in ma_periods if f'MA_{p}' in mas]
                
                # Perfect uptrend: Price > MA10 > MA20 > MA50
                if len(ma_values) == 3 and latest_price > ma_values[0] > ma_values[1] > ma_values[2]:
                    trend_score += 4
                elif len(ma_values) == 3 and latest_price < ma_values[0] < ma_values[1] < ma_values[2]:
                    trend_score += 4
                # Partial alignment
                elif len(ma_values) >= 2:
                    if (latest_price > ma_values[0] > ma_values[1]) or (latest_price < ma_values[0] < ma_values[1]):
                        trend_score += 2
                max_conditions += 4
            
            # 2. Slope consistency analysis
            for period in ma_periods:
                if f'MA_{period}' in mas:
                    ma_series = mas[f'MA_{period}'].dropna()
                    if len(ma_series) >= 10:
                        # Calculate slope over last 10 periods
                        slope = (ma_series.iloc[-1] - ma_series.iloc[-10]) / ma_series.iloc[-10]
                        if abs(slope) > 0.005:  # 0.5% slope threshold
                            trend_score += 1
                        if abs(slope) > 0.01:   # 1% slope threshold
                            trend_score += 1
                        max_conditions += 2
            
            # 3. ADX trend strength
            if 'ADX_14' in df.columns:
                latest_adx = df['ADX_14'].iloc[-1]
                if latest_adx > 30:
                    trend_score += 3
                elif latest_adx > 25:
                    trend_score += 2
                elif latest_adx > 20:
                    trend_score += 1
                max_conditions += 3
            
            # 4. Directional Movement analysis
            if 'DMP_14' in df.columns and 'DMN_14' in df.columns:
                di_plus = df['DMP_14'].iloc[-1]
                di_minus = df['DMN_14'].iloc[-1]
                di_diff = abs(di_plus - di_minus)
                if di_diff > 10:  # Strong directional bias
                    trend_score += 2
                elif di_diff > 5:
                    trend_score += 1
                max_conditions += 2
            
            # 5. Price momentum consistency
            momentum_periods = [5, 10, 20]
            consistent_momentum = 0
            for period in momentum_periods:
                if len(close_prices) >= period:
                    momentum = (close_prices.iloc[-1] - close_prices.iloc[-period]) / close_prices.iloc[-period]
                    if abs(momentum) > 0.01:  # 1% momentum threshold
                        consistent_momentum += 1
                        if abs(momentum) > 0.02:  # 2% threshold
                            consistent_momentum += 1
            
            # Reward consistent momentum across timeframes
            if consistent_momentum >= 4:
                trend_score += 2
            elif consistent_momentum >= 2:
                trend_score += 1
            max_conditions += 2
            
            # Calculate final trend strength
            if max_conditions > 0:
                trend_strength = min(1.0, trend_score / max_conditions)
            else:
                trend_strength = 0.5
            
            return trend_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced trend strength: {e}")
            return 0.5

    def _calculate_enhanced_volatility_regime(self, df: pd.DataFrame) -> float:
        """Enhanced volatility regime with multiple volatility measures"""
        try:
            if 'Close' not in df.columns:
                return 0.5
            
            close_prices = df['Close'].tail(self.volatility_period * 3)
            volatility_scores = []
            
            # 1. Returns-based volatility
            returns = close_prices.pct_change().dropna()
            if len(returns) >= self.volatility_period:
                current_vol = returns.tail(self.volatility_period).std()
                historical_vol = returns.std()
                if historical_vol > 0:
                    vol_ratio = current_vol / historical_vol
                    volatility_scores.append(min(2.0, vol_ratio))  # Cap at 2.0
            
            # 2. ATR-based volatility
            if 'ATRr_14' in df.columns:
                atr_series = df['ATRr_14'].dropna()
                if len(atr_series) >= self.volatility_period:
                    current_atr = atr_series.tail(self.volatility_period).mean()
                    historical_atr = atr_series.mean()
                    if historical_atr > 0:
                        atr_ratio = current_atr / historical_atr
                        volatility_scores.append(min(2.0, atr_ratio))
            
            # 3. Bollinger Band width
            if all(col in df.columns for col in ['BBU_20_2.0', 'BBL_20_2.0', 'BBM_20_2.0']):
                bb_width = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
                bb_series = bb_width.dropna()
                if len(bb_series) >= self.volatility_period:
                    current_bb_width = bb_series.tail(self.volatility_period).mean()
                    historical_bb_width = bb_series.mean()
                    if historical_bb_width > 0:
                        bb_ratio = current_bb_width / historical_bb_width
                        volatility_scores.append(min(2.0, bb_ratio))
            
            # 4. High-Low range volatility
            if all(col in df.columns for col in ['High', 'Low', 'Close']):
                hl_range = (df['High'] - df['Low']) / df['Close']
                range_series = hl_range.dropna()
                if len(range_series) >= self.volatility_period:
                    current_range = range_series.tail(self.volatility_period).mean()
                    historical_range = range_series.mean()
                    if historical_range > 0:
                        range_ratio = current_range / historical_range
                        volatility_scores.append(min(2.0, range_ratio))
            
            # 5. Intraday volatility (if available)
            if all(col in df.columns for col in ['Open', 'Close']):
                gap_volatility = abs(df['Open'] - df['Close'].shift()) / df['Close'].shift()
                gap_series = gap_volatility.dropna()
                if len(gap_series) >= self.volatility_period:
                    current_gap_vol = gap_series.tail(self.volatility_period).mean()
                    historical_gap_vol = gap_series.mean()
                    if historical_gap_vol > 0:
                        gap_ratio = current_gap_vol / historical_gap_vol
                        volatility_scores.append(min(2.0, gap_ratio))
            
            # Calculate weighted average volatility score
            if volatility_scores:
                # Give more weight to recent volatility measures
                weights = [1.5, 1.3, 1.0, 0.8, 0.6][:len(volatility_scores)]
                weighted_score = np.average(volatility_scores, weights=weights)
                return min(1.0, weighted_score)
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error calculating enhanced volatility regime: {e}")
            return 0.5

    def _calculate_enhanced_momentum_regime(self, df: pd.DataFrame) -> float:
        """Enhanced momentum regime with multiple momentum indicators"""
        try:
            momentum_scores = []
            
            # 1. RSI momentum analysis
            if 'RSI_14' in df.columns:
                rsi_series = df['RSI_14'].dropna()
                if len(rsi_series) >= 5:
                    current_rsi = rsi_series.iloc[-1]
                    rsi_change = rsi_series.iloc[-1] - rsi_series.iloc[-5]
                    
                    # RSI extreme momentum
                    extreme_momentum = max(abs(current_rsi - 20), abs(current_rsi - 80)) / 30
                    # RSI change momentum
                    change_momentum = abs(rsi_change) / 20
                    
                    rsi_momentum = (extreme_momentum + change_momentum) / 2
                    momentum_scores.append(min(1.0, rsi_momentum))
            
            # 2. MACD momentum analysis
            if all(col in df.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']):
                macd = df['MACD_12_26_9'].iloc[-1]
                macd_signal = df['MACDs_12_26_9'].iloc[-1]
                macd_histogram = df['MACDh_12_26_9'].iloc[-1]
                
                # MACD line vs signal line divergence
                macd_divergence = abs(macd - macd_signal)
                # Histogram momentum
                histogram_momentum = abs(macd_histogram)
                
                # Normalize MACD momentum
                macd_series = abs(df['MACD_12_26_9'] - df['MACDs_12_26_9']).dropna()
                if len(macd_series) > 0:
                    avg_macd_diff = macd_series.mean()
                    if avg_macd_diff > 0:
                        normalized_macd = min(1.0, macd_divergence / avg_macd_diff)
                        momentum_scores.append(normalized_macd)
            
            # 3. Multi-period price momentum
            if 'Close' in df.columns:
                close_prices = df['Close']
                momentum_periods = [3, 5, 10, 14, 20]
                period_scores = []
                
                for period in momentum_periods:
                    if len(close_prices) >= period:
                        price_momentum = abs(close_prices.iloc[-1] - close_prices.iloc[-period]) / close_prices.iloc[-period]
                        # Scale momentum (multiply by 20 to normalize)
                        scaled_momentum = min(1.0, price_momentum * 20)
                        period_scores.append(scaled_momentum)
                
                if period_scores:
                    # Weight shorter periods more heavily
                    weights = [2.0, 1.5, 1.2, 1.0, 0.8][:len(period_scores)]
                    weighted_momentum = np.average(period_scores, weights=weights)
                    momentum_scores.append(weighted_momentum)
            
            # 4. Stochastic momentum
            if 'STOCHk_14_3_3' in df.columns:
                stoch_k = df['STOCHk_14_3_3'].iloc[-1]
                # Distance from extreme levels (20 and 80)
                stoch_extreme = max(abs(stoch_k - 20), abs(stoch_k - 80))
                stoch_momentum = min(1.0, stoch_extreme / 30)
                momentum_scores.append(stoch_momentum)
            
            # 5. Volume-price momentum
            if 'Volume' in df.columns and df['Volume'].sum() > 0:
                volume_series = df['Volume'].dropna()
                if len(volume_series) >= 10 and len(df['Close']) >= 10:
                    # Volume-weighted price change
                    recent_volume = volume_series.tail(5).mean()
                    historical_volume = volume_series.mean()
                    
                    if historical_volume > 0:
                        volume_ratio = recent_volume / historical_volume
                        price_change = abs(df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
                        
                        # Combine volume and price momentum
                        vp_momentum = min(1.0, volume_ratio * price_change * 10)
                        momentum_scores.append(vp_momentum)
            
            # 6. Acceleration momentum (rate of change of momentum)
            if 'Close' in df.columns and len(df['Close']) >= 20:
                close_prices = df['Close']
                # Calculate momentum over two periods and compare
                recent_momentum = (close_prices.iloc[-1] - close_prices.iloc[-10]) / close_prices.iloc[-10]
                earlier_momentum = (close_prices.iloc[-10] - close_prices.iloc[-20]) / close_prices.iloc[-20]
                
                momentum_acceleration = abs(recent_momentum - earlier_momentum)
                acceleration_score = min(1.0, momentum_acceleration * 10)
                momentum_scores.append(acceleration_score)
            
            # Calculate final momentum score
            if momentum_scores:
                # Use weighted average with emphasis on price-based momentum
                if len(momentum_scores) >= 3:
                    weights = [1.0] * len(momentum_scores)
                    weights[0] = 1.2  # RSI
                    weights[2] = 1.3  # Price momentum (if exists)
                    return np.average(momentum_scores, weights=weights)
                else:
                    return np.mean(momentum_scores)
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error calculating enhanced momentum regime: {e}")
            return 0.5

    def _calculate_volume_regime(self, df: pd.DataFrame) -> float:
        """✅ FIXED: Calculate volume-based regime score with overflow protection"""
        try:
            if 'Volume' not in df.columns or df['Volume'].sum() == 0:
                return 0.5  # Neutral if no volume data
            
            volume_series = df['Volume'].dropna()
            if len(volume_series) < 20:
                return 0.5
            
            volume_scores = []
            
            # 1. Current vs historical volume ratio
            current_volume = volume_series.tail(5).mean()
            historical_volume = volume_series.mean()
            
            if historical_volume > 0:
                volume_ratio = min(2.0, current_volume / historical_volume)
                volume_scores.append(volume_ratio / 2.0)  # Normalize to 0-1
            
            # 2. Volume trend analysis
            if len(volume_series) >= 20:
                recent_vol_avg = volume_series.tail(10).mean()
                earlier_vol_avg = volume_series.tail(20).head(10).mean()
                
                if earlier_vol_avg > 0:
                    volume_trend = min(2.0, recent_vol_avg / earlier_vol_avg)
                    volume_scores.append(volume_trend / 2.0)
            
            # 3. Volume spikes detection
            if len(volume_series) >= 10:
                vol_mean = volume_series.mean()
                vol_std = volume_series.std()
                
                if vol_std > 0:
                    # Count recent volume spikes (above 1.5 standard deviations)
                    spike_threshold = vol_mean + (1.5 * vol_std)
                    recent_spikes = (volume_series.tail(5) > spike_threshold).sum()
                    spike_score = min(1.0, recent_spikes / 3)  # Normalize by max expected spikes
                    volume_scores.append(spike_score)
            
            # 4. ✅ FIXED: Safe On-Balance Volume analysis with overflow protection
            if 'Close' in df.columns and len(df['Close']) >= len(volume_series):
                try:
                    # Safe OBV calculation to prevent overflow
                    price_changes = df['Close'].diff().iloc[-len(volume_series):]
                    obv_changes = []
                    
                    for i, price_change in enumerate(price_changes):
                        if pd.isna(price_change) or i >= len(volume_series):
                            continue
                        
                        vol_value = float(volume_series.iloc[i])
                        
                        # ✅ FIXED: Cap extremely large volumes to prevent overflow
                        if vol_value > 1e12:  # Cap at 1 trillion
                            vol_value = 1e12
                        elif vol_value < 0:  # Ensure positive volume
                            vol_value = abs(vol_value)
                        
                        if price_change > 0:
                            obv_changes.append(vol_value)
                        elif price_change < 0:
                            # ✅ FIXED: Safe negative volume handling to prevent overflow
                            obv_changes.append(-min(vol_value, 1e12))  # Cap negative values too
                        else:
                            obv_changes.append(0)
                    
                    if len(obv_changes) >= 10:
                        # ✅ FIXED: Safe OBV ratio calculation with overflow protection
                        try:
                            recent_obv = sum(obv_changes[-5:])
                            historical_obv_avg = sum(obv_changes) / len(obv_changes)
                            
                            # Only calculate if meaningful volume and avoid division issues
                            if abs(historical_obv_avg) > 1e6 and abs(historical_obv_avg) < 1e11:
                                obv_ratio = abs(recent_obv / (historical_obv_avg * 5))
                                obv_score = min(1.0, obv_ratio)
                                volume_scores.append(obv_score)
                        except (OverflowError, ZeroDivisionError):
                            # Skip OBV score if calculation overflows
                            self.logger.debug("OBV calculation skipped due to overflow")
                            pass
                        
                except (OverflowError, ValueError) as e:
                    self.logger.debug(f"OBV calculation skipped due to overflow: {e}")
                    pass  # Skip OBV if calculation fails
            
            # Return average volume score
            if volume_scores:
                return np.mean(volume_scores)
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error calculating volume regime: {e}")
            return 0.5

    def generate_traditional_signal(self, symbol: str, data_dict: Dict[str, pd.DataFrame], regime: str) -> Optional[Dict[str, Any]]:
        """Generate traditional technical analysis signal based on regime"""
        try:
            execution_df = data_dict.get('EXECUTION')
            if execution_df is None or len(execution_df) < 50:
                return None
            
            # Get current market data
            current_price = execution_df['Close'].iloc[-1]
            atr_value = execution_df.get('ATRr_14', pd.Series([0.001])).iloc[-1]
            
            # Generate signal based on regime
            signal = None
            if regime == "Trending":
                signal = self._generate_trend_following_signal(execution_df)
            elif regime == "Mean-Reverting":
                signal = self._generate_mean_reverting_signal(execution_df)
            elif regime == "High-Volatility":
                signal = self._generate_volatility_breakout_signal(execution_df)
            else:  # Neutral
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
                
                # Add technical confirmation
                signal['confirmations'] = self._get_signal_confirmations(execution_df, signal['direction'])
                
                # Track signal history
                self.signal_history.append(signal.copy())
                if len(self.signal_history) > 100:
                    self.signal_history = self.signal_history[-100:]
                
                self.logger.debug(f"Traditional signal generated: {signal['direction']} ({signal['reason']})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating traditional signal: {e}")
            return None

    def _generate_trend_following_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate trend following signals"""
        try:
            signals = []
            
            # 1. EMA Crossover Strategy
            if 'EMA_20' in df.columns and 'EMA_50' in df.columns and len(df) >= 3:
                ema20_current = df['EMA_20'].iloc[-1]
                ema50_current = df['EMA_50'].iloc[-1]
                ema20_prev = df['EMA_20'].iloc[-2]
                ema50_prev = df['EMA_50'].iloc[-2]
                
                # Bullish crossover
                if ema20_prev <= ema50_prev and ema20_current > ema50_current:
                    signals.append({
                        'direction': 'BUY',
                        'reason': 'EMA_20_cross_above_EMA_50',
                        'strength': 0.8
                    })
                # Bearish crossover
                elif ema20_prev >= ema50_prev and ema20_current < ema50_current:
                    signals.append({
                        'direction': 'SELL',
                        'reason': 'EMA_20_cross_below_EMA_50',
                        'strength': 0.8
                    })
            
            # 2. MACD Trend Signal
            if all(col in df.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9']) and len(df) >= 3:
                macd_current = df['MACD_12_26_9'].iloc[-1]
                signal_current = df['MACDs_12_26_9'].iloc[-1]
                macd_prev = df['MACD_12_26_9'].iloc[-2]
                signal_prev = df['MACDs_12_26_9'].iloc[-2]
                
                # Bullish MACD crossover
                if macd_prev <= signal_prev and macd_current > signal_current and macd_current > 0:
                    signals.append({
                        'direction': 'BUY',
                        'reason': 'MACD_bullish_crossover',
                        'strength': 0.7
                    })
                # Bearish MACD crossover
                elif macd_prev >= signal_prev and macd_current < signal_current and macd_current < 0:
                    signals.append({
                        'direction': 'SELL',
                        'reason': 'MACD_bearish_crossover',
                        'strength': 0.7
                    })
            
            # 3. ADX Strong Trend with Directional Movement
            if all(col in df.columns for col in ['ADX_14', 'DMP_14', 'DMN_14']):
                adx = df['ADX_14'].iloc[-1]
                di_plus = df['DMP_14'].iloc[-1]
                di_minus = df['DMN_14'].iloc[-1]
                
                if adx > 25:  # Strong trend
                    if di_plus > di_minus + 5:  # Strong uptrend
                        signals.append({
                            'direction': 'BUY',
                            'reason': 'ADX_strong_uptrend',
                            'strength': 0.6
                        })
                    elif di_minus > di_plus + 5:  # Strong downtrend
                        signals.append({
                            'direction': 'SELL',
                            'reason': 'ADX_strong_downtrend',
                            'strength': 0.6
                        })
            
            # Return strongest signal
            if signals:
                best_signal = max(signals, key=lambda x: x['strength'])
                return best_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in trend following signal generation: {e}")
            return None

    def _generate_mean_reverting_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate mean reverting signals"""
        try:
            signals = []
            
            # 1. RSI Overbought/Oversold
            if 'RSI_14' in df.columns:
                rsi = df['RSI_14'].iloc[-1]
                
                if rsi < self.rsi_oversold:
                    signals.append({
                        'direction': 'BUY',
                        'reason': f'RSI_oversold_{rsi:.1f}',
                        'strength': 0.7
                    })
                elif rsi > self.rsi_overbought:
                    signals.append({
                        'direction': 'SELL',
                        'reason': f'RSI_overbought_{rsi:.1f}',
                        'strength': 0.7
                    })
            
            # 2. Bollinger Bands Mean Reversion
            if all(col in df.columns for col in ['Close', 'BBU_20_2.0', 'BBL_20_2.0', 'BBM_20_2.0']):
                price = df['Close'].iloc[-1]
                bb_upper = df['BBU_20_2.0'].iloc[-1]
                bb_lower = df['BBL_20_2.0'].iloc[-1]
                
                # Calculate position within bands
                bb_position = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
                
                if bb_position > self.bb_threshold:  # Near upper band
                    signals.append({
                        'direction': 'SELL',
                        'reason': f'BB_upper_reversion_{bb_position:.2f}',
                        'strength': 0.6
                    })
                elif bb_position < (1 - self.bb_threshold):  # Near lower band
                    signals.append({
                        'direction': 'BUY',
                        'reason': f'BB_lower_reversion_{bb_position:.2f}',
                        'strength': 0.6
                    })
            
            # Return strongest signal
            if signals:
                best_signal = max(signals, key=lambda x: x['strength'])
                return best_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in mean reverting signal generation: {e}")
            return None

    def _generate_volatility_breakout_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate volatility breakout signals"""
        try:
            signals = []
            
            # 1. Bollinger Band Breakout
            if all(col in df.columns for col in ['Close', 'BBU_20_2.0', 'BBL_20_2.0']):
                price = df['Close'].iloc[-1]
                bb_upper = df['BBU_20_2.0'].iloc[-1]
                bb_lower = df['BBL_20_2.0'].iloc[-1]
                
                if len(df) >= 2:
                    prev_price = df['Close'].iloc[-2]
                    
                    # Breakout above upper band
                    if prev_price <= bb_upper and price > bb_upper:
                        signals.append({
                            'direction': 'BUY',
                            'reason': 'BB_upper_breakout',
                            'strength': 0.7
                        })
                    # Breakout below lower band
                    elif prev_price >= bb_lower and price < bb_lower:
                        signals.append({
                            'direction': 'SELL',
                            'reason': 'BB_lower_breakout',
                            'strength': 0.7
                        })
            
            # Return strongest signal
            if signals:
                best_signal = max(signals, key=lambda x: x['strength'])
                return best_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in volatility breakout signal generation: {e}")
            return None

    def _generate_neutral_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate neutral market signals"""
        try:
            # In neutral markets, look for clear technical setups
            signals = []
            
            # Simple momentum signal
            if 'RSI_14' in df.columns and 'MACD_12_26_9' in df.columns:
                rsi = df['RSI_14'].iloc[-1]
                macd = df['MACD_12_26_9'].iloc[-1]
                
                # Conservative signals only
                if rsi < 25 and macd > 0:
                    signals.append({
                        'direction': 'BUY',
                        'reason': 'neutral_oversold_momentum',
                        'strength': 0.4
                    })
                elif rsi > 75 and macd < 0:
                    signals.append({
                        'direction': 'SELL',
                        'reason': 'neutral_overbought_momentum',
                        'strength': 0.4
                    })
            
            # Return signal if found
            if signals:
                return max(signals, key=lambda x: x['strength'])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in neutral signal generation: {e}")
            return None

    def _calculate_traditional_confidence(self, signal: Dict[str, Any], 
                                        df: pd.DataFrame, regime: str) -> float:
        """Calculate confidence score for traditional signals"""
        try:
            base_confidence = signal.get('strength', 0.5)
            
            # Regime alignment bonus
            regime_bonus = 0.1 if signal.get('reason', '').lower().find(regime.lower()) != -1 else 0
            
            # Technical confirmation bonus
            confirmations = self._get_signal_confirmations(df, signal['direction'])
            confirmation_bonus = len(confirmations) * 0.05
            
            # Volume confirmation
            volume_bonus = 0.05 if self._has_volume_confirmation(df, signal['direction']) else 0
            
            final_confidence = min(0.95, base_confidence + regime_bonus + confirmation_bonus + volume_bonus)
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating traditional confidence: {e}")
            return 0.5

    def _get_signal_confirmations(self, df: pd.DataFrame, direction: str) -> List[str]:
        """Get technical confirmations for signal"""
        try:
            confirmations = []
            
            if direction.upper() == 'BUY':
                # Look for bullish confirmations
                if 'RSI_14' in df.columns and df['RSI_14'].iloc[-1] < 50:
                    confirmations.append('RSI_bullish')
                if 'MACD_12_26_9' in df.columns and df['MACD_12_26_9'].iloc[-1] > 0:
                    confirmations.append('MACD_bullish')
            else:
                # Look for bearish confirmations
                if 'RSI_14' in df.columns and df['RSI_14'].iloc[-1] > 50:
                    confirmations.append('RSI_bearish')
                if 'MACD_12_26_9' in df.columns and df['MACD_12_26_9'].iloc[-1] < 0:
                    confirmations.append('MACD_bearish')
            
            return confirmations
            
        except Exception as e:
            self.logger.error(f"Error getting signal confirmations: {e}")
            return []

    def _has_volume_confirmation(self, df: pd.DataFrame, direction: str) -> bool:
        """Check for volume confirmation"""
        try:
            if 'Volume' not in df.columns or df['Volume'].sum() == 0:
                return False
            
            # Simple volume confirmation: above average volume
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].tail(20).mean()
            
            return current_volume > avg_volume * 1.2  # 20% above average
            
        except Exception as e:
            self.logger.error(f"Error checking volume confirmation: {e}")
            return False

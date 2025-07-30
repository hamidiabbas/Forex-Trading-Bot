# core/market_intelligence.py (COMPLETE VERSION - All Features Included)
"""
Complete Enhanced Market Intelligence with Full Technical Analysis Integration
This is the complete version with all methods and functionality
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import time

# Smart import for technical analysis
TECHNICAL_ANALYSIS_AVAILABLE = False
try:
    from .technical_analysis_smart import (
        ComprehensiveTechnicalAnalyzer, 
        TrendDirection, 
        SignalStrength, 
        TechnicalSignal
    )
    TECHNICAL_ANALYSIS_AVAILABLE = True
    print("✅ Enhanced Technical Analysis imported successfully")
except ImportError:
    try:
        from .technical_analysis import (
            ComprehensiveTechnicalAnalyzer, 
            TrendDirection, 
            SignalStrength, 
            TechnicalSignal
        )
        TECHNICAL_ANALYSIS_AVAILABLE = True
        print("✅ Standard Technical Analysis imported successfully")
    except ImportError:
        print("⚠️ Technical Analysis not available - using basic intelligence only")
        TECHNICAL_ANALYSIS_AVAILABLE = False

# Placeholder classes if technical analysis not available
if not TECHNICAL_ANALYSIS_AVAILABLE:
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
        confidence: float
        entry_price: float
        stop_loss: float
        take_profit: float
        timeframe: str
        timestamp: datetime
        indicators_used: List[str]
        signal_details: Dict

class MarketRegime(Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    NEUTRAL = "neutral"
    CONSOLIDATING = "consolidating"

class SignalQuality(Enum):
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    POOR = 2
    VERY_POOR = 1

@dataclass
class MarketConditions:
    regime: MarketRegime
    volatility_level: str
    trend_strength: float
    momentum_score: float
    support_resistance_levels: Dict
    market_session: str
    news_impact: str
    correlation_risk: float

@dataclass
class EnhancedSignal:
    symbol: str
    direction: str
    confidence: float
    quality: SignalQuality
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    strategy: str
    timeframe: str
    timestamp: datetime
    technical_analysis: Optional[Dict]
    market_conditions: MarketConditions
    risk_assessment: Dict
    multi_timeframe_confirmation: bool
    news_impact: Dict

class EnhancedMarketIntelligence:
    """
    Complete Enhanced Market Intelligence with Full Technical Analysis Integration
    This is the complete version with all methods and comprehensive functionality
    """
    
    def __init__(self, data_handler, config):
        # CRITICAL: Initialize logger FIRST before any other operations
        self.logger = logging.getLogger(__name__)
        
        # Core components initialization
        self.data_handler = data_handler
        self.config = config
        
        # Enhanced analysis configuration
        self.timeframes = ['M5', 'M15', 'H1', 'H4', 'D1']
        self.analysis_cache = {}
        self.signal_history = []
        self.performance_metrics = {}
        
        # Configuration parameters
        self.min_confidence = getattr(config, 'min_confidence', 0.6)
        self.trend_strength_threshold = getattr(config, 'trend_strength_threshold', 3)
        self.reversal_probability_threshold = getattr(config, 'reversal_probability_threshold', 0.7)
        self.multi_timeframe_confirmation = getattr(config, 'multi_timeframe_confirmation', True)
        self.volume_confirmation_required = getattr(config, 'volume_confirmation_required', False)
        
        # Market sessions
        self.market_sessions = {
            'sydney': {'start': 22, 'end': 6},
            'tokyo': {'start': 0, 'end': 9},
            'london': {'start': 8, 'end': 17},
            'new_york': {'start': 13, 'end': 22}
        }
        
        # Initialize technical analyzer
        self.enhanced_analysis_enabled = False
        self.technical_analyzer = None
        
        if TECHNICAL_ANALYSIS_AVAILABLE:
            try:
                self.technical_analyzer = ComprehensiveTechnicalAnalyzer(
                    logger=self.logger.getChild('TechnicalAnalysis')
                )
                self.enhanced_analysis_enabled = True
                self.logger.info("✅ Enhanced Market Intelligence initialized with Technical Analysis")
            except Exception as e:
                self.logger.error(f"Failed to initialize technical analyzer: {e}")
                self.enhanced_analysis_enabled = False
        else:
            self.logger.warning("⚠️ Technical Analysis not available - using basic intelligence only")
        
        # Initialize market state tracking
        self.current_market_conditions = {}
        self.regime_history = []
        self.signal_performance = {}
        
        # Threading for continuous market monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.logger.info(f"Enhanced Market Intelligence initialized - TA Available: {self.enhanced_analysis_enabled}")
    
    def start_continuous_monitoring(self):
        """Start continuous market monitoring"""
        if self.monitoring_active:
            self.logger.warning("Market monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._continuous_market_analysis,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("🚀 Continuous market monitoring started")
    
    def stop_continuous_monitoring(self):
        """Stop continuous market monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("⏹️ Continuous market monitoring stopped")
    
    def _continuous_market_analysis(self):
        """Continuous market analysis loop"""
        while self.monitoring_active:
            try:
                # Update market conditions for all active symbols
                symbols = getattr(self.config, 'trading_symbols', ['EURUSD', 'GBPUSD', 'USDJPY'])
                
                for symbol in symbols:
                    try:
                        # Get recent data
                        data = self._get_recent_market_data(symbol, 'H1', 100)
                        if data is not None:
                            # Update market conditions
                            conditions = self.analyze_market_conditions(symbol, data)
                            self.current_market_conditions[symbol] = conditions
                    except Exception as e:
                        self.logger.error(f"Error updating conditions for {symbol}: {e}")
                
                # Sleep before next analysis
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in continuous market analysis: {e}")
                time.sleep(30)  # Back off on error
    
    def identify_regime(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> str:
        """
        Enhanced regime identification with multiple confirmations and comprehensive analysis
        """
        try:
            if data is None or len(data) < 50:
                self.logger.warning(f"Insufficient data for regime identification: {len(data) if data is not None else 0}")
                return self._fallback_regime_identification(data)
            
            # Multi-method regime identification
            regime_scores = {}
            
            # Method 1: Enhanced Technical Analysis
            if self.enhanced_analysis_enabled and self.technical_analyzer:
                try:
                    technical_regime = self._identify_regime_technical_analysis(data, symbol)
                    if technical_regime:
                        regime_scores['technical'] = technical_regime
                except Exception as e:
                    self.logger.error(f"Error in technical regime analysis: {e}")
            
            # Method 2: Volatility-based regime
            volatility_regime = self._identify_regime_volatility(data)
            regime_scores['volatility'] = volatility_regime
            
            # Method 3: Trend-based regime
            trend_regime = self._identify_regime_trend(data)
            regime_scores['trend'] = trend_regime
            
            # Method 4: Price action regime
            price_action_regime = self._identify_regime_price_action(data)
            regime_scores['price_action'] = price_action_regime
            
            # Method 5: Volume-based regime (if available)
            if 'Volume' in data.columns and data['Volume'].sum() > 0:
                volume_regime = self._identify_regime_volume(data)
                regime_scores['volume'] = volume_regime
            
            # Combine regime scores using weighted voting
            final_regime = self._combine_regime_scores(regime_scores)
            
            # Store regime history
            regime_entry = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'regime': final_regime,
                'scores': regime_scores,
                'confidence': self._calculate_regime_confidence(regime_scores)
            }
            self.regime_history.append(regime_entry)
            
            # Keep only recent history
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]
            
            self.logger.debug(f"Regime identified for {symbol}: {final_regime}")
            return final_regime
            
        except Exception as e:
            self.logger.error(f"Error in enhanced regime identification: {e}")
            return self._fallback_regime_identification(data)
    
    def _identify_regime_technical_analysis(self, data: pd.DataFrame, symbol: str) -> Optional[str]:
        """Identify regime using technical analysis"""
        try:
            analysis = self.analyze_symbol_comprehensive(symbol, data)
            if not analysis:
                return None
            
            trend_signal = analysis['trading_signal']
            trend_confidence = trend_signal.confidence
            
            # High confidence regime identification
            if trend_confidence > 0.8:
                direction_value = trend_signal.direction.value if hasattr(trend_signal.direction, 'value') else trend_signal.direction
                
                if abs(direction_value) >= 2:  # Strong trend
                    return "Trending"
                elif direction_value == 0:  # Neutral
                    volatility_regime = analysis['indicators']['volatility']['volatility_regime']
                    if volatility_regime == 'high':
                        return "High-Volatility"
                    elif volatility_regime == 'low':
                        return "Mean-Reverting"
                    else:
                        return "Consolidating"
            
            # Medium confidence - check other indicators
            elif trend_confidence > 0.6:
                volatility_indicators = analysis['indicators']['volatility']
                momentum_indicators = analysis['indicators']['momentum']
                
                # Check for ranging market
                bb_position = self._calculate_bb_position(data['Close'].values, volatility_indicators)
                if 0.2 < bb_position < 0.8:  # Price in middle of Bollinger Bands
                    return "Mean-Reverting"
                else:
                    return "Trending"
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in technical regime identification: {e}")
            return None
    
    def _identify_regime_volatility(self, data: pd.DataFrame) -> str:
        """Identify regime based on volatility patterns"""
        try:
            returns = data['Close'].pct_change().dropna()
            
            if len(returns) < 20:
                return "Neutral"
            
            # Calculate multiple volatility measures
            rolling_vol = returns.rolling(20).std()
            current_vol = rolling_vol.iloc[-1]
            avg_vol = rolling_vol.mean()
            vol_percentile = (rolling_vol <= current_vol).mean()
            
            # Volatility clustering
            vol_changes = rolling_vol.pct_change().dropna()
            vol_persistence = vol_changes.autocorr()
            
            # Regime classification
            if current_vol > avg_vol * 2.0:
                return "High-Volatility"
            elif current_vol < avg_vol * 0.5:
                return "Low-Volatility"
            elif vol_percentile > 0.8 and vol_persistence > 0.3:
                return "High-Volatility"
            elif vol_percentile < 0.2 and vol_persistence > 0.3:
                return "Mean-Reverting"
            else:
                return "Neutral"
                
        except Exception as e:
            self.logger.error(f"Error in volatility regime identification: {e}")
            return "Neutral"
    
    def _identify_regime_trend(self, data: pd.DataFrame) -> str:
        """Identify regime based on trend characteristics"""
        try:
            close_prices = data['Close'].values
            
            # Multiple trend measures
            # 1. Linear regression slope
            x = np.arange(len(close_prices))
            slope = np.polyfit(x[-50:], close_prices[-50:], 1)[0]
            
            # 2. Directional movement
            up_moves = 0
            down_moves = 0
            for i in range(1, min(50, len(close_prices))):
                if close_prices[-i] > close_prices[-i-1]:
                    up_moves += 1
                else:
                    down_moves += 1
            
            directional_ratio = up_moves / (up_moves + down_moves)
            
            # 3. Moving average alignment
            if len(close_prices) >= 50:
                ma_20 = pd.Series(close_prices).rolling(20).mean().iloc[-1]
                ma_50 = pd.Series(close_prices).rolling(50).mean().iloc[-1]
                current_price = close_prices[-1]
                
                ma_alignment_score = 0
                if current_price > ma_20 > ma_50:
                    ma_alignment_score = 1
                elif current_price < ma_20 < ma_50:
                    ma_alignment_score = -1
            else:
                ma_alignment_score = 0
            
            # Combine trend measures
            trend_strength = abs(slope) * 1000 + abs(directional_ratio - 0.5) * 2 + abs(ma_alignment_score)
            
            if trend_strength > 0.8:
                return "Trending"
            elif trend_strength < 0.3:
                return "Mean-Reverting"
            else:
                return "Consolidating"
                
        except Exception as e:
            self.logger.error(f"Error in trend regime identification: {e}")
            return "Neutral"
    
    def _identify_regime_price_action(self, data: pd.DataFrame) -> str:
        """Identify regime based on price action patterns"""
        try:
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            # Calculate range characteristics
            recent_period = min(20, len(data))
            recent_ranges = high_prices[-recent_period:] - low_prices[-recent_period:]
            avg_range = np.mean(recent_ranges)
            current_range = high_prices[-1] - low_prices[-1]
            
            # Support/Resistance breaks
            resistance_breaks = 0
            support_breaks = 0
            
            for i in range(1, min(20, len(close_prices))):
                if close_prices[-i] > max(high_prices[-i-10:-i]):
                    resistance_breaks += 1
                if close_prices[-i] < min(low_prices[-i-10:-i]):
                    support_breaks += 1
            
            # Range expansion/contraction
            range_expansion = current_range > avg_range * 1.5
            range_contraction = current_range < avg_range * 0.5
            
            # Price action regime determination
            if resistance_breaks > 2 or support_breaks > 2:
                return "Trending"
            elif range_expansion:
                return "High-Volatility"
            elif range_contraction:
                return "Consolidating"
            else:
                return "Mean-Reverting"
                
        except Exception as e:
            self.logger.error(f"Error in price action regime identification: {e}")
            return "Neutral"
    
    def _identify_regime_volume(self, data: pd.DataFrame) -> str:
        """Identify regime based on volume patterns"""
        try:
            volume = data['Volume'].values
            close_prices = data['Close'].values
            
            # Volume trend
            volume_ma = pd.Series(volume).rolling(20).mean()
            current_volume = volume[-1]
            avg_volume = volume_ma.iloc[-1]
            
            # Volume-price relationship
            price_changes = np.diff(close_prices[-20:])
            volume_changes = np.diff(volume[-20:])
            
            correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
            
            # Volume regime classification
            if current_volume > avg_volume * 2 and abs(correlation) > 0.5:
                return "Trending"
            elif current_volume < avg_volume * 0.5:
                return "Low-Volatility"
            elif abs(correlation) < 0.2:
                return "Consolidating"
            else:
                return "Neutral"
                
        except Exception as e:
            self.logger.error(f"Error in volume regime identification: {e}")
            return "Neutral"
    
    def _combine_regime_scores(self, regime_scores: Dict) -> str:
        """Combine multiple regime scores using weighted voting"""
        try:
            # Weights for different methods
            weights = {
                'technical': 0.35,
                'volatility': 0.25,
                'trend': 0.20,
                'price_action': 0.15,
                'volume': 0.05
            }
            
            # Count votes for each regime
            regime_votes = {}
            
            for method, regime in regime_scores.items():
                if regime and method in weights:
                    weight = weights[method]
                    if regime not in regime_votes:
                        regime_votes[regime] = 0
                    regime_votes[regime] += weight
            
            # Return regime with highest weighted score
            if regime_votes:
                best_regime = max(regime_votes, key=regime_votes.get)
                return best_regime
            else:
                return "Neutral"
                
        except Exception as e:
            self.logger.error(f"Error combining regime scores: {e}")
            return "Neutral"
    
    def _calculate_regime_confidence(self, regime_scores: Dict) -> float:
        """Calculate confidence in regime identification"""
        try:
            unique_regimes = set(regime_scores.values())
            total_methods = len(regime_scores)
            
            if total_methods == 0:
                return 0.0
            
            # Calculate agreement percentage
            agreement_count = 0
            most_common_regime = max(set(regime_scores.values()), key=list(regime_scores.values()).count)
            
            for regime in regime_scores.values():
                if regime == most_common_regime:
                    agreement_count += 1
            
            confidence = agreement_count / total_methods
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating regime confidence: {e}")
            return 0.5
    
    def _fallback_regime_identification(self, data: pd.DataFrame) -> str:
        """Simple fallback regime identification"""
        try:
            if data is None or len(data) < 20:
                return "Neutral"
            
            returns = data['Close'].pct_change().dropna()
            if len(returns) == 0:
                return "Neutral"
            
            volatility = returns.rolling(20, min_periods=1).std().iloc[-1]
            
            if volatility > 0.02:
                return "High-Volatility"
            elif volatility < 0.005:
                return "Mean-Reverting"
            else:
                return "Trending"
                
        except Exception as e:
            self.logger.error(f"Error in fallback regime identification: {e}")
            return "Neutral"
    
    def analyze_symbol_comprehensive(self, symbol: str, price_data: pd.DataFrame, 
                                   timeframe: str = 'H1') -> Optional[Dict]:
        """
        Comprehensive technical analysis for any symbol with caching and error handling
        """
        try:
            if not self.enhanced_analysis_enabled or not self.technical_analyzer:
                self.logger.warning(f"Enhanced analysis not available for {symbol}")
                return None
            
            if len(price_data) < 100:
                self.logger.warning(f"Insufficient data for comprehensive analysis of {symbol}: {len(price_data)}")
                return None
            
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{len(price_data)}_{price_data.index[-1]}"
            if cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key]
                # Check if cache is recent (less than 5 minutes old)
                if (datetime.now() - cached_result['cached_at']).total_seconds() < 300:
                    return cached_result['analysis']
            
            # Perform comprehensive technical analysis
            analysis = self.technical_analyzer.analyze_symbol(
                symbol=symbol,
                timeframe=timeframe,
                price_data=price_data
            )
            
            if analysis:
                # Add to cache
                self.analysis_cache[cache_key] = {
                    'analysis': analysis,
                    'cached_at': datetime.now()
                }
                
                # Clean old cache entries
                self._clean_analysis_cache()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            return None
    
    def _clean_analysis_cache(self):
        """Clean old entries from analysis cache"""
        try:
            current_time = datetime.now()
            keys_to_remove = []
            
            for key, cached_data in self.analysis_cache.items():
                if (current_time - cached_data['cached_at']).total_seconds() > 1800:  # 30 minutes
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.analysis_cache[key]
                
        except Exception as e:
            self.logger.error(f"Error cleaning analysis cache: {e}")
    
    def generate_enhanced_signal(self, symbol: str, data_dict: Dict, regime: str) -> Optional[EnhancedSignal]:
        """
        Generate enhanced trading signal with comprehensive analysis and validation
        """
        try:
            # Validate input data
            if 'EXECUTION' not in data_dict:
                self.logger.warning(f"No execution data available for enhanced signal generation: {symbol}")
                return None
            
            execution_data = data_dict['EXECUTION']
            
            if len(execution_data) < 50:
                self.logger.warning(f"Insufficient execution data for {symbol}: {len(execution_data)}")
                return None
            
            # Enhanced analysis pipeline
            enhanced_signal = None
            
            if self.enhanced_analysis_enabled and self.technical_analyzer:
                try:
                    # Step 1: Comprehensive technical analysis
                    analysis = self.analyze_symbol_comprehensive(symbol, execution_data)
                    
                    if not analysis:
                        return None
                    
                    technical_signal = analysis['trading_signal']
                    
                    # Step 2: Signal quality assessment
                    signal_quality = self._assess_signal_quality(analysis, regime)
                    
                    # Step 3: Multi-timeframe confirmation
                    mtf_confirmation = True
                    if self.multi_timeframe_confirmation:
                        mtf_confirmation = self.check_multi_timeframe_confirmation(symbol, data_dict, {
                            'direction': 'long' if technical_signal.direction.value > 0 else 'short',
                            'confidence': technical_signal.confidence
                        })
                    
                    # Step 4: Market conditions analysis
                    market_conditions = self.analyze_market_conditions(symbol, execution_data)
                    
                    # Step 5: Risk assessment
                    risk_assessment = self._assess_signal_risk(symbol, technical_signal, market_conditions)
                    
                    # Step 6: News impact analysis
                    news_impact = self._analyze_news_impact(symbol)
                    
                    # Step 7: Position sizing
                    position_size = self._calculate_optimal_position_size(symbol, technical_signal, risk_assessment)
                    
                    # Step 8: Signal validation
                    if self._validate_enhanced_signal(technical_signal, signal_quality, mtf_confirmation, risk_assessment):
                        enhanced_signal = EnhancedSignal(
                            symbol=symbol,
                            direction='long' if technical_signal.direction.value > 0 else 'short',
                            confidence=technical_signal.confidence,
                            quality=signal_quality,
                            entry_price=technical_signal.entry_price,
                            stop_loss=technical_signal.stop_loss,
                            take_profit=technical_signal.take_profit,
                            position_size=position_size,
                            strategy=f'Enhanced-TA-{regime}',
                            timeframe='H1',
                            timestamp=technical_signal.timestamp,
                            technical_analysis=analysis,
                            market_conditions=market_conditions,
                            risk_assessment=risk_assessment,
                            multi_timeframe_confirmation=mtf_confirmation,
                            news_impact=news_impact
                        )
                        
                        # Store signal in history
                        self.signal_history.append({
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'signal': enhanced_signal,
                            'regime': regime
                        })
                        
                        # Keep only recent signal history
                        if len(self.signal_history) > 1000:
                            self.signal_history = self.signal_history[-1000:]
                        
                        self.logger.info(f"Enhanced signal generated for {symbol}: {enhanced_signal.direction} "
                                       f"(Confidence: {enhanced_signal.confidence:.2f}, Quality: {signal_quality.name})")
                        
                        return enhanced_signal
                    
                except Exception as e:
                    self.logger.error(f"Error in enhanced signal generation: {e}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced signal for {symbol}: {e}")
            return None
    
    def _assess_signal_quality(self, analysis: Dict, regime: str) -> SignalQuality:
        """Assess the quality of a trading signal"""
        try:
            quality_score = 0
            
            # Technical analysis strength (0-2 points)
            trend_analysis = analysis['signals']['trend']
            trend_strength = trend_analysis['strength'].value if hasattr(trend_analysis['strength'], 'value') else 3
            
            if trend_strength >= 4:
                quality_score += 2
            elif trend_strength >= 3:
                quality_score += 1
            
            # Signal confidence (0-2 points)
            confidence = analysis['trading_signal'].confidence
            if confidence >= 0.8:
                quality_score += 2
            elif confidence >= 0.6:
                quality_score += 1
            
            # Market regime alignment (0-1 point)
            if regime in ['Trending', 'High-Volatility']:
                quality_score += 1
            
            # Map score to quality enum
            if quality_score >= 4:
                return SignalQuality.EXCELLENT
            elif quality_score >= 3:
                return SignalQuality.GOOD
            elif quality_score >= 2:
                return SignalQuality.FAIR
            elif quality_score >= 1:
                return SignalQuality.POOR
            else:
                return SignalQuality.VERY_POOR
                
        except Exception as e:
            self.logger.error(f"Error assessing signal quality: {e}")
            return SignalQuality.FAIR
    
    def _assess_signal_risk(self, symbol: str, technical_signal: TechnicalSignal, 
                           market_conditions: MarketConditions) -> Dict:
        """Assess risk factors for a trading signal"""
        try:
            risk_factors = []
            risk_score = 0.0
            
            # Volatility risk
            if market_conditions.volatility_level == 'high':
                risk_factors.append('high_volatility')
                risk_score += 0.3
            
            # Trend strength risk
            if market_conditions.trend_strength < 0.3:
                risk_factors.append('weak_trend')
                risk_score += 0.2
            
            # News impact risk
            if market_conditions.news_impact == 'high':
                risk_factors.append('news_impact')
                risk_score += 0.4
            
            # Correlation risk
            if market_conditions.correlation_risk > 0.7:
                risk_factors.append('high_correlation')
                risk_score += 0.2
            
            # Market session risk
            if market_conditions.market_session in ['sydney', 'quiet']:
                risk_factors.append('low_liquidity_session')
                risk_score += 0.1
            
            # Calculate risk-reward ratio
            entry_price = technical_signal.entry_price
            stop_loss = technical_signal.stop_loss
            take_profit = technical_signal.take_profit
            
            risk_amount = abs(entry_price - stop_loss)
            reward_amount = abs(take_profit - entry_price)
            
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            if risk_reward_ratio < 1.5:
                risk_factors.append('poor_risk_reward')
                risk_score += 0.3
            
            # Overall risk level
            if risk_score >= 0.8:
                risk_level = 'very_high'
            elif risk_score >= 0.6:
                risk_level = 'high'
            elif risk_score >= 0.4:
                risk_level = 'moderate'
            elif risk_score >= 0.2:
                risk_level = 'low'
            else:
                risk_level = 'very_low'
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'risk_reward_ratio': risk_reward_ratio,
                'approved': risk_score < 0.7,
                'max_position_size': self._calculate_max_position_size_for_risk(risk_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing signal risk: {e}")
            return {
                'risk_score': 0.5,
                'risk_level': 'moderate',
                'risk_factors': ['assessment_error'],
                'risk_reward_ratio': 1.0,
                'approved': False,
                'max_position_size': 0.01
            }
    
    def analyze_market_conditions(self, symbol: str, data: pd.DataFrame) -> MarketConditions:
        """Analyze comprehensive market conditions"""
        try:
            # Regime identification
            regime_str = self.identify_regime(data, symbol)
            regime = MarketRegime(regime_str.lower().replace('-', '_'))
            
            # Volatility analysis
            returns = data['Close'].pct_change().dropna()
            current_volatility = returns.rolling(20).std().iloc[-1] if len(returns) > 20 else 0.01
            avg_volatility = returns.std() if len(returns) > 0 else 0.01
            
            if current_volatility > avg_volatility * 1.5:
                volatility_level = 'high'
            elif current_volatility < avg_volatility * 0.7:
                volatility_level = 'low'
            else:
                volatility_level = 'normal'
            
            # Trend strength
            trend_strength = self._calculate_trend_strength(data)
            
            # Momentum score
            momentum_score = self._calculate_momentum_score(data)
            
            # Support/Resistance levels
            sr_levels = self._calculate_support_resistance_levels(data)
            
            # Market session
            market_session = self._get_current_market_session()
            
            # News impact (placeholder - integrate with news feed)
            news_impact = self._get_news_impact_level(symbol)
            
            # Correlation risk (placeholder - integrate with correlation analysis)
            correlation_risk = self._calculate_correlation_risk(symbol)
            
            return MarketConditions(
                regime=regime,
                volatility_level=volatility_level,
                trend_strength=trend_strength,
                momentum_score=momentum_score,
                support_resistance_levels=sr_levels,
                market_session=market_session,
                news_impact=news_impact,
                correlation_risk=correlation_risk
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            # Return default conditions
            return MarketConditions(
                regime=MarketRegime.NEUTRAL,
                volatility_level='normal',
                trend_strength=0.5,
                momentum_score=0.5,
                support_resistance_levels={},
                market_session='unknown',
                news_impact='low',
                correlation_risk=0.3
            )
    
    def check_multi_timeframe_confirmation(self, symbol: str, data_dict: Dict, 
                                         primary_signal: Dict) -> bool:
        """
        Check for multi-timeframe confirmation with comprehensive analysis
        """
        try:
            if not self.multi_timeframe_confirmation:
                return True
            
            if not self.enhanced_analysis_enabled or not self.technical_analyzer:
                return True  # Don't block signal if enhanced analysis not available
            
            confirmation_results = []
            
            # Check H4 timeframe confirmation
            if 'BIAS' in data_dict:  # H4 data
                h4_confirmation = self._check_timeframe_confirmation(
                    symbol, data_dict['BIAS'], 'H4', primary_signal
                )
                confirmation_results.append(h4_confirmation)
            
            # Check Daily timeframe confirmation if available
            if 'SIGNAL' in data_dict:  # Daily data
                daily_confirmation = self._check_timeframe_confirmation(
                    symbol, data_dict['SIGNAL'], 'D1', primary_signal
                )
                confirmation_results.append(daily_confirmation)
            
            # Need at least one confirmation
            if not confirmation_results:
                return True  # No higher timeframes available
            
            # At least 70% of timeframes should confirm
            confirmation_rate = sum(confirmation_results) / len(confirmation_results)
            confirmed = confirmation_rate >= 0.7
            
            if not confirmed:
                self.logger.info(f"Multi-timeframe confirmation failed for {symbol}: {confirmation_rate:.1%}")
            
            return confirmed
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe confirmation: {e}")
            return True  # Don't block signal on error
    
    def _check_timeframe_confirmation(self, symbol: str, data: pd.DataFrame, 
                                    timeframe: str, primary_signal: Dict) -> bool:
        """Check confirmation from a specific timeframe"""
        try:
            analysis = self.analyze_symbol_comprehensive(symbol, data, timeframe)
            if not analysis:
                return True  # Neutral if can't analyze
            
            tf_signal = analysis['trading_signal']
            
            # Check direction alignment
            primary_direction = primary_signal['direction']
            tf_direction_value = tf_signal.direction.value if hasattr(tf_signal.direction, 'value') else tf_signal.direction
            tf_direction = 'long' if tf_direction_value > 0 else 'short'
            
            # Directions should align or be neutral
            if primary_direction == tf_direction:
                return True
            elif tf_direction_value == 0:  # Neutral is acceptable
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking {timeframe} confirmation: {e}")
            return True
    
    def detect_trend_reversal(self, symbol: str, data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect trend reversal with multiple confirmations and probability assessment
        """
        try:
            if not self.enhanced_analysis_enabled or not self.technical_analyzer:
                return False, 0.0
            
            analysis = self.analyze_symbol_comprehensive(symbol, data)
            if not analysis:
                return False, 0.0
            
            reversal_probability = 0.0
            reversal_signals = 0
            total_checks = 0
            
            # 1. Technical indicator reversal signals
            reversal_analysis = analysis['signals'].get('reversal', {})
            technical_reversal_prob = reversal_analysis.get('probability', 0.0)
            reversal_probability += technical_reversal_prob * 0.4
            total_checks += 1
            
            # 2. Price action reversal patterns
            price_action_reversal, pa_prob = self._detect_price_action_reversal(data)
            if price_action_reversal:
                reversal_signals += 1
                reversal_probability += pa_prob * 0.3
            total_checks += 1
            
            # 3. Volume confirmation reversal
            if 'Volume' in data.columns and data['Volume'].sum() > 0:
                volume_reversal, vol_prob = self._detect_volume_reversal(data)
                if volume_reversal:
                    reversal_signals += 1
                    reversal_probability += vol_prob * 0.2
                total_checks += 1
            
            # 4. Support/Resistance level breaks
            sr_reversal, sr_prob = self._detect_sr_reversal(data)
            if sr_reversal:
                reversal_signals += 1
                reversal_probability += sr_prob * 0.1
            total_checks += 1
            
            # Normalize probability
            if total_checks > 0:
                reversal_probability = min(reversal_probability, 1.0)
            
            # Confirmed reversal requires high probability and multiple signals
            confirmed_reversal = (reversal_probability > self.reversal_probability_threshold and 
                                reversal_signals >= 2)
            
            if confirmed_reversal:
                self.logger.info(f"Trend reversal detected for {symbol}: "
                               f"Probability {reversal_probability:.2f}, Signals {reversal_signals}")
            
            return confirmed_reversal, reversal_probability
            
        except Exception as e:
            self.logger.error(f"Error detecting trend reversal: {e}")
            return False, 0.0
    
    # Additional helper methods (continuing the comprehensive implementation)
    def _detect_price_action_reversal(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """Detect price action reversal patterns"""
        try:
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            reversal_probability = 0.0
            
            # Look for reversal candlestick patterns
            if len(data) >= 3:
                # Doji pattern
                last_candle = data.iloc[-1]
                body_size = abs(last_candle['Close'] - last_candle['Open'])
                candle_range = last_candle['High'] - last_candle['Low']
                
                if body_size < candle_range * 0.1:  # Doji
                    reversal_probability += 0.3
                
                # Hammer/Shooting star
                upper_shadow = last_candle['High'] - max(last_candle['Open'], last_candle['Close'])
                lower_shadow = min(last_candle['Open'], last_candle['Close']) - last_candle['Low']
                
                if lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5:  # Hammer
                    reversal_probability += 0.4
                elif upper_shadow > body_size * 2 and lower_shadow < body_size * 0.5:  # Shooting star
                    reversal_probability += 0.4
            
            # Double top/bottom patterns
            if len(data) >= 10:
                recent_highs = pd.Series(high_prices[-10:]).rolling(3).max()
                recent_lows = pd.Series(low_prices[-10:]).rolling(3).min()
                
                # Check for double top
                if len(recent_highs.dropna()) >= 2:
                    last_two_highs = recent_highs.dropna().tail(2).values
                    if abs(last_two_highs[0] - last_two_highs[1]) < (last_two_highs[0] * 0.01):
                        reversal_probability += 0.3
                
                # Check for double bottom
                if len(recent_lows.dropna()) >= 2:
                    last_two_lows = recent_lows.dropna().tail(2).values
                    if abs(last_two_lows[0] - last_two_lows[1]) < (last_two_lows[0] * 0.01):
                        reversal_probability += 0.3
            
            return reversal_probability > 0.3, reversal_probability
            
        except Exception as e:
            self.logger.error(f"Error detecting price action reversal: {e}")
            return False, 0.0
    
    def _detect_volume_reversal(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """Detect volume-based reversal signals"""
        try:
            volume = data['Volume'].values
            close_prices = data['Close'].values
            
            if len(volume) < 10:
                return False, 0.0
            
            reversal_probability = 0.0
            
            # Volume spike with price reversal
            avg_volume = np.mean(volume[-10:])
            current_volume = volume[-1]
            
            if current_volume > avg_volume * 2:  # Volume spike
                # Check if price is reversing
                price_change = (close_prices[-1] - close_prices[-2]) / close_prices[-2]
                prev_price_change = (close_prices[-2] - close_prices[-3]) / close_prices[-3]
                
                if price_change * prev_price_change < 0:  # Direction change
                    reversal_probability += 0.5
            
            # Declining volume with continued trend (potential exhaustion)
            recent_volumes = volume[-5:]
            volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
            
            if volume_trend < 0:  # Declining volume
                reversal_probability += 0.3
            
            return reversal_probability > 0.3, reversal_probability
            
        except Exception as e:
            self.logger.error(f"Error detecting volume reversal: {e}")
            return False, 0.0
    
    def _detect_sr_reversal(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """Detect support/resistance level reversal"""
        try:
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            current_price = close_prices[-1]
            
            reversal_probability = 0.0
            
            # Calculate support and resistance levels
            recent_period = min(50, len(data))
            recent_highs = pd.Series(high_prices[-recent_period:]).rolling(5).max().dropna().unique()
            recent_lows = pd.Series(low_prices[-recent_period:]).rolling(5).min().dropna().unique()
            
            # Check for resistance rejection
            for resistance in recent_highs:
                if resistance > current_price and abs(current_price - resistance) < resistance * 0.002:
                    # Check if price was rejected from this level
                    if max(high_prices[-3:]) >= resistance * 0.999:
                        reversal_probability += 0.4
                        break
            
            # Check for support bounce
            for support in recent_lows:
                if support < current_price and abs(current_price - support) < support * 0.002:
                    # Check if price bounced from this level
                    if min(low_prices[-3:]) <= support * 1.001:
                        reversal_probability += 0.4
                        break
            
            return reversal_probability > 0.3, reversal_probability
            
        except Exception as e:
            self.logger.error(f"Error detecting S/R reversal: {e}")
            return False, 0.0
    
    # Additional utility methods (continuing comprehensive implementation)
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength score"""
        try:
            close_prices = data['Close'].values
            
            if len(close_prices) < 20:
                return 0.5
            
            # Linear regression slope
            x = np.arange(len(close_prices[-20:]))
            slope = abs(np.polyfit(x, close_prices[-20:], 1)[0])
            
            # Normalize slope
            price_range = np.max(close_prices[-20:]) - np.min(close_prices[-20:])
            normalized_slope = slope / (price_range / 20) if price_range > 0 else 0
            
            return min(normalized_slope, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.5
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate momentum score"""
        try:
            close_prices = data['Close'].values
            
            if len(close_prices) < 10:
                return 0.5
            
            # Rate of change
            roc = (close_prices[-1] - close_prices[-10]) / close_prices[-10]
            
            # Normalize to 0-1 scale
            momentum_score = 0.5 + (roc * 10)  # Scale factor
            return max(0.0, min(1.0, momentum_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {e}")
            return 0.5
    
    def _calculate_support_resistance_levels(self, data: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        try:
            high_prices = data['High'].values
            low_prices = data['Low'].values
            
            recent_period = min(50, len(data))
            recent_highs = pd.Series(high_prices[-recent_period:]).rolling(5).max().dropna().unique()
            recent_lows = pd.Series(low_prices[-recent_period:]).rolling(5).min().dropna().unique()
            
            return {
                'resistance_levels': recent_highs.tolist(),
                'support_levels': recent_lows.tolist(),
                'nearest_resistance': recent_highs.max() if len(recent_highs) > 0 else None,
                'nearest_support': recent_lows.min() if len(recent_lows) > 0 else None
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating S/R levels: {e}")
            return {'resistance_levels': [], 'support_levels': []}
    
    def _get_current_market_session(self) -> str:
        """Get current market session"""
        try:
            current_hour = datetime.now().hour
            
            for session, times in self.market_sessions.items():
                start = times['start']
                end = times['end']
                
                if start <= end:  # Same day
                    if start <= current_hour <= end:
                        return session
                else:  # Crosses midnight
                    if current_hour >= start or current_hour <= end:
                        return session
            
            return 'quiet'
            
        except Exception as e:
            self.logger.error(f"Error getting market session: {e}")
            return 'unknown'
    
    def _get_news_impact_level(self, symbol: str) -> str:
        """Get news impact level for symbol (placeholder)"""
        # This would integrate with news feed in production
        return 'low'
    
    def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk (placeholder)"""
        # This would calculate correlation with other open positions
        return 0.3
    
    def _analyze_news_impact(self, symbol: str) -> Dict:
        """Analyze news impact for symbol (placeholder)"""
        return {
            'impact_level': 'low',
            'upcoming_events': [],
            'sentiment': 'neutral'
        }
    
    def _calculate_optimal_position_size(self, symbol: str, technical_signal: TechnicalSignal, 
                                       risk_assessment: Dict) -> float:
        """Calculate optimal position size"""
        try:
            base_size = 0.01  # Base position size
            
            # Adjust based on signal confidence
            confidence_multiplier = technical_signal.confidence
            
            # Adjust based on risk assessment
            risk_multiplier = 1.0 - (risk_assessment['risk_score'] * 0.5)
            
            # Adjust based on risk-reward ratio
            rr_multiplier = min(risk_assessment['risk_reward_ratio'] / 2.0, 1.5)
            
            optimal_size = base_size * confidence_multiplier * risk_multiplier * rr_multiplier
            
            # Cap at maximum allowed
            max_size = risk_assessment.get('max_position_size', 0.1)
            return min(optimal_size, max_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal position size: {e}")
            return 0.01
    
    def _validate_enhanced_signal(self, technical_signal: TechnicalSignal, 
                                signal_quality: SignalQuality, 
                                mtf_confirmation: bool, 
                                risk_assessment: Dict) -> bool:
        """Validate enhanced signal before execution"""
        try:
            # Minimum confidence threshold
            if technical_signal.confidence < self.min_confidence:
                return False
            
            # Signal quality threshold
            if signal_quality.value < 3:  # Below FAIR
                return False
            
            # Multi-timeframe confirmation required
            if self.multi_timeframe_confirmation and not mtf_confirmation:
                return False
            
            # Risk assessment approval
            if not risk_assessment.get('approved', False):
                return False
            
            # Risk-reward ratio check
            if risk_assessment.get('risk_reward_ratio', 0) < 1.2:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating enhanced signal: {e}")
            return False
    
    def _calculate_max_position_size_for_risk(self, risk_score: float) -> float:
        """Calculate maximum position size based on risk score"""
        try:
            if risk_score >= 0.8:
                return 0.005  # Very high risk
            elif risk_score >= 0.6:
                return 0.01   # High risk
            elif risk_score >= 0.4:
                return 0.02   # Moderate risk
            elif risk_score >= 0.2:
                return 0.05   # Low risk
            else:
                return 0.1    # Very low risk
                
        except Exception as e:
            self.logger.error(f"Error calculating max position size: {e}")
            return 0.01
    
    def _calculate_bb_position(self, close_prices: np.ndarray, volatility_indicators: Dict) -> float:
        """Calculate Bollinger Band position"""
        try:
            bb_upper = volatility_indicators.get('bb_upper', [])
            bb_lower = volatility_indicators.get('bb_lower', [])
            
            if len(bb_upper) == 0 or len(bb_lower) == 0:
                return 0.5
            
            current_price = close_prices[-1]
            upper = bb_upper[-1]
            lower = bb_lower[-1]
            
            if upper == lower:
                return 0.5
            
            position = (current_price - lower) / (upper - lower)
            return max(0.0, min(1.0, position))
            
        except Exception as e:
            self.logger.error(f"Error calculating BB position: {e}")
            return 0.5
    
    def _get_recent_market_data(self, symbol: str, timeframe: str, periods: int) -> Optional[pd.DataFrame]:
        """Get recent market data for symbol"""
        try:
            # This would integrate with your data handler
            if hasattr(self.data_handler, 'get_data'):
                return self.data_handler.get_data(symbol, timeframe, periods)
            elif hasattr(self.data_handler, 'get_recent_data'):
                return self.data_handler.get_recent_data(symbol, timeframe, periods)
            else:
                self.logger.warning(f"No data retrieval method available")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting recent market data: {e}")
            return None
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary of enhanced market intelligence"""
        try:
            total_signals = len(self.signal_history)
            
            if total_signals == 0:
                return {
                    'total_signals_generated': 0,
                    'average_confidence': 0.0,
                    'quality_distribution': {},
                    'regime_distribution': {},
                    'active_regimes': len(set(self.current_market_conditions.values())),
                    'cache_size': len(self.analysis_cache)
                }
            
            # Calculate average confidence
            avg_confidence = np.mean([s['signal'].confidence for s in self.signal_history])
            
            # Quality distribution
            quality_dist = {}
            for signal_entry in self.signal_history:
                quality = signal_entry['signal'].quality.name
                quality_dist[quality] = quality_dist.get(quality, 0) + 1
            
            # Regime distribution
            regime_dist = {}
            for signal_entry in self.signal_history:
                regime = signal_entry['regime']
                regime_dist[regime] = regime_dist.get(regime, 0) + 1
            
            return {
                'total_signals_generated': total_signals,
                'average_confidence': avg_confidence,
                'quality_distribution': quality_dist,
                'regime_distribution': regime_dist,
                'active_regimes': len(set(self.current_market_conditions.values())),
                'cache_size': len(self.analysis_cache),
                'monitoring_active': self.monitoring_active,
                'enhanced_analysis_enabled': self.enhanced_analysis_enabled
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.stop_continuous_monitoring()
            self.analysis_cache.clear()
            self.logger.info("Enhanced Market Intelligence cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# Compatibility alias for existing code
MarketIntelligence = EnhancedMarketIntelligence

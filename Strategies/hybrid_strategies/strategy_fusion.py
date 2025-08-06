# strategies/hybrid_strategies/strategy_fusion.py
from typing import List, Dict, Any, Optional
from strategies.base_strategy import TradingSignal, SignalStrength
import numpy as np

class StrategyFusion:
    """Combines RL and classical signals using advanced fusion techniques"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rl_weight = config.get('rl_weight', 0.7)
        self.classical_weight = config.get('classical_weight', 0.3)
        self.min_consensus = config.get('min_consensus', 0.6)
        self.performance_weights = {}
        
    def fuse_signals(self, signals: List[TradingSignal], 
                    market_regime: str = 'NEUTRAL') -> Optional[TradingSignal]:
        """
        Advanced signal fusion using multiple techniques:
        1. Confidence-weighted voting
        2. Regime-aware weighting
        3. Performance-based dynamic weighting
        """
        if not signals:
            return None
        
        # Separate RL and classical signals
        rl_signals = [s for s in signals if s.strategy_type.startswith('RL-')]
        classical_signals = [s for s in signals if not s.strategy_type.startswith('RL-')]
        
        if not rl_signals and not classical_signals:
            return None
        
        # Calculate regime-specific weights
        regime_weights = self._get_regime_weights(market_regime)
        
        # Fusion method 1: Confidence-weighted consensus
        consensus_signal = self._confidence_weighted_consensus(
            signals, regime_weights)
        
        # Fusion method 2: Performance-weighted ensemble
        performance_signal = self._performance_weighted_fusion(
            rl_signals, classical_signals, regime_weights)
        
        # Meta-fusion: Combine the fusion methods
        final_signal = self._meta_fusion(consensus_signal, performance_signal)
        
        return final_signal
    
    def _get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Dynamic weighting based on market regime"""
        regime_configs = {
            'TRENDING': {'rl_weight': 0.8, 'classical_weight': 0.2},
            'MEAN_REVERTING': {'rl_weight': 0.5, 'classical_weight': 0.5},
            'HIGH_VOLATILITY': {'rl_weight': 0.6, 'classical_weight': 0.4},
            'NEUTRAL': {'rl_weight': 0.7, 'classical_weight': 0.3}
        }
        
        return regime_configs.get(regime, regime_configs['NEUTRAL'])
    
    def _confidence_weighted_consensus(self, signals: List[TradingSignal], 
                                     regime_weights: Dict[str, float]) -> Optional[TradingSignal]:
        """Weighted voting based on signal confidence and regime"""
        if not signals:
            return None
        
        # Group by direction
        direction_scores = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        total_weight = 0.0
        
        for signal in signals:
            # Determine signal weight
            if signal.strategy_type.startswith('RL-'):
                base_weight = regime_weights['rl_weight']
            else:
                base_weight = regime_weights['classical_weight']
            
            # Apply confidence weighting
            weighted_confidence = base_weight * signal.confidence
            
            direction_scores[signal.direction] += weighted_confidence
            total_weight += weighted_confidence
        
        if total_weight == 0:
            return None
        
        # Normalize scores
        for direction in direction_scores:
            direction_scores[direction] /= total_weight
        
        # Find winning direction
        winning_direction = max(direction_scores.keys(), 
                              key=lambda k: direction_scores[k])
        winning_score = direction_scores[winning_direction]
        
        # Check if consensus meets minimum threshold
        if winning_score < self.min_consensus:
            return None
        
        # Create consensus signal
        representative_signal = next(
            (s for s in signals if s.direction == winning_direction), 
            signals[0]
        )
        
        return TradingSignal(
            symbol=representative_signal.symbol,
            direction=winning_direction,
            strategy_type='HYBRID-CONSENSUS',
            confidence=winning_score,
            entry_price=representative_signal.entry_price,
            stop_loss=representative_signal.stop_loss,
            take_profit=representative_signal.take_profit,
            metadata={
                'fusion_method': 'confidence_weighted_consensus',
                'signal_count': len(signals),
                'consensus_score': winning_score
            }
        )
    
    def _performance_weighted_fusion(self, rl_signals: List[TradingSignal],
                                   classical_signals: List[TradingSignal],
                                   regime_weights: Dict[str, float]) -> Optional[TradingSignal]:
        """Fusion based on historical performance"""
        all_signals = rl_signals + classical_signals
        if not all_signals:
            return None
        
        # Calculate performance-adjusted weights
        total_score = 0.0
        signal_scores = []
        
        for signal in all_signals:
            # Get historical performance for this strategy type
            performance_score = self.performance_weights.get(
                signal.strategy_type, 0.5
            )
            
            # Combine with regime weight
            if signal.strategy_type.startswith('RL-'):
                regime_weight = regime_weights['rl_weight']
            else:
                regime_weight = regime_weights['classical_weight']
            
            final_score = (performance_score * signal.confidence * regime_weight)
            signal_scores.append((signal, final_score))
            total_score += final_score
        
        if total_score == 0:
            return None
        
        # Select highest scoring signal
        best_signal, best_score = max(signal_scores, key=lambda x: x[1])
        
        return TradingSignal(
            symbol=best_signal.symbol,
            direction=best_signal.direction,
            strategy_type='HYBRID-PERFORMANCE',
            confidence=best_score / max(total_score, 1.0),
            entry_price=best_signal.entry_price,
            stop_loss=best_signal.stop_loss,
            take_profit=best_signal.take_profit,
            metadata={
                'fusion_method': 'performance_weighted',
                'base_strategy': best_signal.strategy_type,
                'performance_score': best_score
            }
        )

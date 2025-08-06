# core/performance_tracker.py
class StrategyPerformanceTracker:
    """Tracks and analyzes individual strategy performance"""
    
    def __init__(self):
        self.strategy_metrics = {}
        self.regime_performance = {}
    
    def update_performance(self, strategy_name: str, signal: TradingSignal, 
                         outcome: Dict[str, Any], regime: str):
        """Update performance metrics for a strategy"""
        if strategy_name not in self.strategy_metrics:
            self.strategy_metrics[strategy_name] = {
                'total_signals': 0,
                'successful_signals': 0,
                'total_return': 0.0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0
            }
        
        metrics = self.strategy_metrics[strategy_name]
        metrics['total_signals'] += 1
        
        if outcome['profit'] > 0:
            metrics['successful_signals'] += 1
        
        metrics['total_return'] += outcome['profit']
        metrics['win_rate'] = metrics['successful_signals'] / metrics['total_signals']
        metrics['avg_return'] = metrics['total_return'] / metrics['total_signals']
        
        # Update regime-specific performance
        regime_key = f"{strategy_name}_{regime}"
        if regime_key not in self.regime_performance:
            self.regime_performance[regime_key] = []
        
        self.regime_performance[regime_key].append(outcome['profit'])
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Calculate dynamic weights based on recent performance"""
        weights = {}
        for strategy_name, metrics in self.strategy_metrics.items():
            # Base weight on win rate and average return
            base_score = (metrics['win_rate'] * 0.6) + \
                        (min(metrics['avg_return'] * 100, 0.4))
            
            weights[strategy_name] = max(0.1, min(0.9, base_score))
        
        return weights

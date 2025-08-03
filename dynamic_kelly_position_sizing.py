# dynamic_kelly_position_sizing.py - Professional Kelly Criterion Implementation
"""
Professional Dynamic Position Sizing with Kelly Criterion
Advanced Risk Management and Portfolio Optimization
Compatible with Enterprise Trading Bot System
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque
import json
import pickle
from pathlib import Path
import warnings
from scipy import stats
from scipy.optimize import minimize_scalar
import threading
import time

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class TradeHistory:
    """Single trade history record"""
    timestamp: datetime
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_percentage: float
    hold_time: float  # in hours
    strategy: str
    market_conditions: str

@dataclass
class PositionSizingResult:
    """Position sizing calculation result"""
    recommended_size: float
    kelly_fraction: float
    safety_adjusted_kelly: float
    confidence_multiplier: float
    volatility_adjustment: float
    drawdown_adjustment: float
    correlation_adjustment: float
    final_risk_percentage: float
    max_position_value: float
    reasoning: str
    risk_metrics: Dict[str, float]

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    consecutive_losses: int
    volatility: float
    var_95: float  # Value at Risk
    expected_shortfall: float

class ProfessionalKellyPositionSizer:
    """
    Professional-grade position sizing system using Kelly Criterion
    with advanced risk management and dynamic adjustments[1][2]
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Kelly Position Sizer"""
        
        # Configuration
        self.config = config or {}
        
        # Kelly Criterion parameters
        self.kelly_lookback_trades = self.config.get('kelly_lookback_trades', 100)
        self.kelly_safety_factor = self.config.get('kelly_safety_factor', 0.25)  # Use 25% of Kelly
        self.min_kelly_samples = self.config.get('min_kelly_samples', 20)
        
        # Risk management parameters
        self.base_risk_per_trade = self.config.get('base_risk_per_trade', 0.01)  # 1%
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.05)   # 5%
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.20)   # 20%
        self.max_single_symbol_risk = self.config.get('max_single_symbol_risk', 0.10)  # 10%
        
        # Dynamic adjustment parameters
        self.confidence_weight = self.config.get('confidence_weight', 0.3)
        self.volatility_weight = self.config.get('volatility_weight', 0.3)
        self.drawdown_weight = self.config.get('drawdown_weight', 0.4)
        
        # Performance tracking
        self.trade_history: List[TradeHistory] = []
        self.performance_cache = {}
        self.last_performance_update = time.time()
        self.performance_update_interval = 300  # 5 minutes
        
        # Position tracking
        self.current_positions = {}
        self.position_correlations = {}
        self.portfolio_exposure = 0.0
        
        # Risk state
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.current_risk_exposure = 0.0
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Professional Kelly Position Sizer initialized")

    def calculate_optimal_position_size(self, 
                                      signal: Dict[str, Any], 
                                      account_balance: float,
                                      market_data: Dict[str, Any] = None) -> PositionSizingResult:
        """
        Calculate optimal position size using Kelly Criterion with advanced adjustments[1][2]
        
        Args:
            signal: Trading signal with confidence, entry_price, stop_loss, etc.
            account_balance: Current account balance
            market_data: Additional market data for volatility calculation
            
        Returns:
            PositionSizingResult with detailed sizing information
        """
        
        with self.lock:
            try:
                symbol = signal.get('symbol', 'UNKNOWN')
                confidence = signal.get('confidence', 0.5)
                entry_price = signal.get('entry_price', 0.0)
                stop_loss = signal.get('stop_loss', 0.0)
                strategy = signal.get('strategy', 'UNKNOWN')
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Calculate base Kelly fraction
                kelly_fraction = self._calculate_kelly_fraction(symbol, strategy)
                
                # Apply safety factor
                safe_kelly = kelly_fraction * self.kelly_safety_factor
                
                # Calculate risk per unit
                if stop_loss and entry_price:
                    risk_per_unit = abs(entry_price - stop_loss) / entry_price
                else:
                    risk_per_unit = self._estimate_risk_per_unit(symbol, market_data)
                
                # Base position size from Kelly
                if risk_per_unit > 0:
                    base_position_size = (safe_kelly * account_balance) / (risk_per_unit * entry_price)
                else:
                    base_position_size = (self.base_risk_per_trade * account_balance) / entry_price
                
                # Apply dynamic adjustments
                adjustments = self._calculate_dynamic_adjustments(
                    symbol, confidence, market_data, strategy
                )
                
                # Apply all adjustments
                final_position_size = base_position_size
                final_position_size *= adjustments['confidence_multiplier']
                final_position_size *= adjustments['volatility_adjustment']
                final_position_size *= adjustments['drawdown_adjustment']
                final_position_size *= adjustments['correlation_adjustment']
                
                # Position value and risk checks
                position_value = final_position_size * entry_price
                risk_amount = position_value * risk_per_unit
                risk_percentage = risk_amount / account_balance
                
                # Apply hard limits
                max_position_by_risk = (self.max_risk_per_trade * account_balance) / (risk_per_unit * entry_price)
                max_position_by_exposure = (self.max_single_symbol_risk * account_balance) / entry_price
                
                final_position_size = min(
                    final_position_size,
                    max_position_by_risk,
                    max_position_by_exposure
                )
                
                # Ensure minimum viable size
                min_position_size = (0.001 * account_balance) / entry_price  # 0.1% minimum
                final_position_size = max(final_position_size, min_position_size)
                
                # Final risk calculation
                final_position_value = final_position_size * entry_price
                final_risk_amount = final_position_value * risk_per_unit
                final_risk_percentage = final_risk_amount / account_balance
                
                # Create comprehensive result
                result = PositionSizingResult(
                    recommended_size=round(final_position_size, 4),
                    kelly_fraction=kelly_fraction,
                    safety_adjusted_kelly=safe_kelly,
                    confidence_multiplier=adjustments['confidence_multiplier'],
                    volatility_adjustment=adjustments['volatility_adjustment'],
                    drawdown_adjustment=adjustments['drawdown_adjustment'],
                    correlation_adjustment=adjustments['correlation_adjustment'],
                    final_risk_percentage=final_risk_percentage,
                    max_position_value=final_position_value,
                    reasoning=self._generate_sizing_reasoning(kelly_fraction, adjustments, strategy),
                    risk_metrics=self._get_current_risk_metrics()
                )
                
                logger.info(
                    f"Position sizing for {symbol}: Size={final_position_size:.4f}, "
                    f"Risk={final_risk_percentage:.3%}, Kelly={kelly_fraction:.3f}"
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error calculating position size: {e}")
                return self._get_fallback_position_size(account_balance, entry_price)

    def _calculate_kelly_fraction(self, symbol: str, strategy: str) -> float:
        """Calculate Kelly fraction based on historical performance[1]"""
        
        try:
            # Filter relevant trades
            relevant_trades = [
                trade for trade in self.trade_history[-self.kelly_lookback_trades:]
                if (symbol == 'ALL' or trade.symbol == symbol) and
                   (strategy == 'ALL' or trade.strategy == strategy)
            ]
            
            if len(relevant_trades) < self.min_kelly_samples:
                logger.warning(f"Insufficient trade history for Kelly calculation: {len(relevant_trades)}")
                return self.base_risk_per_trade  # Fallback to base risk
            
            # Calculate win rate and average win/loss
            winning_trades = [t for t in relevant_trades if t.pnl > 0]
            losing_trades = [t for t in relevant_trades if t.pnl < 0]
            
            if not losing_trades:
                logger.warning("No losing trades found for Kelly calculation")
                return self.base_risk_per_trade
            
            win_rate = len(winning_trades) / len(relevant_trades)
            avg_win = np.mean([t.pnl_percentage for t in winning_trades]) if winning_trades else 0
            avg_loss = abs(np.mean([t.pnl_percentage for t in losing_trades]))
            
            # Kelly Criterion: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            if avg_loss == 0:
                return self.base_risk_per_trade
            
            b = avg_win / avg_loss  # Win/loss ratio
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Ensure Kelly fraction is reasonable
            kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Cap at 50%
            
            logger.debug(
                f"Kelly calculation: Win rate={win_rate:.3f}, "
                f"Win/Loss ratio={b:.3f}, Kelly={kelly_fraction:.3f}"
            )
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}")
            return self.base_risk_per_trade

    def _calculate_dynamic_adjustments(self, 
                                     symbol: str,
                                     confidence: float,
                                     market_data: Dict[str, Any],
                                     strategy: str) -> Dict[str, float]:
        """Calculate dynamic adjustments based on multiple factors"""
        
        # 1. Confidence-based adjustment
        confidence_multiplier = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0
        
        # 2. Volatility-based adjustment
        volatility_adjustment = self._calculate_volatility_adjustment(symbol, market_data)
        
        # 3. Drawdown-based adjustment
        drawdown_adjustment = self._calculate_drawdown_adjustment()
        
        # 4. Position correlation adjustment
        correlation_adjustment = self._calculate_correlation_adjustment(symbol)
        
        # 5. Time-based adjustment (market session)
        time_adjustment = self._calculate_time_adjustment()
        
        # 6. Strategy-specific adjustment
        strategy_adjustment = self._calculate_strategy_adjustment(strategy)
        
        return {
            'confidence_multiplier': confidence_multiplier,
            'volatility_adjustment': volatility_adjustment,
            'drawdown_adjustment': drawdown_adjustment,
            'correlation_adjustment': correlation_adjustment,
            'time_adjustment': time_adjustment,
            'strategy_adjustment': strategy_adjustment
        }

    def _calculate_volatility_adjustment(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate position size adjustment based on market volatility"""
        
        try:
            if not market_data:
                return 1.0
            
            # Get volatility from market data or calculate from recent trades
            volatility = market_data.get('volatility', None)
            
            if volatility is None:
                # Calculate volatility from recent trade history
                recent_trades = [
                    t for t in self.trade_history[-50:] 
                    if t.symbol == symbol
                ]
                
                if len(recent_trades) >= 10:
                    returns = [t.pnl_percentage for t in recent_trades]
                    volatility = np.std(returns)
                else:
                    return 1.0  # Neutral adjustment
            
            # Adjust based on volatility level
            # Higher volatility = smaller position
            if volatility < 0.01:  # Low volatility (< 1%)
                adjustment = 1.2
            elif volatility < 0.02:  # Medium volatility (1-2%)
                adjustment = 1.0
            elif volatility < 0.05:  # High volatility (2-5%)
                adjustment = 0.8
            else:  # Very high volatility (> 5%)
                adjustment = 0.6
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {e}")
            return 1.0

    def _calculate_drawdown_adjustment(self) -> float:
        """Calculate position size adjustment based on current drawdown"""
        
        try:
            # Update current drawdown
            self._update_drawdown()
            
            if self.current_drawdown <= 0.05:  # Less than 5% drawdown
                return 1.0
            elif self.current_drawdown <= 0.10:  # 5-10% drawdown
                return 0.9
            elif self.current_drawdown <= 0.15:  # 10-15% drawdown
                return 0.8
            elif self.current_drawdown <= 0.20:  # 15-20% drawdown
                return 0.7
            else:  # More than 20% drawdown
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating drawdown adjustment: {e}")
            return 1.0

    def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Calculate adjustment based on position correlations"""
        
        try:
            if not self.current_positions:
                return 1.0
            
            # Check correlation with existing positions
            high_correlation_count = 0
            total_correlation_risk = 0.0
            
            for existing_symbol, position_info in self.current_positions.items():
                if existing_symbol == symbol:
                    continue
                
                correlation = self._get_symbol_correlation(symbol, existing_symbol)
                
                if abs(correlation) > 0.7:  # High correlation
                    high_correlation_count += 1
                    total_correlation_risk += abs(correlation) * position_info.get('risk_amount', 0)
            
            # Reduce position size if high correlation exists
            if high_correlation_count > 0:
                correlation_penalty = min(0.5, high_correlation_count * 0.15)
                return 1.0 - correlation_penalty
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation adjustment: {e}")
            return 1.0

    def _calculate_time_adjustment(self) -> float:
        """Calculate adjustment based on market session/time"""
        
        try:
            current_hour = datetime.now().hour
            
            # Major forex sessions
            if 8 <= current_hour < 17:  # London session
                return 1.0
            elif 13 <= current_hour < 22:  # New York session
                return 1.0
            elif 0 <= current_hour < 9:   # Asian session
                return 0.9
            else:  # Low liquidity hours
                return 0.8
                
        except Exception as e:
            logger.error(f"Error calculating time adjustment: {e}")
            return 1.0

    def _calculate_strategy_adjustment(self, strategy: str) -> float:
        """Calculate adjustment based on strategy type"""
        
        strategy_multipliers = {
            'SCALPING': 0.8,        # Smaller positions for scalping
            'SWING': 1.0,           # Normal positions for swing
            'TREND': 1.2,           # Larger positions for trend following
            'MEAN_REVERSION': 0.9,  # Moderate positions for mean reversion
            'ARBITRAGE': 1.1,       # Slightly larger for arbitrage
            'RL': 1.0,              # Normal for RL strategies
            'ENSEMBLE': 1.1,        # Slightly larger for ensemble
            'MULTIAGENT': 1.15      # Larger for multi-agent (higher confidence)
        }
        
        return strategy_multipliers.get(strategy.upper(), 1.0)

    def _estimate_risk_per_unit(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Estimate risk per unit when stop loss is not provided"""
        
        try:
            # Try to get ATR from market data
            if market_data and 'atr' in market_data:
                return market_data['atr'] * 2  # 2x ATR as risk estimate
            
            # Calculate from recent trade history
            recent_trades = [
                t for t in self.trade_history[-30:] 
                if t.symbol == symbol
            ]
            
            if len(recent_trades) >= 10:
                returns = [abs(t.pnl_percentage) for t in recent_trades]
                return np.percentile(returns, 80)  # 80th percentile of absolute returns
            
            # Default estimates by symbol type
            if 'USD' in symbol or 'EUR' in symbol or 'GBP' in symbol:
                return 0.02  # 2% for major forex pairs
            elif 'XAU' in symbol or 'GOLD' in symbol:
                return 0.03  # 3% for gold
            elif 'JPY' in symbol:
                return 0.025  # 2.5% for yen pairs
            else:
                return 0.025  # 2.5% default
                
        except Exception as e:
            logger.error(f"Error estimating risk per unit: {e}")
            return 0.02

    def add_trade_result(self, trade_result: Dict[str, Any]):
        """Add completed trade to history for Kelly calculation"""
        
        with self.lock:
            try:
                trade = TradeHistory(
                    timestamp=trade_result.get('exit_time', datetime.now()),
                    symbol=trade_result.get('symbol', 'UNKNOWN'),
                    direction=trade_result.get('direction', 'UNKNOWN'),
                    entry_price=trade_result.get('entry_price', 0.0),
                    exit_price=trade_result.get('exit_price', 0.0),
                    position_size=trade_result.get('position_size', 0.0),
                    pnl=trade_result.get('pnl', 0.0),
                    pnl_percentage=trade_result.get('pnl_percentage', 0.0),
                    hold_time=trade_result.get('hold_time_hours', 0.0),
                    strategy=trade_result.get('strategy', 'UNKNOWN'),
                    market_conditions=trade_result.get('market_conditions', 'UNKNOWN')
                )
                
                self.trade_history.append(trade)
                
                # Maintain history size limit
                if len(self.trade_history) > 1000:
                    self.trade_history = self.trade_history[-800:]
                
                # Update consecutive losses counter
                if trade.pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0
                
                # Update daily PnL
                self.daily_pnl += trade.pnl
                
                # Invalidate performance cache
                self.performance_cache.clear()
                
                logger.debug(f"Added trade result: {trade.symbol} PnL={trade.pnl:.2f}")
                
            except Exception as e:
                logger.error(f"Error adding trade result: {e}")

    def update_position(self, symbol: str, position_info: Dict[str, Any]):
        """Update current position information"""
        
        with self.lock:
            self.current_positions[symbol] = position_info
            self._update_portfolio_exposure()

    def close_position(self, symbol: str):
        """Remove position from tracking"""
        
        with self.lock:
            if symbol in self.current_positions:
                del self.current_positions[symbol]
                self._update_portfolio_exposure()

    def _update_portfolio_exposure(self):
        """Update total portfolio exposure"""
        
        self.portfolio_exposure = sum(
            pos.get('position_value', 0) for pos in self.current_positions.values()
        )
        
        self.current_risk_exposure = sum(
            pos.get('risk_amount', 0) for pos in self.current_positions.values()
        )

    def _update_drawdown(self):
        """Update current drawdown calculation"""
        
        try:
            if not self.trade_history:
                return
            
            # Calculate running equity curve
            equity_curve = []
            running_pnl = 0.0
            
            for trade in self.trade_history:
                running_pnl += trade.pnl
                equity_curve.append(running_pnl)
            
            if not equity_curve:
                return
            
            # Calculate peak and current drawdown
            self.peak_equity = max(equity_curve)
            current_equity = equity_curve[-1]
            
            if self.peak_equity > 0:
                self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            else:
                self.current_drawdown = 0.0
                
        except Exception as e:
            logger.error(f"Error updating drawdown: {e}")

    def _update_performance_metrics(self):
        """Update cached performance metrics"""
        
        current_time = time.time()
        if current_time - self.last_performance_update < self.performance_update_interval:
            return  # Use cached metrics
        
        try:
            if len(self.trade_history) < 10:
                return
            
            recent_trades = self.trade_history[-100:]  # Last 100 trades
            
            # Calculate metrics
            winning_trades = [t for t in recent_trades if t.pnl > 0]
            losing_trades = [t for t in recent_trades if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(recent_trades) if recent_trades else 0
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else 0
            
            returns = [t.pnl for t in recent_trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            self.performance_cache = {
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': len(recent_trades),
                'consecutive_losses': self.consecutive_losses
            }
            
            self.last_performance_update = current_time
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def _get_current_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics"""
        
        self._update_performance_metrics()
        
        return {
            'current_drawdown': self.current_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'portfolio_exposure': self.portfolio_exposure,
            'current_risk_exposure': self.current_risk_exposure,
            'daily_pnl': self.daily_pnl,
            **self.performance_cache
        }

    def _get_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols (simplified implementation)"""
        
        # Simplified correlation based on symbol similarity
        # In production, this should use actual price correlation
        
        correlation_matrix = {
            ('EURUSD', 'GBPUSD'): 0.75,
            ('EURUSD', 'AUDUSD'): 0.65,
            ('GBPUSD', 'AUDUSD'): 0.70,
            ('USDJPY', 'EURJPY'): 0.80,
            ('XAUUSD', 'XAGUSD'): 0.85,
        }
        
        pair = tuple(sorted([symbol1, symbol2]))
        return correlation_matrix.get(pair, 0.0)

    def _generate_sizing_reasoning(self, kelly_fraction: float, adjustments: Dict[str, float], strategy: str) -> str:
        """Generate human-readable reasoning for position sizing"""
        
        reasoning_parts = [
            f"Kelly Fraction: {kelly_fraction:.3f}",
            f"Safety Factor: {self.kelly_safety_factor}",
            f"Confidence Adj: {adjustments['confidence_multiplier']:.2f}",
            f"Volatility Adj: {adjustments['volatility_adjustment']:.2f}",
            f"Drawdown Adj: {adjustments['drawdown_adjustment']:.2f}",
            f"Correlation Adj: {adjustments['correlation_adjustment']:.2f}",
            f"Strategy: {strategy}"
        ]
        
        return " | ".join(reasoning_parts)

    def _get_fallback_position_size(self, account_balance: float, entry_price: float) -> PositionSizingResult:
        """Get fallback position size when calculation fails"""
        
        fallback_size = (self.base_risk_per_trade * account_balance) / entry_price
        
        return PositionSizingResult(
            recommended_size=fallback_size,
            kelly_fraction=self.base_risk_per_trade,
            safety_adjusted_kelly=self.base_risk_per_trade,
            confidence_multiplier=1.0,
            volatility_adjustment=1.0,
            drawdown_adjustment=1.0,
            correlation_adjustment=1.0,
            final_risk_percentage=self.base_risk_per_trade,
            max_position_value=fallback_size * entry_price,
            reasoning="Fallback sizing due to calculation error",
            risk_metrics={}
        )

    def get_position_sizing_summary(self) -> Dict[str, Any]:
        """Get comprehensive position sizing summary"""
        
        with self.lock:
            return {
                'configuration': {
                    'kelly_lookback_trades': self.kelly_lookback_trades,
                    'kelly_safety_factor': self.kelly_safety_factor,
                    'base_risk_per_trade': self.base_risk_per_trade,
                    'max_risk_per_trade': self.max_risk_per_trade,
                    'max_portfolio_risk': self.max_portfolio_risk
                },
                'current_state': {
                    'total_trades': len(self.trade_history),
                    'current_drawdown': self.current_drawdown,
                    'consecutive_losses': self.consecutive_losses,
                    'portfolio_exposure': self.portfolio_exposure,
                    'current_risk_exposure': self.current_risk_exposure,
                    'active_positions': len(self.current_positions)
                },
                'performance_metrics': self.performance_cache,
                'recent_kelly_fractions': self._get_recent_kelly_fractions()
            }

    def _get_recent_kelly_fractions(self) -> Dict[str, float]:
        """Get Kelly fractions for different symbols/strategies"""
        
        kelly_fractions = {}
        
        # Calculate for each symbol
        symbols = list(set(t.symbol for t in self.trade_history[-100:]))
        for symbol in symbols[:5]:  # Top 5 most recent symbols
            kelly_fractions[f"{symbol}_kelly"] = self._calculate_kelly_fraction(symbol, 'ALL')
        
        return kelly_fractions

    def save_state(self, filepath: str):
        """Save position sizer state to file"""
        
        try:
            state = {
                'config': self.config,
                'trade_history': [
                    {
                        'timestamp': trade.timestamp.isoformat(),
                        'symbol': trade.symbol,
                        'direction': trade.direction,
                        'entry_price': trade.entry_price,
                        'exit_price': trade.exit_price,
                        'position_size': trade.position_size,
                        'pnl': trade.pnl,
                        'pnl_percentage': trade.pnl_percentage,
                        'hold_time': trade.hold_time,
                        'strategy': trade.strategy,
                        'market_conditions': trade.market_conditions
                    }
                    for trade in self.trade_history
                ],
                'current_positions': self.current_positions,
                'current_drawdown': self.current_drawdown,
                'peak_equity': self.peak_equity,
                'consecutive_losses': self.consecutive_losses,
                'daily_pnl': self.daily_pnl
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"Position sizer state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def load_state(self, filepath: str):
        """Load position sizer state from file"""
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore trade history
            self.trade_history = []
            for trade_data in state.get('trade_history', []):
                trade = TradeHistory(
                    timestamp=datetime.fromisoformat(trade_data['timestamp']),
                    symbol=trade_data['symbol'],
                    direction=trade_data['direction'],
                    entry_price=trade_data['entry_price'],
                    exit_price=trade_data['exit_price'],
                    position_size=trade_data['position_size'],
                    pnl=trade_data['pnl'],
                    pnl_percentage=trade_data['pnl_percentage'],
                    hold_time=trade_data['hold_time'],
                    strategy=trade_data['strategy'],
                    market_conditions=trade_data['market_conditions']
                )
                self.trade_history.append(trade)
            
            # Restore other state
            self.current_positions = state.get('current_positions', {})
            self.current_drawdown = state.get('current_drawdown', 0.0)
            self.peak_equity = state.get('peak_equity', 0.0)
            self.consecutive_losses = state.get('consecutive_losses', 0)
            self.daily_pnl = state.get('daily_pnl', 0.0)
            
            logger.info(f"Position sizer state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")

# Integration helper functions

def create_kelly_position_sizer(config: Dict[str, Any] = None) -> ProfessionalKellyPositionSizer:
    """Factory function to create Kelly position sizer"""
    
    default_config = {
        'kelly_lookback_trades': 100,
        'kelly_safety_factor': 0.25,
        'min_kelly_samples': 20,
        'base_risk_per_trade': 0.01,
        'max_risk_per_trade': 0.05,
        'max_portfolio_risk': 0.20,
        'max_single_symbol_risk': 0.10,
        'confidence_weight': 0.3,
        'volatility_weight': 0.3,
        'drawdown_weight': 0.4
    }
    
    if config:
        default_config.update(config)
    
    return ProfessionalKellyPositionSizer(default_config)

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'kelly_lookback_trades': 50,
        'kelly_safety_factor': 0.20,
        'base_risk_per_trade': 0.015,
        'max_risk_per_trade': 0.06
    }
    
    # Create position sizer
    kelly_sizer = create_kelly_position_sizer(config)
    
    # Example signal
    signal = {
        'symbol': 'EURUSD',
        'confidence': 0.75,
        'entry_price': 1.0500,
        'stop_loss': 1.0450,
        'strategy': 'SWING'
    }
    
    # Calculate position size
    result = kelly_sizer.calculate_optimal_position_size(
        signal=signal,
        account_balance=10000,
        market_data={'volatility': 0.015, 'atr': 0.008}
    )
    
    print("Kelly Position Sizing Result:")
    print(f"Recommended Size: {result.recommended_size:.4f}")
    print(f"Kelly Fraction: {result.kelly_fraction:.3f}")
    print(f"Risk Percentage: {result.final_risk_percentage:.3%}")
    print(f"Reasoning: {result.reasoning}")
    
    # Add some example trades for Kelly calculation
    example_trades = [
        {'symbol': 'EURUSD', 'pnl': 150, 'pnl_percentage': 0.015, 'strategy': 'SWING'},
        {'symbol': 'EURUSD', 'pnl': -100, 'pnl_percentage': -0.010, 'strategy': 'SWING'},
        {'symbol': 'EURUSD', 'pnl': 200, 'pnl_percentage': 0.020, 'strategy': 'SWING'},
        {'symbol': 'EURUSD', 'pnl': -80, 'pnl_percentage': -0.008, 'strategy': 'SWING'},
        {'symbol': 'EURUSD', 'pnl': 300, 'pnl_percentage': 0.030, 'strategy': 'SWING'}
    ]
    
    for trade in example_trades:
        kelly_sizer.add_trade_result(trade)
    
    # Recalculate with trade history
    result2 = kelly_sizer.calculate_optimal_position_size(
        signal=signal,
        account_balance=10000,
        market_data={'volatility': 0.015, 'atr': 0.008}
    )
    
    print("\nUpdated Kelly Position Sizing (with history):")
    print(f"Recommended Size: {result2.recommended_size:.4f}")
    print(f"Kelly Fraction: {result2.kelly_fraction:.3f}")
    print(f"Risk Percentage: {result2.final_risk_percentage:.3%}")
    
    # Get summary
    summary = kelly_sizer.get_position_sizing_summary()
    print(f"\nSystem Summary:")
    print(f"Total Trades: {summary['current_state']['total_trades']}")
    print(f"Current Drawdown: {summary['current_state']['current_drawdown']:.3%}")

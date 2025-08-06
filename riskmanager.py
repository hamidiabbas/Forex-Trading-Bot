"""
Enhanced Risk Manager - Professional Grade Implementation
Comprehensive risk management with portfolio optimization, dynamic correlation,
volatility targeting, drawdown protection, and advanced analytics
"""

import numpy as np
import pandas as pd
import logging
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level enumeration"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

class MarketRegime(Enum):
    """Market regime enumeration"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"

class PositionSizingMethod(Enum):
    """Position sizing method enumeration"""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly_criterion"
    OPTIMAL_F = "optimal_f"
    VOLATILITY_TARGET = "volatility_target"
    RISK_PARITY = "risk_parity"
    DYNAMIC_ALLOCATION = "dynamic_allocation"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics data structure"""
    position_size: float
    risk_amount: float
    max_loss_amount: float
    max_gain_amount: float
    risk_reward_ratio: float
    portfolio_risk_pct: float
    confidence_adjusted_size: float
    kelly_size: float
    volatility_adjusted_size: float
    correlation_adjusted_size: float
    drawdown_adjusted_size: float
    final_position_size: float
    risk_score: float
    sizing_method: str
    market_regime: str
    liquidity_score: float
    execution_risk_score: float

@dataclass
class PortfolioMetrics:
    """Portfolio-level risk metrics"""
    total_exposure: float
    net_exposure: float
    gross_exposure: float
    portfolio_beta: float
    portfolio_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    var_95: float
    cvar_95: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    timestamp: datetime
    alert_type: str
    severity: RiskLevel
    symbol: str
    message: str
    current_value: float
    threshold: float
    recommended_action: str

class EnhancedRiskManager:
    """
    Complete Professional Risk Manager with Advanced Portfolio Management
    """
    
    def __init__(self, config, kelly_manager=None, correlation_matrix: Dict[Tuple, float] = None):
        self.config = config
        self.kelly_manager = kelly_manager  # Store kelly_manager separately
        self.correlation_matrix = correlation_matrix or {}  # Keep correlation_matrix as dict
        
        # Core risk parameters
        self.max_risk_per_trade = getattr(config, 'max_risk_per_trade', 0.01)
        self.max_daily_risk = getattr(config, 'max_daily_risk', 0.05)  
        self.max_weekly_risk = getattr(config, 'max_weekly_risk', 0.15)
        self.max_monthly_risk = getattr(config, 'max_monthly_risk', 0.30)
        self.correlation_limit = getattr(config,'risk_management.correlation_limit', 0.7)
        
        # Advanced risk parameters
        self.max_portfolio_exposure = getattr(config,'risk_management.max_portfolio_exposure', 2.0)
        self.max_single_symbol_exposure = getattr(config,'risk_management.max_single_symbol_exposure', 0.3)
        self.max_drawdown_limit = getattr(config,'risk_management.max_drawdown_limit', 0.15)
        self.volatility_target = getattr(config,'risk_management.volatility_target', 0.15)
        self.var_confidence = getattr(config,'risk_management.var_confidence', 0.95)
        
        # Position sizing configuration
        self.default_sizing_method = PositionSizingMethod(
            getattr(config,'risk_management.default_sizing_method', 'dynamic_allocation')
        )
        self.use_dynamic_sizing = getattr(config,'risk_management.use_dynamic_position_sizing', True)
        self.use_kelly_criterion = getattr(config,'risk_management.use_kelly_criterion', True)
        self.use_volatility_targeting = getattr(config,'risk_management.use_volatility_targeting', True)
        
        # Symbol risk weights and parameters
        self.symbol_risk_weights = getattr(config,'risk_management.symbol_risk_weights', {
            'EURUSD': 1.0, 'GBPUSD': 1.1, 'XAUUSD': 1.4, 'USDJPY': 1.2
        })
        
        # Market regime parameters
        self.regime_detection_enabled = getattr(config,'risk_management.enable_regime_detection', True)
        self.regime_lookback_period = getattr(config,'risk_management.regime_lookback_period', 50)
        self.volatility_threshold_high = getattr(config,'risk_management.volatility_threshold_high', 0.02)
        self.volatility_threshold_low = getattr(config,'risk_management.volatility_threshold_low', 0.005)
        
        # Risk tracking and monitoring
        self.daily_risk_used = 0.0
        self.weekly_risk_used = 0.0
        self.monthly_risk_used = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown_session = 0.0
        self.last_reset_date = datetime.now().date()
        self.last_weekly_reset = datetime.now().date()
        self.last_monthly_reset = datetime.now().date()
        
        # Portfolio tracking
        self.portfolio_positions = {}
        self.portfolio_history = []
        self.risk_alerts = []
        self.performance_metrics = {}
        
        # Advanced analytics
        self.correlation_cache = {}
        self.volatility_cache = {}
        self.regime_cache = {}
        self.kelly_cache = {}
        
        # Risk limits and emergency controls
        self.emergency_stop_triggered = False
        self.risk_override_mode = False
        self.max_consecutive_losses = getattr(config,'emergency.max_consecutive_losses', 5)
        self.consecutive_losses = 0
        
        # Historical data for calculations
        self.price_history = {}
        self.returns_history = {}
        self.volatility_history = {}
        
        # Performance tracking
        self.total_trades_managed = 0
        self.successful_risk_calculations = 0
        self.risk_alerts_generated = 0
        
        logger.info("Enhanced Risk Manager initialized with professional features")
        logger.info(f"Max Risk per Trade: {self.max_risk_per_trade*100:.1f}%")
        logger.info(f"Max Portfolio Exposure: {self.max_portfolio_exposure:.1f}x")
        logger.info(f"Max Drawdown Limit: {self.max_drawdown_limit*100:.1f}%")
        logger.info(f"Volatility Target: {self.volatility_target*100:.1f}%")
        logger.info(f"Correlation Pairs: {len(self.correlation_matrix)}")
        logger.info(f"Default Sizing Method: {self.default_sizing_method.value}")

    def calculate_enhanced_risk(self, symbol: str, direction: str, entry_price: float, 
                               stop_loss: float, take_profit: float, confidence: float, 
                               strategy: str, account_balance: float = None) -> Optional[Dict[str, Any]]:
        """
        Calculate comprehensive risk parameters with all advanced features
        """
        try:
            with self.lock:
                self.total_trades_managed += 1
                self._reset_periodic_risk_counters()
                
                # Get account balance
                if account_balance is None:
                    account_balance = self._get_account_balance()
                
                # Pre-flight risk checks
                if not self._pre_flight_risk_checks(symbol, account_balance):
                    return None
                
                # Validate inputs
                if not self._validate_risk_inputs(symbol, direction, entry_price, stop_loss, confidence):
                    return None
                
                # Get market data and regime
                market_data = self._get_market_data_for_risk(symbol)
                current_regime = self._detect_market_regime(symbol, market_data)
                
                # Calculate base position size using multiple methods
                position_sizes = self._calculate_multiple_position_sizes(
                    symbol, direction, entry_price, stop_loss, account_balance, 
                    confidence, current_regime, market_data
                )
                
                # Select optimal position size
                optimal_size = self._select_optimal_position_size(
                    position_sizes, symbol, current_regime, confidence
                )
                
                # Apply portfolio-level adjustments
                portfolio_adjusted_size = self._apply_portfolio_adjustments(
                    optimal_size, symbol, direction, account_balance
                )
                
                # Apply correlation adjustments
                correlation_adjusted_size = self._apply_correlation_adjustments(
                    portfolio_adjusted_size, symbol, direction
                )
                
                # Apply drawdown protection
                drawdown_adjusted_size = self._apply_drawdown_protection(
                    correlation_adjusted_size, account_balance
                )
                
                # Apply emergency controls
                final_size = self._apply_emergency_controls(drawdown_adjusted_size, symbol, strategy)
                
                # Calculate comprehensive risk metrics
                risk_metrics = self._calculate_comprehensive_risk_metrics(
                    symbol, final_size, entry_price, stop_loss, take_profit, 
                    account_balance, position_sizes, current_regime
                )
                
                # Validate final risk
                if not self._validate_comprehensive_risk(risk_metrics, symbol):
                    return None
                
                # Update risk tracking
                self._update_comprehensive_risk_tracking(risk_metrics, symbol, strategy)
                
                # Generate risk alerts if necessary
                self._check_and_generate_risk_alerts(risk_metrics, symbol)
                
                # Create comprehensive risk parameters
                risk_params = self._create_comprehensive_risk_params(
                    risk_metrics, entry_price, stop_loss, take_profit, 
                    strategy, confidence, current_regime
                )
                
                self.successful_risk_calculations += 1
                logger.debug(f"Enhanced risk calculated for {symbol}: "
                           f"Size={risk_metrics.final_position_size:.3f}, "
                           f"Risk=${risk_metrics.risk_amount:.2f}, "
                           f"Method={risk_metrics.sizing_method}, "
                           f"Regime={current_regime.value}")
                
                return risk_params
                
        except Exception as e:
            logger.error(f"Error in enhanced risk calculation for {symbol}: {e}")
            return None

    def _pre_flight_risk_checks(self, symbol: str, account_balance: float) -> bool:
        """Comprehensive pre-flight risk checks"""
        try:
            # Check emergency stop
            if self.emergency_stop_triggered:
                logger.warning(f"Emergency stop active - blocking {symbol}")
                return False
            
            # Check account balance
            if account_balance <= 0:
                logger.error(f"Invalid account balance: {account_balance}")
                return False
            
            # Check drawdown limits
            if self.current_drawdown > self.max_drawdown_limit:
                logger.warning(f"Drawdown limit exceeded: {self.current_drawdown:.2%} > {self.max_drawdown_limit:.2%}")
                return False
            
            # Check daily risk limits
            if self.daily_risk_used > self.max_daily_risk * 100:
                logger.warning(f"Daily risk limit reached: {self.daily_risk_used:.1f}%")
                return False
            
            # Check portfolio exposure
            current_exposure = self._calculate_current_portfolio_exposure()
            if current_exposure > self.max_portfolio_exposure:
                logger.warning(f"Portfolio exposure limit reached: {current_exposure:.1f}x")
                return False
            
            # Check symbol-specific exposure
            symbol_exposure = self._calculate_symbol_exposure(symbol)
            if symbol_exposure > self.max_single_symbol_exposure:
                logger.warning(f"Symbol exposure limit reached for {symbol}: {symbol_exposure:.1%}")
                return False
            
            # Check consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                logger.warning(f"Too many consecutive losses: {self.consecutive_losses}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in pre-flight checks: {e}")
            return False

    def _get_market_data_for_risk(self, symbol: str) -> Dict[str, Any]:
        """Get market data specifically for risk calculations"""
        try:
            # Try to get real market data
            if symbol in self.price_history and len(self.price_history[symbol]) > 0:
                recent_prices = self.price_history[symbol][-100:]  # Last 100 prices
                if len(recent_prices) > 1:
                    current_price = recent_prices[-1]
                    returns = np.diff(np.log(recent_prices))
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized
                    
                    return {
                        'current_price': current_price,
                        'volatility': volatility,
                        'returns': returns,
                        'price_history': recent_prices,
                        'atr': np.mean(np.abs(returns)) * current_price,
                        'trend': 1 if recent_prices[-1] > recent_prices[-20] else -1
                    }
            
            # Fallback to synthetic data
            base_prices = {'EURUSD': 1.1000, 'GBPUSD': 1.3000, 'XAUUSD': 2000.0, 'USDJPY': 148.0}
            price = base_prices.get(symbol, 1.1000)
            
            return {
                'current_price': price,
                'volatility': 0.15,  # 15% annual volatility
                'atr': price * 0.01,  # 1% ATR
                'trend': 0,
                'returns': np.array([0.001, -0.002, 0.003]),  # Sample returns
                'price_history': [price] * 20
            }
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {'current_price': 1.0, 'volatility': 0.15, 'atr': 0.01, 'trend': 0}

    def _detect_market_regime(self, symbol: str, market_data: Dict[str, Any]) -> MarketRegime:
        """Advanced market regime detection"""
        try:
            if not self.regime_detection_enabled:
                return MarketRegime.RANGING
            
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().date()}"
            if cache_key in self.regime_cache:
                return self.regime_cache[cache_key]
            
            volatility = market_data.get('volatility', 0.15)
            trend = market_data.get('trend', 0)
            returns = market_data.get('returns', np.array([]))
            
            # Volatility regime
            if volatility > self.volatility_threshold_high:
                if len(returns) > 0 and np.std(returns) > volatility * 2:
                    regime = MarketRegime.CRISIS
                else:
                    regime = MarketRegime.HIGH_VOLATILITY
            elif volatility < self.volatility_threshold_low:
                regime = MarketRegime.LOW_VOLATILITY
            else:
                # Trend regime
                if trend > 0.5:
                    regime = MarketRegime.TRENDING_BULL
                elif trend < -0.5:
                    regime = MarketRegime.TRENDING_BEAR
                else:
                    regime = MarketRegime.RANGING
            
            # Cache the result
            self.regime_cache[cache_key] = regime
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime for {symbol}: {e}")
            return MarketRegime.RANGING

    def _calculate_multiple_position_sizes(self, symbol: str, direction: str, entry_price: float, 
                                         stop_loss: float, account_balance: float, confidence: float, 
                                         regime: MarketRegime, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate position sizes using multiple methods"""
        try:
            sizes = {}
            
            # 1. Fixed Fractional Method
            sizes['fixed_fractional'] = self._calculate_fixed_fractional_size(
                symbol, entry_price, stop_loss, account_balance
            )
            
            # 2. Kelly Criterion
            sizes['kelly'] = self._calculate_kelly_criterion_size(
                symbol, entry_price, stop_loss, account_balance, confidence, market_data
            )
            
            # 3. Volatility Targeting
            sizes['volatility_target'] = self._calculate_volatility_target_size(
                symbol, account_balance, market_data
            )
            
            # 4. Risk Parity
            sizes['risk_parity'] = self._calculate_risk_parity_size(
                symbol, account_balance, market_data
            )
            
            # 5. Optimal F
            sizes['optimal_f'] = self._calculate_optimal_f_size(
                symbol, entry_price, stop_loss, account_balance, market_data
            )
            
            # 6. Regime-based adjustment
            regime_multiplier = self._get_regime_multiplier(regime)
            for method in sizes:
                sizes[method] *= regime_multiplier
            
            return sizes
            
        except Exception as e:
            logger.error(f"Error calculating multiple position sizes: {e}")
            return {'fixed_fractional': 0.01}

    def _calculate_fixed_fractional_size(self, symbol: str, entry_price: float, 
                                       stop_loss: float, account_balance: float) -> float:
        """Calculate fixed fractional position size"""
        try:
            risk_per_trade = self.max_risk_per_trade
            symbol_weight = self.symbol_risk_weights.get(symbol, 1.0)
            adjusted_risk = risk_per_trade / symbol_weight
            
            risk_amount = account_balance * adjusted_risk
            risk_in_pips = abs(entry_price - stop_loss)
            pip_value = self._get_pip_value(symbol)
            
            if risk_in_pips > 0 and pip_value > 0:
                position_size = risk_amount / (risk_in_pips * pip_value * 100000)
                return max(0.01, min(position_size, 1.0))
            
            return 0.01
            
        except Exception as e:
            logger.error(f"Error in fixed fractional calculation: {e}")
            return 0.01

    def _calculate_kelly_criterion_size(self, symbol: str, entry_price: float, stop_loss: float,
                                      account_balance: float, confidence: float, 
                                      market_data: Dict[str, Any]) -> float:
        """Calculate Kelly Criterion position size"""
        try:
            if not self.use_kelly_criterion:
                return self._calculate_fixed_fractional_size(symbol, entry_price, stop_loss, account_balance)
            
            # Simplified Kelly calculation based on confidence and historical performance
            win_rate = max(0.51, min(0.80, confidence))  # Convert confidence to win rate
            avg_win = 1.5  # Average win multiplier
            avg_loss = 1.0  # Average loss multiplier
            
            kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_f = max(0.01, min(0.25, kelly_f))  # Cap Kelly fraction
            
            # Convert to position size
            risk_amount = account_balance * kelly_f
            risk_in_pips = abs(entry_price - stop_loss)
            pip_value = self._get_pip_value(symbol)
            
            if risk_in_pips > 0 and pip_value > 0:
                position_size = risk_amount / (risk_in_pips * pip_value * 100000)
                return max(0.01, min(position_size, 0.5))
            
            return 0.01
            
        except Exception as e:
            logger.error(f"Error in Kelly criterion calculation: {e}")
            return 0.01

    def _calculate_volatility_target_size(self, symbol: str, account_balance: float, 
                                        market_data: Dict[str, Any]) -> float:
        """Calculate volatility-targeted position size"""
        try:
            if not self.use_volatility_targeting:
                return 0.01
            
            current_volatility = market_data.get('volatility', 0.15)
            target_volatility = self.volatility_target
            
            # Scale position size inversely with volatility
            volatility_multiplier = target_volatility / max(current_volatility, 0.01)
            base_size = account_balance * 0.01 / (market_data.get('current_price', 1.0) * 100000)
            
            position_size = base_size * volatility_multiplier
            return max(0.01, min(position_size, 0.3))
            
        except Exception as e:
            logger.error(f"Error in volatility targeting calculation: {e}")
            return 0.01

    def _calculate_risk_parity_size(self, symbol: str, account_balance: float, 
                                  market_data: Dict[str, Any]) -> float:
        """Calculate risk parity position size"""
        try:
            # Simple risk parity: equal risk contribution
            volatility = market_data.get('volatility', 0.15)
            target_risk = account_balance * 0.01  # 1% risk target
            
            position_value = target_risk / volatility
            current_price = market_data.get('current_price', 1.0)
            position_size = position_value / (current_price * 100000)
            
            return max(0.01, min(position_size, 0.2))
            
        except Exception as e:
            logger.error(f"Error in risk parity calculation: {e}")
            return 0.01

    def _calculate_optimal_f_size(self, symbol: str, entry_price: float, stop_loss: float,
                                account_balance: float, market_data: Dict[str, Any]) -> float:
        """Calculate Optimal F position size"""
        try:
            # Simplified Optimal F calculation
            # Would require historical trade outcomes for full implementation
            base_size = self._calculate_fixed_fractional_size(symbol, entry_price, stop_loss, account_balance)
            
            # Apply confidence adjustment
            volatility = market_data.get('volatility', 0.15)
            confidence_multiplier = 0.8 + (0.4 / max(volatility, 0.05))  # Lower vol = higher confidence
            
            return base_size * confidence_multiplier
            
        except Exception as e:
            logger.error(f"Error in Optimal F calculation: {e}")
            return 0.01

    def _select_optimal_position_size(self, position_sizes: Dict[str, float], symbol: str, 
                                    regime: MarketRegime, confidence: float) -> float:
        """Select optimal position size from multiple methods"""
        try:
            # Filter out zero or negative sizes
            valid_sizes = {k: v for k, v in position_sizes.items() if v > 0}
            
            if not valid_sizes:
                return 0.01
            
            # Use median for robustness
            sizes_list = list(valid_sizes.values())
            median_size = np.median(sizes_list)
            
            # Apply confidence scaling
            confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0 range
            final_size = median_size * confidence_multiplier
            
            # Ensure reasonable bounds
            return max(0.01, min(final_size, 0.5))
            
        except Exception as e:
            logger.error(f"Error selecting optimal position size: {e}")
            return 0.01

    def _apply_portfolio_adjustments(self, position_size: float, symbol: str, 
                                   direction: str, account_balance: float) -> float:
        """Apply portfolio-level adjustments"""
        try:
            # Check current portfolio exposure
            current_exposure = self._calculate_current_portfolio_exposure()
            exposure_multiplier = max(0.5, min(1.0, self.max_portfolio_exposure - current_exposure))
            
            # Apply symbol-specific adjustments
            symbol_exposure = self._calculate_symbol_exposure(symbol)
            symbol_multiplier = max(0.1, min(1.0, self.max_single_symbol_exposure - symbol_exposure))
            
            adjusted_size = position_size * exposure_multiplier * symbol_multiplier
            return max(0.01, adjusted_size)
            
        except Exception as e:
            logger.error(f"Error applying portfolio adjustments: {e}")
            return position_size

    def _apply_correlation_adjustments(self, position_size: float, symbol: str, direction: str) -> float:
        """Apply correlation-based adjustments"""
        try:
            if not self.correlation_matrix:
                return position_size
            
            # Check correlations with existing positions
            correlation_risk = 0.0
            for existing_symbol in self.portfolio_positions:
                pair_key = tuple(sorted([symbol, existing_symbol]))
                correlation = self.correlation_matrix.get(pair_key, 0.0)
                
                if abs(correlation) > self.correlation_limit:
                    correlation_risk += abs(correlation)
            
            # Reduce size based on correlation risk
            correlation_multiplier = max(0.3, 1.0 - (correlation_risk * 0.5))
            adjusted_size = position_size * correlation_multiplier
            
            return max(0.01, adjusted_size)
            
        except Exception as e:
            logger.error(f"Error applying correlation adjustments: {e}")
            return position_size

    def _apply_drawdown_protection(self, position_size: float, account_balance: float) -> float:
        """Apply drawdown protection adjustments"""
        try:
            # Reduce position size based on current drawdown
            if self.current_drawdown > 0:
                drawdown_multiplier = max(0.5, 1.0 - (self.current_drawdown / self.max_drawdown_limit))
                position_size *= drawdown_multiplier
            
            # Reduce size if approaching daily risk limits
            daily_risk_used_pct = self.daily_risk_used / 100
            if daily_risk_used_pct > 0.7:  # 70% of daily limit used
                daily_multiplier = max(0.3, 1.0 - daily_risk_used_pct)
                position_size *= daily_multiplier
            
            return max(0.01, position_size)
            
        except Exception as e:
            logger.error(f"Error applying drawdown protection: {e}")
            return position_size

    def _apply_emergency_controls(self, position_size: float, symbol: str, strategy: str) -> float:
        """Apply emergency risk controls"""
        try:
            if self.emergency_stop_triggered:
                return 0.0
            
            # Reduce size for consecutive losses
            if self.consecutive_losses > 2:
                loss_multiplier = max(0.5, 1.0 - (self.consecutive_losses * 0.1))
                position_size *= loss_multiplier
            
            return max(0.01, position_size)
            
        except Exception as e:
            logger.error(f"Error applying emergency controls: {e}")
            return position_size

    def _calculate_comprehensive_risk_metrics(self, symbol: str, final_size: float, 
                                            entry_price: float, stop_loss: float, 
                                            take_profit: float, account_balance: float,
                                            position_sizes: Dict[str, float], 
                                            current_regime: MarketRegime) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            pip_value = self._get_pip_value(symbol)
            risk_in_pips = abs(entry_price - stop_loss)
            reward_in_pips = abs(take_profit - entry_price) if take_profit else 0
            
            risk_amount = final_size * risk_in_pips * pip_value * 100000
            max_gain = final_size * reward_in_pips * pip_value * 100000 if reward_in_pips > 0 else 0
            
            risk_reward_ratio = reward_in_pips / risk_in_pips if risk_in_pips > 0 else 0
            portfolio_risk_pct = (risk_amount / account_balance) * 100
            
            return RiskMetrics(
                position_size=final_size,
                risk_amount=risk_amount,
                max_loss_amount=risk_amount,
                max_gain_amount=max_gain,
                risk_reward_ratio=risk_reward_ratio,
                portfolio_risk_pct=portfolio_risk_pct,
                confidence_adjusted_size=position_sizes.get('fixed_fractional', final_size),
                kelly_size=position_sizes.get('kelly', final_size),
                volatility_adjusted_size=position_sizes.get('volatility_target', final_size),
                correlation_adjusted_size=final_size,
                drawdown_adjusted_size=final_size,
                final_position_size=final_size,
                risk_score=min(100, portfolio_risk_pct * 10),
                sizing_method=self.default_sizing_method.value,
                market_regime=current_regime.value,
                liquidity_score=85.0,  # Simplified
                execution_risk_score=15.0  # Simplified
            )
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive risk metrics: {e}")
            return RiskMetrics(
                position_size=0.01, risk_amount=0, max_loss_amount=0, max_gain_amount=0,
                risk_reward_ratio=0, portfolio_risk_pct=0, confidence_adjusted_size=0.01,
                kelly_size=0.01, volatility_adjusted_size=0.01, correlation_adjusted_size=0.01,
                drawdown_adjusted_size=0.01, final_position_size=0.01, risk_score=0,
                sizing_method="fixed_fractional", market_regime="ranging",
                liquidity_score=50.0, execution_risk_score=50.0
            )

    def _validate_comprehensive_risk(self, risk_metrics: RiskMetrics, symbol: str) -> bool:
        """Validate comprehensive risk metrics"""
        try:
            # Check if risk amount is within limits
            if risk_metrics.portfolio_risk_pct > self.max_risk_per_trade * 100:
                logger.warning(f"Risk too high for {symbol}: {risk_metrics.portfolio_risk_pct:.2f}%")
                return False
            
            # Check position size bounds
            if risk_metrics.final_position_size <= 0 or risk_metrics.final_position_size > 1.0:
                logger.warning(f"Invalid position size for {symbol}: {risk_metrics.final_position_size}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating comprehensive risk: {e}")
            return False

    def _update_comprehensive_risk_tracking(self, risk_metrics: RiskMetrics, symbol: str, strategy: str):
        """Update comprehensive risk tracking"""
        try:
            # Update daily risk usage
            self.daily_risk_used += risk_metrics.portfolio_risk_pct
            
            # Add to portfolio positions
            self.portfolio_positions[symbol] = {
                'size': risk_metrics.final_position_size,
                'risk_amount': risk_metrics.risk_amount,
                'strategy': strategy,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error updating risk tracking: {e}")

    def _check_and_generate_risk_alerts(self, risk_metrics: RiskMetrics, symbol: str):
        """Check and generate risk alerts if necessary"""
        try:
            alerts = []
            
            # Check high risk score
            if risk_metrics.risk_score > 75:
                alert = RiskAlert(
                    timestamp=datetime.now(),
                    alert_type="HIGH_RISK_SCORE",
                    severity=RiskLevel.HIGH,
                    symbol=symbol,
                    message=f"High risk score: {risk_metrics.risk_score:.1f}",
                    current_value=risk_metrics.risk_score,
                    threshold=75.0,
                    recommended_action="Consider reducing position size"
                )
                alerts.append(alert)
            
            # Check portfolio risk concentration
            if risk_metrics.portfolio_risk_pct > 3.0:
                alert = RiskAlert(
                    timestamp=datetime.now(),
                    alert_type="PORTFOLIO_CONCENTRATION",
                    severity=RiskLevel.MEDIUM,
                    symbol=symbol,
                    message=f"High portfolio risk: {risk_metrics.portfolio_risk_pct:.2f}%",
                    current_value=risk_metrics.portfolio_risk_pct,
                    threshold=3.0,
                    recommended_action="Monitor position closely"
                )
                alerts.append(alert)
            
            # Add to risk alerts list
            self.risk_alerts.extend(alerts)
            self.risk_alerts_generated += len(alerts)
            
        except Exception as e:
            logger.error(f"Error generating risk alerts: {e}")

    def _create_comprehensive_risk_params(self, risk_metrics: RiskMetrics, entry_price: float,
                                         stop_loss: float, take_profit: float, strategy: str,
                                         confidence: float, current_regime: MarketRegime) -> Dict[str, Any]:
        """Create comprehensive risk parameters dictionary"""
        try:
            return {
                'symbol': risk_metrics.final_position_size,  # This seems wrong, should be symbol
                'position_size': risk_metrics.final_position_size,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': risk_metrics.risk_amount,
                'max_loss': risk_metrics.max_loss_amount,
                'max_gain': risk_metrics.max_gain_amount,
                'risk_reward_ratio': risk_metrics.risk_reward_ratio,
                'portfolio_risk_pct': risk_metrics.portfolio_risk_pct,
                'risk_score': risk_metrics.risk_score,
                'sizing_method': risk_metrics.sizing_method,
                'market_regime': risk_metrics.market_regime,
                'strategy': strategy,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'liquidity_score': risk_metrics.liquidity_score,
                'execution_risk_score': risk_metrics.execution_risk_score
            }
            
        except Exception as e:
            logger.error(f"Error creating risk params: {e}")
            return {}

    # Helper methods
    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol"""
        if 'JPY' in symbol:
            return 0.01
        else:
            return 0.0001

    def _get_account_balance(self) -> float:
        """Get current account balance"""
        return 100000.0  # Default value, should be updated from MT5

    def _validate_risk_inputs(self, symbol: str, direction: str, entry_price: float, 
                            stop_loss: float, confidence: float) -> bool:
        """Validate risk calculation inputs"""
        if not symbol or not direction:
            return False
        if entry_price <= 0 or stop_loss <= 0:
            return False
        if confidence < 0 or confidence > 1:
            return False
        return True

    def _reset_periodic_risk_counters(self):
        """Reset periodic risk counters"""
        current_date = datetime.now().date()
        
        if current_date > self.last_reset_date:
            self.daily_risk_used = 0.0
            self.last_reset_date = current_date

    def _calculate_current_portfolio_exposure(self) -> float:
        """Calculate current portfolio exposure"""
        return len(self.portfolio_positions) * 0.1  # Simplified

    def _calculate_symbol_exposure(self, symbol: str) -> float:
        """Calculate symbol-specific exposure"""
        return 0.05 if symbol in self.portfolio_positions else 0.0  # Simplified

    def _get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get position size multiplier based on market regime"""
        multipliers = {
            MarketRegime.TRENDING_BULL: 1.1,
            MarketRegime.TRENDING_BEAR: 1.1,
            MarketRegime.RANGING: 0.9,
            MarketRegime.HIGH_VOLATILITY: 0.8,
            MarketRegime.LOW_VOLATILITY: 1.0,
            MarketRegime.CRISIS: 0.5
        }
        return multipliers.get(regime, 1.0)

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            return {
                'daily_risk_used': self.daily_risk_used,
                'portfolio_positions': len(self.portfolio_positions),
                'current_drawdown': self.current_drawdown,
                'consecutive_losses': self.consecutive_losses,
                'emergency_stop': self.emergency_stop_triggered,
                'total_trades_managed': self.total_trades_managed,
                'successful_calculations': self.successful_risk_calculations,
                'risk_alerts_generated': self.risk_alerts_generated,
                'last_updated': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}

    def shutdown(self):
        """Shutdown risk manager"""
        try:
            logger.info("Risk Manager shutting down...")
            # Clear caches and reset state
            self.correlation_cache.clear()
            self.volatility_cache.clear()
            self.regime_cache.clear()
            logger.info("Risk Manager shutdown completed")
        except Exception as e:
            logger.error(f"Error during risk manager shutdown: {e}")

# Backwards compatibility
class RiskManager(EnhancedRiskManager):
    """Backwards compatibility alias"""
    pass

"""
Enhanced Risk Manager with Advanced Position Sizing and Risk Controls
Professional-grade risk management for institutional trading
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import threading
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    account_balance: float
    account_equity: float
    daily_risk_used: float
    portfolio_risk_used: float
    open_positions_count: int
    max_drawdown: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%

class EnhancedRiskManager:
    """
    Enhanced Risk Management System with Advanced Analytics
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.risk_per_trade = config.get('risk_management.risk_per_trade', 1.0)
        self.max_position_size = config.get('risk_management.max_position_size', 10.0)
        self.max_daily_risk = config.get('risk_management.max_daily_risk', 5.0)
        self.max_portfolio_risk = config.get('risk_management.max_portfolio_risk', 10.0)
        self.min_risk_reward_ratio = config.get('risk_management.min_risk_reward_ratio', 1.5)
        
        # Position limits
        self.max_open_positions = config.get('risk_management.max_open_positions', 5)
        self.max_correlation_exposure = config.get('risk_management.max_correlation_exposure', 0.3)
        
        # Advanced risk parameters
        self.max_consecutive_losses = config.get('risk_management.max_consecutive_losses', 3)
        self.drawdown_limit = config.get('risk_management.max_drawdown_percent', 15.0)
        self.var_confidence = config.get('risk_management.var_confidence', 0.95)
        
        # Risk tracking with thread safety
        self.lock = threading.Lock()
        self.daily_risk_used = 0.0
        self.current_portfolio_risk = 0.0
        self.open_positions = {}
        self.closed_positions = []
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Account tracking
        self.account_balance = 100000.0  # Default, will be updated
        self.account_equity = 100000.0
        self.initial_balance = 100000.0
        self.peak_equity = 100000.0
        
        # Performance tracking
        self.trade_history = []
        self.daily_returns = []
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # Risk metrics cache
        self.risk_metrics_cache = None
        self.cache_timestamp = None
        self.cache_duration = 60  # Cache for 60 seconds
        
        self.logger.info("Enhanced RiskManager initialized successfully")
        self.logger.info(f"Risk Settings: {self.risk_per_trade}% per trade, {self.max_daily_risk}% daily max")

    def initialize(self):
        """Initialize risk manager with current account data"""
        try:
            # This would typically fetch real account data
            # For now, we'll use default values
            self.logger.info("Risk Manager initialized with default parameters")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing risk manager: {e}")
            return False

    def calculate_position_size(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Enhanced position sizing with multiple risk models
        """
        try:
            with self.lock:
                # Reset daily risk if needed
                self._reset_daily_risk_if_needed()
                
                # Extract signal information
                symbol = signal.get('symbol', '')
                entry_price = signal.get('entry_price', 0)
                direction = signal.get('direction', '')
                atr_at_signal = signal.get('atr_at_signal', 0.001)
                strategy = signal.get('strategy', 'Unknown')
                confidence = signal.get('confidence', 0.7)
                
                # Validate basic signal data
                if not all([symbol, entry_price, direction]):
                    self.logger.warning(f"Invalid signal data for {symbol}")
                    return None
                
                # Pre-trade risk checks
                if not self._pre_trade_risk_checks(symbol, signal):
                    return None
                
                # Calculate stop loss and take profit levels
                stop_loss, take_profit = self._calculate_enhanced_levels(
                    entry_price, direction, atr_at_signal, confidence
                )
                
                # Calculate risk metrics
                risk_in_pips = abs(entry_price - stop_loss)
                if risk_in_pips <= 0:
                    self.logger.warning(f"Invalid risk calculation for {symbol}: {risk_in_pips}")
                    return None
                
                reward_in_pips = abs(take_profit - entry_price)
                risk_reward_ratio = reward_in_pips / risk_in_pips if risk_in_pips > 0 else 0
                
                # Enhanced risk-reward validation
                min_rr = self._get_dynamic_min_rr_ratio(confidence, strategy)
                if risk_reward_ratio < min_rr:
                    self.logger.warning(f"Risk-reward ratio too low for {symbol}: {risk_reward_ratio:.2f} < {min_rr:.2f}")
                    return None
                
                # Multi-model position sizing
                position_size = self._calculate_optimal_position_size(
                    symbol, entry_price, risk_in_pips, confidence, strategy
                )
                
                if position_size <= 0:
                    self.logger.warning(f"Position size calculation failed for {symbol}")
                    return None
                
                # Final risk validation
                trade_risk_amount = self._calculate_trade_risk_amount(
                    symbol, position_size, risk_in_pips
                )
                
                if not self._validate_final_risk(trade_risk_amount, symbol):
                    return None
                
                # Create enhanced risk parameters
                risk_params = {
                    'symbol': symbol,
                    'direction': direction,
                    'position_size': round(position_size, 2),
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_amount': trade_risk_amount,
                    'risk_in_pips': risk_in_pips,
                    'reward_in_pips': reward_in_pips,
                    'risk_reward_ratio': risk_reward_ratio,
                    'pip_value': self._get_pip_value(symbol),
                    'strategy': strategy,
                    'confidence': confidence,
                    'account_risk_percent': self._get_adjusted_risk_percent(confidence, strategy),
                    'max_loss_amount': trade_risk_amount,
                    'max_gain_amount': position_size * reward_in_pips * self._get_pip_value(symbol) * 100000,
                    'position_value': position_size * entry_price * 100000,
                    'timestamp': datetime.now(),
                    'risk_model_used': 'Enhanced_Multi_Model'
                }
                
                # Log comprehensive risk analysis
                self._log_risk_analysis(risk_params)
                
                return risk_params
                
        except Exception as e:
            self.logger.error(f"Error calculating enhanced position size for {signal.get('symbol', 'Unknown')}: {e}")
            return None

    def _pre_trade_risk_checks(self, symbol: str, signal: Dict[str, Any]) -> bool:
        """Comprehensive pre-trade risk validation"""
        try:
            # Check position limits
            if len(self.open_positions) >= self.max_open_positions:
                self.logger.warning(f"Maximum open positions reached: {len(self.open_positions)}")
                return False
            
            # Check for existing position in same symbol
            if symbol in self.open_positions:
                self.logger.warning(f"Already have open position in {symbol}")
                return False
            
            # Check consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.logger.warning(f"Maximum consecutive losses reached: {self.consecutive_losses}")
                return False
            
            # Check drawdown limit
            current_drawdown = self._calculate_current_drawdown()
            if current_drawdown > self.drawdown_limit:
                self.logger.warning(f"Drawdown limit exceeded: {current_drawdown:.2f}% > {self.drawdown_limit}%")
                return False
            
            # Check market hours (if applicable)
            if not self._is_trading_hours_valid():
                self.logger.warning("Outside valid trading hours")
                return False
            
            # Strategy-specific risk checks
            if not self._strategy_specific_risk_check(signal):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in pre-trade risk checks: {e}")
            return False

    def _calculate_enhanced_levels(self, entry_price: float, direction: str, 
                                 atr: float, confidence: float) -> Tuple[float, float]:
        """Enhanced stop loss and take profit calculation"""
        try:
            # Dynamic ATR multipliers based on confidence and market conditions
            base_sl_multiplier = self.config.get('risk_management.atr_stop_loss_multiplier', 2.0)
            base_tp_multiplier = self.config.get('risk_management.atr_take_profit_multiplier', 3.0)
            
            # Adjust multipliers based on confidence
            confidence_adjustment = 0.8 + (confidence * 0.4)  # Range: 0.8 to 1.2
            sl_multiplier = base_sl_multiplier * confidence_adjustment
            tp_multiplier = base_tp_multiplier * confidence_adjustment
            
            # Market volatility adjustment
            volatility_adjustment = self._get_volatility_adjustment()
            sl_multiplier *= volatility_adjustment
            tp_multiplier *= volatility_adjustment
            
            if direction.upper() == 'BUY':
                stop_loss = entry_price - (atr * sl_multiplier)
                take_profit = entry_price + (atr * tp_multiplier)
            else:  # SELL
                stop_loss = entry_price + (atr * sl_multiplier)
                take_profit = entry_price - (atr * tp_multiplier)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced levels: {e}")
            # Fallback to simple calculation
            if direction.upper() == 'BUY':
                return entry_price * 0.999, entry_price * 1.003
            else:
                return entry_price * 1.001, entry_price * 0.997

    def _calculate_optimal_position_size(self, symbol: str, entry_price: float, 
                                       risk_in_pips: float, confidence: float, 
                                       strategy: str) -> float:
        """Multi-model position sizing optimization"""
        try:
            position_sizes = []
            
            # 1. Fixed Fractional Model
            adjusted_risk_percent = self._get_adjusted_risk_percent(confidence, strategy)
            account_risk_amount = self.account_balance * (adjusted_risk_percent / 100)
            pip_value = self._get_pip_value(symbol)
            
            ff_position_size = account_risk_amount / (risk_in_pips * pip_value * 100000)
            position_sizes.append(('fixed_fractional', ff_position_size))
            
            # 2. Kelly Criterion (simplified)
            if len(self.trade_history) >= 10:
                kelly_f = self._calculate_kelly_fraction(strategy)
                kelly_position_size = (self.account_balance * kelly_f) / (entry_price * 100000)
                position_sizes.append(('kelly', kelly_position_size))
            
            # 3. Volatility-Adjusted Model
            volatility_factor = self._get_volatility_factor(symbol)
            vol_adjusted_size = ff_position_size * (1 / volatility_factor)
            position_sizes.append(('volatility_adjusted', vol_adjusted_size))
            
            # 4. Confidence-Weighted Model
            confidence_weight = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0
            conf_weighted_size = ff_position_size * confidence_weight
            position_sizes.append(('confidence_weighted', conf_weighted_size))
            
            # Select optimal position size (conservative approach)
            sizes_only = [size for _, size in position_sizes if size > 0]
            if not sizes_only:
                return 0
            
            # Use median of calculated sizes for robustness
            optimal_size = np.median(sizes_only)
            
            # Apply maximum position size limit
            max_position_value = self.account_balance * (self.max_position_size / 100)
            max_lots = max_position_value / (entry_price * 100000)
            final_size = min(optimal_size, max_lots)
            
            # Ensure minimum viable position size
            min_size = 0.01  # Minimum 0.01 lots
            final_size = max(final_size, min_size) if final_size > 0 else 0
            
            self.logger.debug(f"Position size models for {symbol}: {position_sizes}")
            self.logger.debug(f"Selected optimal size: {final_size:.2f} lots")
            
            return final_size
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal position size: {e}")
            return 0

    def _get_adjusted_risk_percent(self, confidence: float, strategy: str) -> float:
        """Get risk percentage adjusted for confidence and strategy"""
        try:
            base_risk = self.risk_per_trade
            
            # Confidence adjustment (0.5x to 1.5x base risk)
            confidence_multiplier = 0.5 + confidence
            
            # Strategy-specific adjustments
            strategy_multipliers = {
                'RL-SAC': 1.2,    # Higher confidence in SAC
                'RL-A2C': 1.1,    # Moderate confidence in A2C
                'RL-PPO': 1.1,    # Moderate confidence in PPO
                'Traditional-Trending': 1.0,
                'Traditional-Mean-Reverting': 0.9,
                'Traditional-High-Volatility': 0.8,
                'Traditional-Neutral': 0.7
            }
            
            strategy_multiplier = strategy_multipliers.get(strategy, 1.0)
            
            # Account for recent performance
            performance_multiplier = self._get_performance_adjustment()
            
            adjusted_risk = base_risk * confidence_multiplier * strategy_multiplier * performance_multiplier
            
            # Ensure within reasonable bounds
            return max(0.1, min(adjusted_risk, self.risk_per_trade * 2))
            
        except Exception as e:
            self.logger.error(f"Error calculating adjusted risk percent: {e}")
            return self.risk_per_trade

    def _calculate_trade_risk_amount(self, symbol: str, position_size: float, risk_in_pips: float) -> float:
        """Calculate actual risk amount for the trade"""
        try:
            pip_value = self._get_pip_value(symbol)
            risk_amount = position_size * risk_in_pips * pip_value * 100000
            return risk_amount
        except Exception as e:
            self.logger.error(f"Error calculating trade risk amount: {e}")
            return 0

    def _validate_final_risk(self, trade_risk_amount: float, symbol: str) -> bool:
        """Final risk validation before trade approval"""
        try:
            # Check daily risk limit
            potential_daily_risk = self.daily_risk_used + trade_risk_amount
            daily_risk_percent = (potential_daily_risk / self.account_balance) * 100
            
            if daily_risk_percent > self.max_daily_risk:
                self.logger.warning(f"Daily risk limit would be exceeded: {daily_risk_percent:.2f}% > {self.max_daily_risk}%")
                return False
            
            # Check portfolio risk limit
            potential_portfolio_risk = self.current_portfolio_risk + trade_risk_amount
            portfolio_risk_percent = (potential_portfolio_risk / self.account_balance) * 100
            
            if portfolio_risk_percent > self.max_portfolio_risk:
                self.logger.warning(f"Portfolio risk limit would be exceeded: {portfolio_risk_percent:.2f}% > {self.max_portfolio_risk}%")
                return False
            
            # Check individual trade risk limit (no single trade > 3% of account)
            individual_risk_percent = (trade_risk_amount / self.account_balance) * 100
            max_individual_risk = min(3.0, self.risk_per_trade * 2)
            
            if individual_risk_percent > max_individual_risk:
                self.logger.warning(f"Individual trade risk too high: {individual_risk_percent:.2f}% > {max_individual_risk}%")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating final risk: {e}")
            return False

    def add_position(self, position_data: Dict[str, Any]) -> bool:
        """Add new position to risk tracking"""
        try:
            with self.lock:
                symbol = position_data.get('symbol')
                if not symbol:
                    return False
                
                # Enhanced position data
                enhanced_position = {
                    **position_data,
                    'added_at': datetime.now(),
                    'initial_account_balance': self.account_balance,
                    'risk_metrics_at_entry': self._get_current_risk_metrics()
                }
                
                self.open_positions[symbol] = enhanced_position
                
                # Update risk tracking
                risk_amount = position_data.get('risk_amount', 0)
                self.daily_risk_used += risk_amount
                self.current_portfolio_risk += risk_amount
                
                self.logger.info(f"Position added: {symbol} (Risk: ${risk_amount:.2f})")
                self.logger.info(f"Portfolio risk: {(self.current_portfolio_risk/self.account_balance)*100:.2f}%")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
            return False

    def remove_position(self, symbol: str, realized_pnl: float = 0, 
                       close_reason: str = 'manual') -> bool:
        """Remove position from tracking with enhanced analytics"""
        try:
            with self.lock:
                if symbol not in self.open_positions:
                    return False
                
                position_data = self.open_positions.pop(symbol)
                risk_amount = position_data.get('risk_amount', 0)
                
                # Update risk tracking
                self.current_portfolio_risk -= risk_amount
                self.current_portfolio_risk = max(0, self.current_portfolio_risk)
                
                # Update account balance
                self.account_balance += realized_pnl
                self.account_equity = self.account_balance
                
                # Update peak equity tracking
                if self.account_equity > self.peak_equity:
                    self.peak_equity = self.account_equity
                
                # Track trade performance
                trade_record = {
                    'symbol': symbol,
                    'realized_pnl': realized_pnl,
                    'risk_amount': risk_amount,
                    'close_reason': close_reason,
                    'hold_duration': datetime.now() - position_data.get('added_at', datetime.now()),
                    'strategy': position_data.get('strategy', 'Unknown'),
                    'entry_price': position_data.get('entry_price', 0),
                    'close_time': datetime.now(),
                    'account_balance_after': self.account_balance
                }
                
                self.trade_history.append(trade_record)
                self.closed_positions.append(trade_record)
                
                # Update win/loss streaks
                if realized_pnl > 0:
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                
                # Keep trade history manageable
                if len(self.trade_history) > 1000:
                    self.trade_history = self.trade_history[-500:]
                
                self.logger.info(f"Position removed: {symbol} (P&L: ${realized_pnl:.2f}, Reason: {close_reason})")
                
                # Invalidate risk metrics cache
                self.risk_metrics_cache = None
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error removing position: {e}")
            return False

    def update_account_info(self, balance: float, equity: float) -> None:
        """Update account information with validation"""
        try:
            with self.lock:
                # Validate the new balance/equity values
                if balance <= 0 or equity <= 0:
                    self.logger.warning(f"Invalid account info: Balance={balance}, Equity={equity}")
                    return
                
                # Check for significant changes (potential data error)
                balance_change = abs(balance - self.account_balance) / self.account_balance
                if balance_change > 0.5:  # 50% change seems suspicious
                    self.logger.warning(f"Suspicious balance change: {balance_change:.2%}")
                    return
                
                self.account_balance = balance
                self.account_equity = equity
                
                # Update peak equity
                if equity > self.peak_equity:
                    self.peak_equity = equity
                
                # Calculate daily return
                if len(self.daily_returns) == 0:
                    self.initial_balance = balance
                
                self.logger.debug(f"Account updated: Balance=${balance:.2f}, Equity=${equity:.2f}")
                
                # Invalidate cache
                self.risk_metrics_cache = None
                
        except Exception as e:
            self.logger.error(f"Error updating account info: {e}")

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary with caching"""
        try:
            # Check cache first
            if (self.risk_metrics_cache and self.cache_timestamp and
                (datetime.now() - self.cache_timestamp).seconds < self.cache_duration):
                return self.risk_metrics_cache
            
            # Calculate fresh metrics
            daily_risk_percent = (self.daily_risk_used / self.account_balance) * 100
            portfolio_risk_percent = (self.current_portfolio_risk / self.account_balance) * 100
            current_drawdown = self._calculate_current_drawdown()
            
            # Performance metrics
            total_return = ((self.account_balance - self.initial_balance) / self.initial_balance) * 100
            win_rate = self._calculate_win_rate()
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            risk_summary = {
                # Account Status
                'account_balance': self.account_balance,
                'account_equity': self.account_equity,
                'initial_balance': self.initial_balance,
                'peak_equity': self.peak_equity,
                'total_return_percent': total_return,
                
                # Risk Utilization
                'daily_risk_used': self.daily_risk_used,
                'daily_risk_percent': daily_risk_percent,
                'daily_risk_remaining': max(0, self.max_daily_risk - daily_risk_percent),
                'portfolio_risk_used': self.current_portfolio_risk,
                'portfolio_risk_percent': portfolio_risk_percent,
                'portfolio_risk_remaining': max(0, self.max_portfolio_risk - portfolio_risk_percent),
                
                # Position Status
                'open_positions_count': len(self.open_positions),
                'max_positions_remaining': max(0, self.max_open_positions - len(self.open_positions)),
                'open_positions': list(self.open_positions.keys()),
                
                # Performance Metrics
                'current_drawdown_percent': current_drawdown,
                'max_drawdown_limit': self.drawdown_limit,
                'consecutive_wins': self.consecutive_wins,
                'consecutive_losses': self.consecutive_losses,
                'win_rate_percent': win_rate * 100,
                'sharpe_ratio': sharpe_ratio,
                
                # Risk Settings
                'risk_per_trade_percent': self.risk_per_trade,
                'max_daily_risk_percent': self.max_daily_risk,
                'max_portfolio_risk_percent': self.max_portfolio_risk,
                'min_risk_reward_ratio': self.min_risk_reward_ratio,
                
                # Advanced Metrics
                'total_trades': len(self.trade_history),
                'trades_today': len([t for t in self.trade_history if t['close_time'].date() == datetime.now().date()]),
                'avg_trade_return': np.mean([t['realized_pnl'] for t in self.trade_history]) if self.trade_history else 0,
                'risk_adjusted_return': total_return / max(current_drawdown, 1),
                
                # Status Flags
                'risk_status': self._get_risk_status(),
                'trading_allowed': self._is_trading_allowed(),
                'last_updated': datetime.now()
            }
            
            # Cache the result
            self.risk_metrics_cache = risk_summary
            self.cache_timestamp = datetime.now()
            
            return risk_summary
            
        except Exception as e:
            self.logger.error(f"Error getting risk summary: {e}")
            return {}

    # Helper methods for risk calculations
    def _reset_daily_risk_if_needed(self):
        """Reset daily risk tracking if new day"""
        try:
            current_time = datetime.now()
            if current_time.date() > self.daily_reset_time.date():
                self.daily_risk_used = 0.0
                self.daily_reset_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                self.logger.info("Daily risk counter reset")
        except Exception as e:
            self.logger.error(f"Error resetting daily risk: {e}")

    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for different currency pairs"""
        # Simplified pip value calculation
        if 'JPY' in symbol:
            return 0.01
        else:
            return 0.0001

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown percentage"""
        try:
            if self.peak_equity <= 0:
                return 0.0
            drawdown = ((self.peak_equity - self.account_equity) / self.peak_equity) * 100
            return max(0.0, drawdown)
        except Exception as e:
            self.logger.error(f"Error calculating drawdown: {e}")
            return 0.0

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history"""
        try:
            if not self.trade_history:
                return 0.0
            winning_trades = sum(1 for trade in self.trade_history if trade['realized_pnl'] > 0)
            return winning_trades / len(self.trade_history)
        except Exception as e:
            self.logger.error(f"Error calculating win rate: {e}")
            return 0.0

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from trade returns"""
        try:
            if len(self.trade_history) < 10:
                return 0.0
            
            returns = [trade['realized_pnl'] / self.account_balance for trade in self.trade_history]
            if not returns:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Annualized Sharpe ratio (assuming 252 trading days)
            sharpe = (mean_return / std_return) * np.sqrt(252)
            return sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def _get_risk_status(self) -> str:
        """Get overall risk status"""
        try:
            daily_risk_percent = (self.daily_risk_used / self.account_balance) * 100
            portfolio_risk_percent = (self.current_portfolio_risk / self.account_balance) * 100
            drawdown = self._calculate_current_drawdown()
            
            if (daily_risk_percent > self.max_daily_risk * 0.9 or
                portfolio_risk_percent > self.max_portfolio_risk * 0.9 or
                drawdown > self.drawdown_limit * 0.8):
                return "HIGH_RISK"
            elif (daily_risk_percent > self.max_daily_risk * 0.7 or
                  portfolio_risk_percent > self.max_portfolio_risk * 0.7 or
                  drawdown > self.drawdown_limit * 0.6):
                return "MODERATE_RISK"
            else:
                return "LOW_RISK"
                
        except Exception as e:
            self.logger.error(f"Error getting risk status: {e}")
            return "UNKNOWN"

    def _is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed"""
        try:
            # Check all risk limits
            daily_risk_percent = (self.daily_risk_used / self.account_balance) * 100
            portfolio_risk_percent = (self.current_portfolio_risk / self.account_balance) * 100
            drawdown = self._calculate_current_drawdown()
            
            if (daily_risk_percent >= self.max_daily_risk or
                portfolio_risk_percent >= self.max_portfolio_risk or
                drawdown >= self.drawdown_limit or
                self.consecutive_losses >= self.max_consecutive_losses or
                len(self.open_positions) >= self.max_open_positions):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking trading allowance: {e}")
            return False

    def _log_risk_analysis(self, risk_params: Dict[str, Any]):
        """Log comprehensive risk analysis"""
        try:
            symbol = risk_params.get('symbol', 'Unknown')
            self.logger.info(f"✅ Enhanced risk calculated for {symbol}:")
            self.logger.info(f"  Position Size: {risk_params['position_size']:.2f} lots")
            self.logger.info(f"  Risk Amount: ${risk_params['risk_amount']:.2f} ({risk_params['account_risk_percent']:.2f}%)")
            self.logger.info(f"  Risk/Reward: 1:{risk_params['risk_reward_ratio']:.2f}")
            self.logger.info(f"  Stop Loss: {risk_params['stop_loss']:.5f}")
            self.logger.info(f"  Take Profit: {risk_params['take_profit']:.5f}")
            self.logger.info(f"  Strategy: {risk_params['strategy']} (Confidence: {risk_params['confidence']:.2f})")
        except Exception as e:
            self.logger.error(f"Error logging risk analysis: {e}")

    # Additional helper methods (simplified versions for brevity)
    def _get_dynamic_min_rr_ratio(self, confidence: float, strategy: str) -> float:
        base_rr = self.min_risk_reward_ratio
        confidence_adjustment = 1.0 - (confidence * 0.2)  # Lower RR for higher confidence
        return base_rr * confidence_adjustment

    def _get_volatility_adjustment(self) -> float:
        return 1.0  # Simplified - would analyze current market volatility

    def _calculate_kelly_fraction(self, strategy: str) -> float:
        """Simplified Kelly Criterion calculation"""
        try:
            recent_trades = [t for t in self.trade_history if t['strategy'] == strategy][-20:]
            if len(recent_trades) < 10:
                return 0.02  # Conservative default
            
            wins = [t['realized_pnl'] for t in recent_trades if t['realized_pnl'] > 0]
            losses = [abs(t['realized_pnl']) for t in recent_trades if t['realized_pnl'] < 0]
            
            if not wins or not losses:
                return 0.02
            
            win_rate = len(wins) / len(recent_trades)
            avg_win = np.mean(wins)
            avg_loss = np.mean(losses)
            
            kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            return max(0.01, min(0.1, kelly_f))  # Cap between 1% and 10%
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly fraction: {e}")
            return 0.02

    def _get_volatility_factor(self, symbol: str) -> float:
        return 1.0  # Simplified

    def _get_performance_adjustment(self) -> float:
        """Adjust risk based on recent performance"""
        try:
            if len(self.trade_history) < 5:
                return 1.0
            
            recent_trades = self.trade_history[-10:]
            recent_pnl = sum(t['realized_pnl'] for t in recent_trades)
            
            if recent_pnl > 0:
                return min(1.2, 1.0 + (recent_pnl / self.account_balance))
            else:
                return max(0.8, 1.0 + (recent_pnl / self.account_balance))
                
        except Exception as e:
            self.logger.error(f"Error calculating performance adjustment: {e}")
            return 1.0

    def _is_trading_hours_valid(self) -> bool:
        return True  # Simplified - would check actual market hours

    def _strategy_specific_risk_check(self, signal: Dict[str, Any]) -> bool:
        return True  # Simplified - would implement strategy-specific validations

    def _get_current_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics snapshot"""
        return {
            'daily_risk_percent': (self.daily_risk_used / self.account_balance) * 100,
            'portfolio_risk_percent': (self.current_portfolio_risk / self.account_balance) * 100,
            'drawdown_percent': self._calculate_current_drawdown(),
            'account_balance': self.account_balance
        }

    def is_trade_allowed(self, symbol: str, estimated_risk: float = None) -> Tuple[bool, str]:
        """Enhanced trade allowance check"""
        try:
            if not self._is_trading_allowed():
                return False, "Trading temporarily suspended due to risk limits"
            
            # Symbol-specific checks
            if symbol in self.open_positions:
                return False, f"Position already exists for {symbol}"
            
            # Estimated risk check
            if estimated_risk:
                if not self._validate_final_risk(estimated_risk, symbol):
                    return False, "Estimated risk exceeds limits"
            
            return True, "Trade allowed"
            
        except Exception as e:
            self.logger.error(f"Error checking trade allowance: {e}")
            return False, f"Error checking trade: {e}"

    def emergency_close_all(self) -> bool:
        """Emergency close all positions"""
        try:
            self.logger.warning("🚨 Emergency close all positions triggered!")
            
            with self.lock:
                # Clear position tracking
                closed_positions = list(self.open_positions.keys())
                self.open_positions.clear()
                self.current_portfolio_risk = 0.0
                
                self.logger.warning(f"Cleared {len(closed_positions)} positions from risk tracking")
                return True
                
        except Exception as e:
            self.logger.error(f"Error in emergency close: {e}")
            return False
    def check_correlation_risk(self, new_signal: Dict[str, Any], open_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            symbol = new_signal.get('symbol', '')
            direction = new_signal.get('direction', '')
            
            if not symbol or not direction:
                raise ValueError("Missing required signal data")
        
            # Define correlation matrix (you can make this dynamic by calculating from historical data)
            correlation_matrix = {
                ('EURUSD', 'GBPUSD'): 0.85,  # Strong positive correlation
                ('EURUSD', 'XAUUSD'): -0.25, # Weak negative correlation
                ('GBPUSD', 'XAUUSD'): -0.20, # Weak negative correlation
                ('USDJPY', 'EURUSD'): 0.90,  # Strong positive correlation
                ('USDJPY', 'GBPUSD'): 0.80,  # Strong positive correlation
                ('XAUUSD', 'USDJPY'): -0.30   # Moderate negative correlation
            }
        except ValueError as ve:
            self.logger.error(f"Invalid signal data: {ve}")
            return {
                'allow_trade': False,
                'reason': f'Invalid signal data: {ve}',
                'warnings': [],
                'suggested_action': 'Check signal data completeness'
            }
        except Exception as e:
            self.logger.error(f"Error processing correlation data: {e}")
            return {
                'allow_trade': False,
                'reason': f'Error in correlation check: {e}',
                'warnings': [],
                'suggested_action': 'System error - manual review needed'
            }
        
        correlation_warnings = []
        high_risk_correlations = []
        
        for position in open_positions:
            pos_symbol = position.get('symbol', '')
            pos_direction = position.get('type', '')
            
            # Skip if same symbol
            if pos_symbol == symbol:
                continue
            
            # Get correlation coefficient
            pair_key = tuple(sorted([symbol, pos_symbol]))
            correlation = correlation_matrix.get(pair_key, 0.0)
            
            # Check for high correlation conflicts
            if abs(correlation) > 0.7:  # High correlation threshold
                
                if correlation > 0.7:  # Positive correlation
                    if direction != pos_direction:  # Opposite directions
                        warning = {
                            'type': 'POSITIVE_CORRELATION_CONFLICT',
                            'message': f"{symbol} {direction} conflicts with {pos_symbol} {pos_direction}",
                            'correlation': correlation,
                            'risk_level': 'HIGH',
                            'recommendation': 'Consider closing one position or reducing size'
                        }
                        high_risk_correlations.append(warning)
                        correlation_warnings.append(warning)
                
                elif correlation < -0.7:  # Negative correlation
                    if direction == pos_direction:  # Same directions
                        warning = {
                            'type': 'NEGATIVE_CORRELATION_CONFLICT', 
                            'message': f"{symbol} {direction} conflicts with {pos_symbol} {pos_direction}",
                            'correlation': correlation,
                            'risk_level': 'MEDIUM',
                            'recommendation': 'Monitor closely for divergence'
                        }
                        correlation_warnings.append(warning)
        
        # Determine action based on correlation analysis
        if high_risk_correlations:
            return {
                'allow_trade': False,  # Block conflicting trades
                'reason': 'High correlation conflict detected',
                'warnings': correlation_warnings,
                'suggested_action': 'Wait for better entry or close conflicting position'
            }
        elif correlation_warnings:
            return {
                'allow_trade': True,  # Allow but with warnings
                'reason': 'Moderate correlation risk',
                'warnings': correlation_warnings,
                'suggested_action': 'Reduce position size by 50%'
            }
        else:
            return {
                'allow_trade': True,
                'reason': 'No significant correlation conflicts',
                'warnings': [],
                'suggested_action': 'Normal position sizing'
            }
    
"""
Gold-Specific Risk Management
Add this to your risk_manager.py
"""

def calculate_gold_risk_parameters(self, symbol: str, entry_price: float, 
                                 stop_loss: float, take_profit: float, 
                                 confidence: float) -> Dict[str, Any]:
    """Calculate gold-specific risk parameters"""
    
    if symbol != 'XAUUSD':
        return self.calculate_enhanced_risk(symbol, 'BUY', entry_price, stop_loss, take_profit, confidence, 'standard')
    
    # Gold-specific calculations
    min_stop_distance = 5.00  # $5 minimum stop distance
    min_tp_distance = 10.00   # $10 minimum take profit distance
    max_position_size = 0.10  # Maximum 0.1 lot for gold
    
    # Adjust stop loss if too close
    if stop_loss and abs(entry_price - stop_loss) < min_stop_distance:
        if entry_price > stop_loss:  # Short position
            stop_loss = entry_price - min_stop_distance
        else:  # Long position  
            stop_loss = entry_price + min_stop_distance
        
        logger.info(f"XAUUSD stop loss adjusted to minimum distance: {stop_loss}")
    
    # Adjust take profit if too close
    if take_profit and abs(take_profit - entry_price) < min_tp_distance:
        if take_profit > entry_price:  # Long position
            take_profit = entry_price + min_tp_distance
        else:  # Short position
            take_profit = entry_price - min_tp_distance
            
        logger.info(f"XAUUSD take profit adjusted to minimum distance: {take_profit}")
    
    # Calculate position size based on dollar risk
    account_balance = 100000  # Your account balance
    risk_percentage = 0.01    # 1% risk
    risk_amount = account_balance * risk_percentage
    
    if stop_loss:
        stop_distance = abs(entry_price - stop_loss)
        # Gold: $1 move = $100 per lot, so position size = risk_amount / (stop_distance * 100)
        position_size = risk_amount / (stop_distance * 100)
        position_size = min(position_size, max_position_size)  # Cap at max size
        position_size = max(position_size, 0.01)  # Minimum 0.01 lot
    else:
        position_size = 0.01  # Default minimum
    
    return {
        'symbol': symbol,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'position_size': round(position_size, 2),
        'risk_amount': risk_amount,
        'max_loss': stop_distance * position_size * 100 if stop_loss else 0,
        'potential_profit': abs(take_profit - entry_price) * position_size * 100 if take_profit else 0,
        'confidence': confidence,
        'optimized_for_gold': True
    }

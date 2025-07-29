"""
Performance Monitor for Trading Bot Analytics
Professional-grade performance tracking and analysis
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import threading
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

@dataclass
class TradeMetrics:
    """Trade performance metrics data structure"""
    symbol: str
    strategy: str
    direction: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    profit_loss: float
    return_percent: float
    hold_duration_minutes: float
    risk_amount: float
    reward_amount: float
    confidence: float
    slippage_pips: float
    execution_time_seconds: float

@dataclass
class PerformanceSnapshot:
    """Performance snapshot data structure"""
    timestamp: datetime
    account_balance: float
    account_equity: float
    total_return_percent: float
    daily_return_percent: float
    max_drawdown_percent: float
    win_rate_percent: float
    profit_factor: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int

class PerformanceMonitor:
    """
    Professional performance monitoring and analytics system
    """
    
    def __init__(self, config):
        self.config = config  
        self.logger = logging.getLogger(__name__)
        
        # Threading for thread-safe operations
        self.lock = threading.Lock()
        
        # Performance tracking settings
        self.update_interval = config.get('performance.update_interval', 300)  # 5 minutes
        self.metrics_retention_days = config.get('performance.metrics_retention_days', 30)
        self.enable_detailed_logging = config.get('performance.detailed_logging', True)
        
        # Core performance data
        self.trade_records = []
        self.signal_records = []
        self.execution_records = []
        self.performance_snapshots = []
        self.daily_returns = deque(maxlen=252)  # Keep 1 year of daily returns
        
        # Real-time metrics
        self.current_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'peak_equity': 100000.0,  # Default starting equity
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'last_updated': datetime.now()
        }
        
        # Strategy-specific performance tracking
        self.strategy_performance = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        })
        
        # Symbol-specific performance tracking
        self.symbol_performance = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0
        })
        
        # Time-based performance tracking
        self.hourly_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0.0})
        self.daily_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0.0})
        self.weekly_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0.0})
        self.monthly_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0.0})
        
        # Advanced analytics
        self.risk_metrics = {
            'var_95': 0.0,  # Value at Risk 95%
            'var_99': 0.0,  # Value at Risk 99%
            'expected_shortfall': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'current_streak': 0,
            'streak_type': 'none'  # 'win', 'loss', 'none'
        }
        
        # Performance alerts
        self.alert_thresholds = {
            'max_drawdown_percent': config.get('performance.max_drawdown_alert', 10.0),
            'consecutive_losses': config.get('performance.max_consecutive_losses_alert', 5),
            'daily_loss_percent': config.get('performance.daily_loss_alert', 3.0)
        }
        
        # File storage for persistence
        self.data_file = Path('logs/performance_data.json')
        self.snapshots_file = Path('logs/performance_snapshots.json')
        
        # Load historical data if exists
        self._load_historical_data()
        
        self.logger.info("Performance Monitor initialized successfully")
        self.logger.info(f"Monitoring enabled: Update interval {self.update_interval}s, Retention {self.metrics_retention_days} days")

    def record_signal(self, signal: Dict[str, Any], risk_params: Dict[str, Any], 
                     execution_result: Dict[str, Any]) -> None:
        """Record signal generation and execution for performance analysis"""
        try:
            with self.lock:
                # Create signal record
                signal_record = {
                    'timestamp': datetime.now(),
                    'symbol': signal.get('symbol', 'Unknown'),
                    'strategy': signal.get('strategy', 'Unknown'),
                    'direction': signal.get('direction', 'Unknown'),
                    'confidence': signal.get('confidence', 0.0),
                    'entry_price': signal.get('entry_price', 0.0),
                    'risk_amount': risk_params.get('risk_amount', 0.0),
                    'expected_profit': risk_params.get('max_gain_amount', 0.0),
                    'executed': execution_result.get('success', False),
                    'execution_time': execution_result.get('execution_time_seconds', 0.0),
                    'slippage_pips': execution_result.get('slippage_pips', 0.0),
                    'ticket': execution_result.get('ticket', None)
                }
                
                self.signal_records.append(signal_record)
                
                # Keep records manageable
                if len(self.signal_records) > 10000:
                    self.signal_records = self.signal_records[-5000:]
                
                # Update execution statistics
                if execution_result.get('success', False):
                    self._update_execution_stats(signal_record)
                
                self.logger.debug(f"Signal recorded: {signal.get('strategy', 'Unknown')} "
                                f"{signal.get('direction', 'Unknown')} {signal.get('symbol', 'Unknown')}")
                
        except Exception as e:
            self.logger.error(f"Error recording signal: {e}")

    def record_trade_closure(self, trade_data: Dict[str, Any]) -> None:
        """Record trade closure for comprehensive performance analysis"""
        try:
            with self.lock:
                # Create trade metrics
                entry_time = trade_data.get('entry_time', datetime.now())
                exit_time = trade_data.get('exit_time', datetime.now())
                hold_duration = (exit_time - entry_time).total_seconds() / 60  # minutes
                
                profit_loss = trade_data.get('profit_loss', 0.0)
                risk_amount = trade_data.get('risk_amount', 1.0)  # Avoid division by zero
                return_percent = (profit_loss / risk_amount) * 100 if risk_amount > 0 else 0.0
                
                trade_metrics = TradeMetrics(
                    symbol=trade_data.get('symbol', 'Unknown'),
                    strategy=trade_data.get('strategy', 'Unknown'),
                    direction=trade_data.get('direction', 'Unknown'),
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=trade_data.get('entry_price', 0.0),
                    exit_price=trade_data.get('exit_price', 0.0),
                    position_size=trade_data.get('position_size', 0.0),
                    profit_loss=profit_loss,
                    return_percent=return_percent,
                    hold_duration_minutes=hold_duration,
                    risk_amount=risk_amount,
                    reward_amount=trade_data.get('reward_amount', 0.0),
                    confidence=trade_data.get('confidence', 0.0),
                    slippage_pips=trade_data.get('slippage_pips', 0.0),
                    execution_time_seconds=trade_data.get('execution_time_seconds', 0.0)
                )
                
                self.trade_records.append(trade_metrics)
                
                # Update all performance metrics
                self._update_trade_metrics(trade_metrics)
                self._update_strategy_performance(trade_metrics)
                self._update_symbol_performance(trade_metrics)
                self._update_time_based_performance(trade_metrics)
                self._update_risk_metrics(trade_metrics)
                self._check_performance_alerts(trade_metrics)
                
                # Keep records manageable
                if len(self.trade_records) > 10000:
                    self.trade_records = self.trade_records[-5000:]
                
                self.logger.info(f"Trade closed: {trade_metrics.symbol} {trade_metrics.direction} "
                               f"P&L: ${profit_loss:.2f} ({return_percent:.2f}%)")
                
        except Exception as e:
            self.logger.error(f"Error recording trade closure: {e}")

    def _update_execution_stats(self, signal_record: Dict[str, Any]) -> None:
        """Update execution statistics"""
        try:
            execution_record = {
                'timestamp': signal_record['timestamp'],
                'strategy': signal_record['strategy'],
                'execution_time': signal_record['execution_time'],
                'slippage_pips': signal_record['slippage_pips'],
                'success': signal_record['executed']
            }
            
            self.execution_records.append(execution_record)
            
            # Keep records manageable
            if len(self.execution_records) > 1000:
                self.execution_records = self.execution_records[-500:]
                
        except Exception as e:
            self.logger.error(f"Error updating execution stats: {e}")

    def _update_trade_metrics(self, trade: TradeMetrics) -> None:
        """Update core trade metrics"""
        try:
            self.current_metrics['total_trades'] += 1
            
            if trade.profit_loss > 0:
                self.current_metrics['winning_trades'] += 1
                self.current_metrics['total_profit'] += trade.profit_loss
                self._update_win_streak()
            else:
                self.current_metrics['losing_trades'] += 1
                self.current_metrics['total_loss'] += abs(trade.profit_loss)
                self._update_loss_streak()
            
            # Update win rate
            total_trades = self.current_metrics['total_trades']
            if total_trades > 0:
                self.current_metrics['win_rate'] = (self.current_metrics['winning_trades'] / total_trades) * 100
            
            # Update profit factor
            total_loss = self.current_metrics['total_loss']
            if total_loss > 0:
                self.current_metrics['profit_factor'] = self.current_metrics['total_profit'] / total_loss
            
            # Update drawdown metrics
            self._update_drawdown_metrics(trade.profit_loss)
            
            # Add to daily returns for Sharpe ratio calculation
            self.daily_returns.append(trade.return_percent / 100)
            
            # Update Sharpe ratio
            self._update_sharpe_ratio()
            
            self.current_metrics['last_updated'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating trade metrics: {e}")

    def _update_strategy_performance(self, trade: TradeMetrics) -> None:
        """Update strategy-specific performance metrics"""
        try:
            strategy = trade.strategy
            perf = self.strategy_performance[strategy]
            
            perf['trades'] += 1
            perf['total_pnl'] += trade.profit_loss
            
            if trade.profit_loss > 0:
                perf['wins'] += 1
                current_wins = [t.profit_loss for t in self.trade_records 
                               if t.strategy == strategy and t.profit_loss > 0]
                perf['avg_win'] = np.mean(current_wins) if current_wins else 0
            else:
                perf['losses'] += 1
                current_losses = [abs(t.profit_loss) for t in self.trade_records 
                                 if t.strategy == strategy and t.profit_loss < 0]
                perf['avg_loss'] = np.mean(current_losses) if current_losses else 0
            
            # Update win rate
            if perf['trades'] > 0:
                perf['win_rate'] = (perf['wins'] / perf['trades']) * 100
            
            # Update profit factor
            if perf['avg_loss'] > 0:
                perf['profit_factor'] = perf['avg_win'] / perf['avg_loss']
                
        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {e}")

    def _update_symbol_performance(self, trade: TradeMetrics) -> None:
        """Update symbol-specific performance metrics"""
        try:
            symbol = trade.symbol
            perf = self.symbol_performance[symbol]
            
            perf['trades'] += 1
            perf['total_pnl'] += trade.profit_loss
            
            if trade.profit_loss > 0:
                perf['wins'] += 1
            else:
                perf['losses'] += 1
            
            # Update win rate
            if perf['trades'] > 0:
                perf['win_rate'] = (perf['wins'] / perf['trades']) * 100
                
        except Exception as e:
            self.logger.error(f"Error updating symbol performance: {e}")

    def _update_time_based_performance(self, trade: TradeMetrics) -> None:
        """Update time-based performance analytics"""
        try:
            trade_time = trade.exit_time or datetime.now()
            
            # Hourly performance
            hour_key = trade_time.hour
            self.hourly_performance[hour_key]['trades'] += 1
            self.hourly_performance[hour_key]['pnl'] += trade.profit_loss
            
            # Daily performance
            day_key = trade_time.date()
            self.daily_performance[day_key]['trades'] += 1
            self.daily_performance[day_key]['pnl'] += trade.profit_loss
            
            # Weekly performance
            week_key = f"{trade_time.year}-W{trade_time.isocalendar()[1]}"
            self.weekly_performance[week_key]['trades'] += 1
            self.weekly_performance[week_key]['pnl'] += trade.profit_loss
            
            # Monthly performance
            month_key = f"{trade_time.year}-{trade_time.month:02d}"
            self.monthly_performance[month_key]['trades'] += 1
            self.monthly_performance[month_key]['pnl'] += trade.profit_loss
            
        except Exception as e:
            self.logger.error(f"Error updating time-based performance: {e}")

    def _update_risk_metrics(self, trade: TradeMetrics) -> None:
        """Update advanced risk metrics"""
        try:
            # Update VaR calculations if we have enough data
            if len(self.trade_records) >= 30:
                returns = [t.return_percent for t in self.trade_records[-100:]]  # Last 100 trades
                if returns:
                    self.risk_metrics['var_95'] = np.percentile(returns, 5)
                    self.risk_metrics['var_99'] = np.percentile(returns, 1)
                    
                    # Expected shortfall (average of worst 5% returns)
                    worst_5_percent = [r for r in returns if r <= self.risk_metrics['var_95']]
                    if worst_5_percent:
                        self.risk_metrics['expected_shortfall'] = np.mean(worst_5_percent)
            
            # Update Sortino ratio (similar to Sharpe but using downside deviation)
            if len(self.daily_returns) >= 10:
                returns_array = np.array(list(self.daily_returns))
                negative_returns = returns_array[returns_array < 0]
                
                if len(negative_returns) > 0:
                    downside_deviation = np.std(negative_returns)
                    if downside_deviation > 0:
                        mean_return = np.mean(returns_array)
                        self.risk_metrics['sortino_ratio'] = mean_return / downside_deviation * np.sqrt(252)
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")

    def _update_drawdown_metrics(self, profit_loss: float) -> None:
        """Update drawdown tracking"""
        try:
            # Update peak equity
            current_equity = self.current_metrics.get('peak_equity', 100000) + profit_loss
            
            if current_equity > self.current_metrics['peak_equity']:
                self.current_metrics['peak_equity'] = current_equity
            
            # Calculate current drawdown
            current_drawdown = ((self.current_metrics['peak_equity'] - current_equity) / 
                              self.current_metrics['peak_equity']) * 100
            
            self.current_metrics['current_drawdown'] = current_drawdown
            
            # Update max drawdown
            if current_drawdown > self.current_metrics['max_drawdown']:
                self.current_metrics['max_drawdown'] = current_drawdown
                
        except Exception as e:
            self.logger.error(f"Error updating drawdown metrics: {e}")

    def _update_win_streak(self) -> None:
        """Update winning streak tracking"""
        try:
            if self.risk_metrics['streak_type'] == 'win':
                self.risk_metrics['current_streak'] += 1
            else:
                self.risk_metrics['streak_type'] = 'win'
                self.risk_metrics['current_streak'] = 1
            
            if self.risk_metrics['current_streak'] > self.risk_metrics['max_consecutive_wins']:
                self.risk_metrics['max_consecutive_wins'] = self.risk_metrics['current_streak']
                
        except Exception as e:
            self.logger.error(f"Error updating win streak: {e}")

    def _update_loss_streak(self) -> None:
        """Update losing streak tracking"""
        try:
            if self.risk_metrics['streak_type'] == 'loss':
                self.risk_metrics['current_streak'] += 1
            else:
                self.risk_metrics['streak_type'] = 'loss'
                self.risk_metrics['current_streak'] = 1
            
            if self.risk_metrics['current_streak'] > self.risk_metrics['max_consecutive_losses']:
                self.risk_metrics['max_consecutive_losses'] = self.risk_metrics['current_streak']
                
        except Exception as e:
            self.logger.error(f"Error updating loss streak: {e}")

    def _update_sharpe_ratio(self) -> None:
        """Update Sharpe ratio calculation"""
        try:
            if len(self.daily_returns) >= 10:
                returns_array = np.array(list(self.daily_returns))
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                
                if std_return > 0:
                    # Annualized Sharpe ratio (assuming 252 trading days)
                    self.current_metrics['sharpe_ratio'] = (mean_return / std_return) * np.sqrt(252)
                    
        except Exception as e:
            self.logger.error(f"Error updating Sharpe ratio: {e}")

    def _check_performance_alerts(self, trade: TradeMetrics) -> None:
        """Check for performance alerts and warnings"""
        try:
            alerts = []
            
            # Check drawdown alert
            current_drawdown = self.current_metrics.get('current_drawdown', 0)
            if current_drawdown > self.alert_thresholds['max_drawdown_percent']:
                alerts.append(f"High drawdown alert: {current_drawdown:.2f}%")
            
            # Check consecutive losses
            if (self.risk_metrics['streak_type'] == 'loss' and 
                self.risk_metrics['current_streak'] >= self.alert_thresholds['consecutive_losses']):
                alerts.append(f"Consecutive losses alert: {self.risk_metrics['current_streak']} losses")
            
            # Check daily loss
            today = datetime.now().date()
            daily_pnl = self.daily_performance.get(today, {}).get('pnl', 0)
            if daily_pnl < 0:
                daily_loss_percent = abs(daily_pnl) / self.current_metrics.get('peak_equity', 100000) * 100
                if daily_loss_percent > self.alert_thresholds['daily_loss_percent']:
                    alerts.append(f"Daily loss alert: -{daily_loss_percent:.2f}%")
            
            # Log alerts
            for alert in alerts:
                self.logger.warning(f"⚠️ Performance Alert: {alert}")
                
        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}")

    def update(self) -> None:
        """Update performance metrics and create snapshots"""
        try:
            with self.lock:
                # Create performance snapshot
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.now(),
                    account_balance=self.current_metrics.get('peak_equity', 100000),
                    account_equity=self.current_metrics.get('peak_equity', 100000),
                    total_return_percent=self._calculate_total_return(),
                    daily_return_percent=self._calculate_daily_return(),
                    max_drawdown_percent=self.current_metrics.get('max_drawdown', 0),
                    win_rate_percent=self.current_metrics.get('win_rate', 0),
                    profit_factor=self.current_metrics.get('profit_factor', 0),
                    sharpe_ratio=self.current_metrics.get('sharpe_ratio', 0),
                    total_trades=self.current_metrics.get('total_trades', 0),
                    winning_trades=self.current_metrics.get('winning_trades', 0),
                    losing_trades=self.current_metrics.get('losing_trades', 0)
                )
                
                self.performance_snapshots.append(snapshot)
                
                # Keep snapshots manageable
                if len(self.performance_snapshots) > 10000:
                    self.performance_snapshots = self.performance_snapshots[-5000:]
                
                # Save to file periodically
                if len(self.performance_snapshots) % 10 == 0:
                    self._save_performance_data()
                
                self.logger.debug("Performance metrics updated")
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    def _calculate_total_return(self) -> float:
        """Calculate total return percentage"""
        try:
            total_pnl = self.current_metrics.get('total_profit', 0) - self.current_metrics.get('total_loss', 0)
            initial_balance = 100000.0  # This should come from config or be tracked
            return (total_pnl / initial_balance) * 100
        except Exception as e:
            self.logger.error(f"Error calculating total return: {e}")
            return 0.0

    def _calculate_daily_return(self) -> float:
        """Calculate daily return percentage"""
        try:
            today = datetime.now().date()
            daily_pnl = self.daily_performance.get(today, {}).get('pnl', 0)
            current_balance = self.current_metrics.get('peak_equity', 100000)
            return (daily_pnl / current_balance) * 100
        except Exception as e:
            self.logger.error(f"Error calculating daily return: {e}")
            return 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            with self.lock:
                # Calculate additional metrics
                avg_win = 0
                avg_loss = 0
                
                if self.current_metrics['winning_trades'] > 0:
                    wins = [t.profit_loss for t in self.trade_records if t.profit_loss > 0]
                    avg_win = np.mean(wins) if wins else 0
                
                if self.current_metrics['losing_trades'] > 0:
                    losses = [abs(t.profit_loss) for t in self.trade_records if t.profit_loss < 0]
                    avg_loss = np.mean(losses) if losses else 0
                
                # Execution statistics
                execution_stats = self._get_execution_statistics()
                
                # Top performing strategies
                top_strategies = self._get_top_strategies(limit=5)
                
                # Recent performance trend
                recent_trend = self._get_recent_performance_trend()
                
                return {
                    # Core metrics
                    'total_trades': self.current_metrics.get('total_trades', 0),
                    'winning_trades': self.current_metrics.get('winning_trades', 0),
                    'losing_trades': self.current_metrics.get('losing_trades', 0),
                    'win_rate_percent': self.current_metrics.get('win_rate', 0),
                    'profit_factor': self.current_metrics.get('profit_factor', 0),
                    'sharpe_ratio': self.current_metrics.get('sharpe_ratio', 0),
                    
                    # Financial metrics
                    'total_profit': self.current_metrics.get('total_profit', 0),
                    'total_loss': self.current_metrics.get('total_loss', 0),
                    'net_profit': self.current_metrics.get('total_profit', 0) - self.current_metrics.get('total_loss', 0),
                    'average_win': avg_win,
                    'average_loss': avg_loss,
                    'total_return_percent': self._calculate_total_return(),
                    'daily_return_percent': self._calculate_daily_return(),
                    
                    # Risk metrics
                    'max_drawdown_percent': self.current_metrics.get('max_drawdown', 0),
                    'current_drawdown_percent': self.current_metrics.get('current_drawdown', 0),
                    'var_95_percent': self.risk_metrics['var_95'],
                    'var_99_percent': self.risk_metrics['var_99'],
                    'sortino_ratio': self.risk_metrics['sortino_ratio'],
                    'calmar_ratio': self.risk_metrics['calmar_ratio'],
                    
                    # Streak metrics
                    'max_consecutive_wins': self.risk_metrics['max_consecutive_wins'],
                    'max_consecutive_losses': self.risk_metrics['max_consecutive_losses'],
                    'current_streak': self.risk_metrics['current_streak'],
                    'current_streak_type': self.risk_metrics['streak_type'],
                    
                    # Execution metrics
                    'execution_statistics': execution_stats,
                    
                    # Strategy performance
                    'top_strategies': top_strategies,
                    
                    # Recent performance
                    'recent_trend': recent_trend,
                    
                    # Metadata
                    'last_updated': self.current_metrics.get('last_updated', datetime.now()),
                    'monitoring_duration_days': (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)).days,
                    'data_points': len(self.trade_records)
                }
                
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}

    def _get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution performance statistics"""
        try:
            if not self.execution_records:
                return {}
            
            execution_times = [r['execution_time'] for r in self.execution_records if r['execution_time'] > 0]
            slippages = [r['slippage_pips'] for r in self.execution_records if r['slippage_pips'] >= 0]
            
            return {
                'avg_execution_time_seconds': np.mean(execution_times) if execution_times else 0,
                'max_execution_time_seconds': np.max(execution_times) if execution_times else 0,
                'avg_slippage_pips': np.mean(slippages) if slippages else 0,
                'max_slippage_pips': np.max(slippages) if slippages else 0,
                'execution_success_rate': len([r for r in self.execution_records if r['success']]) / len(self.execution_records) * 100
            }
        except Exception as e:
            self.logger.error(f"Error getting execution statistics: {e}")
            return {}

    def _get_top_strategies(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing strategies"""
        try:
            strategies = []
            for strategy, perf in self.strategy_performance.items():
                if perf['trades'] >= 3:  # Only include strategies with at least 3 trades
                    strategies.append({
                        'strategy': strategy,
                        'total_trades': perf['trades'],
                        'win_rate': perf['win_rate'],
                        'total_pnl': perf['total_pnl'],
                        'profit_factor': perf['profit_factor']
                    })
            
            # Sort by total PnL
            strategies.sort(key=lambda x: x['total_pnl'], reverse=True)
            return strategies[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting top strategies: {e}")
            return []

    def _get_recent_performance_trend(self, days: int = 7) -> Dict[str, Any]:
        """Get recent performance trend"""
        try:
            cutoff_date = datetime.now().date() - timedelta(days=days)
            recent_trades = [t for t in self.trade_records if t.exit_time and t.exit_time.date() >= cutoff_date]
            
            if not recent_trades:
                return {}
            
            recent_pnl = sum(t.profit_loss for t in recent_trades)
            recent_wins = len([t for t in recent_trades if t.profit_loss > 0])
            recent_total = len(recent_trades)
            
            return {
                'period_days': days,
                'total_trades': recent_total,
                'winning_trades': recent_wins,
                'win_rate': (recent_wins / recent_total * 100) if recent_total > 0 else 0,
                'total_pnl': recent_pnl,
                'avg_daily_pnl': recent_pnl / days,
                'trend': 'positive' if recent_pnl > 0 else 'negative' if recent_pnl < 0 else 'neutral'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting recent performance trend: {e}")
            return {}

    def _save_performance_data(self) -> None:
        """Save performance data to file"""
        try:
            # Prepare data for JSON serialization
            data = {
                'current_metrics': self.current_metrics.copy(),
                'risk_metrics': self.risk_metrics.copy(),
                'strategy_performance': dict(self.strategy_performance),
                'symbol_performance': dict(self.symbol_performance),
                'last_saved': datetime.now().isoformat()
            }
            
            # Convert datetime objects to ISO format
            if 'last_updated' in data['current_metrics']:
                data['current_metrics']['last_updated'] = data['current_metrics']['last_updated'].isoformat()
            
            # Save to file
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}")

    def _load_historical_data(self) -> None:
        """Load historical performance data"""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load metrics
                self.current_metrics.update(data.get('current_metrics', {}))
                self.risk_metrics.update(data.get('risk_metrics', {}))
                
                # Convert datetime strings back to datetime objects
                if 'last_updated' in self.current_metrics and isinstance(self.current_metrics['last_updated'], str):
                    self.current_metrics['last_updated'] = datetime.fromisoformat(self.current_metrics['last_updated'])
                
                # Load strategy and symbol performance
                strategy_data = data.get('strategy_performance', {})
                for strategy, perf in strategy_data.items():
                    self.strategy_performance[strategy].update(perf)
                
                symbol_data = data.get('symbol_performance', {})
                for symbol, perf in symbol_data.items():
                    self.symbol_performance[symbol].update(perf)
                
                self.logger.info("Historical performance data loaded successfully")
                
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")

    def reset_metrics(self) -> bool:
        """Reset all performance metrics"""
        try:
            with self.lock:
                self.trade_records.clear()
                self.signal_records.clear()
                self.execution_records.clear()
                self.performance_snapshots.clear()
                self.daily_returns.clear()
                
                # Reset metrics
                self.current_metrics = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit': 0.0,
                    'total_loss': 0.0,
                    'max_drawdown': 0.0,
                    'current_drawdown': 0.0,
                    'peak_equity': 100000.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'last_updated': datetime.now()
                }
                
                self.strategy_performance.clear()
                self.symbol_performance.clear()
                self.hourly_performance.clear()
                self.daily_performance.clear()
                self.weekly_performance.clear()
                self.monthly_performance.clear()
                
                # Reset risk metrics
                self.risk_metrics = {
                    'var_95': 0.0,
                    'var_99': 0.0,
                    'expected_shortfall': 0.0,
                    'calmar_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'max_consecutive_losses': 0,
                    'max_consecutive_wins': 0,
                    'current_streak': 0,
                    'streak_type': 'none'
                }
                
                self.logger.info("Performance metrics reset successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Error resetting metrics: {e}")
            return False

    def export_performance_report(self, filepath: str = None) -> str:
        """Export comprehensive performance report"""
        try:
            if filepath is None:
                filepath = f"logs/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = {
                'report_generated': datetime.now().isoformat(),
                'summary': self.get_summary(),
                'strategy_performance': dict(self.strategy_performance),
                'symbol_performance': dict(self.symbol_performance),
                'recent_trades': [asdict(t) for t in self.trade_records[-50:]] if self.trade_records else [],
                'execution_statistics': self._get_execution_statistics(),
                'risk_analysis': self.risk_metrics.copy()
            }
            
            # Save report
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Performance report exported to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting performance report: {e}")
            return ""

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics for dashboard"""
        try:
            with self.lock:
                return {
                    'timestamp': datetime.now().isoformat(),
                    'total_trades': self.current_metrics.get('total_trades', 0),
                    'win_rate': self.current_metrics.get('win_rate', 0),
                    'profit_factor': self.current_metrics.get('profit_factor', 0),
                    'current_drawdown': self.current_metrics.get('current_drawdown', 0),
                    'daily_pnl': self._calculate_daily_return(),
                    'current_streak': self.risk_metrics['current_streak'],
                    'streak_type': self.risk_metrics['streak_type'],
                    'sharpe_ratio': self.current_metrics.get('sharpe_ratio', 0),
                    'active_strategies': len(self.strategy_performance),
                    'last_trade_time': self.trade_records[-1].exit_time.isoformat() if self.trade_records else None
                }
        except Exception as e:
            self.logger.error(f"Error getting real-time metrics: {e}")
            return {}

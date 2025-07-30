# core/complete_dynamic_position_manager.py
"""
Complete Dynamic Position Manager
Integrates with your existing ExecutionManager
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import threading
import time

class PositionAction(Enum):
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    CLOSE_PARTIAL = "close_partial"
    CLOSE_ALL = "close_all"
    MODIFY_SL = "modify_sl"
    MODIFY_TP = "modify_tp"
    TRAIL_STOP = "trail_stop"

class PositionStatus(Enum):
    ACTIVE = "active"
    SCALING = "scaling"
    CLOSING = "closing"
    EMERGENCY_EXIT = "emergency_exit"

class RiskLevel(Enum):
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class PositionActionCommand:
    position_id: str
    action: PositionAction
    percentage: float
    price_level: Optional[float] = None
    reason: str = ""
    urgency: int = 1  # 1-5, 5 being most urgent
    created_at: datetime = field(default_factory=datetime.now)
    executed: bool = False

@dataclass
class DynamicPosition:
    position_id: str
    symbol: str
    direction: str  # 'long' or 'short'
    original_size: float
    current_size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    created_at: datetime
    last_updated: datetime
    status: PositionStatus
    
    # Dynamic management fields
    unrealized_pnl: float = 0.0
    peak_profit: float = 0.0
    max_drawdown: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    action_history: List[PositionActionCommand] = field(default_factory=list)
    
    def update_metrics(self, current_price: float, atr: float = None):
        """Update position metrics"""
        self.current_price = current_price
        self.last_updated = datetime.now()
        
        # Calculate PnL
        if self.direction == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.current_size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.current_size
        
        # Update peak profit
        if self.unrealized_pnl > self.peak_profit:
            self.peak_profit = self.unrealized_pnl
        
        # Update drawdown
        drawdown_from_peak = self.peak_profit - self.unrealized_pnl if self.peak_profit > 0 else 0
        if drawdown_from_peak > self.max_drawdown:
            self.max_drawdown = drawdown_from_peak
        
        # Update risk level
        pnl_percentage = (self.unrealized_pnl / (self.original_size * self.entry_price)) * 100
        
        if pnl_percentage < -5.0:
            self.risk_level = RiskLevel.EMERGENCY
        elif pnl_percentage < -3.0:
            self.risk_level = RiskLevel.CRITICAL
        elif pnl_percentage < -2.0:
            self.risk_level = RiskLevel.HIGH
        elif pnl_percentage < -1.0:
            self.risk_level = RiskLevel.MODERATE
        else:
            self.risk_level = RiskLevel.LOW

class CompleteDynamicPositionManager:
    """
    Complete Dynamic Position Manager that integrates with your ExecutionManager
    """
    
    def __init__(self, execution_manager, market_intelligence, risk_manager, config: Dict):
        self.execution_manager = execution_manager
        self.market_intelligence = market_intelligence
        self.risk_manager = risk_manager
        self.config = config
        
        # Position storage
        self.active_positions: Dict[str, DynamicPosition] = {}
        self.pending_actions: List[PositionActionCommand] = []
        
        # Threading for real-time monitoring
        self.monitoring_active = False
        self.analysis_thread = None
        self.execution_thread = None
        
        # Configuration
        self.analysis_interval = config.get('analysis_interval_seconds', 5)
        self.emergency_exit_threshold = config.get('emergency_exit_threshold', -5.0)
        
        # Performance tracking
        self.total_actions_executed = 0
        self.successful_actions = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Dynamic Position Manager initialized")
    
    def start_dynamic_monitoring(self):
        """Start dynamic position monitoring"""
        if self.monitoring_active:
            self.logger.warning("Dynamic monitoring already active")
            return
        
        self.monitoring_active = True
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(
            target=self._continuous_analysis,
            daemon=True
        )
        self.analysis_thread.start()
        
        # Start execution thread
        self.execution_thread = threading.Thread(
            target=self._continuous_execution,
            daemon=True
        )
        self.execution_thread.start()
        
        self.logger.info("🚀 Dynamic position monitoring started")
    
    def stop_dynamic_monitoring(self):
        """Stop dynamic monitoring"""
        self.monitoring_active = False
        
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        if self.execution_thread:
            self.execution_thread.join(timeout=5)
        
        self.logger.info("⏹️ Dynamic position monitoring stopped")
    
    def add_position(self, position_data: Dict) -> str:
        """Add position to dynamic management"""
        try:
            position = DynamicPosition(
                position_id=position_data['position_id'],
                symbol=position_data['symbol'],
                direction=position_data['direction'],
                original_size=position_data['size'],
                current_size=position_data['size'],
                entry_price=position_data['entry_price'],
                current_price=position_data['entry_price'],
                stop_loss=position_data.get('stop_loss', 0),
                take_profit=position_data.get('take_profit', 0),
                created_at=datetime.now(),
                last_updated=datetime.now(),
                status=PositionStatus.ACTIVE
            )
            
            self.active_positions[position.position_id] = position
            
            self.logger.info(f"Position {position.position_id} added to dynamic management")
            return position.position_id
            
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
            return None
    
    def _continuous_analysis(self):
        """Continuous analysis of all positions"""
        while self.monitoring_active:
            try:
                if not self.active_positions:
                    time.sleep(self.analysis_interval)
                    continue
                
                for position_id, position in list(self.active_positions.items()):
                    try:
                        actions = self._analyze_single_position(position)
                        if actions:
                            self.pending_actions.extend(actions)
                    except Exception as e:
                        self.logger.error(f"Error analyzing position {position_id}: {e}")
                
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                self.logger.error(f"Error in continuous analysis: {e}")
                time.sleep(self.analysis_interval)
    
    def _analyze_single_position(self, position: DynamicPosition) -> List[PositionActionCommand]:
        """Analyze single position and generate actions"""
        try:
            # Get current market data
            current_price = self._get_current_price(position.symbol)
            if not current_price:
                return []
            
            # Get market data for analysis
            price_data = self._get_recent_price_data(position.symbol, 100)
            if price_data is None:
                return []
            
            # Update position metrics
            atr = self._calculate_atr(price_data)
            position.update_metrics(current_price, atr)
            
            # Generate actions based on analysis
            actions = []
            
            # 1. EMERGENCY EXIT CHECK
            if self._should_emergency_exit(position, price_data):
                actions.append(PositionActionCommand(
                    position_id=position.position_id,
                    action=PositionAction.CLOSE_ALL,
                    percentage=1.0,
                    reason=f"Emergency exit: Risk level {position.risk_level.name}",
                    urgency=5
                ))
                return actions
            
            # 2. TREND REVERSAL CHECK
            reversal_detected, reversal_probability = self.market_intelligence.detect_trend_reversal(
                position.symbol, price_data
            )
            
            if reversal_detected and reversal_probability > 0.7:
                # Partial or full exit based on PnL
                if position.unrealized_pnl > 0:
                    # Close profitable position on reversal
                    actions.append(PositionActionCommand(
                        position_id=position.position_id,
                        action=PositionAction.CLOSE_ALL,
                        percentage=1.0,
                        reason=f"Trend reversal detected: {reversal_probability:.2f}",
                        urgency=4
                    ))
                else:
                    # Scale out losing position
                    actions.append(PositionActionCommand(
                        position_id=position.position_id,
                        action=PositionAction.SCALE_OUT,
                        percentage=0.5,
                        reason=f"Partial exit on reversal: {reversal_probability:.2f}",
                        urgency=4
                    ))
            
            # 3. PROFIT TAKING
            pnl_percentage = (position.unrealized_pnl / (position.original_size * position.entry_price)) * 100
            
            if pnl_percentage >= 2.0 and not self._has_taken_profit_at_level(position, 2.0):
                actions.append(PositionActionCommand(
                    position_id=position.position_id,
                    action=PositionAction.SCALE_OUT,
                    percentage=0.25,
                    reason=f"Profit taking at {pnl_percentage:.1f}%",
                    urgency=2
                ))
            
            # 4. TRAILING STOP
            if position.unrealized_pnl > 0:
                new_stop = self._calculate_trailing_stop(position, current_price, atr)
                if new_stop != position.stop_loss:
                    actions.append(PositionActionCommand(
                        position_id=position.position_id,
                        action=PositionAction.MODIFY_SL,
                        percentage=0.0,
                        price_level=new_stop,
                        reason="Trailing stop adjustment",
                        urgency=3
                    ))
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error in position analysis: {e}")
            return []
    
    def _should_emergency_exit(self, position: DynamicPosition, price_data: pd.DataFrame) -> bool:
        """Determine if emergency exit is required"""
        try:
            # Critical loss threshold
            pnl_percentage = (position.unrealized_pnl / (position.original_size * position.entry_price)) * 100
            if pnl_percentage < self.emergency_exit_threshold:
                return True
            
            # Risk level check
            if position.risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in emergency exit check: {e}")
            return False
    
    def _calculate_trailing_stop(self, position: DynamicPosition, current_price: float, atr: float) -> float:
        """Calculate trailing stop level"""
        try:
            trail_distance = atr * 2.0
            
            if position.direction == 'long':
                trailing_stop = current_price - trail_distance
                return max(position.stop_loss, trailing_stop)
            else:
                trailing_stop = current_price + trail_distance
                return min(position.stop_loss, trailing_stop)
                
        except Exception as e:
            self.logger.error(f"Error calculating trailing stop: {e}")
            return position.stop_loss
    
    def _has_taken_profit_at_level(self, position: DynamicPosition, level: float) -> bool:
        """Check if profit has been taken at specific level"""
        for action in position.action_history:
            if (action.action == PositionAction.SCALE_OUT and 
                action.executed and 
                f"{level}%" in action.reason):
                return True
        return False
    
    def _continuous_execution(self):
        """Continuous execution of pending actions"""
        while self.monitoring_active:
            try:
                if not self.pending_actions:
                    time.sleep(1)
                    continue
                
                # Sort by urgency
                self.pending_actions.sort(key=lambda x: x.urgency, reverse=True)
                
                # Execute top priority action
                if self.pending_actions:
                    action = self.pending_actions.pop(0)
                    success = self._execute_action(action)
                    
                    if success:
                        self.successful_actions += 1
                        # Add to position history
                        if action.position_id in self.active_positions:
                            self.active_positions[action.position_id].action_history.append(action)
                    
                    self.total_actions_executed += 1
                
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error in continuous execution: {e}")
                time.sleep(1)
    
    def _execute_action(self, action: PositionActionCommand) -> bool:
        """Execute single action through ExecutionManager"""
        try:
            position = self.active_positions.get(action.position_id)
            if not position:
                return False
            
            self.logger.info(f"Executing {action.action.value} for {action.position_id}: {action.reason}")
            
            if action.action == PositionAction.CLOSE_ALL:
                # Use your existing close_position method
                success = self.execution_manager.close_position(action.position_id, percentage=1.0)
                if success:
                    del self.active_positions[action.position_id]
                return success
            
            elif action.action == PositionAction.SCALE_OUT:
                success = self.execution_manager.close_position(action.position_id, percentage=action.percentage)
                if success:
                    position.current_size *= (1 - action.percentage)
                return success
            
            elif action.action == PositionAction.MODIFY_SL:
                success = self.execution_manager.modify_stop_loss(action.position_id, action.price_level)
                if success:
                    position.stop_loss = action.price_level
                return success
            
            action.executed = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing action: {e}")
            return False
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from your data handler"""
        try:
            price_data = self.execution_manager.data_handler.get_current_price(symbol)
            if price_data:
                return (price_data['bid'] + price_data['ask']) / 2
            return None
        except:
            return None
    
    def _get_recent_price_data(self, symbol: str, periods: int) -> Optional[pd.DataFrame]:
        """Get recent price data"""
        try:
            return self.execution_manager.data_handler.get_data(symbol, 'H1', periods)
        except:
            return None
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR"""
        try:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            return true_range.rolling(period).mean().iloc[-1]
        except:
            return 0.001
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        return {
            'active_positions': len(self.active_positions),
            'total_actions_executed': self.total_actions_executed,
            'successful_actions': self.successful_actions,
            'success_rate': (self.successful_actions / max(1, self.total_actions_executed)) * 100,
            'pending_actions': len(self.pending_actions),
            'monitoring_active': self.monitoring_active
        }

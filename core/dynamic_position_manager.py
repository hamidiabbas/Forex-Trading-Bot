# core/dynamic_position_manager.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import threading
import time
from concurrent.futures import ThreadPoolExecutor

class PositionAction(Enum):
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    CLOSE_PARTIAL = "close_partial"
    CLOSE_ALL = "close_all"
    MODIFY_SL = "modify_sl"
    MODIFY_TP = "modify_tp"
    TIGHTEN_SL = "tighten_sl"
    TRAIL_STOP = "trail_stop"

class PositionStatus(Enum):
    ACTIVE = "active"
    SCALING = "scaling"
    CLOSING = "closing"
    EMERGENCY_EXIT = "emergency_exit"
    PAUSED = "paused"

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
    execution_result: Optional[Dict] = None

@dataclass
class PositionMetrics:
    unrealized_pnl: float
    unrealized_pnl_percentage: float
    duration_minutes: int
    current_risk_level: RiskLevel
    drawdown_from_peak: float
    peak_profit: float
    risk_reward_ratio: float
    volatility_adjusted_return: float

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
    scaling_plan: Dict
    risk_metrics: PositionMetrics
    action_history: List[PositionActionCommand] = field(default_factory=list)
    peak_profit: float = 0.0
    max_drawdown: float = 0.0
    last_analysis_time: datetime = field(default_factory=datetime.now)
    
    # Trend tracking
    trend_confidence: float = 0.0
    momentum_score: float = 0.0
    reversal_probability: float = 0.0
    
    def update_metrics(self, current_price: float, atr: float = None):
        """Update position metrics with current market data"""
        self.current_price = current_price
        self.last_updated = datetime.now()
        
        # Calculate PnL
        if self.direction == 'long':
            pnl = (current_price - self.entry_price) * self.current_size
            pnl_percentage = ((current_price / self.entry_price) - 1) * 100
        else:
            pnl = (self.entry_price - current_price) * self.current_size
            pnl_percentage = ((self.entry_price / current_price) - 1) * 100
        
        # Update peak profit and drawdown
        if pnl > self.peak_profit:
            self.peak_profit = pnl
        
        drawdown_from_peak = self.peak_profit - pnl if self.peak_profit > 0 else 0
        if drawdown_from_peak > self.max_drawdown:
            self.max_drawdown = drawdown_from_peak
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(pnl_percentage, drawdown_from_peak)
        
        # Duration
        duration = (datetime.now() - self.created_at).total_seconds() / 60
        
        # Risk-reward ratio
        distance_to_sl = abs(current_price - self.stop_loss)
        distance_to_tp = abs(self.take_profit - current_price)
        risk_reward = distance_to_tp / distance_to_sl if distance_to_sl > 0 else 0
        
        # Volatility adjusted return
        volatility_adj_return = pnl_percentage / (atr / current_price * 100) if atr else pnl_percentage
        
        # Update metrics
        self.risk_metrics = PositionMetrics(
            unrealized_pnl=pnl,
            unrealized_pnl_percentage=pnl_percentage,
            duration_minutes=int(duration),
            current_risk_level=risk_level,
            drawdown_from_peak=drawdown_from_peak,
            peak_profit=self.peak_profit,
            risk_reward_ratio=risk_reward,
            volatility_adjusted_return=volatility_adj_return
        )
    
    def _calculate_risk_level(self, pnl_percentage: float, drawdown: float) -> RiskLevel:
        """Calculate current risk level based on position performance"""
        
        # Critical conditions
        if pnl_percentage < -5.0 or drawdown > self.peak_profit * 0.8:
            return RiskLevel.EMERGENCY
        elif pnl_percentage < -3.0 or drawdown > self.peak_profit * 0.6:
            return RiskLevel.CRITICAL
        elif pnl_percentage < -2.0 or drawdown > self.peak_profit * 0.4:
            return RiskLevel.HIGH
        elif pnl_percentage < -1.0 or drawdown > self.peak_profit * 0.2:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

class CompleteDynamicPositionManager:
    """
    Complete Dynamic Position Manager with real-time analysis and execution
    """
    
    def __init__(self, trading_engine, technical_analyzer, risk_manager, config: Dict):
        self.trading_engine = trading_engine
        self.technical_analyzer = technical_analyzer
        self.risk_manager = risk_manager
        self.config = config
        
        # Position storage
        self.active_positions: Dict[str, DynamicPosition] = {}
        self.position_history: List[DynamicPosition] = []
        self.pending_actions: List[PositionActionCommand] = []
        
        # Threading for real-time monitoring
        self.monitoring_active = False
        self.analysis_thread = None
        self.execution_thread = None
        self.thread_executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.total_actions_executed = 0
        self.successful_actions = 0
        self.emergency_exits = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.analysis_interval = config.get('analysis_interval_seconds', 5)
        self.max_positions = config.get('max_positions', 10)
        self.emergency_exit_threshold = config.get('emergency_exit_threshold', -10.0)
        
    def start_dynamic_monitoring(self):
        """Start the dynamic position monitoring system"""
        
        if self.monitoring_active:
            self.logger.warning("Dynamic monitoring already active")
            return
        
        self.monitoring_active = True
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(
            target=self._continuous_position_analysis,
            daemon=True
        )
        self.analysis_thread.start()
        
        # Start execution thread
        self.execution_thread = threading.Thread(
            target=self._continuous_action_execution,
            daemon=True
        )
        self.execution_thread.start()
        
        self.logger.info("Dynamic position monitoring started")
    
    def stop_dynamic_monitoring(self):
        """Stop the dynamic monitoring system"""
        
        self.monitoring_active = False
        
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        if self.execution_thread:
            self.execution_thread.join(timeout=5)
            
        self.thread_executor.shutdown(wait=True)
        
        self.logger.info("Dynamic position monitoring stopped")
    
    def add_position(self, position_data: Dict) -> str:
        """Add a new position to dynamic management"""
        
        try:
            # Create scaling plan
            scaling_plan = self._create_initial_scaling_plan(position_data)
            
            # Initialize position metrics
            initial_metrics = PositionMetrics(
                unrealized_pnl=0.0,
                unrealized_pnl_percentage=0.0,
                duration_minutes=0,
                current_risk_level=RiskLevel.LOW,
                drawdown_from_peak=0.0,
                peak_profit=0.0,
                risk_reward_ratio=self._calculate_initial_rr(position_data),
                volatility_adjusted_return=0.0
            )
            
            # Create dynamic position
            position = DynamicPosition(
                position_id=position_data['position_id'],
                symbol=position_data['symbol'],
                direction=position_data['direction'],
                original_size=position_data['size'],
                current_size=position_data['size'],
                entry_price=position_data['entry_price'],
                current_price=position_data['entry_price'],
                stop_loss=position_data['stop_loss'],
                take_profit=position_data['take_profit'],
                created_at=datetime.now(),
                last_updated=datetime.now(),
                status=PositionStatus.ACTIVE,
                scaling_plan=scaling_plan,
                risk_metrics=initial_metrics
            )
            
            # Add to active positions
            self.active_positions[position.position_id] = position
            
            self.logger.info(f"Added position {position.position_id} to dynamic management")
            return position.position_id
            
        except Exception as e:
            self.logger.error(f"Error adding position to dynamic management: {e}")
            return None
    
    def _continuous_position_analysis(self):
        """Continuous analysis of all active positions"""
        
        while self.monitoring_active:
            try:
                if not self.active_positions:
                    time.sleep(self.analysis_interval)
                    continue
                
                # Analyze all positions in parallel
                analysis_futures = []
                
                for position_id, position in self.active_positions.items():
                    future = self.thread_executor.submit(
                        self._analyze_single_position, 
                        position
                    )
                    analysis_futures.append((position_id, future))
                
                # Process analysis results
                for position_id, future in analysis_futures:
                    try:
                        actions = future.result(timeout=30)
                        if actions:
                            self.pending_actions.extend(actions)
                            self.logger.debug(f"Generated {len(actions)} actions for position {position_id}")
                    except Exception as e:
                        self.logger.error(f"Error analyzing position {position_id}: {e}")
                
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                self.logger.error(f"Error in continuous position analysis: {e}")
                time.sleep(self.analysis_interval * 2)  # Back off on error
    
    def _analyze_single_position(self, position: DynamicPosition) -> List[PositionActionCommand]:
        """Comprehensive analysis of a single position"""
        
        try:
            # Get current market data
            current_price = self._get_current_price(position.symbol)
            if not current_price:
                return []
            
            # Get technical analysis
            price_data = self._get_recent_price_data(position.symbol, 100)
            if price_data is None or len(price_data) < 50:
                return []
            
            # Perform complete technical analysis
            technical_analysis = self.technical_analyzer.analyze_symbol(
                symbol=position.symbol,
                timeframe='H1',
                price_data=price_data
            )
            
            # Update position metrics
            atr = technical_analysis['indicators']['volatility']['atr_14'][-1]
            position.update_metrics(current_price, atr)
            
            # Update trend indicators
            trend_analysis = technical_analysis['signals']['trend']
            position.trend_confidence = trend_analysis.get('confidence', 0.0)
            position.momentum_score = technical_analysis['signals']['momentum'].get('score', 0.0)
            position.reversal_probability = technical_analysis['signals']['reversal'].get('probability', 0.0)
            
            # Generate action recommendations
            actions = self._generate_position_actions(position, technical_analysis)
            
            position.last_analysis_time = datetime.now()
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error analyzing position {position.position_id}: {e}")
            return []
    
    def _generate_position_actions(self, position: DynamicPosition, 
                                 technical_analysis: Dict) -> List[PositionActionCommand]:
        """Generate specific actions based on position and market analysis"""[1][2]
        
        actions = []
        
        try:
            # Extract key metrics
            trend_direction = technical_analysis['signals']['trend']['direction'].value
            reversal_prob = position.reversal_probability
            momentum_score = position.momentum_score
            pnl_pct = position.risk_metrics.unrealized_pnl_percentage
            risk_level = position.risk_metrics.current_risk_level
            
            # 1. EMERGENCY EXIT CONDITIONS
            if self._should_emergency_exit(position, technical_analysis):
                actions.append(PositionActionCommand(
                    position_id=position.position_id,
                    action=PositionAction.CLOSE_ALL,
                    percentage=1.0,
                    reason=f"Emergency exit: Risk level {risk_level.name}, Reversal prob {reversal_prob:.2f}",
                    urgency=5
                ))
                return actions  # Emergency exit overrides all other actions
            
            # 2. TREND REVERSAL DETECTION
            if self._detect_confirmed_reversal(position, technical_analysis):
                if position.risk_metrics.unrealized_pnl > 0:
                    # Close profitable position on reversal
                    actions.append(PositionActionCommand(
                        position_id=position.position_id,
                        action=PositionAction.CLOSE_ALL,
                        percentage=1.0,
                        reason=f"Confirmed trend reversal: {reversal_prob:.2f} probability",
                        urgency=4
                    ))
                else:
                    # Scale out losing position on reversal
                    actions.append(PositionActionCommand(
                        position_id=position.position_id,
                        action=PositionAction.SCALE_OUT,
                        percentage=0.5,
                        reason=f"Partial exit on reversal signal: {reversal_prob:.2f}",
                        urgency=4
                    ))
            
            # 3. SCALING IN ON TREND CONTINUATION
            elif self._should_scale_in(position, technical_analysis):
                scale_percentage = self._calculate_scale_in_size(position, technical_analysis)
                actions.append(PositionActionCommand(
                    position_id=position.position_id,
                    action=PositionAction.SCALE_IN,
                    percentage=scale_percentage,
                    reason=f"Trend continuation: confidence {position.trend_confidence:.2f}",
                    urgency=2
                ))
            
            # 4. PROFIT TAKING LEVELS
            profit_actions = self._check_profit_taking_levels(position)
            actions.extend(profit_actions)
            
            # 5. DYNAMIC STOP LOSS MANAGEMENT
            sl_actions = self._manage_dynamic_stop_loss(position, technical_analysis)
            actions.extend(sl_actions)
            
            # 6. POSITION SIZE OPTIMIZATION
            size_actions = self._optimize_position_size(position, technical_analysis)
            actions.extend(size_actions)
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error generating actions for position {position.position_id}: {e}")
            return []
    
    def _should_emergency_exit(self, position: DynamicPosition, analysis: Dict) -> bool:
        """Determine if emergency exit is required"""[1][2]
        
        try:
            # Critical loss threshold
            if position.risk_metrics.unrealized_pnl_percentage < self.emergency_exit_threshold:
                return True
            
            # High reversal probability with significant loss
            if (position.reversal_probability > 0.8 and 
                position.risk_metrics.unrealized_pnl_percentage < -2.0):
                return True
            
            # Risk level critical or emergency
            if position.risk_metrics.current_risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]:
                return True
            
            # Strong opposing trend with momentum
            current_trend = analysis['signals']['trend']['direction'].value
            if ((position.direction == 'long' and current_trend < -1) or
                (position.direction == 'short' and current_trend > 1)):
                if position.momentum_score < -0.7:
                    return True
            
            # Maximum drawdown exceeded
            if (position.max_drawdown > position.original_size * 0.05 and
                position.risk_metrics.drawdown_from_peak > position.peak_profit * 0.75):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in emergency exit check: {e}")
            return False
    
    def _detect_confirmed_reversal(self, position: DynamicPosition, analysis: Dict) -> bool:
        """Detect confirmed trend reversal with multiple confirmations"""[2][3]
        
        try:
            confirmations = 0
            required_confirmations = 2
            
            # 1. High reversal probability
            if position.reversal_probability > 0.6:
                confirmations += 1
            
            # 2. Trend direction change
            trend_direction = analysis['signals']['trend']['direction'].value
            if ((position.direction == 'long' and trend_direction < 0) or
                (position.direction == 'short' and trend_direction > 0)):
                confirmations += 1
            
            # 3. Momentum confirmation
            if position.momentum_score < -0.5:
                confirmations += 1
            
            # 4. Support/Resistance break
            sr_levels = analysis['market_structure']['sr_levels']
            current_price = position.current_price
            
            if position.direction == 'long':
                # Check if price broke below key support
                nearest_support = sr_levels.get('nearest_support')
                if nearest_support and current_price < nearest_support * 0.998:
                    confirmations += 1
            else:
                # Check if price broke above key resistance
                nearest_resistance = sr_levels.get('nearest_resistance')
                if nearest_resistance and current_price > nearest_resistance * 1.002:
                    confirmations += 1
            
            # 5. Volume confirmation (if available)
            volume_analysis = analysis['indicators'].get('volume', {})
            if volume_analysis.get('volume_available', False):
                if volume_analysis.get('volume_analysis', {}).get('breakout_volume', False):
                    confirmations += 1
            
            return confirmations >= required_confirmations
            
        except Exception as e:
            self.logger.error(f"Error detecting reversal: {e}")
            return False
    
    def _should_scale_in(self, position: DynamicPosition, analysis: Dict) -> bool:
        """Determine if position should be scaled in"""[2]
        
        try:
            # Don't scale in if position is losing
            if position.risk_metrics.unrealized_pnl_percentage < -0.5:
                return False
            
            # Don't scale in if reversal probability is high
            if position.reversal_probability > 0.4:
                return False
            
            # Check if we can increase position size (risk management)
            if not self.risk_manager.can_increase_position(position.symbol, 0.3):
                return False
            
            # Trend must be in our favor
            trend_direction = analysis['signals']['trend']['direction'].value
            trend_strength = analysis['signals']['trend']['strength'].value
            
            if position.direction == 'long':
                trend_favorable = trend_direction > 0 and trend_strength >= 3
            else:
                trend_favorable = trend_direction < 0 and trend_strength >= 3
            
            if not trend_favorable:
                return False
            
            # High trend confidence
            if position.trend_confidence < 0.7:
                return False
            
            # Positive momentum
            if position.momentum_score < 0.3:
                return False
            
            # Check scaling plan limits
            scaling_plan = position.scaling_plan
            max_scale_ins = scaling_plan.get('max_scale_ins', 2)
            current_scale_ins = len([a for a in position.action_history 
                                   if a.action == PositionAction.SCALE_IN and a.executed])
            
            if current_scale_ins >= max_scale_ins:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking scale in conditions: {e}")
            return False
    
    def _calculate_scale_in_size(self, position: DynamicPosition, analysis: Dict) -> float:
        """Calculate optimal scale-in size"""[2]
        
        try:
            # Base scale-in percentage
            base_percentage = 0.3
            
            # Adjust based on trend strength
            trend_strength = analysis['signals']['trend']['strength'].value
            strength_multiplier = min(trend_strength / 3.0, 1.5)
            
            # Adjust based on confidence
            confidence_multiplier = position.trend_confidence
            
            # Adjust based on current profit
            profit_multiplier = 1.0
            if position.risk_metrics.unrealized_pnl_percentage > 2.0:
                profit_multiplier = 1.2
            elif position.risk_metrics.unrealized_pnl_percentage > 1.0:
                profit_multiplier = 1.1
            
            # Calculate final percentage
            scale_percentage = (base_percentage * 
                              strength_multiplier * 
                              confidence_multiplier * 
                              profit_multiplier)
            
            # Cap at maximum
            return min(scale_percentage, 0.5)
            
        except Exception as e:
            self.logger.error(f"Error calculating scale in size: {e}")
            return 0.2  # Default safe size
    
    def _check_profit_taking_levels(self, position: DynamicPosition) -> List[PositionActionCommand]:
        """Check and generate profit-taking actions"""[2]
        
        actions = []
        
        try:
            pnl_pct = position.risk_metrics.unrealized_pnl_percentage
            
            # Progressive profit taking levels
            profit_levels = [
                (1.5, 0.2, "First profit level"),      # 1.5% profit -> scale out 20%
                (3.0, 0.25, "Second profit level"),    # 3.0% profit -> scale out 25%
                (5.0, 0.3, "Third profit level"),      # 5.0% profit -> scale out 30%
                (8.0, 0.25, "Final profit level")      # 8.0% profit -> scale out 25%
            ]
            
            for profit_threshold, scale_out_pct, reason in profit_levels:
                if pnl_pct >= profit_threshold:
                    # Check if this level already taken
                    level_already_taken = any(
                        action.executed and 
                        action.action == PositionAction.SCALE_OUT and
                        reason in action.reason
                        for action in position.action_history
                    )
                    
                    if not level_already_taken:
                        actions.append(PositionActionCommand(
                            position_id=position.position_id,
                            action=PositionAction.SCALE_OUT,
                            percentage=scale_out_pct,
                            reason=f"{reason}: {pnl_pct:.1f}% profit",
                            urgency=2
                        ))
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error checking profit levels: {e}")
            return []
    
    def _manage_dynamic_stop_loss(self, position: DynamicPosition, 
                                analysis: Dict) -> List[PositionActionCommand]:
        """Manage dynamic stop-loss adjustments"""[1][2]
        
        actions = []
        
        try:
            current_price = position.current_price
            current_sl = position.stop_loss
            atr = analysis['indicators']['volatility']['atr_14'][-1]
            
            # Calculate different stop-loss levels
            atr_stop = self._calculate_atr_stop(position, current_price, atr)
            trailing_stop = self._calculate_trailing_stop(position, current_price, atr)
            breakeven_stop = position.entry_price
            
            # Determine best stop-loss level
            new_stop_loss = None
            reason = ""
            
            # If position is profitable, use trailing stop
            if position.risk_metrics.unrealized_pnl > 0:
                if position.direction == 'long':
                    new_stop_loss = max(current_sl, trailing_stop, breakeven_stop)
                else:
                    new_stop_loss = min(current_sl, trailing_stop, breakeven_stop)
                reason = "Trailing stop adjustment"
            
            # If position is at breakeven, move to breakeven
            elif abs(position.risk_metrics.unrealized_pnl_percentage) < 0.1:
                new_stop_loss = breakeven_stop
                reason = "Move stop to breakeven"
            
            # If volatility increased significantly, widen stop
            else:
                volatility_regime = analysis['indicators']['volatility']['volatility_regime']
                if volatility_regime == 'high':
                    new_stop_loss = atr_stop
                    reason = "Volatility adjustment"
            
            # Execute stop-loss modification if needed
            if new_stop_loss and abs(new_stop_loss - current_sl) > 0.0001:
                actions.append(PositionActionCommand(
                    position_id=position.position_id,
                    action=PositionAction.MODIFY_SL,
                    percentage=0.0,
                    price_level=new_stop_loss,
                    reason=reason,
                    urgency=3
                ))
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error managing stop loss: {e}")
            return []
    
    def _calculate_trailing_stop(self, position: DynamicPosition, 
                               current_price: float, atr: float) -> float:
        """Calculate trailing stop level"""[2]
        
        try:
            # ATR-based trailing distance
            trail_distance = atr * 2.0
            
            # Adjust based on volatility
            volatility_multiplier = 1.0
            if position.risk_metrics.current_risk_level == RiskLevel.HIGH:
                volatility_multiplier = 1.5
            elif position.risk_metrics.current_risk_level == RiskLevel.LOW:
                volatility_multiplier = 0.8
            
            trail_distance *= volatility_multiplier
            
            if position.direction == 'long':
                trailing_stop = current_price - trail_distance
                # Only move stop up, never down
                return max(position.stop_loss, trailing_stop)
            else:
                trailing_stop = current_price + trail_distance
                # Only move stop down, never up
                return min(position.stop_loss, trailing_stop)
                
        except Exception as e:
            self.logger.error(f"Error calculating trailing stop: {e}")
            return position.stop_loss
    
    def _continuous_action_execution(self):
        """Continuous execution of pending actions"""
        
        while self.monitoring_active:
            try:
                if not self.pending_actions:
                    time.sleep(1)
                    continue
                
                # Sort actions by urgency (highest first)
                self.pending_actions.sort(key=lambda x: x.urgency, reverse=True)
                
                # Execute actions
                actions_to_remove = []
                
                for action in self.pending_actions[:5]:  # Process max 5 actions per cycle
                    try:
                        success = self._execute_single_action(action)
                        action.executed = True
                        
                        if success:
                            self.successful_actions += 1
                            if action.action == PositionAction.CLOSE_ALL:
                                self.emergency_exits += 1
                        
                        self.total_actions_executed += 1
                        actions_to_remove.append(action)
                        
                        # Add to position history
                        if action.position_id in self.active_positions:
                            self.active_positions[action.position_id].action_history.append(action)
                        
                        # Small delay between executions
                        time.sleep(0.1)
                        
                    except Exception as e:
                        self.logger.error(f"Error executing action {action.action.value}: {e}")
                        actions_to_remove.append(action)
                
                # Remove processed actions
                for action in actions_to_remove:
                    if action in self.pending_actions:
                        self.pending_actions.remove(action)
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in continuous action execution: {e}")
                time.sleep(2)
    
    def _execute_single_action(self, action: PositionActionCommand) -> bool:
        """Execute a single position action"""
        
        try:
            position = self.active_positions.get(action.position_id)
            if not position:
                self.logger.warning(f"Position {action.position_id} not found for action {action.action.value}")
                return False
            
            self.logger.info(f"Executing {action.action.value} for position {action.position_id}: {action.reason}")
            
            if action.action == PositionAction.CLOSE_ALL:
                return self._execute_close_all(position, action)
            
            elif action.action in [PositionAction.SCALE_OUT, PositionAction.CLOSE_PARTIAL]:
                return self._execute_scale_out(position, action)
            
            elif action.action == PositionAction.SCALE_IN:
                return self._execute_scale_in(position, action)
            
            elif action.action == PositionAction.MODIFY_SL:
                return self._execute_modify_sl(position, action)
            
            elif action.action == PositionAction.MODIFY_TP:
                return self._execute_modify_tp(position, action)
            
            else:
                self.logger.warning(f"Unknown action type: {action.action.value}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing action {action.action.value}: {e}")
            action.execution_result = {"error": str(e)}
            return False
    
    def _execute_close_all(self, position: DynamicPosition, action: PositionActionCommand) -> bool:
        """Execute complete position closure"""
        
        try:
            success = self.trading_engine.close_position(
                position_id=position.position_id,
                percentage=1.0
            )
            
            if success:
                # Move to history
                position.status = PositionStatus.CLOSING
                self.position_history.append(position)
                del self.active_positions[position.position_id]
                
                action.execution_result = {"status": "success", "closed_size": position.current_size}
                self.logger.info(f"Successfully closed position {position.position_id}")
                return True
            else:
                action.execution_result = {"status": "failed", "error": "Trading engine returned false"}
                return False
                
        except Exception as e:
            action.execution_result = {"status": "error", "error": str(e)}
            return False
    
    def _execute_scale_out(self, position: DynamicPosition, action: PositionActionCommand) -> bool:
        """Execute partial position closure"""
        
        try:
            success = self.trading_engine.close_position(
                position_id=position.position_id,
                percentage=action.percentage
            )
            
            if success:
                # Update position size
                closed_size = position.current_size * action.percentage
                position.current_size -= closed_size
                position.status = PositionStatus.SCALING
                
                action.execution_result = {
                    "status": "success", 
                    "closed_size": closed_size,
                    "remaining_size": position.current_size
                }
                
                self.logger.info(f"Scaled out {action.percentage:.1%} of position {position.position_id}")
                return True
            else:
                action.execution_result = {"status": "failed", "error": "Trading engine returned false"}
                return False
                
        except Exception as e:
            action.execution_result = {"status": "error", "error": str(e)}
            return False
    
    def _execute_scale_in(self, position: DynamicPosition, action: PositionActionCommand) -> bool:
        """Execute position size increase"""
        
        try:
            additional_size = position.original_size * action.percentage
            
            success = self.trading_engine.add_to_position(
                position_id=position.position_id,
                additional_size=additional_size,
                direction=position.direction
            )
            
            if success:
                # Update position size
                position.current_size += additional_size
                position.status = PositionStatus.SCALING
                
                action.execution_result = {
                    "status": "success",
                    "added_size": additional_size,
                    "new_size": position.current_size
                }
                
                self.logger.info(f"Scaled in {action.percentage:.1%} to position {position.position_id}")
                return True
            else:
                action.execution_result = {"status": "failed", "error": "Trading engine returned false"}
                return False
                
        except Exception as e:
            action.execution_result = {"status": "error", "error": str(e)}
            return False
    
    def _execute_modify_sl(self, position: DynamicPosition, action: PositionActionCommand) -> bool:
        """Execute stop-loss modification"""
        
        try:
            success = self.trading_engine.modify_stop_loss(
                position_id=position.position_id,
                new_stop_loss=action.price_level
            )
            
            if success:
                old_sl = position.stop_loss
                position.stop_loss = action.price_level
                
                action.execution_result = {
                    "status": "success",
                    "old_stop_loss": old_sl,
                    "new_stop_loss": action.price_level
                }
                
                self.logger.info(f"Modified stop loss for position {position.position_id}: {old_sl} -> {action.price_level}")
                return True
            else:
                action.execution_result = {"status": "failed", "error": "Trading engine returned false"}
                return False
                
        except Exception as e:
            action.execution_result = {"status": "error", "error": str(e)}
            return False
    
    # Helper methods for data retrieval (integrate with your data manager)
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        try:
            # This should integrate with your data manager/broker API
            return self.trading_engine.get_current_price(symbol)
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _get_recent_price_data(self, symbol: str, periods: int) -> Optional[pd.DataFrame]:
        """Get recent OHLCV data for symbol"""
        try:
            # This should integrate with your data manager
            return self.trading_engine.get_price_data(symbol, 'H1', periods)
        except Exception as e:
            self.logger.error(f"Error getting price data for {symbol}: {e}")
            return None
    
    def _create_initial_scaling_plan(self, position_data: Dict) -> Dict:
        """Create initial scaling plan for position"""[2]
        
        return {
            'max_scale_ins': 2,
            'max_scale_outs': 4,
            'scale_in_levels': self._calculate_scale_in_levels(position_data),
            'scale_out_levels': self._calculate_scale_out_levels(position_data),
            'emergency_exit_threshold': -10.0,
            'max_position_multiplier': 2.0
        }
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary of dynamic position management"""
        
        active_count = len(self.active_positions)
        total_unrealized_pnl = sum(p.risk_metrics.unrealized_pnl for p in self.active_positions.values())
        
        risk_distribution = {}
        for level in RiskLevel:
            count = sum(1 for p in self.active_positions.values() 
                       if p.risk_metrics.current_risk_level == level)
            risk_distribution[level.name] = count
        
        success_rate = (self.successful_actions / max(self.total_actions_executed, 1)) * 100
        
        return {
            'active_positions': active_count,
            'total_unrealized_pnl': total_unrealized_pnl,
            'risk_distribution': risk_distribution,
            'total_actions_executed': self.total_actions_executed,
            'successful_actions': self.successful_actions,
            'success_rate_percent': success_rate,
            'emergency_exits': self.emergency_exits,
            'pending_actions': len(self.pending_actions),
            'monitoring_active': self.monitoring_active
        }

# Example usage and integration
if __name__ == "__main__":
    # This would be integrated with your main trading system
    
    config = {
        'analysis_interval_seconds': 5,
        'max_positions': 10,
        'emergency_exit_threshold': -8.0
    }
    
    # Initialize components (these would be your actual implementations)
    # trading_engine = YourTradingEngine()
    # technical_analyzer = ComprehensiveTechnicalAnalyzer()
    # risk_manager = YourRiskManager()
    
    # position_manager = CompleteDynamicPositionManager(
    #     trading_engine, technical_analyzer, risk_manager, config
    # )
    
    # position_manager.start_dynamic_monitoring()
    
    print("Dynamic Position Manager implementation complete!")

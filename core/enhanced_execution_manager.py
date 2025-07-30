# core/enhanced_execution_manager.py
"""
Enhanced Execution Manager that integrates dynamic position management
Extends your existing ExecutionManager
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

class EnhancedExecutionManager:
    """
    Enhanced Execution Manager with dynamic features
    Extends your existing ExecutionManager functionality
    """
    
    def __init__(self, config, market_intelligence, risk_manager, data_handler):
        # Preserve your existing initialization
        self.config = config
        self.market_intelligence = market_intelligence
        self.risk_manager = risk_manager
        self.data_handler = data_handler
        self.logger = logging.getLogger(__name__)
        
        # Dynamic position manager will be set after initialization
        self.position_manager = None
        self.dynamic_monitoring_active = False
        
        # Your existing execution logic
        self.active_positions = {}
        self.execution_history = []
        
        self.logger.info("Enhanced Execution Manager initialized")
    
    def set_position_manager(self, position_manager):
        """Set the dynamic position manager"""
        self.position_manager = position_manager
        self.logger.info("Dynamic position manager connected")
    
    def start_dynamic_monitoring(self):
        """Start dynamic position monitoring"""
        if self.position_manager:
            self.position_manager.start_dynamic_monitoring()
            self.dynamic_monitoring_active = True
            self.logger.info("🚀 Dynamic position monitoring started")
    
    def stop_dynamic_monitoring(self):
        """Stop dynamic position monitoring"""
        if self.position_manager:
            self.position_manager.stop_dynamic_monitoring()
            self.dynamic_monitoring_active = False
            self.logger.info("⏹️ Dynamic position monitoring stopped")
    
    def execute_trade_with_dynamic_features(self, signal: Dict, risk_params: Dict) -> Optional[Dict]:
        """
        Execute trade with dynamic position management
        Enhances your existing execute_trade method
        """
        try:
            self.logger.info(f"Executing enhanced trade for {signal['symbol']}")
            
            # Use your existing trade execution logic
            result = self._execute_traditional_trade(signal, risk_params)
            
            if result and result.get('success'):
                # Add to dynamic management
                if self.position_manager:
                    position_data = {
                        'position_id': result.get('position_id', f"{signal['symbol']}_{int(time.time())}"),
                        'symbol': signal['symbol'],
                        'direction': signal['direction'],
                        'size': risk_params.get('position_size', 0.1),
                        'entry_price': result.get('fill_price', signal['entry_price']),
                        'stop_loss': signal.get('stop_loss', 0),
                        'take_profit': signal.get('take_profit', 0)
                    }
                    
                    position_id = self.position_manager.add_position(position_data)
                    if position_id:
                        self.logger.info(f"Position {position_id} added to dynamic management")
                
                return result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced trade execution: {e}")
            return None
    
    def _execute_traditional_trade(self, signal: Dict, risk_params: Dict) -> Optional[Dict]:
        """
        Your existing trade execution logic
        Replace this with your actual implementation
        """
        try:
            # This is where your existing trade execution happens
            # For now, return a mock successful result
            position_id = f"{signal['symbol']}_{int(time.time())}"
            
            # Mock execution result
            result = {
                'success': True,
                'position_id': position_id,
                'fill_price': signal['entry_price'],
                'size': risk_params.get('position_size', 0.1),
                'timestamp': datetime.now()
            }
            
            # Add to your existing position tracking
            self.active_positions[position_id] = {
                'symbol': signal['symbol'],
                'direction': signal['direction'],
                'size': result['size'],
                'entry_price': result['fill_price'],
                'created_at': datetime.now()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in traditional trade execution: {e}")
            return None
    
    def manage_positions(self):
        """
        Enhanced position management that includes dynamic features
        Extends your existing manage_positions method
        """
        try:
            # Your existing position management logic here
            self._manage_traditional_positions()
            
            # Dynamic management is handled automatically by the position manager
            # Log performance if available
            if self.position_manager:
                performance = self.position_manager.get_performance_summary()
                if performance['total_actions_executed'] > 0:
                    self.logger.debug(f"Dynamic management: {performance['total_actions_executed']} actions, "
                                    f"{performance['success_rate']:.1f}% success rate")
            
        except Exception as e:
            self.logger.error(f"Error in enhanced position management: {e}")
    
    def _manage_traditional_positions(self):
        """Your existing position management logic"""
        # Replace with your actual position management code
        pass
    
    def close_position(self, position_id: str, percentage: float = 1.0) -> bool:
        """
        Enhanced close position method
        Replace with your actual implementation
        """
        try:
            self.logger.info(f"Closing {percentage:.1%} of position {position_id}")
            
            # Your actual position closing logic here
            # For now, return mock success
            
            if percentage >= 1.0:
                # Full closure
                if position_id in self.active_positions:
                    del self.active_positions[position_id]
            else:
                # Partial closure
                if position_id in self.active_positions:
                    current_size = self.active_positions[position_id]['size']
                    new_size = current_size * (1 - percentage)
                    self.active_positions[position_id]['size'] = new_size
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {e}")
            return False
    
    def modify_stop_loss(self, position_id: str, new_stop_loss: float) -> bool:
        """
        Modify stop loss for position
        Replace with your actual implementation
        """
        try:
            self.logger.info(f"Modifying stop loss for {position_id} to {new_stop_loss}")
            
            # Your actual stop loss modification logic here
            if position_id in self.active_positions:
                self.active_positions[position_id]['stop_loss'] = new_stop_loss
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error modifying stop loss for {position_id}: {e}")
            return False
    
    def get_position_performance(self) -> Dict:
        """Get combined performance summary"""
        traditional_performance = {
            'active_traditional_positions': len(self.active_positions),
            'execution_history_count': len(self.execution_history)
        }
        
        if self.position_manager:
            dynamic_performance = self.position_manager.get_performance_summary()
            return {**traditional_performance, **dynamic_performance}
        
        return traditional_performance

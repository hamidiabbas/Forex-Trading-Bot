"""
Simplified RL Model Manager without stable-baselines3
"""
import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SimpleRLModelManager:
    """
    Simplified RL Model Manager that returns random signals for testing
    """
    
    def __init__(self, config):
        self.config = config
        self.is_loaded = False
        self.confidence_threshold = 0.6
        
        logger.info("✅ Simple RL Model Manager initialized (using random signals for testing)")
    
    def load_model(self) -> bool:
        """Simulate model loading"""
        self.is_loaded = True
        logger.info("✅ Simple RL model 'loaded' (simulated)")
        return True
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate random signal for testing"""
        try:
            if not self.is_loaded:
                return None
            
            # Generate random signal
            np.random.seed(hash(symbol + str(int(datetime.now().timestamp()))) % 2**32)
            
            action = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])
            confidence = np.random.uniform(0.5, 0.9)
            
            if action == 'HOLD' or confidence < self.confidence_threshold:
                return None
            
            current_price = market_data.get('current_price', 1.0)
            atr = market_data.get('atr', current_price * 0.01)
            
            if action == 'BUY':
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            else:
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
            
            signal = {
                'symbol': symbol,
                'direction': action,
                'confidence': confidence,
                'strategy': 'RL-Simple',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now()
            }
            
            logger.debug(f"Simple RL signal: {symbol} {action} (confidence: {confidence:.2f})")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating simple RL signal: {e}")
            return None
    
    def update_model_performance(self, signal: Dict[str, Any], trade_result: Dict[str, Any]) -> None:
        """Log model performance (simplified)"""
        logger.debug(f"Simple RL performance update: {signal.get('symbol')} {signal.get('direction')}")
    
    def shutdown(self) -> None:
        """Simple shutdown"""
        logger.info("✅ Simple RL Model Manager shutdown completed")

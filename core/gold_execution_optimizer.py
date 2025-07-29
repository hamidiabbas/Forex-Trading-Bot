"""
XAUUSD (Gold) Execution Optimizer
Specialized handling for gold trading with proper slippage and parameters
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class GoldExecutionOptimizer:
    """
    Specialized execution optimizer for XAUUSD trading
    """
    
    def __init__(self):
        self.symbol_specs = {
            'XAUUSD': {
                'min_slippage': 50,           # 50 points minimum slippage
                'max_slippage': 200,          # 200 points maximum slippage
                'min_stop_distance': 500,     # $5.00 minimum stop distance
                'min_tp_distance': 1000,      # $10.00 minimum take profit distance
                'max_retry_attempts': 5,      # More attempts for gold
                'retry_delay': 2,             # 2 seconds between retries
                'spread_tolerance': 100,      # $1.00 maximum spread tolerance
                'volatility_buffer': 1.5,     # 1.5x volatility buffer
                'requote_slippage_increase': 25  # Increase slippage by 25 points per requote
            }
        }
        
        logger.info("✅ Gold Execution Optimizer initialized")
    
    def get_optimized_slippage(self, symbol: str, attempt: int = 1, base_slippage: int = 20) -> int:
        """Calculate optimized slippage for symbol and attempt"""
        if symbol != 'XAUUSD':
            return base_slippage
        
        specs = self.symbol_specs['XAUUSD']
        
        # Base slippage for gold
        optimized_slippage = specs['min_slippage']
        
        # Increase slippage with each retry attempt
        retry_increase = (attempt - 1) * specs['requote_slippage_increase']
        optimized_slippage += retry_increase
        
        # Cap at maximum slippage
        optimized_slippage = min(optimized_slippage, specs['max_slippage'])
        
        logger.info(f"XAUUSD slippage for attempt {attempt}: {optimized_slippage} points")
        return optimized_slippage
    
    def validate_and_adjust_levels(self, symbol: str, entry_price: float, 
                                  stop_loss: float, take_profit: float) -> Dict[str, float]:
        """Validate and adjust stop loss and take profit levels for gold"""
        if symbol != 'XAUUSD':
            return {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'adjusted': False
            }
        
        specs = self.symbol_specs['XAUUSD']
        min_stop_dist = specs['min_stop_distance'] / 100  # Convert points to price
        min_tp_dist = specs['min_tp_distance'] / 100
        
        adjusted = False
        original_sl = stop_loss
        original_tp = take_profit
        
        # Adjust stop loss if too close
        if stop_loss:
            current_sl_distance = abs(entry_price - stop_loss)
            if current_sl_distance < min_stop_dist:
                if entry_price > stop_loss:  # Long position
                    stop_loss = entry_price - min_stop_dist
                else:  # Short position
                    stop_loss = entry_price + min_stop_dist
                adjusted = True
                logger.info(f"XAUUSD stop loss adjusted: {original_sl:.5f} → {stop_loss:.5f}")
        
        # Adjust take profit if too close
        if take_profit:
            current_tp_distance = abs(take_profit - entry_price)
            if current_tp_distance < min_tp_dist:
                if take_profit > entry_price:  # Long position
                    take_profit = entry_price + min_tp_dist
                else:  # Short position
                    take_profit = entry_price - min_tp_dist
                adjusted = True
                logger.info(f"XAUUSD take profit adjusted: {original_tp:.5f} → {take_profit:.5f}")
        
        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'adjusted': adjusted
        }
    
    def should_skip_trade(self, symbol: str, current_spread: float, market_volatility: float = 0.01) -> bool:
        """Determine if trade should be skipped due to poor market conditions"""
        if symbol != 'XAUUSD':
            return False
        
        specs = self.symbol_specs['XAUUSD']
        max_spread = specs['spread_tolerance'] / 100  # Convert to price
        
        # Skip if spread is too wide
        if current_spread > max_spread:
            logger.warning(f"XAUUSD trade skipped - spread too wide: ${current_spread:.2f} > ${max_spread:.2f}")
            return True
        
        # Skip if volatility is extremely high
        if market_volatility > 0.05:  # 5% volatility threshold
            logger.warning(f"XAUUSD trade skipped - volatility too high: {market_volatility:.3f}")
            return True
        
        return False
    
    def get_execution_parameters(self, symbol: str, attempt: int = 1) -> Dict[str, Any]:
        """Get comprehensive execution parameters for symbol"""
        if symbol != 'XAUUSD':
            return {
                'slippage': 20,
                'max_attempts': 3,
                'retry_delay': 1,
                'fill_policy': 'IOC'
            }
        
        specs = self.symbol_specs['XAUUSD']
        
        return {
            'slippage': self.get_optimized_slippage(symbol, attempt),
            'max_attempts': specs['max_retry_attempts'],
            'retry_delay': specs['retry_delay'],
            'fill_policy': 'FOK',  # Fill or Kill for gold
            'price_deviation': self.get_optimized_slippage(symbol, attempt),
            'execution_timeout': 10,  # 10 seconds timeout for gold
            'market_execution': True  # Use market execution for gold
        }

# Global instance
gold_optimizer = GoldExecutionOptimizer()

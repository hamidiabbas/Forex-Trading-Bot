# strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class SignalStrength(Enum):
    WEAK = 0.3
    MODERATE = 0.6
    STRONG = 0.9

@dataclass
class TradingSignal:
    """Standardized signal format for all strategies"""
    symbol: str
    direction: str  # 'BUY', 'SELL', 'HOLD'
    strategy_type: str
    confidence: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    metadata: Dict[str, Any] = None
    timestamp: str = None

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.is_active = True
        self.performance_history = []
    
    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate trading signal from market data"""
        pass
    
    @abstractmethod
    def update_performance(self, signal: TradingSignal, outcome: Dict[str, Any]):
        """Update strategy performance metrics"""
        pass
    
    @abstractmethod
    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate signal before execution"""
        pass

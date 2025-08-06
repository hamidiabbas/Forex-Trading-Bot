# multi_agent_rl_system.py - Complete Professional Implementation

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import time
from collections import deque
import asyncio
from enum import Enum

from stable_baselines3 import SAC, PPO, A2C, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime types"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"

@dataclass
class AgentConfig:
    """Configuration for individual trading agents"""
    name: str
    strategy_type: str  # 'scalping', 'swing', 'trend', 'arbitrage'
    timeframe: str  # 'M1', 'M5', 'H1', 'H4', 'D1'
    risk_tolerance: float  # 0.0 to 1.0
    learning_rate: float
    batch_size: int
    memory_size: int
    update_frequency: int
    target_return: float
    max_drawdown: float
    position_size_limits: Tuple[float, float]  # (min, max)

@dataclass  
class TradingSignal:
    """Structured trading signal from agents"""
    agent_name: str
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: float
    reasoning: str
    timestamp: pd.Timestamp
    timeframe: str
    risk_reward_ratio: float

@dataclass
class MarketState:
    """Current market state information"""
    symbol: str
    current_price: float
    volatility: float
    trend_strength: float
    volume: float
    regime: MarketRegime
    technical_indicators: Dict[str, float]
    sentiment_score: float
    timestamp: pd.Timestamp

class BaseAgent(ABC):
    """Abstract base class for trading agents"""[1]
    
    def __init__(self, config: AgentConfig, environment):
        self.config = config
        self.environment = environment
        self.performance_history = deque(maxlen=1000)
        self.trade_history = []
        self.current_position = 0.0
        self.total_return = 0.0
        self.win_rate = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        
        # Agent state
        self.is_training = True
        self.is_active = True
        self.last_update = pd.Timestamp.now()
        self.expertise_score = 1.0
        
        logger.info(f"Initialized {self.config.name} agent")
    
    @abstractmethod
    def predict(self, observation: np.ndarray, market_state: MarketState) -> TradingSignal:
        """Generate trading signal from observation"""
        pass
    
    @abstractmethod
    def train(self, experiences: List[Dict]) -> Dict[str, float]:
        """Train the agent on experiences"""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> bool:
        """Save agent model"""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> bool:
        """Load agent model"""
        pass
    
    def update_performance(self, reward: float, action: str):
        """Update agent performance metrics"""
        self.performance_history.append(reward)
        
        if len(self.performance_history) >= 100:
            recent_returns = list(self.performance_history)[-100:]
            self.total_return = sum(recent_returns)
            self.sharpe_ratio = np.mean(recent_returns) / (np.std(recent_returns) + 1e-10)
            
            # Calculate win rate
            winning_trades = [r for r in recent_returns if r > 0]
            self.win_rate = len(winning_trades) / len(recent_returns)
            
            # Update expertise score based on recent performance
            self.expertise_score = min(max(0.1, self.sharpe_ratio + 1.0), 3.0)

    def get_agent_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            'name': self.config.name,
            'strategy_type': self.config.strategy_type,
            'total_return': self.total_return,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'expertise_score': self.expertise_score,
            'is_active': self.is_active,
            'last_update': self.last_update
        }

class ScalpingAgent(BaseAgent):
    """High-frequency scalping agent using SAC"""[2]
    
    def __init__(self, config: AgentConfig, environment):
        super().__init__(config, environment)
        
        # SAC model for continuous action space
        self.model = SAC(
            'MlpPolicy',
            environment,
            learning_rate=config.learning_rate,
            buffer_size=config.memory_size,
            learning_starts=1000,
            batch_size=config.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=config.update_frequency,
            gradient_steps=1,
            target_update_interval=1,
            use_sde=False,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0
        )
        
        # Scalping-specific parameters
        self.min_profit_pips = 2.0
        self.max_holding_time = 300  # 5 minutes max
        self.entry_signals = deque(maxlen=20)
        self.market_microstructure_features = {}
        
    def predict(self, observation: np.ndarray, market_state: MarketState) -> TradingSignal:
        """Generate scalping signal based on market microstructure"""
        
        try:
            # Analyze market microstructure
            microstructure_score = self._analyze_market_microstructure(market_state)
            
            # Get model prediction
            action, _states = self.model.predict(observation, deterministic=False)
            
            # Convert continuous action to discrete trading decision
            position_change = float(action[0])  # -1 to 1
            base_confidence = min(abs(position_change), 1.0)
            
            # Adjust confidence based on market conditions
            if market_state.regime == MarketRegime.VOLATILE:
                confidence = base_confidence * 0.7  # Reduce in volatile markets
            elif market_state.regime == MarketRegime.CALM:
                confidence = base_confidence * 1.2  # Increase in calm markets
            else:
                confidence = base_confidence
                
            # Apply microstructure filter
            confidence *= microstructure_score
            confidence = min(confidence, 1.0)
            
            # Determine action
            if position_change > 0.1 and confidence > 0.6:
                trading_action = 'BUY'
            elif position_change < -0.1 and confidence > 0.6:
                trading_action = 'SELL'
            else:
                trading_action = 'HOLD'
            
            # Dynamic position sizing
            base_size = self.config.risk_tolerance * 0.01
            volatility_adjustment = 1.0 / (1.0 + market_state.volatility)
            position_size = base_size * confidence * volatility_adjustment
            
            # Ensure within limits
            position_size = np.clip(position_size, 
                                  self.config.position_size_limits[0],
                                  self.config.position_size_limits[1])
            
            # Current price
            current_price = market_state.current_price
            
            # Set tight stops for scalping
            spread_estimate = current_price * 0.00005  # 0.5 pips
            if trading_action == 'BUY':
                stop_loss = current_price - (self.min_profit_pips * 0.0001 + spread_estimate)
                take_profit = current_price + (self.min_profit_pips * 2 * 0.0001)
            elif trading_action == 'SELL':
                stop_loss = current_price + (self.min_profit_pips * 0.0001 + spread_estimate)
                take_profit = current_price - (self.min_profit_pips * 2 * 0.0001)
            else:
                stop_loss = None
                take_profit = None
            
            # Calculate risk-reward ratio
            if stop_loss and take_profit:
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                risk_reward_ratio = reward / (risk + 1e-10)
            else:
                risk_reward_ratio = 0.0
            
            signal = TradingSignal(
                agent_name=self.config.name,
                symbol=market_state.symbol,
                action=trading_action,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=f"SAC scalping: action={position_change:.3f}, confidence={confidence:.3f}, microstructure={microstructure_score:.3f}",
                timestamp=market_state.timestamp,
                timeframe=self.config.timeframe,
                risk_reward_ratio=risk_reward_ratio
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in {self.config.name} prediction: {e}")
            return self._get_neutral_signal(market_state)
    
    def _analyze_market_microstructure(self, market_state: MarketState) -> float:
        """Analyze market microstructure for scalping opportunities"""
        
        try:
            score = 1.0
            
            # Volume analysis
            if market_state.volume > 0:
                volume_score = min(market_state.volume / 1000000, 2.0)  # Normalize volume
                score *= (0.5 + 0.5 * volume_score)
            
            # Volatility filter
            if market_state.volatility < 0.001:  # Too low volatility
                score *= 0.3
            elif market_state.volatility > 0.01:  # Too high volatility
                score *= 0.5
            
            # Technical indicators
            rsi = market_state.technical_indicators.get('rsi', 50)
            if 30 < rsi < 70:  # Avoid oversold/overbought
                score *= 1.0
            else:
                score *= 0.6
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error analyzing microstructure: {e}")
            return 0.5
    
    def train(self, experiences: List[Dict]) -> Dict[str, float]:
        """Train SAC model"""
        
        if not self.is_training or len(experiences) < self.config.batch_size:
            return {}
        
        try:
            # Train for a few steps
            info = self.model.learn(total_timesteps=self.config.update_frequency, 
                                  reset_num_timesteps=False)
            
            return {
                'loss': 0.0,
                'q_value': 0.0,
                'policy_loss': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error training {self.config.name}: {e}")
            return {}
    
    def save_model(self, path: str) -> bool:
        """Save SAC model"""
        try:
            self.model.save(f"{path}/{self.config.name}")
            return True
        except Exception as e:
            logger.error(f"Error saving {self.config.name}: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load SAC model"""
        try:
            self.model = SAC.load(f"{path}/{self.config.name}")
            return True
        except Exception as e:
            logger.error(f"Error loading {self.config.name}: {e}")
            return False
    
    def _get_neutral_signal(self, market_state: MarketState) -> TradingSignal:
        """Return neutral/hold signal"""
        return TradingSignal(
            agent_name=self.config.name,
            symbol=market_state.symbol,
            action='HOLD',
            confidence=0.0,
            entry_price=market_state.current_price,
            stop_loss=None,
            take_profit=None,
            position_size=0.0,
            reasoning="Error or neutral signal",
            timestamp=market_state.timestamp,
            timeframe=self.config.timeframe,
            risk_reward_ratio=0.0
        )

class SwingAgent(BaseAgent):
    """Medium-term swing trading agent using PPO"""[2]
    
    def __init__(self, config: AgentConfig, environment):
        super().__init__(config, environment)
        
        # PPO model for policy optimization
        self.model = PPO(
            'MlpPolicy',
            environment,
            learning_rate=config.learning_rate,
            n_steps=2048,
            batch_size=config.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[512, 512]),
            verbose=0
        )
        
        # Swing trading parameters
        self.min_profit_target = 50  # 50 pips minimum
        self.max_holding_time = 24 * 60 * 60  # 24 hours max
        self.trend_confirmation_periods = 3
        self.support_resistance_levels = []
        
    def predict(self, observation: np.ndarray, market_state: MarketState) -> TradingSignal:
        """Generate swing trading signal"""
        
        try:
            # Analyze swing trading conditions
            swing_score = self._analyze_swing_conditions(market_state)
            
            # Get model prediction
            action, _states = self.model.predict(observation, deterministic=True)
            
            # PPO typically outputs discrete actions
            action_value = int(action[0]) if hasattr(action, '__getitem__') else int(action)
            
            # Map to trading actions (0: HOLD, 1: BUY, 2: SELL)
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            trading_action = action_map.get(action_value, 'HOLD')
            
            # Calculate confidence based on trend strength and swing conditions
            trend_confidence = min(abs(market_state.trend_strength), 1.0)
            base_confidence = trend_confidence * swing_score
            
            # Adjust for market regime
            if market_state.regime == MarketRegime.TRENDING:
                confidence = base_confidence * 1.3
            elif market_state.regime == MarketRegime.RANGING:
                confidence = base_confidence * 0.7
            else:
                confidence = base_confidence
                
            confidence = min(confidence, 1.0)
            
            # Filter weak signals
            if confidence < 0.5:
                trading_action = 'HOLD'
                confidence = 0.0
            
            # Position sizing for swing trades
            base_size = self.config.risk_tolerance * 0.02
            sentiment_adjustment = 1.0 + (market_state.sentiment_score * 0.2)
            position_size = base_size * confidence * sentiment_adjustment
            position_size = np.clip(position_size,
                                  self.config.position_size_limits[0],
                                  self.config.position_size_limits[1])
            
            # Current price
            current_price = market_state.current_price
            
            # Set wider stops for swing trading
            atr_estimate = market_state.volatility * current_price * 20  # Approximate ATR
            stop_distance = max(atr_estimate * 2, self.min_profit_target * 0.0001)
            
            if trading_action == 'BUY':
                stop_loss = current_price - stop_distance
                take_profit = current_price + (stop_distance * 2)  # 1:2 risk-reward
            elif trading_action == 'SELL':
                stop_loss = current_price + stop_distance
                take_profit = current_price - (stop_distance * 2)
            else:
                stop_loss = None
                take_profit = None
            
            # Risk-reward calculation
            if stop_loss and take_profit:
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                risk_reward_ratio = reward / (risk + 1e-10)
            else:
                risk_reward_ratio = 0.0
            
            signal = TradingSignal(
                agent_name=self.config.name,
                symbol=market_state.symbol,
                action=trading_action,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=f"PPO swing: action={action_value}, trend={market_state.trend_strength:.3f}, swing_score={swing_score:.3f}",
                timestamp=market_state.timestamp,
                timeframe=self.config.timeframe,
                risk_reward_ratio=risk_reward_ratio
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in {self.config.name} prediction: {e}")
            return self._get_neutral_signal(market_state)
    
    def _analyze_swing_conditions(self, market_state: MarketState) -> float:
        """Analyze conditions for swing trading"""
        
        try:
            score = 1.0
            
            # Trend strength analysis
            if abs(market_state.trend_strength) > 0.7:
                score *= 1.2
            elif abs(market_state.trend_strength) < 0.3:
                score *= 0.6
            
            # Volatility filter
            if 0.005 < market_state.volatility < 0.02:
                score *= 1.0
            else:
                score *= 0.7
            
            # Technical indicators
            macd = market_state.technical_indicators.get('macd', 0)
            rsi = market_state.technical_indicators.get('rsi', 50)
            
            # MACD momentum
            if abs(macd) > 0.001:
                score *= 1.1
            
            # RSI divergence opportunities
            if 25 < rsi < 75:
                score *= 1.0
            else:
                score *= 1.2  # Potential reversal
            
            return min(score, 2.0)
            
        except Exception as e:
            logger.warning(f"Error analyzing swing conditions: {e}")
            return 0.5
    
    def train(self, experiences: List[Dict]) -> Dict[str, float]:
        """Train PPO model"""
        
        if not self.is_training:
            return {}
        
        try:
            info = self.model.learn(total_timesteps=self.config.update_frequency,
                                  reset_num_timesteps=False)
            
            return {
                'loss': 0.0,
                'value_loss': 0.0,
                'policy_loss': 0.0,
                'explained_variance': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error training {self.config.name}: {e}")
            return {}
    
    def save_model(self, path: str) -> bool:
        """Save PPO model"""
        try:
            self.model.save(f"{path}/{self.config.name}")
            return True
        except Exception as e:
            logger.error(f"Error saving {self.config.name}: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load PPO model"""
        try:
            self.model = PPO.load(f"{path}/{self.config.name}")
            return True
        except Exception as e:
            logger.error(f"Error loading {self.config.name}: {e}")
            return False
    
    def _get_neutral_signal(self, market_state: MarketState) -> TradingSignal:
        """Return neutral signal"""
        return TradingSignal(
            agent_name=self.config.name,
            symbol=market_state.symbol,
            action='HOLD',
            confidence=0.0,
            entry_price=market_state.current_price,
            stop_loss=None,
            take_profit=None,
            position_size=0.0,
            reasoning="Error or neutral signal",
            timestamp=market_state.timestamp,
            timeframe=self.config.timeframe,
            risk_reward_ratio=0.0
        )

class TrendAgent(BaseAgent):
    """Long-term trend following agent using A2C"""[2]
    
    def __init__(self, config: AgentConfig, environment):
        super().__init__(config, environment)
        
        # A2C model for advantage actor-critic
        self.model = A2C(
            'MlpPolicy',
            environment,
            learning_rate=config.learning_rate,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.25,
            max_grad_norm=0.5,
            rms_prop_eps=1e-5,
            use_rms_prop=True,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0
        )
        
        # Trend following parameters
        self.min_trend_strength = 0.7
        self.trend_confirmation_bars = 10
        self.max_position_hold = 7 * 24 * 60 * 60  # 7 days
        self.trend_history = deque(maxlen=100)
        
    def predict(self, observation: np.ndarray, market_state: MarketState) -> TradingSignal:
        """Generate trend following signal"""
        
        try:
            # Analyze trend conditions
            trend_score = self._analyze_trend_conditions(market_state)
            
            # Get model prediction
            action, _states = self.model.predict(observation, deterministic=True)
            
            # A2C outputs discrete actions
            action_value = int(action[0]) if hasattr(action, '__getitem__') else int(action)
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            trading_action = action_map.get(action_value, 'HOLD')
            
            # Calculate confidence based on trend strength
            trend_strength = abs(market_state.trend_strength)
            base_confidence = trend_strength * trend_score
            
            # Adjust for market regime
            if market_state.regime == MarketRegime.TRENDING:
                confidence = base_confidence * 1.5
            elif market_state.regime == MarketRegime.RANGING:
                confidence = base_confidence * 0.3
            else:
                confidence = base_confidence
            
            confidence = min(confidence, 1.0)
            
            # Only trade on strong trends
            if confidence < self.min_trend_strength:
                trading_action = 'HOLD'
                confidence = 0.0
            
            # Position sizing for trend following
            base_size = self.config.risk_tolerance * 0.05
            momentum_adjustment = 1.0 + min(trend_strength, 0.5)
            position_size = base_size * confidence * momentum_adjustment
            position_size = np.clip(position_size,
                                  self.config.position_size_limits[0],
                                  self.config.position_size_limits[1])
            
            # Current price
            current_price = market_state.current_price
            
            # Wide stops for trend following
            atr_estimate = market_state.volatility * current_price * 20
            stop_distance = max(atr_estimate * 3, 100 * 0.0001)  # 3 ATR or 100 pips
            
            if trading_action == 'BUY':
                stop_loss = current_price - stop_distance
                take_profit = current_price + (stop_distance * 3)  # 1:3 risk-reward
            elif trading_action == 'SELL':
                stop_loss = current_price + stop_distance
                take_profit = current_price - (stop_distance * 3)
            else:
                stop_loss = None
                take_profit = None
            
            # Risk-reward calculation
            if stop_loss and take_profit:
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                risk_reward_ratio = reward / (risk + 1e-10)
            else:
                risk_reward_ratio = 0.0
            
            signal = TradingSignal(
                agent_name=self.config.name,
                symbol=market_state.symbol,
                action=trading_action,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=f"A2C trend: strength={trend_strength:.3f}, score={trend_score:.3f}, regime={market_state.regime}",
                timestamp=market_state.timestamp,
                timeframe=self.config.timeframe,
                risk_reward_ratio=risk_reward_ratio
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in {self.config.name} prediction: {e}")
            return self._get_neutral_signal(market_state)
    
    def _analyze_trend_conditions(self, market_state: MarketState) -> float:
        """Analyze trend following conditions"""
        
        try:
            score = 1.0
            
            # Add current trend to history
            self.trend_history.append(market_state.trend_strength)
            
            # Trend persistence
            if len(self.trend_history) >= 10:
                recent_trends = list(self.trend_history)[-10:]
                trend_consistency = sum(1 for t in recent_trends if t * market_state.trend_strength > 0) / len(recent_trends)
                score *= (0.5 + trend_consistency)
            
            # Trend acceleration
            if len(self.trend_history) >= 5:
                recent_trend = np.mean(list(self.trend_history)[-5:])
                older_trend = np.mean(list(self.trend_history)[-10:-5]) if len(self.trend_history) >= 10 else recent_trend
                
                if abs(recent_trend) > abs(older_trend):
                    score *= 1.2  # Accelerating trend
            
            # Volume confirmation
            if market_state.volume > 0:
                # Higher volume should support trend
                volume_support = min(market_state.volume / 500000, 1.5)
                score *= volume_support
            
            # Technical indicators
            adx = market_state.technical_indicators.get('adx', 25)
            if adx > 30:  # Strong trend
                score *= 1.3
            elif adx < 20:  # Weak trend
                score *= 0.5
            
            return min(score, 2.0)
            
        except Exception as e:
            logger.warning(f"Error analyzing trend conditions: {e}")
            return 0.5
    
    def train(self, experiences: List[Dict]) -> Dict[str, float]:
        """Train A2C model"""
        
        if not self.is_training:
            return {}
        
        try:
            info = self.model.learn(total_timesteps=self.config.update_frequency,
                                  reset_num_timesteps=False)
            
            return {
                'loss': 0.0,
                'value_loss': 0.0,
                'policy_loss': 0.0,
                'entropy_loss': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error training {self.config.name}: {e}")
            return {}
    
    def save_model(self, path: str) -> bool:
        """Save A2C model"""
        try:
            self.model.save(f"{path}/{self.config.name}")
            return True
        except Exception as e:
            logger.error(f"Error saving {self.config.name}: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load A2C model"""
        try:
            self.model = A2C.load(f"{path}/{self.config.name}")
            return True
        except Exception as e:
            logger.error(f"Error loading {self.config.name}: {e}")
            return False
    
    def _get_neutral_signal(self, market_state: MarketState) -> TradingSignal:
        """Return neutral signal"""
        return TradingSignal(
            agent_name=self.config.name,
            symbol=market_state.symbol,
            action='HOLD',
            confidence=0.0,
            entry_price=market_state.current_price,
            stop_loss=None,
            take_profit=None,
            position_size=0.0,
            reasoning="Error or neutral signal",
            timestamp=market_state.timestamp,
            timeframe=self.config.timeframe,
            risk_reward_ratio=0.0
        )

class ArbitrageAgent(BaseAgent):
    """Arbitrage opportunities detection agent"""[2]
    
    def __init__(self, config: AgentConfig, environment):
        super().__init__(config, environment)
        
        # TD3 for continuous control
        self.model = TD3(
            'MlpPolicy',
            environment,
            learning_rate=config.learning_rate,
            buffer_size=config.memory_size,
            learning_starts=1000,
            batch_size=config.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs=dict(net_arch=[400, 300]),
            verbose=0
        )
        
        # Arbitrage parameters
        self.min_spread = 0.0001  # 1 pip minimum
        self.execution_delay = 0.1  # 100ms execution time
        self.correlation_threshold = 0.8
        self.price_feeds = {}
        
    def predict(self, observation: np.ndarray, market_state: MarketState) -> TradingSignal:
        """Generate arbitrage signal"""
        
        try:
            # Look for arbitrage opportunities
            arbitrage_score = self._detect_arbitrage_opportunities(market_state)
            
            if arbitrage_score < 0.8:  # No significant arbitrage
                return self._get_neutral_signal(market_state)
            
            # Get model prediction
            action, _states = self.model.predict(observation, deterministic=True)
            
            position_change = float(action[0])
            confidence = min(abs(position_change) * arbitrage_score, 1.0)
            
            # Determine action
            if position_change > 0.1:
                trading_action = 'BUY'
            elif position_change < -0.1:
                trading_action = 'SELL'
            else:
                trading_action = 'HOLD'
            
            # High-frequency position sizing
            base_size = self.config.risk_tolerance * 0.005  # Small positions
            position_size = base_size * confidence * arbitrage_score
            position_size = np.clip(position_size,
                                  self.config.position_size_limits[0],
                                  self.config.position_size_limits[1])
            
            # Tight stops for arbitrage
            current_price = market_state.current_price
            stop_distance = self.min_spread * 2
            
            if trading_action == 'BUY':
                stop_loss = current_price - stop_distance
                take_profit = current_price + stop_distance
            elif trading_action == 'SELL':
                stop_loss = current_price + stop_distance
                take_profit = current_price - stop_distance
            else:
                stop_loss = None
                take_profit = None
            
            # Risk-reward calculation
            risk_reward_ratio = 1.0 if stop_loss and take_profit else 0.0
            
            signal = TradingSignal(
                agent_name=self.config.name,
                symbol=market_state.symbol,
                action=trading_action,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=f"TD3 arbitrage: score={arbitrage_score:.3f}, action={position_change:.3f}",
                timestamp=market_state.timestamp,
                timeframe=self.config.timeframe,
                risk_reward_ratio=risk_reward_ratio
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in {self.config.name} prediction: {e}")
            return self._get_neutral_signal(market_state)
    
    def _detect_arbitrage_opportunities(self, market_state: MarketState) -> float:
        """Detect arbitrage opportunities"""
        
        try:
            # Store current price
            self.price_feeds[market_state.timestamp] = market_state.current_price
            
            # Clean old prices (keep last 100)
            if len(self.price_feeds) > 100:
                oldest_key = min(self.price_feeds.keys())
                del self.price_feeds[oldest_key]
            
            if len(self.price_feeds) < 10:
                return 0.0
            
            # Calculate price spread and momentum
            prices = list(self.price_feeds.values())
            recent_prices = prices[-5:]
            
            # Price momentum deviation
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            
            # Arbitrage score based on price inefficiencies
            score = 0.0
            
            # Momentum-based opportunities
            if abs(momentum) > volatility * 2:
                score += 0.4
            
            # Volatility opportunities
            if volatility > 0.001:  # Sufficient movement
                score += 0.3
            
            # Time-based opportunities (market opening/closing)
            hour = market_state.timestamp.hour
            if hour in [8, 9, 15, 16, 21, 22]:  # Major session opens/closes
                score += 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error detecting arbitrage: {e}")
            return 0.0
    
    def train(self, experiences: List[Dict]) -> Dict[str, float]:
        """Train TD3 model"""
        
        if not self.is_training:
            return {}
        
        try:
            info = self.model.learn(total_timesteps=self.config.update_frequency,
                                  reset_num_timesteps=False)
            
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error training {self.config.name}: {e}")
            return {}
    
    def save_model(self, path: str) -> bool:
        """Save TD3 model"""
        try:
            self.model.save(f"{path}/{self.config.name}")
            return True
        except Exception as e:
            logger.error(f"Error saving {self.config.name}: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load TD3 model"""
        try:
            self.model = TD3.load(f"{path}/{self.config.name}")
            return True
        except Exception as e:
            logger.error(f"Error loading {self.config.name}: {e}")
            return False
    
    def _get_neutral_signal(self, market_state: MarketState) -> TradingSignal:
        """Return neutral signal"""
        return TradingSignal(
            agent_name=self.config.name,
            symbol=market_state.symbol,
            action='HOLD',
            confidence=0.0,
            entry_price=market_state.current_price,
            stop_loss=None,
            take_profit=None,
            position_size=0.0,
            reasoning="No arbitrage opportunity",
            timestamp=market_state.timestamp,
            timeframe=self.config.timeframe,
            risk_reward_ratio=0.0
        )

class CoordinatorAgent:
    """Central coordinator for managing multiple trading agents"""[1]
    
    def __init__(self, config: dict):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.signal_queue = queue.Queue()
        self.performance_tracker = {}
        self.coordination_weights = {}
        self.consensus_threshold = getattr(config,'consensus_threshold', 0.6)
        self.max_portfolio_risk = getattr(config,'max_portfolio_risk', 0.1)
        
        # Coordination strategies
        self.coordination_strategy = getattr(config,'coordination_strategy', 'weighted_voting')
        self.risk_manager = PortfolioRiskManager(config)
        
        # Performance tracking
        self.total_signals = 0
        self.successful_signals = 0
        self.coordination_history = deque(maxlen=1000)
        self.active_positions = {}
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector()
        
        logger.info("Multi-Agent Coordinator initialized")
    
    def add_agent(self, agent: BaseAgent):
        """Add a trading agent to the system"""
        self.agents[agent.config.name] = agent
        self.coordination_weights[agent.config.name] = 1.0 / (len(self.agents) or 1)
        self.performance_tracker[agent.config.name] = {
            'signals_generated': 0,
            'successful_trades': 0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }
        
        logger.info(f"Added agent: {agent.config.name}")
    
    async def coordinate_decision(self, symbol: str, observation: np.ndarray, 
                                 market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Coordinate decisions from all agents"""
        
        try:
            # Create market state
            market_state = self._create_market_state(symbol, market_data)
            
            # Get signals from all active agents
            agent_signals = await self._collect_agent_signals(observation, market_state)
            
            if not agent_signals:
                return None
            
            # Apply coordination strategy
            coordinated_signal = self._apply_coordination_strategy(agent_signals, market_state)
            
            if not coordinated_signal:
                return None
            
            # Risk management check
            if not self.risk_manager.validate_signal(coordinated_signal, self.active_positions):
                logger.info(f"Signal rejected by risk manager: {coordinated_signal.reasoning}")
                return None
            
            # Update coordination history
            self._update_coordination_history(agent_signals, coordinated_signal)
            
            return coordinated_signal
            
        except Exception as e:
            logger.error(f"Error coordinating decision: {e}")
            return None
    
    async def _collect_agent_signals(self, observation: np.ndarray, 
                                    market_state: MarketState) -> List[TradingSignal]:
        """Collect signals from all active agents"""
        
        signals = []
        
        # Use ThreadPoolExecutor for parallel signal generation
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            future_to_agent = {
                executor.submit(agent.predict, observation, market_state): agent_name
                for agent_name, agent in self.agents.items() if agent.is_active
            }
            
            for future in as_completed(future_to_agent, timeout=5):
                agent_name = future_to_agent[future]
                try:
                    signal = future.result()
                    if signal and signal.action != 'HOLD':
                        signals.append(signal)
                        self.performance_tracker[agent_name]['signals_generated'] += 1
                except Exception as e:
                    logger.warning(f"Error getting signal from {agent_name}: {e}")
        
        return signals
    
    def _create_market_state(self, symbol: str, market_data: Dict[str, Any]) -> MarketState:
        """Create market state from raw market data"""
        
        try:
            # Extract basic market information
            current_price = market_data.get('close', 0.0)
            volume = market_data.get('volume', 0.0)
            
            # Calculate volatility (simple estimate)
            prices = market_data.get('prices', [current_price])
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns)
            else:
                volatility = 0.01
            
            # Calculate trend strength
            if len(prices) >= 20:
                trend_strength = self._calculate_trend_strength(prices)
            else:
                trend_strength = 0.0
            
            # Detect market regime
            regime = self.regime_detector.detect_regime(prices, volatility)
            
            # Extract technical indicators
            technical_indicators = market_data.get('indicators', {})
            
            # Extract sentiment
            sentiment_score = market_data.get('sentiment', 0.0)
            
            return MarketState(
                symbol=symbol,
                current_price=current_price,
                volatility=volatility,
                trend_strength=trend_strength,
                volume=volume,
                regime=regime,
                technical_indicators=technical_indicators,
                sentiment_score=sentiment_score,
                timestamp=pd.Timestamp.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating market state: {e}")
            return MarketState(
                symbol=symbol,
                current_price=0.0,
                volatility=0.01,
                trend_strength=0.0,
                volume=0.0,
                regime=MarketRegime.CALM,
                technical_indicators={},
                sentiment_score=0.0,
                timestamp=pd.Timestamp.now()
            )
    
    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength using linear regression"""
        
        try:
            x = np.arange(len(prices))
            y = np.array(prices)
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Normalize slope relative to price level
            normalized_slope = slope / np.mean(prices)
            
            # R-squared for trend strength
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Combine slope and R-squared
            trend_strength = normalized_slope * r_squared
            
            return np.clip(trend_strength, -1.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _apply_coordination_strategy(self, signals: List[TradingSignal], 
                                   market_state: MarketState) -> Optional[TradingSignal]:
        """Apply coordination strategy to combine agent signals"""
        
        if not signals:
            return None
        
        try:
            if self.coordination_strategy == 'weighted_voting':
                return self._weighted_voting_strategy(signals, market_state)
            elif self.coordination_strategy == 'consensus':
                return self._consensus_strategy(signals, market_state)
            elif self.coordination_strategy == 'expert_selection':
                return self._expert_selection_strategy(signals, market_state)
            elif self.coordination_strategy == 'risk_weighted':
                return self._risk_weighted_strategy(signals, market_state)
            else:
                return self._weighted_voting_strategy(signals, market_state)
                
        except Exception as e:
            logger.error(f"Error applying coordination strategy: {e}")
            return None
    
    def _weighted_voting_strategy(self, signals: List[TradingSignal], 
                                 market_state: MarketState) -> Optional[TradingSignal]:
        """Weighted voting coordination strategy"""
        
        if not signals:
            return None
        
        # Separate by action
        buy_signals = [s for s in signals if s.action == 'BUY']
        sell_signals = [s for s in signals if s.action == 'SELL']
        
        # Calculate weighted votes
        buy_weight = sum(s.confidence * self._get_agent_weight(s.agent_name) for s in buy_signals)
        sell_weight = sum(s.confidence * self._get_agent_weight(s.agent_name) for s in sell_signals)
        
        # Determine final action
        if buy_weight > sell_weight and buy_weight > self.consensus_threshold:
            final_action = 'BUY'
            relevant_signals = buy_signals
            final_confidence = min(buy_weight, 1.0)
        elif sell_weight > buy_weight and sell_weight > self.consensus_threshold:
            final_action = 'SELL'
            relevant_signals = sell_signals
            final_confidence = min(sell_weight, 1.0)
        else:
            return None  # No consensus
        
        # Combine signal parameters
        return self._combine_signals(relevant_signals, final_action, final_confidence, market_state)
    
    def _consensus_strategy(self, signals: List[TradingSignal], 
                           market_state: MarketState) -> Optional[TradingSignal]:
        """Consensus-based coordination strategy"""
        
        if len(signals) < 2:
            return None
        
        # Count actions
        actions = [s.action for s in signals]
        buy_count = actions.count('BUY')
        sell_count = actions.count('SELL')
        total_count = len(actions)
        
        # Require strong consensus (>60% agreement)
        if buy_count / total_count > 0.6:
            final_action = 'BUY'
            relevant_signals = [s for s in signals if s.action == 'BUY']
        elif sell_count / total_count > 0.6:
            final_action = 'SELL'
            relevant_signals = [s for s in signals if s.action == 'SELL']
        else:
            return None
        
        # Average confidence
        avg_confidence = np.mean([s.confidence for s in relevant_signals])
        
        return self._combine_signals(relevant_signals, final_action, avg_confidence, market_state)
    
    def _expert_selection_strategy(self, signals: List[TradingSignal], 
                                  market_state: MarketState) -> Optional[TradingSignal]:
        """Expert selection coordination strategy"""
        
        if not signals:
            return None
        
        # Find the agent with highest expertise score for current market regime
        best_signal = None
        best_score = 0.0
        
        for signal in signals:
            agent = self.agents.get(signal.agent_name)
            if not agent:
                continue
            
            # Calculate regime-specific expertise
            expertise_score = self._calculate_regime_expertise(agent, market_state)
            adjusted_score = expertise_score * signal.confidence
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_signal = signal
        
        return best_signal
    
    def _risk_weighted_strategy(self, signals: List[TradingSignal], 
                               market_state: MarketState) -> Optional[TradingSignal]:
        """Risk-weighted coordination strategy"""
        
        if not signals:
            return None
        
        # Filter signals by risk-reward ratio
        good_signals = [s for s in signals if s.risk_reward_ratio >= 1.5]
        
        if not good_signals:
            return None
        
        # Weight by inverse of risk and agent performance
        weighted_buy = 0.0
        weighted_sell = 0.0
        
        for signal in good_signals:
            agent_weight = self._get_agent_weight(signal.agent_name)
            risk_weight = min(signal.risk_reward_ratio / 2.0, 2.0)  # Cap at 2x
            total_weight = signal.confidence * agent_weight * risk_weight
            
            if signal.action == 'BUY':
                weighted_buy += total_weight
            elif signal.action == 'SELL':
                weighted_sell += total_weight
        
        # Choose action and combine signals
        if weighted_buy > weighted_sell and weighted_buy > 0.5:
            relevant_signals = [s for s in good_signals if s.action == 'BUY']
            return self._combine_signals(relevant_signals, 'BUY', min(weighted_buy, 1.0), market_state)
        elif weighted_sell > weighted_buy and weighted_sell > 0.5:
            relevant_signals = [s for s in good_signals if s.action == 'SELL']
            return self._combine_signals(relevant_signals, 'SELL', min(weighted_sell, 1.0), market_state)
        
        return None
    
    def _combine_signals(self, signals: List[TradingSignal], action: str, 
                        confidence: float, market_state: MarketState) -> TradingSignal:
        """Combine multiple signals into one coordinated signal"""
        
        if not signals:
            return None
        
        # Average position size (risk-adjusted)
        total_risk = sum(abs(s.entry_price - s.stop_loss) * s.position_size for s in signals if s.stop_loss)
        if total_risk > self.max_portfolio_risk:
            # Scale down position sizes
            scale_factor = self.max_portfolio_risk / total_risk
            avg_position_size = np.mean([s.position_size for s in signals]) * scale_factor
        else:
            avg_position_size = np.mean([s.position_size for s in signals])
        
        # Weighted average stops and targets
        weights = [s.confidence for s in signals]
        total_weight = sum(weights)
        
        if total_weight > 0:
            if action == 'BUY':
                avg_stop_loss = np.average([s.stop_loss for s in signals if s.stop_loss], weights=weights[:len([s for s in signals if s.stop_loss])])
                avg_take_profit = np.average([s.take_profit for s in signals if s.take_profit], weights=weights[:len([s for s in signals if s.take_profit])])
            else:
                avg_stop_loss = np.average([s.stop_loss for s in signals if s.stop_loss], weights=weights[:len([s for s in signals if s.stop_loss])])
                avg_take_profit = np.average([s.take_profit for s in signals if s.take_profit], weights=weights[:len([s for s in signals if s.take_profit])])
        else:
            avg_stop_loss = signals[0].stop_loss
            avg_take_profit = signals[0].take_profit
        
        # Risk-reward ratio
        if avg_stop_loss and avg_take_profit:
            risk = abs(market_state.current_price - avg_stop_loss)
            reward = abs(avg_take_profit - market_state.current_price)
            risk_reward_ratio = reward / (risk + 1e-10)
        else:
            risk_reward_ratio = 0.0
        
        # Combined reasoning
        agent_names = [s.agent_name for s in signals]
        reasoning = f"Coordinated signal from {len(signals)} agents: {', '.join(agent_names)} using {self.coordination_strategy}"
        
        return TradingSignal(
            agent_name="Coordinator",
            symbol=market_state.symbol,
            action=action,
            confidence=confidence,
            entry_price=market_state.current_price,
            stop_loss=avg_stop_loss,
            take_profit=avg_take_profit,
            position_size=avg_position_size,
            reasoning=reasoning,
            timestamp=market_state.timestamp,
            timeframe="MULTI",
            risk_reward_ratio=risk_reward_ratio
        )
    
    def _get_agent_weight(self, agent_name: str) -> float:
        """Get dynamic weight for agent based on recent performance"""
        
        agent = self.agents.get(agent_name)
        if not agent:
            return 0.1
        
        # Base weight
        base_weight = self.coordination_weights.get(agent_name, 1.0)
        
        # Performance adjustment
        performance_data = self.performance_tracker[agent_name]
        
        # Sharpe ratio adjustment
        sharpe_adjustment = 1.0 + min(max(agent.sharpe_ratio, -1.0), 2.0) * 0.2
        
        # Win rate adjustment
        win_rate_adjustment = 0.5 + agent.win_rate
        
        # Expertise score
        expertise_adjustment = agent.expertise_score / 2.0
        
        # Combined weight
        final_weight = base_weight * sharpe_adjustment * win_rate_adjustment * expertise_adjustment
        
        return max(0.1, min(final_weight, 3.0))
    
    def _calculate_regime_expertise(self, agent: BaseAgent, market_state: MarketState) -> float:
        """Calculate agent expertise for current market regime"""
        
        # Strategy-regime matching
        strategy_regime_mapping = {
            'scalping': {
                MarketRegime.VOLATILE: 1.2,
                MarketRegime.CALM: 0.8,
                MarketRegime.TRENDING: 0.9,
                MarketRegime.RANGING: 1.1
            },
            'swing': {
                MarketRegime.VOLATILE: 0.8,
                MarketRegime.CALM: 1.1,
                MarketRegime.TRENDING: 1.2,
                MarketRegime.RANGING: 1.0
            },
            'trend': {
                MarketRegime.VOLATILE: 0.7,
                MarketRegime.CALM: 0.9,
                MarketRegime.TRENDING: 1.5,
                MarketRegime.RANGING: 0.5
            },
            'arbitrage': {
                MarketRegime.VOLATILE: 1.3,
                MarketRegime.CALM: 0.7,
                MarketRegime.TRENDING: 0.8,
                MarketRegime.RANGING: 1.0
            }
        }
        
        strategy_type = agent.config.strategy_type
        regime_multiplier = strategy_regime_mapping.get(strategy_type, {}).get(market_state.regime, 1.0)
        
        return agent.expertise_score * regime_multiplier
    
    def _update_coordination_history(self, agent_signals: List[TradingSignal], 
                                   coordinated_signal: TradingSignal):
        """Update coordination history for analysis"""
        
        coordination_record = {
            'timestamp': coordinated_signal.timestamp,
            'agent_count': len(agent_signals),
            'agent_signals': [s.agent_name for s in agent_signals],
            'coordinated_action': coordinated_signal.action,
            'coordinated_confidence': coordinated_signal.confidence,
            'coordination_strategy': self.coordination_strategy
        }
        
        self.coordination_history.append(coordination_record)
    
    def update_agent_performance(self, agent_name: str, trade_result: Dict[str, Any]):
        """Update agent performance based on trade results"""
        
        if agent_name not in self.performance_tracker:
            return
        
        agent = self.agents.get(agent_name)
        if not agent:
            return
        
        # Update performance metrics
        profit_loss = trade_result.get('profit_loss', 0.0)
        agent.update_performance(profit_loss, trade_result.get('action', 'HOLD'))
        
        # Update tracker
        tracker = self.performance_tracker[agent_name]
        tracker['total_return'] += profit_loss
        if profit_loss > 0:
            tracker['successful_trades'] += 1
        
        # Update coordination weights
        self._rebalance_coordination_weights()
    
    def _rebalance_coordination_weights(self):
        """Rebalance coordination weights based on recent performance"""
        
        total_expertise = sum(agent.expertise_score for agent in self.agents.values())
        
        if total_expertise > 0:
            for agent_name, agent in self.agents.items():
                self.coordination_weights[agent_name] = agent.expertise_score / total_expertise
        else:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(self.agents)
            for agent_name in self.agents:
                self.coordination_weights[agent_name] = equal_weight
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'total_agents': len(self.agents),
            'active_agents': sum(1 for agent in self.agents.values() if agent.is_active),
            'coordination_strategy': self.coordination_strategy,
            'total_signals': self.total_signals,
            'successful_signals': self.successful_signals,
            'success_rate': self.successful_signals / max(self.total_signals, 1),
            'agent_weights': self.coordination_weights.copy(),
            'agent_performance': {name: agent.get_agent_state() for name, agent in self.agents.items()},
            'active_positions': len(self.active_positions),
            'last_update': pd.Timestamp.now()
        }

class PortfolioRiskManager:
    """Portfolio-level risk management"""[2]
    
    def __init__(self, config: dict):
        self.config = config
        self.max_portfolio_risk = getattr(config,'max_portfolio_risk', 0.1)
        self.max_single_position_risk = getattr(config,'max_single_position_risk', 0.02)
        self.max_correlation_risk = getattr(config,'max_correlation_risk', 0.8)
        self.position_history = deque(maxlen=1000)
        
    def validate_signal(self, signal: TradingSignal, active_positions: Dict) -> bool:
        """Validate signal against portfolio risk limits"""
        
        try:
            # Single position risk check
            if not self._check_single_position_risk(signal):
                return False
            
            # Portfolio risk check
            if not self._check_portfolio_risk(signal, active_positions):
                return False
            
            # Correlation risk check
            if not self._check_correlation_risk(signal, active_positions):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    def _check_single_position_risk(self, signal: TradingSignal) -> bool:
        """Check if single position risk is within limits"""
        
        if signal.stop_loss:
            risk_per_unit = abs(signal.entry_price - signal.stop_loss)
            total_risk = risk_per_unit * signal.position_size
            
            if total_risk > self.max_single_position_risk:
                logger.warning(f"Single position risk too high: {total_risk:.4f} > {self.max_single_position_risk}")
                return False
        
        return True
    
    def _check_portfolio_risk(self, signal: TradingSignal, active_positions: Dict) -> bool:
        """Check if portfolio risk is within limits"""
        
        # Calculate current portfolio risk
        current_risk = sum(pos.get('risk', 0) for pos in active_positions.values())
        
        # Add new signal risk
        if signal.stop_loss:
            signal_risk = abs(signal.entry_price - signal.stop_loss) * signal.position_size
            total_risk = current_risk + signal_risk
            
            if total_risk > self.max_portfolio_risk:
                logger.warning(f"Portfolio risk too high: {total_risk:.4f} > {self.max_portfolio_risk}")
                return False
        
        return True
    
    def _check_correlation_risk(self, signal: TradingSignal, active_positions: Dict) -> bool:
        """Check correlation risk between positions"""
        
        # Simple correlation check based on symbol similarity
        symbol_base = signal.symbol[:3]  # e.g., EUR from EURUSD
        
        correlated_positions = 0
        for pos_id, position in active_positions.items():
            if position.get('symbol', '')[:3] == symbol_base:
                correlated_positions += 1
        
        # Limit correlated positions
        if correlated_positions >= 3:  # Max 3 correlated positions
            logger.warning(f"Too many correlated positions for {symbol_base}")
            return False
        
        return True

class MarketRegimeDetector:
    """Detect current market regime"""[2]
    
    def __init__(self):
        self.regime_history = deque(maxlen=100)
        
    def detect_regime(self, prices: List[float], volatility: float) -> MarketRegime:
        """Detect current market regime"""
        
        try:
            if len(prices) < 20:
                return MarketRegime.CALM
            
            # Calculate trend strength
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            normalized_slope = abs(slope) / np.mean(prices)
            
            # Volatility threshold
            vol_threshold_low = 0.005
            vol_threshold_high = 0.02
            
            # Trend threshold
            trend_threshold = 0.001
            
            # Determine regime
            if volatility > vol_threshold_high:
                regime = MarketRegime.VOLATILE
            elif volatility < vol_threshold_low:
                regime = MarketRegime.CALM
            elif normalized_slope > trend_threshold:
                regime = MarketRegime.TRENDING
            else:
                regime = MarketRegime.RANGING
            
            # Store regime history
            self.regime_history.append(regime)
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.CALM

# Factory function to create agents
def create_agent(agent_type: str, config: AgentConfig, environment) -> BaseAgent:
    """Factory function to create different types of agents"""
    
    agent_classes = {
        'scalping': ScalpingAgent,
        'swing': SwingAgent,
        'trend': TrendAgent,
        'arbitrage': ArbitrageAgent
    }
    
    agent_class = agent_classes.get(agent_type)
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent_class(config, environment)

# Main Multi-Agent System
class MultiAgentTradingSystem:
    """Complete Multi-Agent Reinforcement Learning Trading System"""[1]
    
    def __init__(self, config_file: str):
        """Initialize the multi-agent system"""
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Initialize coordinator
        self.coordinator = CoordinatorAgent(getattr(self.config,'coordinator', {}))
        
        # Initialize agents
        self.agents = []
        self._initialize_agents()
        
        # System state
        self.is_running = False
        self.performance_metrics = {}
        
        logger.info("Multi-Agent Trading System initialized")
    
    def _initialize_agents(self):
        """Initialize all agents from configuration"""
        
        from tradingenvironment import TradingEnvironment
        
        agent_configs = getattr(self.config,'agents', [])
        
        for agent_config in agent_configs:
            try:
                # Create agent configuration
                config = AgentConfig(**agent_config)
                
                # Create environment (placeholder - you'll need to implement this)
                env = DummyVecEnv([lambda: Monitor(TradingEnvironment(None))])
                
                # Create agent
                agent = create_agent(config.strategy_type, config, env)
                
                # Add to coordinator
                self.coordinator.add_agent(agent)
                self.agents.append(agent)
                
                logger.info(f"Created {config.strategy_type} agent: {config.name}")
                
            except Exception as e:
                logger.error(f"Error creating agent: {e}")
    
    async def run_trading_session(self, market_data_stream):
        """Run the trading session with live market data"""
        
        self.is_running = True
        logger.info("Starting multi-agent trading session")
        
        try:
            while self.is_running:
                # Get market data
                market_data = await market_data_stream.get_next()
                
                if not market_data:
                    await asyncio.sleep(1)
                    continue
                
                # Process each symbol
                for symbol, data in market_data.items():
                    # Create observation (you'll need to implement this based on your data format)
                    observation = self._create_observation(data)
                    
                    # Get coordinated decision
                    signal = await self.coordinator.coordinate_decision(symbol, observation, data)
                    
                    if signal:
                        # Execute trade (implement your execution logic)
                        await self._execute_trade(signal)
                        
                        logger.info(f"Executed trade: {signal.action} {signal.symbol} @ {signal.entry_price}")
                
                # Performance monitoring
                await self._monitor_performance()
                
                # Short sleep to prevent excessive CPU usage
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in trading session: {e}")
        finally:
            self.is_running = False
            logger.info("Trading session ended")
    
    def _create_observation(self, market_data: Dict) -> np.ndarray:
        """Create observation array from market data"""
        
        # This is a placeholder - implement based on your feature engineering
        features = []
        
        # Price features
        features.extend([
            market_data.get('open', 0),
            market_data.get('high', 0),
            market_data.get('low', 0),
            market_data.get('close', 0),
            market_data.get('volume', 0)
        ])
        
        # Technical indicators
        indicators = market_data.get('indicators', {})
        features.extend([
            indicators.get('rsi', 50),
            indicators.get('macd', 0),
            indicators.get('adx', 25)
        ])
        
        return np.array(features, dtype=np.float32)
    
    async def _execute_trade(self, signal: TradingSignal):
        """Execute the trading signal"""
        
        # Placeholder for trade execution logic
        # You'll need to implement this based on your broker/platform
        pass
    
    async def _monitor_performance(self):
        """Monitor system performance"""
        
        # Update performance metrics
        system_status = self.coordinator.get_system_status()
        self.performance_metrics = system_status
        
        # Log performance periodically
        if int(time.time()) % 300 == 0:  # Every 5 minutes
            logger.info(f"System Status: {system_status['success_rate']:.2%} success rate, "
                       f"{system_status['active_agents']} active agents")
    
    def save_system_state(self, filepath: str):
        """Save the complete system state"""
        
        try:
            # Save all agent models
            models_dir = Path(filepath) / 'models'
            models_dir.mkdir(parents=True, exist_ok=True)
            
            for agent in self.agents:
                agent.save_model(str(models_dir))
            
            # Save system configuration and state
            system_state = {
                'config': self.config,
                'coordinator_state': self.coordinator.get_system_status(),
                'performance_metrics': self.performance_metrics
            }
            
            with open(Path(filepath) / 'system_state.json', 'w') as f:
                json.dump(system_state, f, indent=2, default=str)
            
            logger.info(f"System state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
    
    def load_system_state(self, filepath: str):
        """Load the complete system state"""
        
        try:
            # Load system state
            with open(Path(filepath) / 'system_state.json', 'r') as f:
                system_state = json.load(f)
            
            # Load agent models
            models_dir = Path(filepath) / 'models'
            
            for agent in self.agents:
                agent.load_model(str(models_dir))
            
            logger.info(f"System state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading system state: {e}")
    
    def stop_system(self):
        """Stop the trading system"""
        self.is_running = False
        logger.info("Stopping multi-agent trading system")

# Example configuration
EXAMPLE_CONFIG = {
    "coordinator": {
        "consensus_threshold": 0.6,
        "max_portfolio_risk": 0.1,
        "coordination_strategy": "weighted_voting"
    },
    "agents": [
        {
            "name": "ScalpingAgent_1",
            "strategy_type": "scalping",
            "timeframe": "M1",
            "risk_tolerance": 0.8,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "memory_size": 50000,
            "update_frequency": 100,
            "target_return": 0.02,
            "max_drawdown": 0.05,
            "position_size_limits": [0.001, 0.01]
        },
        {
            "name": "SwingAgent_1",
            "strategy_type": "swing",
            "timeframe": "H1",
            "risk_tolerance": 0.6,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "memory_size": 100000,
            "update_frequency": 500,
            "target_return": 0.05,
            "max_drawdown": 0.10,
            "position_size_limits": [0.005, 0.03]
        },
        {
            "name": "TrendAgent_1",
            "strategy_type": "trend",
            "timeframe": "H4",
            "risk_tolerance": 0.5,
            "learning_rate": 0.0003,
            "batch_size": 16,
            "memory_size": 200000,
            "update_frequency": 1000,
            "target_return": 0.10,
            "max_drawdown": 0.15,
            "position_size_limits": [0.01, 0.05]
        },
        {
            "name": "ArbitrageAgent_1",
            "strategy_type": "arbitrage",
            "timeframe": "M1",
            "risk_tolerance": 0.9,
            "learning_rate": 0.0001,
            "batch_size": 128,
            "memory_size": 25000,
            "update_frequency": 50,
            "target_return": 0.01,
            "max_drawdown": 0.02,
            "position_size_limits": [0.0005, 0.005]
        }
    ]
}

# Usage example
async def main():
    """Example usage of the Multi-Agent Trading System"""
    
    # Save example configuration
    with open('multi_agent_config.json', 'w') as f:
        json.dump(EXAMPLE_CONFIG, f, indent=2)
    
    # Initialize system
    system = MultiAgentTradingSystem('multi_agent_config.json')
    
    # Example market data stream (implement your own)
    class MockMarketDataStream:
        async def get_next(self):
            # Return mock market data
            return {
                'EURUSD': {
                    'open': 1.1000,
                    'high': 1.1020,
                    'low': 1.0980,
                    'close': 1.1010,
                    'volume': 1000000,
                    'indicators': {
                        'rsi': 55,
                        'macd': 0.001,
                        'adx': 30
                    },
                    'sentiment': 0.1
                }
            }
    
    # Run trading session
    market_stream = MockMarketDataStream()
    await system.run_trading_session(market_stream)

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())

# multi_agent_integration.py - Integration with Main Trading System

from multi_agent_rl_system import MultiAgentTradingSystem, TradingSignal
import asyncio
import logging

logger = logging.getLogger(__name__)

class MultiAgentIntegration:
    """Integration layer for multi-agent system with main trading bot"""
    
    def __init__(self, main_bot, config_file: str):
        self.main_bot = main_bot
        self.multi_agent_system = MultiAgentTradingSystem(config_file)
        self.is_enabled = False
        
    async def get_multi_agent_signal(self, symbol: str, market_data: dict) -> dict:
        """Get signal from multi-agent system"""
        
        if not self.is_enabled:
            return None
        
        try:
            # Create observation from market data
            observation = self._prepare_observation(market_data)
            
            # Get coordinated signal
            signal = await self.multi_agent_system.coordinator.coordinate_decision(
                symbol, observation, market_data
            )
            
            if signal:
                return {
                    'symbol': signal.symbol,
                    'direction': signal.action,
                    'strategy': 'MULTI_AGENT',
                    'confidence': signal.confidence,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'position_size': signal.position_size,
                    'reasoning': signal.reasoning
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting multi-agent signal: {e}")
            return None
    
    def _prepare_observation(self, market_data: dict):
        """Prepare observation for multi-agent system"""
        
        # Convert market data to observation format
        # This should match your feature engineering
        features = []
        
        # Add price features
        features.extend([
            market_data.get('Open', 0),
            market_data.get('High', 0),
            market_data.get('Low', 0),
            market_data.get('Close', 0),
            market_data.get('Volume', 0)
        ])
        
        # Add technical indicators
        for indicator_name in ['RSI', 'MACD', 'ADX', 'BB_upper', 'BB_lower']:
            features.append(market_data.get(indicator_name, 0))
        
        return np.array(features, dtype=np.float32)
    
    def enable_multi_agent(self):
        """Enable multi-agent system"""
        self.is_enabled = True
        logger.info("Multi-agent system enabled")
    
    def disable_multi_agent(self):
        """Disable multi-agent system"""
        self.is_enabled = False
        logger.info("Multi-agent system disabled")

# در فایل main.py اضافه کن:
"""
# Add this to your TradingBot class

from multi_agent_integration import MultiAgentIntegration

class TradingBot:
    def __init__(self, config):
        # ... existing code ...
        
        # Initialize multi-agent system
        self.multi_agent = MultiAgentIntegration(self, 'multi_agent_config.json')
        
    async def get_trading_signals(self, symbol, data):
        signals = []
        
        # ... existing signal generation ...
        
        # Add multi-agent signal
        multi_agent_signal = await self.multi_agent.get_multi_agent_signal(symbol, data)
        if multi_agent_signal:
            signals.append(multi_agent_signal)
        
        return signals
"""

# enhanced_tradingenvironment.py - Production Ready Trading Environment
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class EnhancedTradingEnvironment(gym.Env):
    """Enhanced Trading Environment for RL training"""
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any] = None):
        super().__init__()
        
        self.data = data.copy()
        self.config = config or {}
        
        # Environment parameters
        self.initial_balance = self.config.get('initial_balance', 10000.0)
        self.transaction_cost = self.config.get('transaction_cost', 0.0001)
        self.max_drawdown_limit = self.config.get('max_drawdown_limit', 0.3)
        
        # State variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Tracking
        self.trades_history = []
        self.equity_curve = []
        self.current_drawdown = 0.0
        self.peak_equity = self.initial_balance
        
        # Spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        n_features = min(50, len(self.data.columns) * 2)  # Dynamic feature count
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        
        self.prepare_data()
        logger.info(f"Enhanced Trading Environment initialized - {len(self.data)} data points")
    
    def prepare_data(self):
        """Prepare data for training"""
        try:
            if 'Returns' not in self.data.columns:
                self.data['Returns'] = self.data['Close'].pct_change()
            
            self.data = self.data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            if not isinstance(self.data.index, pd.DatetimeIndex):
                self.data.index = pd.date_range(start='2020-01-01', periods=len(self.data), freq='H')
            
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 50
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        self.trades_history = []
        self.equity_curve = [self.initial_balance]
        self.current_drawdown = 0.0
        self.peak_equity = self.initial_balance
        
        observation = self.get_observation()
        info = self.get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step"""
        self.current_step += 1
        
        if self.current_step >= len(self.data):
            return self.get_observation(), 0.0, True, False, self.get_info()
        
        position_change, position_size_target = np.clip(action, self.action_space.low, self.action_space.high)
        current_price = self.data.iloc[self.current_step]['Close']
        
        if self.position != 0:
            self.update_unrealized_pnl(current_price)
        
        reward = 0.0
        if abs(position_change) > 0.01:
            reward = self.execute_trade(position_change, position_size_target, current_price)
        
        self.update_portfolio()
        reward += self.calculate_step_reward()
        
        done, truncated = self.check_termination()
        
        observation = self.get_observation()
        info = self.get_info()
        
        return observation, reward, done, truncated, info
    
    def execute_trade(self, position_change, position_size, current_price):
        """Execute trading action"""
        try:
            reward = 0.0
            
            if self.position != 0 and np.sign(position_change) != np.sign(self.position):
                reward += self.close_position(current_price)
            
            if abs(position_change) > 0.01:
                reward += self.open_position(position_change, position_size, current_price)
            
            return reward
        except:
            return -0.1
    
    def close_position(self, current_price):
        """Close current position"""
        if self.position == 0:
            return 0.0
        
        if self.position > 0:
            pnl = (current_price - self.entry_price) * abs(self.position_size)
        else:
            pnl = (self.entry_price - current_price) * abs(self.position_size)
        
        cost = abs(self.position_size) * current_price * self.transaction_cost
        net_pnl = pnl - cost
        
        self.balance += net_pnl
        self.realized_pnl += net_pnl
        self.unrealized_pnl = 0.0
        
        self.trades_history.append({
            'entry_price': self.entry_price,
            'exit_price': current_price,
            'net_pnl': net_pnl
        })
        
        self.position = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        
        return net_pnl / self.initial_balance * 100
    
    def open_position(self, position_change, position_size, current_price):
        """Open new position"""
        self.position = np.sign(position_change)
        available_capital = self.balance * position_size
        self.position_size = (available_capital / current_price) * self.position
        self.entry_price = current_price
        
        cost = abs(self.position_size) * current_price * self.transaction_cost
        self.balance -= cost
        
        return 0.001
    
    def update_unrealized_pnl(self, current_price):
        """Update unrealized P&L"""
        if self.position == 0 or self.entry_price == 0:
            self.unrealized_pnl = 0.0
            return
        
        if self.position > 0:
            self.unrealized_pnl = (current_price - self.entry_price) * abs(self.position_size)
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * abs(self.position_size)
    
    def update_portfolio(self):
        """Update portfolio metrics"""
        self.equity = self.balance + self.unrealized_pnl
        
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        
        self.equity_curve.append(self.equity)
    
    def calculate_step_reward(self):
        """Calculate step-wise reward"""
        reward = 0.0
        
        if len(self.equity_curve) > 1:
            equity_change = self.equity_curve[-1] - self.equity_curve[-2]
            reward += equity_change / self.initial_balance * 5
        
        if self.current_drawdown > 0.1:
            reward -= self.current_drawdown * 2
        
        return reward
    
    def check_termination(self):
        """Check termination conditions"""
        done = False
        truncated = False
        
        if self.current_step >= len(self.data) - 1:
            done = True
        
        if self.current_drawdown >= self.max_drawdown_limit:
            done = True
        
        if self.equity <= self.initial_balance * 0.1:
            done = True
        
        return done, truncated
    
    def get_observation(self):
        """Get current observation"""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        
        current_data = self.data.iloc[self.current_step]
        close = current_data['Close']
        
        # Market features (dynamic based on available data)
        market_features = []
        
        # Basic OHLCV
        market_features.extend([
            current_data.get('Open', close) / close,
            current_data.get('High', close) / close,
            current_data.get('Low', close) / close,
            1.0,  # Close/Close = 1
            np.log(current_data.get('Volume', 1000000) + 1) / 10
        ])
        
        # Technical indicators (if available)
        indicators = ['RSI_14', 'MACD', 'MACD_Signal', 'ATR_14', 'SMA_20', 'EMA_20']
        for indicator in indicators:
            value = current_data.get(indicator, 0)
            if 'RSI' in indicator:
                market_features.append((value - 50) / 50)
            elif 'MACD' in indicator or 'ATR' in indicator:
                market_features.append(np.tanh(value * 1000) if value != 0 else 0)
            else:
                market_features.append((value - close) / close if close > 0 and value > 0 else 0)
        
        # Recent returns
        if self.current_step >= 5:
            recent_returns = self.data['Returns'].iloc[self.current_step-4:self.current_step+1].fillna(0)
            market_features.extend(recent_returns.tolist())
        else:
            market_features.extend([0.0] * 5)
        
        # Portfolio features
        portfolio_features = [
            float(self.position),
            self.position_size / 10000 if self.position_size != 0 else 0.0,
            (self.balance - self.initial_balance) / self.initial_balance,
            (self.equity - self.initial_balance) / self.initial_balance,
            self.unrealized_pnl / self.initial_balance,
            self.realized_pnl / self.initial_balance,
            self.current_drawdown,
            len(self.trades_history) / 100.0,
            float(self.current_step) / len(self.data),
            0.0
        ]
        
        # Combine and ensure fixed size
        all_features = market_features + portfolio_features
        target_size = self.observation_space.shape[0]
        
        if len(all_features) > target_size:
            all_features = all_features[:target_size]
        elif len(all_features) < target_size:
            all_features.extend([0.0] * (target_size - len(all_features)))
        
        observation = np.array(all_features, dtype=np.float32)
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def get_info(self):
        """Get environment info"""
        current_price = self.data.iloc[self.current_step]['Close'] if self.current_step < len(self.data) else 0
        
        return {
            'step': self.current_step,
            'current_price': float(current_price),
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'total_trades': len(self.trades_history),
            'current_drawdown': self.current_drawdown,
            'total_return': (self.equity - self.initial_balance) / self.initial_balance
        }

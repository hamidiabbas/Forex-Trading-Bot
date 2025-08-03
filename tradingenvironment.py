
# tradingenvironment.py - Complete Trading Environment
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data.copy()
        self.current_step = 0
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_profit = 0
        self.trade_count = 0
        self.winning_trades = 0
        
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)
        
        self._prepare_features()
        logger.info(f"TradingEnvironment initialized with {len(self.data)} data points")
    
    def _prepare_features(self):
        feature_columns = [
            'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'ATR_14',
            'EMA_20', 'EMA_50', 'BB_upper', 'BB_lower', 'BB_middle',
            'High_Low_Ratio', 'Close_Open_Ratio', 'Price_Range',
            'Volume_Ratio', 'Returns', 'Volatility_20', 'Momentum_10'
        ]
        
        for col in feature_columns:
            if col not in self.data.columns:
                if 'RSI' in col:
                    self.data[col] = 50
                elif 'MACD' in col:
                    self.data[col] = 0
                elif 'ATR' in col:
                    self.data[col] = 0.001
                elif 'EMA' in col or 'BB' in col:
                    self.data[col] = self.data['Close']
                else:
                    self.data[col] = 0.0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 50
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_profit = 0
        self.trade_count = 0
        self.winning_trades = 0
        
        observation = self._get_observation()
        info = self._get_info()
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        current_price = self.data['Close'].iloc[self.current_step]
        
        reward = self._calculate_reward(action, current_price)
        self._update_position(action, current_price)
        
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, truncated, info
    
    def _calculate_reward(self, action: int, current_price: float) -> float:
        try:
            reward = 0.0
            
            if self.position != 0 and action == 0:  # Close position
                if self.position == 1:  # Long
                    profit = (current_price - self.entry_price) / self.entry_price
                else:  # Short
                    profit = (self.entry_price - current_price) / self.entry_price
                
                reward = profit * 100
                self.total_profit += profit
                self.trade_count += 1
                
                if profit > 0:
                    self.winning_trades += 1
                    reward *= 1.2
            
            elif self.position != 0:  # Holding position
                if self.position == 1:
                    unrealized_profit = (current_price - self.entry_price) / self.entry_price
                else:
                    unrealized_profit = (self.entry_price - current_price) / self.entry_price
                reward = unrealized_profit * 10
            
            return float(reward)
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def _update_position(self, action: int, current_price: float):
        try:
            if action == 1 and self.position <= 0:  # Buy
                self.position = 1
                self.entry_price = current_price
            elif action == 2 and self.position >= 0:  # Sell
                self.position = -1
                self.entry_price = current_price
            elif action == 0:  # Hold/Close
                self.position = 0
                self.entry_price = 0
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def _get_observation(self) -> np.ndarray:
        try:
            if self.current_step >= len(self.data):
                self.current_step = len(self.data) - 1
            
            current_data = self.data.iloc[self.current_step]
            obs = []
            
            # Technical indicators (first 24)
            feature_cols = ['RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'ATR_14',
                          'EMA_20', 'EMA_50', 'High_Low_Ratio', 'Close_Open_Ratio',
                          'Price_Range', 'Volume_Ratio', 'Returns', 'Volatility_20']
            
            for col in feature_cols[:24]:
                value = current_data.get(col, 0.0)
                if col in ['RSI_14']:
                    value = (value - 50) / 50
                elif col in ['MACD_12_26_9', 'MACDs_12_26_9']:
                    value = np.tanh(value * 1000)
                obs.append(float(value))
            
            # Position info (8 more features)
            obs.extend([
                float(self.position),
                float(self.entry_price / current_data['Close'] - 1) if self.entry_price > 0 else 0.0,
                float(self.total_profit),
                float(self.trade_count / 100.0),
                float(self.winning_trades / max(self.trade_count, 1)),
                float(self.balance / self.initial_balance - 1),
                float(self.current_step / len(self.data)),
                0.0  # padding
            ])
            
            while len(obs) < 32:
                obs.append(0.0)
            
            observation = np.array(obs[:32], dtype=np.float32)
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
            return observation
            
        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            return np.zeros(32, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        try:
            win_rate = self.winning_trades / max(self.trade_count, 1)
            current_price = self.data['Close'].iloc[self.current_step] if self.current_step < len(self.data) else 0
            
            return {
                'current_step': self.current_step,
                'current_position': self.position,
                'current_price': current_price,
                'total_profit': self.total_profit,
                'trade_count': self.trade_count,
                'win_rate': win_rate
            }
        except Exception as e:
            logger.error(f"Error creating info: {e}")
            return {'error': str(e)}

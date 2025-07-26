"""
/******************************************************************************
 *
 * FILE NAME:           trading_environment_adv.py
 *
 * PURPOSE:
 *
 * This module defines an advanced trading environment with a continuous
 * action space, allowing the RL agent to learn sophisticated position sizing.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 26, 2025
 *
 * VERSION:             63.0
 *
 ******************************************************************************/
"""
import gymnasium as gym
import numpy as np

class AdvancedTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=100000, transaction_cost_pct=0.001):
        super().__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        
        # Continuous Action Space: -1 (Max Sell) to 1 (Max Buy)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space includes market data + 2 values for position state
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(df.shape[1] + 2,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position_size = 0
        self.entry_price = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position_size = 0
        self.entry_price = 0
        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.df.iloc[self.current_step].values
        # Add position info to what the agent sees
        is_long = 1 if self.position_size > 0 else 0
        is_short = 1 if self.position_size < 0 else 0
        position_info = np.array([is_long, is_short])
        return np.concatenate([obs, position_info]).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        self._take_action(action[0])
        
        # Calculate new equity and reward
        new_equity = self.balance + (self.position_size * self.df['Close'].iloc[self.current_step])
        reward = new_equity - self.equity
        self.equity = new_equity
        
        return self._get_observation(), reward, done, False, {}

    def _take_action(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Action is positive: BUY signal
        if action > 0:
            if self.position_size < 0: # If short, close position first
                self.balance += self.position_size * -1 * self.entry_price
                self.balance -= self.transaction_cost_pct * self.balance
                self.position_size = 0
            
            trade_value = self.equity * action
            trade_size_units = trade_value / current_price
            self.balance -= trade_value
            self.position_size += trade_size_units
            self.entry_price = current_price

        # Action is negative: SELL signal
        elif action < 0:
            if self.position_size > 0: # If long, close position first
                self.balance += self.position_size * current_price
                self.balance -= self.transaction_cost_pct * self.balance
                self.position_size = 0

            trade_value = self.equity * abs(action)
            trade_size_units = trade_value / current_price
            self.balance += trade_value
            self.position_size -= trade_size_units
            self.entry_price = current_price
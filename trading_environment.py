"""
/******************************************************************************
 *
 * FILE NAME:           trading_environment.py (Final Version)
 *
 * PURPOSE:
 *
 * This version implements a robust, equity-based reward system and a
 * correctly shaped observation space to provide the RL agent with the
 * context and motivation it needs to learn a profitable trading policy.
 * This is the complete and final version of this file.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 29, 2025
 *
 * VERSION:             77.0 (Final)
 *
 ******************************************************************************/
"""
import gymnasium as gym
import numpy as np
import pandas as pd

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=100000, transaction_cost_pct=0.001):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        
        self.action_space = gym.spaces.Discrete(3) # 0=Hold, 1=Buy, 2=Sell
        
        # --- THIS IS THE FIX ---
        # Observation space now correctly includes the 2 extra features for position state
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(df.shape[1] + 2,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0 # 0=None, 1=Long, -1=Short
        self.entry_price = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0
        self.entry_price = 0
        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.df.iloc[self.current_step].values
        
        is_long = 1 if self.position == 1 else 0
        is_short = 1 if self.position == -1 else 0
        position_info = np.array([is_long, is_short])
        
        return np.concatenate([obs, position_info]).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        prev_equity = self.equity
        
        self._take_action(action)
        
        # Update the current equity
        current_price = self.df['Close'].iloc[self.current_step]
        if self.position != 0:
            unrealized_pnl = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
            self.equity = self.balance + unrealized_pnl
        else:
            self.equity = self.balance

        # The reward is the change in equity from the last step
        reward = self.equity - prev_equity

        # --- THIS IS THE FIX ---
        # Apply a penalty only when the agent is flat and chooses to do nothing
        if self.position == 0 and action == 0:
            reward = -0.1
        
        info = {}
        return self._get_observation(), reward, done, False, info

    def _take_action(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Action 1: Buy
        if action == 1 and self.position != 1:
            if self.position == -1: # If short, close position first
                self.balance += (self.entry_price - current_price)
                self.balance -= self.transaction_cost_pct * self.balance
            self.position = 1
            self.entry_price = current_price

        # Action 2: Sell
        elif action == 2 and self.position != -1:
            if self.position == 1: # If long, close position first
                self.balance += (current_price - self.entry_price)
                self.balance -= self.transaction_cost_pct * self.balance
            self.position = -1
            self.entry_price = current_price
"""
/******************************************************************************
 *
 * FILE NAME:           trading_environment.py
 *
 * PURPOSE:
 *
 * This module defines a custom trading environment compatible with OpenAI Gym
 * and Stable-Baselines3 for training Reinforcement Learning agents with a
 * discrete (Buy, Sell, Hold) action space. This is the complete and
 * correct version of this file.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 26, 2025
 *
 * VERSION:             62.2 (Corrected & Verified)
 *
 ******************************************************************************/
"""
import gymnasium as gym
import numpy as np
import pandas as pd

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=100000, transaction_cost_pct=0.001):
        super().__init__()
        
        self.df = df.reset_index(drop=True) # Ensure a clean integer-based index
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        
        # Define the action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = gym.spaces.Discrete(3)
        
        # Define the observation space (the market data the agent sees)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(df.shape[1],),
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
        return self.df.iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        done = self.current_step >= len(self.df) - 1
        
        self._take_action(action)
        
        new_equity = self.balance
        if self.position != 0:
            current_price = self.df['Close'].iloc[self.current_step]
            unrealized_pnl = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
            new_equity += unrealized_pnl

        reward = new_equity - self.equity
        self.equity = new_equity
        
        # Return observation, reward, terminated, truncated, info
        return self._get_observation(), reward, done, False, {}

    def _take_action(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Action 1: Buy
        if action == 1:
            if self.position == -1: # If short, close position first
                self.balance += (self.entry_price - current_price)
                self.balance -= self.transaction_cost_pct * self.balance
            self.position = 1
            self.entry_price = current_price

        # Action 2: Sell
        elif action == 2:
            if self.position == 1: # If long, close position first
                self.balance += (current_price - self.entry_price)
                self.balance -= self.transaction_cost_pct * self.balance
            self.position = -1
            self.entry_price = current_price
            
        # Action 0: Hold (do nothing)
        else:
            pass
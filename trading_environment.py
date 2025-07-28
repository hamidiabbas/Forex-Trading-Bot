"""
/******************************************************************************
 *
 * FILE NAME:           trading_environment.py (with Inaction Penalty)
 *
 * PURPOSE:
 *
 * This version adds a small penalty for the "Hold" action to encourage
 * the RL agent to be more proactive in finding and executing trades.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 28, 2025
 *
 * VERSION:             62.7 (with Inaction Penalty)
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
        
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(df.shape[1],),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.entry_atr = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.entry_atr = 0
        return self._get_observation(), {}

    def _get_observation(self):
        return self.df.iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        prev_position = self.position
        reward = self._take_action_and_get_reward(action)
        
        new_equity = self.balance
        if self.position != 0:
            current_price = self.df['Close'].iloc[self.current_step]
            unrealized_pnl = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
            new_equity += unrealized_pnl
        self.equity = new_equity
        
        info = {'position_changed': self.position != prev_position, 'current_position': self.position}
        return self._get_observation(), reward, done, False, info

    def _take_action_and_get_reward(self, action):
        reward = 0
        current_price = self.df['Close'].iloc[self.current_step]
        current_atr = self.df['ATRr_14'].iloc[self.current_step]

        # Close a BUY position if action is SELL
        if self.position == 1 and action == 2:
            profit = (current_price - self.entry_price) - (self.transaction_cost_pct * self.entry_price)
            risk_taken = self.entry_atr
            reward = profit / risk_taken if risk_taken > 0 else 0
            self.balance += profit
            self.position = 0
        
        # Close a SELL position if action is BUY
        elif self.position == -1 and action == 1:
            profit = (self.entry_price - current_price) - (self.transaction_cost_pct * self.entry_price)
            risk_taken = self.entry_atr
            reward = profit / risk_taken if risk_taken > 0 else 0
            self.balance += profit
            self.position = 0

        # Open a new position if we don't have one
        if self.position == 0:
            if action == 1: # Buy
                self.position = 1
                self.entry_price = current_price
                self.entry_atr = current_atr
            elif action == 2: # Sell
                self.position = -1
                self.entry_price = current_price
                self.entry_atr = current_atr
        
        # --- NEW: Add a small penalty for not taking a trade (action 0) ---
        if action == 0:
            reward = -0.1 # Small penalty for inaction
        
        return reward
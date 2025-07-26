"""
/******************************************************************************
 *
 * FILE NAME:           trading_environment.py
 *
 * PURPOSE:
 *
 * This module defines a custom trading environment compatible with OpenAI Gym
 * and Stable-Baselines3 for training Reinforcement Learning agents. It
 * simulates the actions and rewards of forex trading.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 26, 2025
 *
 * VERSION:             61.0
 *
 ******************************************************************************/
"""
import gymnasium as gym
import numpy as np

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=100000, transaction_cost_pct=0.001):
        super().__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        
        # Define the action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = gym.spaces.Discrete(3)
        
        # Define the observation space (the market data the agent sees)
        # It's the number of features in our dataframe
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
        
        # If we reach the end of the data, the episode is done
        done = self.current_step >= len(self.df) - 1
        
        # Execute the chosen action
        self._take_action(action)
        
        # Calculate the reward
        new_equity = self.balance
        if self.position != 0:
            current_price = self.df['Close'].iloc[self.current_step]
            unrealized_pnl = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
            new_equity += unrealized_pnl

        reward = new_equity - self.equity
        self.equity = new_equity
        
        return self._get_observation(), reward, done, False, {}

    def _take_action(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Action 1: Buy
        if action == 1:
            # If we are in a short position, close it first
            if self.position == -1:
                self.balance += (self.entry_price - current_price)
                self.balance -= self.transaction_cost_pct * self.balance
            # Open a new long position
            self.position = 1
            self.entry_price = current_price

        # Action 2: Sell
        elif action == 2:
            # If we are in a long position, close it first
            if self.position == 1:
                self.balance += (current_price - self.entry_price)
                self.balance -= self.transaction_cost_pct * self.balance
            # Open a new short position
            self.position = -1
            self.entry_price = current_price
            
        # Action 0: Hold (do nothing)
        else:
            pass
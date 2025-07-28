import gymnasium as gym
import numpy as np
import pandas as pd

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=100000, transaction_cost_pct=0.001):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.max_position_size = 0.1
        self.action_space = gym.spaces.Discrete(3)  # 0=Hold/Close, 1=Buy, 2=Sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(df.shape[1] + 3,),
            dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0  # 0=None, 1=Long, -1=Short
        self.entry_price = 0
        self.position_size = 0
        self.total_trades = 0
        self.winning_trades = 0
        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.df.iloc[self.current_step].values
        position_info = np.array([
            float(self.position),
            self.entry_price / self.df['Close'].iloc[self.current_step] if self.entry_price > 0 else 0,
            self._get_unrealized_pnl() / self.initial_balance
        ])
        return np.concatenate([obs, position_info]).astype(np.float32)

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {}

        prev_equity = self.equity
        prev_position = self.position
        reward = self._take_action(action)
        self.current_step += 1
        self._update_equity()
        equity_change = self.equity - prev_equity
        reward = equity_change / self.initial_balance

        if prev_position == 0 and self.position != 0:
            reward += 0.01
        elif prev_position != 0 and self.position == 0:
            if equity_change > 0:
                reward += 0.1
            self.total_trades += 1
            if equity_change > 0:
                self.winning_trades += 1

        if action == 0 and self.position == 0 and self.current_step > 5:
            recent_returns = np.diff(self.df['Close'].iloc[self.current_step-5:self.current_step+1])
            if abs(np.mean(recent_returns)) > np.std(recent_returns):
                reward -= 0.005

        done = self.current_step >= len(self.df) - 1
        info = {
            'current_position': self.position,
            'balance': self.balance,
            'equity': self.equity,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades)
        }
        return self._get_observation(), reward, done, False, info

    def _take_action(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        reward = 0
        if action == 0 and self.position != 0:
            reward = self._close_position(current_price)
        elif action == 1:
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.position_size = self.max_position_size * self.balance
                self.balance -= self.position_size * self.transaction_cost_pct
            elif self.position == -1:
                reward = self._close_position(current_price)
                self.position = 1
                self.entry_price = current_price
                self.position_size = self.max_position_size * self.balance
                self.balance -= self.position_size * self.transaction_cost_pct
        elif action == 2:
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
                self.position_size = self.max_position_size * self.balance
                self.balance -= self.position_size * self.transaction_cost_pct
            elif self.position == 1:
                reward = self._close_position(current_price)
                self.position = -1
                self.entry_price = current_price
                self.position_size = self.max_position_size * self.balance
                self.balance -= self.position_size * self.transaction_cost_pct
        return reward

    def _close_position(self, current_price):
        if self.position == 0:
            return 0
        if self.position == 1:
            pnl = (current_price - self.entry_price) * (self.position_size / self.entry_price)
        else:
            pnl = (self.entry_price - current_price) * (self.position_size / self.entry_price)
        transaction_cost = self.position_size * self.transaction_cost_pct
        net_pnl = pnl - transaction_cost
        self.balance += net_pnl
        self.position = 0
        self.entry_price = 0
        self.position_size = 0
        return net_pnl / self.initial_balance

    def _get_unrealized_pnl(self):
        if self.position == 0:
            return 0
        current_price = self.df['Close'].iloc[self.current_step]
        if self.position == 1:
            return (current_price - self.entry_price) * (self.position_size / self.entry_price)
        else:
            return (self.entry_price - current_price) * (self.position_size / self.entry_price)

    def _update_equity(self):
        self.equity = self.balance + self._get_unrealized_pnl()

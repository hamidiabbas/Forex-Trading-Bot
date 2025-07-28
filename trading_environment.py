import gymnasium as gym
import numpy as np
import pandas as pd

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=100000, transaction_cost_pct=0.001):
        super().__init__()
        
        # FIXED: Remove non-numeric columns before processing
        self.df = df.select_dtypes(include=[np.number]).reset_index(drop=True)
        
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.max_position_size = 0.1  # 10% of balance per trade
        
        # FIXED: Proper action space - 0=Close/Hold, 1=Buy, 2=Sell
        self.action_space = gym.spaces.Discrete(3)
        
        # FIXED: Correct observation space with position info
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.df.shape[1] + 3,),  # +3 for position, entry_price, unrealized_pnl
            dtype=np.float32
        )
        
        # Initialize action tracking for diagnostics
        self.action_count = {0: 0, 1: 0, 2: 0}
        self.consecutive_holds = 0
        
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
        
        # Reset action tracking for new episode
        self.action_count = {0: 0, 1: 0, 2: 0}
        self.consecutive_holds = 0
        
        return self._get_observation(), {}

    def _get_observation(self):
        """FIXED: Proper state representation with trading context"""
        obs = self.df.iloc[self.current_step].values
        
        # Add position information to state
        position_info = np.array([
            float(self.position),  # Current position (-1, 0, 1)
            self.entry_price / self.df['Close'].iloc[self.current_step] if self.entry_price > 0 else 0,
            self._get_unrealized_pnl() / self.initial_balance  # Normalized unrealized P&L
        ])
        
        return np.concatenate([obs, position_info]).astype(np.float32)

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {}
            
        prev_equity = self.equity
        prev_position = self.position
        
        # DIAGNOSTIC: Log trading activity every 100 steps
        if self.current_step % 100 == 0:
            action_names = {0: 'HOLD/CLOSE', 1: 'BUY', 2: 'SELL'}
            print(f"Step {self.current_step}: Action={action_names[action]}, Position={self.position}, Trades={self.total_trades}")
        
        # Track action distribution
        self.action_count[action] += 1
        
        # Execute action
        self._take_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Update equity
        self._update_equity()
        
        # FIXED: Improved reward calculation that ENCOURAGES trading
        equity_change = self.equity - prev_equity
        reward = equity_change / self.initial_balance  # Normalize to prevent scale issues
        
        # CRITICAL: Track consecutive holds and penalize excessive inactivity
        if action == 0 and self.position == 0:
            self.consecutive_holds += 1
            # Strong penalty for too many consecutive holds
            if self.consecutive_holds > 10:
                reward -= 0.05  # Escalating penalty for inactivity
            elif self.consecutive_holds > 20:
                reward -= 0.1   # Even stronger penalty
        else:
            self.consecutive_holds = 0  # Reset if any trading action taken
        
        # FIXED: Incentive structure that promotes good trading behavior
        if prev_position == 0 and self.position != 0:  # Opening position
            reward += 0.02  # Bonus for taking action
            print(f"  🟢 TRADE OPENED: {('LONG' if self.position == 1 else 'SHORT')} at {self.entry_price:.5f}")
        elif prev_position != 0 and self.position == 0:  # Closing position
            trade_pnl = equity_change
            if trade_pnl > 0:
                reward += 0.1  # Large bonus for profitable trade
                self.winning_trades += 1
                print(f"  ✅ PROFITABLE TRADE CLOSED: +${trade_pnl:.2f}")
            else:
                reward += 0.01  # Small reward even for attempting trades
                print(f"  ❌ LOSING TRADE CLOSED: ${trade_pnl:.2f}")
            self.total_trades += 1
        
        # Additional reward shaping: encourage trading in trending markets
        if action != 0 and self.current_step > 5:
            recent_returns = np.diff(self.df['Close'].iloc[self.current_step-5:self.current_step+1])
            if len(recent_returns) > 0:
                trend_strength = abs(np.mean(recent_returns))
                if trend_strength > np.std(recent_returns):  # Clear trend detected
                    reward += 0.005  # Small bonus for trading in trending conditions
        
        done = self.current_step >= len(self.df) - 1
        
        # DIAGNOSTIC: Log trading summary at episode end
        if done:
            total_actions = sum(self.action_count.values())
            if total_actions > 0:
                hold_pct = (self.action_count[0] / total_actions) * 100
                buy_pct = (self.action_count[1] / total_actions) * 100
                sell_pct = (self.action_count[2] / total_actions) * 100
                
                print(f"\n=== EPISODE TRADING SUMMARY ===")
                print(f"Total Actions: {total_actions}")
                print(f"Hold/Close: {self.action_count[0]} ({hold_pct:.1f}%)")
                print(f"Buy: {self.action_count[1]} ({buy_pct:.1f}%)")
                print(f"Sell: {self.action_count[2]} ({sell_pct:.1f}%)")
                print(f"Total Trades Executed: {self.total_trades}")
                print(f"Win Rate: {(self.winning_trades/max(1,self.total_trades))*100:.1f}%")
                print(f"Final Equity: ${self.equity:.2f}")
                print(f"Total Return: {((self.equity - self.initial_balance)/self.initial_balance)*100:.2f}%")
                print(f"Max Consecutive Holds: {getattr(self, 'max_consecutive_holds', 0)}")
                
                # Assess trading activity level
                trading_activity = (buy_pct + sell_pct)
                if trading_activity < 5:
                    print(f"🚨 WARNING: Very low trading activity ({trading_activity:.1f}%)")
                elif trading_activity < 15:
                    print(f"⚠️ CAUTION: Low trading activity ({trading_activity:.1f}%)")
                else:
                    print(f"✅ GOOD: Active trading detected ({trading_activity:.1f}%)")
                
                print("=" * 30)
        
        # Track maximum consecutive holds for analysis
        if not hasattr(self, 'max_consecutive_holds'):
            self.max_consecutive_holds = 0
        self.max_consecutive_holds = max(self.max_consecutive_holds, self.consecutive_holds)
        
        info = {
            'current_position': self.position,
            'balance': self.balance,
            'equity': self.equity,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'action_distribution': self.action_count.copy(),
            'consecutive_holds': self.consecutive_holds,
            'trading_activity_pct': ((self.action_count[1] + self.action_count[2]) / max(1, sum(self.action_count.values()))) * 100
        }
        
        return self._get_observation(), reward, done, False, info

    def _take_action(self, action):
        """FIXED: Proper action handling with correct transaction costs"""
        current_price = self.df['Close'].iloc[self.current_step]
        
        # Action 0: Close position or hold
        if action == 0:
            if self.position != 0:
                self._close_position(current_price)
                print(f"  🔄 POSITION CLOSED at {current_price:.5f}")
        
        # Action 1: Buy (long position)
        elif action == 1:
            if self.position == 0:  # Open long
                self.position = 1
                self.entry_price = current_price
                self.position_size = self.max_position_size * self.balance
                transaction_cost = self.position_size * self.transaction_cost_pct
                self.balance -= transaction_cost
            elif self.position == -1:  # Close short and open long
                self._close_position(current_price)
                self.position = 1
                self.entry_price = current_price
                self.position_size = self.max_position_size * self.balance
                transaction_cost = self.position_size * self.transaction_cost_pct
                self.balance -= transaction_cost
        
        # Action 2: Sell (short position)
        elif action == 2:
            if self.position == 0:  # Open short
                self.position = -1
                self.entry_price = current_price
                self.position_size = self.max_position_size * self.balance
                transaction_cost = self.position_size * self.transaction_cost_pct
                self.balance -= transaction_cost
            elif self.position == 1:  # Close long and open short
                self._close_position(current_price)
                self.position = -1
                self.entry_price = current_price
                self.position_size = self.max_position_size * self.balance
                transaction_cost = self.position_size * self.transaction_cost_pct
                self.balance -= transaction_cost

    def _close_position(self, current_price):
        """FIXED: Proper position closing with correct P&L calculation"""
        if self.position == 0:
            return
            
        # Calculate P&L
        if self.position == 1:  # Closing long
            pnl = (current_price - self.entry_price) * (self.position_size / self.entry_price)
        else:  # Closing short
            pnl = (self.entry_price - current_price) * (self.position_size / self.entry_price)
        
        # Apply transaction cost
        transaction_cost = self.position_size * self.transaction_cost_pct
        net_pnl = pnl - transaction_cost
        
        # Update balance
        self.balance += net_pnl
        
        # Reset position
        self.position = 0
        self.entry_price = 0
        self.position_size = 0

    def _get_unrealized_pnl(self):
        """Calculate current unrealized P&L"""
        if self.position == 0:
            return 0
            
        current_price = self.df['Close'].iloc[self.current_step]
        if self.position == 1:  # Long position
            return (current_price - self.entry_price) * (self.position_size / self.entry_price)
        else:  # Short position
            return (self.entry_price - current_price) * (self.position_size / self.entry_price)

    def _update_equity(self):
        """Update current equity including unrealized P&L"""
        self.equity = self.balance + self._get_unrealized_pnl()

    def get_trading_stats(self):
        """Get comprehensive trading statistics"""
        total_actions = sum(self.action_count.values())
        if total_actions == 0:
            return None
            
        return {
            'total_actions': total_actions,
            'hold_percentage': (self.action_count[0] / total_actions) * 100,
            'buy_percentage': (self.action_count[1] / total_actions) * 100,
            'sell_percentage': (self.action_count[2] / total_actions) * 100,
            'trading_activity': ((self.action_count[1] + self.action_count[2]) / total_actions) * 100,
            'total_trades': self.total_trades,
            'win_rate': (self.winning_trades / max(1, self.total_trades)) * 100,
            'final_equity': self.equity,
            'total_return': ((self.equity - self.initial_balance) / self.initial_balance) * 100,
            'consecutive_holds': self.consecutive_holds,
            'max_consecutive_holds': getattr(self, 'max_consecutive_holds', 0)
        }

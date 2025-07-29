"""
Complete Trading Environment without gym/stable-baselines3 dependencies
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TradingEnvironment:
    """
    Complete trading environment without external ML dependencies
    """
    
    def __init__(self, symbol: str, config: Dict[str, Any], lookback_period: int = 50):
        self.symbol = symbol
        self.config = config
        self.lookback_period = lookback_period
        
        # Environment parameters
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.position_size = 0.0
        self.entry_price = 0.0
        self.current_step = 0
        self.max_steps = 1000
        
        # Market data
        self.df = None
        self.current_price = 0.0
        
        # Initialize market data
        self._initialize_market_data()
        
        logger.info(f"✅ Trading environment initialized for {symbol} (No external ML deps)")
    
    def _initialize_market_data(self) -> None:
        """Initialize market data for the environment"""
        try:
            # Create synthetic market data for testing
            self._create_synthetic_data()
                
        except Exception as e:
            logger.warning(f"Error loading data: {e}, using synthetic data")
            self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> None:
        """Create synthetic market data for testing"""
        np.random.seed(42)
        
        # Generate synthetic price data
        n_points = 2000
        dates = pd.date_range(start='2023-01-01', periods=n_points, freq='15min')
        
        # Base prices for different symbols
        base_prices = {
            'EURUSD': 1.1000,
            'GBPUSD': 1.3000,
            'XAUUSD': 2000.0,
            'USDJPY': 148.0
        }
        base_price = base_prices.get(self.symbol, 1.1000)
        
        # Random walk with trend
        returns = np.random.normal(0.0001, 0.01, n_points)
        price = base_price
        
        ohlc_data = []
        for i in range(n_points):
            open_price = price
            high_price = open_price + abs(np.random.normal(0, 0.002))
            low_price = open_price - abs(np.random.normal(0, 0.002))
            close_price = open_price + returns[i]
            
            # Ensure OHLC logic
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            ohlc_data.append({
                'time': dates[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'tick_volume': np.random.randint(50, 200)
            })
            
            price = close_price
        
        self.df = pd.DataFrame(ohlc_data)
        self._add_simple_indicators()
        logger.info(f"✅ Created {len(self.df)} synthetic data points for {self.symbol}")
    
    def _add_simple_indicators(self) -> None:
        """Add simple technical indicators"""
        try:
            # Simple Moving Averages
            self.df['sma_20'] = self.df['close'].rolling(window=20).mean()
            self.df['sma_50'] = self.df['close'].rolling(window=50).mean()
            
            # RSI (simplified)
            delta = self.df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            self.df['bb_middle'] = self.df['close'].rolling(window=20).mean()
            bb_std = self.df['close'].rolling(window=20).std()
            self.df['bb_upper'] = self.df['bb_middle'] + (bb_std * 2)
            self.df['bb_lower'] = self.df['bb_middle'] - (bb_std * 2)
            
            # MACD (simplified)
            exp1 = self.df['close'].ewm(span=12).mean()
            exp2 = self.df['close'].ewm(span=26).mean()
            self.df['macd'] = exp1 - exp2
            self.df['macd_signal'] = self.df['macd'].ewm(span=9).mean()
            self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']
            
            # ATR (simplified)
            high_low = self.df['high'] - self.df['low']
            high_close = np.abs(self.df['high'] - self.df['close'].shift())
            low_close = np.abs(self.df['low'] - self.df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            self.df['atr'] = true_range.rolling(14).mean()
            
            # Fill NaN values
            self.df = self.df.fillna(method='bfill').fillna(0)
            
            logger.debug(f"✅ Simple indicators added to {self.symbol} data")
            
        except Exception as e:
            logger.error(f"Error adding simple indicators: {e}")
            # Fill with default values if calculation fails
            for col in ['sma_20', 'sma_50', 'rsi', 'bb_middle', 'bb_upper', 'bb_lower', 
                       'macd', 'macd_signal', 'macd_histogram', 'atr']:
                if col not in self.df.columns:
                    self.df[col] = 0.0
    
    def reset(self) -> Dict[str, Any]:
        """Reset the environment"""
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.current_step = self.lookback_period
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, {'episode_end': True}
        
        # Get current market data
        current_data = self.df.iloc[self.current_step]
        self.current_price = current_data['close']
        
        # Execute action
        reward = self._execute_action(action)
        
        # Update net worth
        self._update_net_worth()
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= min(len(self.df) - 1, self.max_steps) or 
                self.net_worth <= self.initial_balance * 0.5)
        
        # Create info dictionary
        info = {
            'balance': self.balance,
            'net_worth': self.net_worth,
            'position': self.position,
            'current_price': self.current_price,
            'trade_executed': abs(action - 1) > 0.5 and self.position != (action - 1),
            'trade_profit': 0.0
        }
        
        return self._get_observation(), reward, done, info
    
    def _execute_action(self, action: int) -> float:
        """Execute trading action and return reward"""
        reward = 0.0
        prev_net_worth = self.net_worth
        
        # Close existing position if action is different
        if self.position != 0 and action != (self.position + 1):
            reward += self._close_position()
        
        # Execute new action
        if action == 1 and self.position == 0:  # Buy
            self._open_position(1)
        elif action == 2 and self.position == 0:  # Sell
            self._open_position(-1)
        # action == 0 is hold, no action needed
        
        # Calculate reward based on net worth change
        self._update_net_worth()
        reward += (self.net_worth - prev_net_worth) * 10  # Scale reward
        
        # Add small penalty for holding to encourage action
        if action == 0:
            reward -= 0.01
        
        return reward
    
    def _open_position(self, direction: int) -> None:
        """Open a new position"""
        self.position = direction
        self.entry_price = self.current_price
        
        # Calculate position size (risk 2% of balance)
        risk_amount = self.balance * 0.02
        self.position_size = risk_amount / (abs(self.current_price * 0.01))  # Assume 1% stop loss
        
        logger.debug(f"Opened {direction} position at {self.current_price:.5f}, size: {self.position_size:.3f}")
    
    def _close_position(self) -> float:
        """Close current position and return profit"""
        if self.position == 0:
            return 0.0
        
        # Calculate profit
        if self.position == 1:  # Long position
            profit = (self.current_price - self.entry_price) * self.position_size * 100000
        else:  # Short position
            profit = (self.entry_price - self.current_price) * self.position_size * 100000
        
        # Update balance
        self.balance += profit
        
        logger.debug(f"Closed position at {self.current_price:.5f}, profit: ${profit:.2f}")
        
        # Reset position
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        
        return profit / 100  # Scale reward
    
    def _update_net_worth(self) -> None:
        """Update net worth including unrealized P&L"""
        self.net_worth = self.balance
        
        if self.position != 0:
            # Add unrealized P&L
            if self.position == 1:  # Long
                unrealized_pnl = (self.current_price - self.entry_price) * self.position_size * 100000
            else:  # Short
                unrealized_pnl = (self.entry_price - self.current_price) * self.position_size * 100000
            
            self.net_worth += unrealized_pnl
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation as dictionary"""
        if self.current_step < self.lookback_period:
            return {
                'current_price': 0,
                'rsi': 50,
                'sma_20': 0,
                'sma_50': 0,
                'atr': 0,
                'macd': 0,
                'position': 0,
                'net_worth_ratio': 1.0
            }
        
        # Get recent data
        current_data = self.df.iloc[self.current_step]
        
        try:
            observation = {
                'current_price': float(current_data['close']),
                'rsi': float(current_data.get('rsi', 50)),
                'sma_20': float(current_data.get('sma_20', current_data['close'])),
                'sma_50': float(current_data.get('sma_50', current_data['close'])),
                'atr': float(current_data.get('atr', current_data['close'] * 0.01)),
                'macd': float(current_data.get('macd', 0)),
                'macd_signal': float(current_data.get('macd_signal', 0)),
                'bb_upper': float(current_data.get('bb_upper', current_data['close'])),
                'bb_lower': float(current_data.get('bb_lower', current_data['close'])),
                'position': self.position,
                'net_worth_ratio': self.net_worth / self.initial_balance,
                'timestamp': current_data['time']
            }
            
            return observation
            
        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            return {
                'current_price': 1.0,
                'rsi': 50,
                'position': 0,
                'net_worth_ratio': 1.0
            }
    
    def get_current_market_data(self) -> Dict[str, Any]:
        """Get current market data for signal generation"""
        try:
            if self.current_step >= len(self.df):
                return None
            
            current = self.df.iloc[self.current_step]
            
            return {
                'symbol': self.symbol,
                'current_price': float(current['close']),
                'open': float(current['open']),
                'high': float(current['high']),
                'low': float(current['low']),
                'close': float(current['close']),
                'volume': int(current.get('tick_volume', 100)),
                'rsi': float(current.get('rsi', 50)),
                'sma_20': float(current.get('sma_20', current['close'])),
                'sma_50': float(current.get('sma_50', current['close'])),
                'atr': float(current.get('atr', current['close'] * 0.01)),
                'macd': float(current.get('macd', 0)),
                'macd_signal': float(current.get('macd_signal', 0)),
                'macd_histogram': float(current.get('macd_histogram', 0)),
                'bb_upper': float(current.get('bb_upper', current['close'])),
                'bb_middle': float(current.get('bb_middle', current['close'])),
                'bb_lower': float(current.get('bb_lower', current['close'])),
                'bb_position': self._calculate_bb_position(current),
                'trend_direction': self._get_trend_direction(),
                'volatility': float(current.get('atr', current['close'] * 0.01)) / current['close'],
                'timestamp': current['time']
            }
            
        except Exception as e:
            logger.error(f"Error getting current market data: {e}")
            return None
    
    def _calculate_bb_position(self, current_data) -> float:
        """Calculate position within Bollinger Bands"""
        try:
            bb_upper = current_data.get('bb_upper', current_data['close'])
            bb_lower = current_data.get('bb_lower', current_data['close'])
            price = current_data['close']
            
            if bb_upper > bb_lower:
                return (price - bb_lower) / (bb_upper - bb_lower)
            return 0.5
        except Exception:
            return 0.5
    
    def _get_trend_direction(self) -> str:
        """Get current trend direction"""
        try:
            if self.current_step < 20:
                return 'NEUTRAL'
            
            current = self.df.iloc[self.current_step]
            sma_20 = current.get('sma_20', current['close'])
            sma_50 = current.get('sma_50', current['close'])
            
            if current['close'] > sma_20 > sma_50:
                return 'BULLISH'
            elif current['close'] < sma_20 < sma_50:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
        except Exception:
            return 'NEUTRAL'

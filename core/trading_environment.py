"""
Complete Production Trading Environment with Full ML Integration
Supports gym interface and stable-baselines3 compatibility
"""
import numpy as np
import pandas as pd
import gym
from gym import spaces
import MetaTrader5 as mt5
import talib
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import threading
import pickle
import os

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Complete production trading environment with full ML integration
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, symbol: str, config: Dict[str, Any], lookback_period: int = 50):
        super(TradingEnvironment, self).__init__()
        
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
        self.max_steps = 2000
        
        # Market data
        self.df = None
        self.current_price = 0.0
        self.price_history = []
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 64 comprehensive features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32
        )
        
        # Performance tracking
        self.total_profit = 0.0
        self.successful_trades = 0
        self.total_trades = 0
        self.max_drawdown = 0.0
        self.episode_trades = []
        
        # Threading safety
        self.lock = threading.Lock()
        
        # Initialize market data
        self._initialize_market_data()
        
        logger.info(f"✅ Production Trading Environment initialized for {symbol}")
        logger.info(f"   Observation Space: {self.observation_space.shape}")
        logger.info(f"   Action Space: {self.action_space.n}")
        logger.info(f"   Max Steps: {self.max_steps}")
    
    def _initialize_market_data(self) -> None:
        """Initialize comprehensive market data"""
        try:
            # Try to get real market data from MT5
            if not mt5.initialize():
                logger.warning("MT5 not available, using synthetic data")
                self._create_synthetic_data()
                return
            
            # Get real market data
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M15, 0, 3000)
            
            if rates is not None and len(rates) > 0:
                self.df = pd.DataFrame(rates)
                self.df['time'] = pd.to_datetime(self.df['time'], unit='s')
                self._add_comprehensive_indicators()
                logger.info(f"✅ Loaded {len(self.df)} real market data points for {self.symbol}")
            else:
                logger.warning(f"No real data available for {self.symbol}, using synthetic data")
                self._create_synthetic_data()
                
        except Exception as e:
            logger.warning(f"Error loading real data: {e}, using synthetic data")
            self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> None:
        """Create sophisticated synthetic market data"""
        np.random.seed(42)
        
        # Generate realistic price data with regime changes
        n_points = 3000
        dates = pd.date_range(start='2023-01-01', periods=n_points, freq='15min')
        
        # Base prices for different symbols
        base_prices = {
            'EURUSD': 1.1000,
            'GBPUSD': 1.3000,
            'XAUUSD': 2000.0,
            'USDJPY': 148.0
        }
        price = base_prices.get(self.symbol, 1.1000)
        
        # Generate returns with multiple regimes
        returns = []
        volatility_regimes = np.random.choice([0.005, 0.01, 0.02], n_points, p=[0.6, 0.3, 0.1])
        trend_regimes = np.random.choice([-0.0001, 0, 0.0001], n_points, p=[0.2, 0.6, 0.2])
        
        for i in range(n_points):
            vol = volatility_regimes[i]
            trend = trend_regimes[i]
            ret = np.random.normal(trend, vol)
            returns.append(ret)
        
        # Generate OHLC data
        ohlc_data = []
        for i in range(n_points):
            open_price = price
            return_val = returns[i]
            close_price = open_price * (1 + return_val)
            
            # Generate realistic high/low
            intrabar_vol = abs(return_val) * 1.5
            high_price = max(open_price, close_price) * (1 + intrabar_vol * np.random.uniform(0.3, 1.0))
            low_price = min(open_price, close_price) * (1 - intrabar_vol * np.random.uniform(0.3, 1.0))
            
            # Generate volume
            base_volume = 100
            volume_noise = np.random.uniform(0.5, 2.0)
            volume = int(base_volume * volume_noise)
            
            ohlc_data.append({
                'time': dates[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'tick_volume': volume,
                'real_volume': volume * 10
            })
            
            price = close_price
        
        self.df = pd.DataFrame(ohlc_data)
        self._add_comprehensive_indicators()
        logger.info(f"✅ Created {len(self.df)} synthetic data points for {self.symbol}")
    
    def _add_comprehensive_indicators(self) -> None:
        """Add comprehensive technical indicators using TA-Lib"""
        try:
            # Price arrays
            high = self.df['high'].values
            low = self.df['low'].values
            close = self.df['close'].values
            open_prices = self.df['open'].values
            volume = self.df['tick_volume'].values
            
            # ===== TREND INDICATORS =====
            
            # Moving Averages
            self.df['sma_10'] = talib.SMA(close, timeperiod=10)
            self.df['sma_20'] = talib.SMA(close, timeperiod=20)
            self.df['sma_50'] = talib.SMA(close, timeperiod=50)
            self.df['sma_100'] = talib.SMA(close, timeperiod=100)
            self.df['sma_200'] = talib.SMA(close, timeperiod=200)
            
            # Exponential Moving Averages
            self.df['ema_12'] = talib.EMA(close, timeperiod=12)
            self.df['ema_26'] = talib.EMA(close, timeperiod=26)
            self.df['ema_50'] = talib.EMA(close, timeperiod=50)
            
            # Weighted Moving Average
            self.df['wma_14'] = talib.WMA(close, timeperiod=14)
            
            # MACD Family
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            self.df['macd'] = macd
            self.df['macd_signal'] = macd_signal
            self.df['macd_histogram'] = macd_hist
            
            # Parabolic SAR
            self.df['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            
            # Average Directional Index
            self.df['adx'] = talib.ADX(high, low, close, timeperiod=14)
            self.df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            self.df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            # ===== MOMENTUM INDICATORS =====
            
            # RSI Family
            self.df['rsi_14'] = talib.RSI(close, timeperiod=14)
            self.df['rsi_21'] = talib.RSI(close, timeperiod=21)
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            self.df['stoch_k'] = slowk
            self.df['stoch_d'] = slowd
            
            # Stochastic RSI
            fastk, fastd = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3)
            self.df['stochrsi_k'] = fastk
            self.df['stochrsi_d'] = fastd
            
            # Williams %R
            self.df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            
            # Rate of Change
            self.df['roc'] = talib.ROC(close, timeperiod=10)
            
            # Momentum
            self.df['momentum'] = talib.MOM(close, timeperiod=10)
            
            # Commodity Channel Index
            self.df['cci'] = talib.CCI(high, low, close, timeperiod=14)
            
            # Money Flow Index
            self.df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
            
            # ===== VOLATILITY INDICATORS =====
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            self.df['bb_upper'] = bb_upper
            self.df['bb_middle'] = bb_middle
            self.df['bb_lower'] = bb_lower
            self.df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            self.df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            # Average True Range
            self.df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            self.df['atr_21'] = talib.ATR(high, low, close, timeperiod=21)
            
            # True Range
            self.df['trange'] = talib.TRANGE(high, low, close)
            
            # ===== VOLUME INDICATORS =====
            
            # On Balance Volume
            self.df['obv'] = talib.OBV(close, volume)
            
            # Accumulation/Distribution
            self.df['ad'] = talib.AD(high, low, close, volume)
            
            # Chaikin A/D Oscillator
            self.df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
            
            # ===== PRICE TRANSFORM =====
            
            # Typical Price
            self.df['typprice'] = talib.TYPPRICE(high, low, close)
            
            # Weighted Close Price
            self.df['wclprice'] = talib.WCLPRICE(high, low, close)
            
            # ===== CUSTOM INDICATORS =====
            
            # Price changes
            self.df['price_change'] = close - np.roll(close, 1)
            self.df['price_change_pct'] = talib.ROC(close, timeperiod=1)
            
            # High-Low ratios
            self.df['hl_ratio'] = (high - low) / close
            self.df['oc_ratio'] = (close - open_prices) / open_prices
            
            # Volatility measures
            self.df['volatility_10'] = talib.STDDEV(close, timeperiod=10)
            self.df['volatility_20'] = talib.STDDEV(close, timeperiod=20)
            
            # Support/Resistance levels
            self.df['support'] = talib.MIN(low, timeperiod=20)
            self.df['resistance'] = talib.MAX(high, timeperiod=20)
            
            # Volume ratios
            self.df['volume_sma'] = talib.SMA(volume.astype(float), timeperiod=20)
            self.df['volume_ratio'] = volume / self.df['volume_sma']
            
            # Market structure
            self.df['higher_highs'] = (high > np.roll(high, 1)).astype(int)
            self.df['lower_lows'] = (low < np.roll(low, 1)).astype(int)
            
            # Fill NaN values
            self.df = self.df.fillna(method='bfill').fillna(0)
            
            logger.debug(f"✅ Comprehensive indicators added: {len(self.df.columns)} features")
            
        except Exception as e:
            logger.error(f"Error adding comprehensive indicators: {e}")
            # Add minimal indicators if TA-Lib fails
            self._add_minimal_indicators()
    
    def _add_minimal_indicators(self) -> None:
        """Add minimal indicators if TA-Lib fails"""
        try:
            # Basic moving averages
            self.df['sma_20'] = self.df['close'].rolling(window=20).mean()
            self.df['sma_50'] = self.df['close'].rolling(window=50).mean()
            
            # Basic RSI
            delta = self.df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Basic Bollinger Bands
            self.df['bb_middle'] = self.df['close'].rolling(window=20).mean()
            bb_std = self.df['close'].rolling(window=20).std()
            self.df['bb_upper'] = self.df['bb_middle'] + (bb_std * 2)
            self.df['bb_lower'] = self.df['bb_middle'] - (bb_std * 2)
            
            # Fill with zeros for missing indicators
            indicator_names = [
                'macd', 'macd_signal', 'macd_histogram', 'atr', 'adx',
                'stoch_k', 'stoch_d', 'williams_r', 'cci', 'momentum'
            ]
            
            for indicator in indicator_names:
                if indicator not in self.df.columns:
                    self.df[indicator] = 0.0
            
            self.df = self.df.fillna(0)
            logger.warning("Using minimal indicators due to TA-Lib issues")
            
        except Exception as e:
            logger.error(f"Error adding minimal indicators: {e}")
    
    def reset(self) -> np.ndarray:
        """Reset the environment"""
        with self.lock:
            self.balance = self.initial_balance
            self.net_worth = self.initial_balance
            self.position = 0
            self.position_size = 0.0
            self.entry_price = 0.0
            self.current_step = self.lookback_period
            self.total_profit = 0.0
            self.episode_trades = []
            
            return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        with self.lock:
            if self.current_step >= len(self.df) - 1:
                return self._get_observation(), 0, True, {'episode_end': True}
            
            # Get current market data
            current_data = self.df.iloc[self.current_step]
            self.current_price = current_data['close']
            self.price_history.append(self.current_price)
            
            # Execute action
            reward = self._execute_action(action)
            
            # Update net worth and track performance
            self._update_net_worth()
            self._update_performance_metrics()
            
            # Move to next step
            self.current_step += 1
            
            # Check if episode is done
            done = (self.current_step >= min(len(self.df) - 1, self.max_steps) or 
                    self.net_worth <= self.initial_balance * 0.5 or
                    self.net_worth >= self.initial_balance * 3.0)  # Stop at 3x gain too
            
            # Create comprehensive info
            info = {
                'balance': self.balance,
                'net_worth': self.net_worth,
                'position': self.position,
                'current_price': self.current_price,
                'total_profit': self.total_profit,
                'successful_trades': self.successful_trades,
                'total_trades': self.total_trades,
                'max_drawdown': self.max_drawdown,
                'trade_executed': abs(action - 1) > 0.5 and self.position != (action - 1),
                'win_rate': (self.successful_trades / max(1, self.total_trades)) * 100,
                'profit_factor': self._calculate_profit_factor(),
                'sharpe_ratio': self._calculate_sharpe_ratio()
            }
            
            return self._get_observation(), reward, done, info
    
    def _execute_action(self, action: int) -> float:
        """Execute trading action with realistic slippage and costs"""
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
        
        # Calculate reward
        self._update_net_worth()
        net_worth_change = self.net_worth - prev_net_worth
        
        # Reward shaping
        reward += net_worth_change * 100  # Scale for better learning
        
        # Penalty for excessive trading
        if action != 0:
            reward -= 2  # Small trading cost
        
        # Bonus for profitable trades
        if net_worth_change > 0:
            reward += net_worth_change * 50  # Bonus for profits
        
        return reward
    
    def _open_position(self, direction: int) -> None:
        """Open position with realistic sizing"""
        self.position = direction
        self.entry_price = self.current_price
        
        # Dynamic position sizing based on volatility
        current_atr = self.df.iloc[self.current_step].get('atr', self.current_price * 0.01)
        volatility_factor = current_atr / self.current_price
        
        # Risk 1-3% of balance based on volatility
        risk_pct = np.clip(0.01 / volatility_factor, 0.01, 0.03)
        risk_amount = self.balance * risk_pct
        
        # Calculate position size
        if self.symbol == 'XAUUSD':
            self.position_size = risk_amount / (current_atr * 100)
        else:
            pip_value = 10 if 'JPY' not in self.symbol else 1
            pip_size = 0.0001 if 'JPY' not in self.symbol else 0.01
            atr_pips = current_atr / pip_size
            self.position_size = risk_amount / (atr_pips * pip_value)
        
        # Limit position size
        self.position_size = np.clip(self.position_size, 0.01, 1.0)
        
        logger.debug(f"Opened {direction} position: size={self.position_size:.3f}, price={self.entry_price:.5f}")
    
    def _close_position(self) -> float:
        """Close position and return reward"""
        if self.position == 0:
            return 0.0
        
        # Calculate profit with realistic slippage
        slippage = np.random.normal(0, 0.0001)  # Random slippage
        exit_price = self.current_price + (slippage * self.position)
        
        if self.position == 1:  # Long position
            if self.symbol == 'XAUUSD':
                profit = (exit_price - self.entry_price) * self.position_size * 100
            else:
                pip_value = 10 if 'JPY' not in self.symbol else 1
                pip_size = 0.0001 if 'JPY' not in self.symbol else 0.01
                pips = (exit_price - self.entry_price) / pip_size
                profit = pips * pip_value * self.position_size
        else:  # Short position
            if self.symbol == 'XAUUSD':
                profit = (self.entry_price - exit_price) * self.position_size * 100
            else:
                pip_value = 10 if 'JPY' not in self.symbol else 1
                pip_size = 0.0001 if 'JPY' not in self.symbol else 0.01
                pips = (self.entry_price - exit_price) / pip_size
                profit = pips * pip_value * self.position_size
        
        # Apply spread cost
        spread_cost = self.position_size * 2  # $2 spread cost per lot
        profit -= spread_cost
        
        # Update balance and tracking
        self.balance += profit
        self.total_profit += profit
        self.total_trades += 1
        
        if profit > 0:
            self.successful_trades += 1
        
        # Record trade
        trade_record = {
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'position': self.position,
            'size': self.position_size,
            'profit': profit,
            'timestamp': self.current_step
        }
        self.episode_trades.append(trade_record)
        
        logger.debug(f"Closed position: profit=${profit:.2f}, exit_price={exit_price:.5f}")
        
        # Reset position
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        
        return profit / 10  # Scale reward
    
    def _update_net_worth(self) -> None:
        """Update net worth with unrealized P&L"""
        self.net_worth = self.balance
        
        if self.position != 0:
            # Calculate unrealized P&L
            if self.position == 1:  # Long
                if self.symbol == 'XAUUSD':
                    unrealized = (self.current_price - self.entry_price) * self.position_size * 100
                else:
                    pip_value = 10 if 'JPY' not in self.symbol else 1
                    pip_size = 0.0001 if 'JPY' not in self.symbol else 0.01
                    pips = (self.current_price - self.entry_price) / pip_size
                    unrealized = pips * pip_value * self.position_size
            else:  # Short
                if self.symbol == 'XAUUSD':
                    unrealized = (self.entry_price - self.current_price) * self.position_size * 100
                else:
                    pip_value = 10 if 'JPY' not in self.symbol else 1
                    pip_size = 0.0001 if 'JPY' not in self.symbol else 0.01
                    pips = (self.entry_price - self.current_price) / pip_size
                    unrealized = pips * pip_value * self.position_size
            
            self.net_worth += unrealized
    
    def _update_performance_metrics(self) -> None:
        """Update performance tracking metrics"""
        try:
            # Update drawdown
            if len(self.price_history) > 1:
                peak = max(self.price_history)
                current_dd = (peak - self.net_worth) / peak
                self.max_drawdown = max(self.max_drawdown, current_dd)
        except Exception as e:
            logger.debug(f"Error updating performance metrics: {e}")
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        try:
            if not self.episode_trades:
                return 1.0
            
            profits = [t['profit'] for t in self.episode_trades if t['profit'] > 0]
            losses = [abs(t['profit']) for t in self.episode_trades if t['profit'] < 0]
            
            total_profit = sum(profits) if profits else 0
            total_loss = sum(losses) if losses else 1
            
            return total_profit / total_loss if total_loss > 0 else 1.0
        except Exception:
            return 1.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.episode_trades) < 2:
                return 0.0
            
            returns = [t['profit'] / self.initial_balance for t in self.episode_trades]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            return mean_return / std_return if std_return > 0 else 0.0
        except Exception:
            return 0.0
    
    def _get_observation(self) -> np.ndarray:
        """Get comprehensive 64-feature observation"""
        if self.current_step < self.lookback_period:
            return np.zeros(64, dtype=np.float32)
        
        try:
            # Get current market data
            current = self.df.iloc[self.current_step]
            recent = self.df.iloc[self.current_step-10:self.current_step]  # Last 10 bars
            
            # Price features (8)
            price_features = [
                current['close'] / current['open'] - 1,  # Daily return
                (current['high'] - current['low']) / current['close'],  # Daily range
                current['close'] / current.get('sma_20', current['close']) - 1,  # Price vs SMA20
                current['close'] / current.get('sma_50', current['close']) - 1,  # Price vs SMA50
                current.get('bb_position', 0.5) - 0.5,  # BB position centered
                current.get('hl_ratio', 0),  # High-low ratio
                current.get('oc_ratio', 0),  # Open-close ratio
                (current['close'] - recent['close'].iloc[0]) / recent['close'].iloc[0]  # 10-bar return
            ]
            
            # Technical indicators (16)
            technical_features = [
                (current.get('rsi_14', 50) - 50) / 50,  # RSI normalized
                current.get('macd', 0) / current['close'],  # MACD normalized
                current.get('macd_histogram', 0) / current['close'],  # MACD hist normalized
                (current.get('stoch_k', 50) - 50) / 50,  # Stoch K normalized
                (current.get('stoch_d', 50) - 50) / 50,  # Stoch D normalized
                (current.get('williams_r', -50) + 50) / 50,  # Williams R normalized
                current.get('cci', 0) / 100,  # CCI normalized
                current.get('momentum', 0) / current['close'],  # Momentum normalized
                (current.get('adx', 25) - 25) / 25,  # ADX normalized
                current.get('atr', 0) / current['close'],  # ATR normalized
                current.get('roc', 0),  # Rate of change
                current.get('mfi', 50) / 100 - 0.5,  # MFI normalized
                current.get('plus_di', 25) / 50 - 0.5,  # +DI normalized
                current.get('minus_di', 25) / 50 - 0.5,  # -DI normalized
                current.get('volatility_20', 0) / current['close'],  # Volatility
                current.get('bb_width', 0)  # BB width
            ]
            
            # Volume features (4)
            volume_features = [
                current.get('volume_ratio', 1) - 1,  # Volume ratio
                current.get('obv', 0) / 1000000,  # OBV scaled
                current.get('ad', 0) / 1000000,  # A/D scaled
                current.get('adosc', 0) / 1000  # ADOSC scaled
            ]
            
            # Trend features (8)
            trend_features = [
                1 if current.get('sma_20', 0) > current.get('sma_50', 0) else -1,  # MA trend
                1 if current['close'] > current.get('sma_20', 0) else -1,  # Price above SMA20
                1 if current.get('ema_12', 0) > current.get('ema_26', 0) else -1,  # EMA crossover
                1 if current.get('macd', 0) > current.get('macd_signal', 0) else -1,  # MACD signal
                current.get('higher_highs', 0) - 0.5,  # Higher highs
                current.get('lower_lows', 0) - 0.5,  # Lower lows
                (current['close'] - current.get('support', current['close'])) / current['close'],  # Support distance
                (current.get('resistance', current['close']) - current['close']) / current['close']  # Resistance distance
            ]
            
            # Position and portfolio features (8)
            portfolio_features = [
                self.position,  # Current position
                self.position_size if self.position != 0 else 0,  # Position size
                (self.net_worth - self.initial_balance) / self.initial_balance,  # Portfolio return
                (self.balance - self.initial_balance) / self.initial_balance,  # Realized return
                self.max_drawdown,  # Max drawdown
                self.successful_trades / max(1, self.total_trades),  # Win rate
                self.current_step / self.max_steps,  # Time progress
                len(self.episode_trades) / 100  # Trade frequency
            ]
            
            # Market microstructure features (8)
            microstructure_features = [
                recent['close'].pct_change().mean(),  # Mean return
                recent['close'].pct_change().std(),  # Return volatility
                recent['high'].rolling(3).max().iloc[-1] / current['close'] - 1,  # Recent high
                1 - recent['low'].rolling(3).min().iloc[-1] / current['close'],  # Recent low
                recent['tick_volume'].mean() / recent['tick_volume'].rolling(20).mean().iloc[-1] - 1,  # Volume trend
                (recent['close'] > recent['open']).sum() / len(recent) - 0.5,  # Bullish bars ratio
                recent['atr'].iloc[-1] / recent['atr'].mean() - 1 if 'atr' in recent.columns else 0,  # ATR trend
                np.corrcoef(range(len(recent)), recent['close'])[0, 1] if len(recent) > 1 else 0  # Price trend correlation
            ]
            
            # Time features (12)
            current_time = current['time'] if pd.notna(current['time']) else datetime.now()
            hour = current_time.hour if hasattr(current_time, 'hour') else 12
            day_of_week = current_time.weekday() if hasattr(current_time, 'weekday') else 2
            
            time_features = [
                np.sin(2 * np.pi * hour / 24),  # Hour sin
                np.cos(2 * np.pi * hour / 24),  # Hour cos
                np.sin(2 * np.pi * day_of_week / 7),  # Day sin
                np.cos(2 * np.pi * day_of_week / 7),  # Day cos
                1 if 8 <= hour <= 17 else 0,  # London/NY session
                1 if 13 <= hour <= 17 else 0,  # Overlap session
                1 if hour < 6 or hour > 22 else 0,  # Low liquidity
                1 if day_of_week < 5 else 0,  # Weekday
                1 if day_of_week == 4 else 0,  # Friday
                1 if day_of_week == 0 else 0,  # Monday
                hour / 24,  # Hour normalized
                day_of_week / 7  # Day normalized
            ]
            
            # Combine all features
            all_features = (
                price_features + technical_features + volume_features + 
                trend_features + portfolio_features + microstructure_features + 
                time_features
            )
            
            # Ensure exactly 64 features
            if len(all_features) > 64:
                all_features = all_features[:64]
            elif len(all_features) < 64:
                all_features.extend([0.0] * (64 - len(all_features)))
            
            # Convert to numpy array and handle NaN/inf
            observation = np.array(all_features, dtype=np.float32)
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
            observation = np.clip(observation, -10.0, 10.0)  # Clip extreme values
            
            return observation
            
        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            return np.zeros(64, dtype=np.float32)
    
    def render(self, mode='human') -> None:
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Price: {self.current_price:.5f}, "
                  f"Position: {self.position}, Net Worth: ${self.net_worth:.2f}, "
                  f"Total Trades: {self.total_trades}, Win Rate: {(self.successful_trades/max(1,self.total_trades)*100):.1f}%")
    
    def close(self) -> None:
        """Close the environment"""
        if hasattr(self, 'df'):
            del self.df
        logger.info("Trading environment closed")
    
    def save_episode_data(self, filename: str) -> None:
        """Save episode data for analysis"""
        try:
            episode_data = {
                'trades': self.episode_trades,
                'final_balance': self.balance,
                'final_net_worth': self.net_worth,
                'total_profit': self.total_profit,
                'successful_trades': self.successful_trades,
                'total_trades': self.total_trades,
                'max_drawdown': self.max_drawdown,
                'symbol': self.symbol,
                'steps': self.current_step
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(episode_data, f)
                
            logger.info(f"Episode data saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving episode data: {e}")

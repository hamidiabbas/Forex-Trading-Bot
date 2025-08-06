# rl_backtest_fixed.py - FINAL WORKING VERSION
"""
FIXED: Clean RL Model Backtesting Framework
Resolves numpy.skew error and pandas indexing issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import json
import logging
from dataclasses import dataclass
from scipy import stats  # ✅ FIXED: Import scipy.stats for skew/kurtosis

# RL imports
try:
    from stable_baselines3 import PPO, SAC
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Simple backtesting configuration"""
    initial_capital: float = 10000.0
    commission_rate: float = 0.0001  # 0.01% per trade
    spread_cost: float = 0.0002      # 0.02% spread
    slippage_rate: float = 0.0001    # 0.01% slippage
    max_position_size: float = 0.1   # 10% of capital per trade

class SimpleBacktester:
    """Clean, simple backtesting engine with realistic costs"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.reset()
        
    def reset(self):
        """Reset all backtesting state"""
        self.capital = self.config.initial_capital
        self.position = 0.0  # -1, 0, 1 for short, flat, long
        self.position_value = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        
        # Performance tracking
        self.equity_curve = []
        self.trades = []
        
        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.total_costs = 0.0
        self.peak_equity = self.config.initial_capital
        self.max_drawdown = 0.0
        
    def calculate_costs(self, price: float, size: float) -> float:
        """Calculate total trading costs"""
        trade_value = abs(price * size)
        commission = trade_value * self.config.commission_rate
        spread = trade_value * self.config.spread_cost
        slippage = trade_value * self.config.slippage_rate * abs(np.random.normal(0, 0.5))
        return commission + spread + slippage
    
    def execute_trade(self, action: float, price: float, timestamp: datetime) -> bool:
        """Execute trade based on RL model action"""
        # Convert action to position target (-1 to 1)
        if isinstance(action, np.ndarray):
            action = float(action[0])
        
        # Map action to position (simple threshold approach)
        if action > 0.3:
            target_position = 1.0  # Long
        elif action < -0.3:
            target_position = -1.0  # Short
        else:
            target_position = 0.0  # Flat
        
        # Check if we need to change position
        if abs(target_position - self.position) < 0.1:
            return False  # No change needed
        
        # Close existing position first if changing direction
        if self.position != 0 and target_position != self.position:
            self._close_position(price, timestamp)
        
        # Open new position if needed
        if target_position != 0 and self.position == 0:
            return self._open_position(target_position, price, timestamp)
        
        return False
    
    def _open_position(self, direction: float, price: float, timestamp: datetime) -> bool:
        """Open a new position"""
        # Calculate position size (percentage of capital)
        position_value = self.capital * self.config.max_position_size
        
        # Calculate costs
        costs = self.calculate_costs(price, position_value / price)
        
        # Check if we have enough capital
        if costs > self.capital * 0.05:  # Don't use more than 5% for costs
            return False
        
        # Execute position
        self.position = direction
        self.position_value = position_value
        self.entry_price = price
        self.entry_time = timestamp
        
        # Deduct costs
        self.capital -= costs
        self.total_costs += costs
        
        return True
    
    def _close_position(self, price: float, timestamp: datetime):
        """Close current position"""
        if self.position == 0:
            return
        
        # Calculate P&L
        if self.position > 0:  # Long position
            pnl = (price - self.entry_price) * (self.position_value / self.entry_price)
        else:  # Short position
            pnl = (self.entry_price - price) * (self.position_value / self.entry_price)
        
        # Calculate exit costs
        exit_costs = self.calculate_costs(price, self.position_value / self.entry_price)
        net_pnl = pnl - exit_costs
        
        # Update capital
        self.capital += self.position_value + net_pnl
        self.total_costs += exit_costs
        
        # Record trade
        duration = (timestamp - self.entry_time).total_seconds() / 3600 if self.entry_time else 0
        self.trades.append({
            'entry_time': self.entry_time,
            'exit_time': timestamp,
            'entry_price': self.entry_price,
            'exit_price': price,
            'side': 'long' if self.position > 0 else 'short',
            'pnl': net_pnl,
            'duration_hours': duration
        })
        
        # Update stats
        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1
        
        # Reset position
        self.position = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.entry_time = None
    
    def update_equity(self, price: float, timestamp: datetime):
        """Update equity curve and drawdown tracking"""
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        if self.position != 0:
            if self.position > 0:
                unrealized_pnl = (price - self.entry_price) * (self.position_value / self.entry_price)
            else:
                unrealized_pnl = (self.entry_price - price) * (self.position_value / self.entry_price)
        
        # Total equity
        total_equity = self.capital + self.position_value + unrealized_pnl
        
        # Track equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'unrealized_pnl': unrealized_pnl
        })
        
        # Update drawdown
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity
        else:
            current_drawdown = (self.peak_equity - total_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not self.equity_curve:
            return {
                'total_return': 0.0,
                'final_equity': self.config.initial_capital,
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'total_costs': 0.0,
                'cost_ratio': 0.0,
                'avg_trade_pnl': 0.0
            }
        
        final_equity = self.equity_curve[-1]['equity']
        total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital
        
        # Calculate daily returns for risk metrics
        equity_values = [point['equity'] for point in self.equity_curve]
        daily_returns = []
        for i in range(1, len(equity_values)):
            if equity_values[i-1] > 0:
                daily_return = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                daily_returns.append(daily_return)
        
        # Risk metrics
        if len(daily_returns) > 1:
            volatility = np.std(daily_returns) * np.sqrt(252)
            mean_return = np.mean(daily_returns)
            sharpe_ratio = mean_return / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        return {
            'total_return': total_return,
            'final_equity': final_equity,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'max_drawdown': self.max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'total_costs': self.total_costs,
            'cost_ratio': self.total_costs / self.config.initial_capital,
            'avg_trade_pnl': np.mean([t['pnl'] for t in self.trades]) if self.trades else 0
        }

class RLModelTester:
    """Test RL models with proper observation generation"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.backtester = SimpleBacktester(self.config)
        
    def create_observation(self, data: pd.DataFrame, index: int) -> Optional[np.ndarray]:
        """
        Create exactly 50 features for RL model - FIXED VERSION
        """
        if index < 50:  # Need enough history
            return None
        
        try:
            # Get current and recent data - FIXED indexing
            current = data.iloc[index]
            recent = data.iloc[max(0, index-50):index].copy()  # ✅ FIXED: Safe slicing
            
            features = []
            
            # 1. Basic price features (5 features)
            close = float(current['Close'])
            features.extend([
                float(current['Open']) / close - 1 if close > 0 else 0,
                float(current['High']) / close - 1 if close > 0 else 0,
                float(current['Low']) / close - 1 if close > 0 else 0,
                0.0,  # Close/Close - 1 = 0
                np.log(max(float(current.get('Volume', 1000000)), 1) + 1) / 20
            ])
            
            # 2. Recent price changes (10 features) - FIXED
            try:
                recent_returns = recent['Close'].pct_change().fillna(0)
                if len(recent_returns) >= 10:
                    last_10_returns = recent_returns.iloc[-10:].values
                else:
                    last_10_returns = list(recent_returns.values) + [0.0] * (10 - len(recent_returns))
                features.extend([float(x) for x in last_10_returns[:10]])
            except:
                features.extend([0.0] * 10)
            
            # 3. Moving averages (8 features) - FIXED
            for period in [5, 10, 20, 50]:
                try:
                    if len(recent) >= period:
                        ma_data = recent['Close'].iloc[-period:]  # ✅ FIXED: Safe slicing
                        ma = float(ma_data.mean())
                        features.append((close - ma) / close if close > 0 else 0)
                        features.append(ma / close - 1 if close > 0 and ma > 0 else 0)
                    else:
                        features.extend([0.0, 0.0])
                except:
                    features.extend([0.0, 0.0])
            
            # 4. Technical indicators (12 features) - FIXED
            # RSI - FIXED
            try:
                if len(recent) >= 14:
                    delta = recent['Close'].diff().fillna(0)
                    gain = delta.where(delta > 0, 0)
                    loss = (-delta.where(delta < 0, 0))
                    avg_gain = gain.rolling(14, min_periods=1).mean().iloc[-1]
                    avg_loss = loss.rolling(14, min_periods=1).mean().iloc[-1]
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        features.append((float(rsi) - 50) / 50)
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
            
            # ATR - FIXED
            try:
                if len(recent) >= 14:
                    high_low = recent['High'] - recent['Low']
                    high_close = np.abs(recent['High'] - recent['Close'].shift(1))
                    low_close = np.abs(recent['Low'] - recent['Close'].shift(1))
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = float(true_range.rolling(14, min_periods=1).mean().iloc[-1])
                    features.append(min(atr / close, 0.1) if close > 0 else 0.001)
                else:
                    features.append(0.001)
            except:
                features.append(0.001)
            
            # MACD - FIXED
            try:
                if len(recent) >= 26:
                    ema_12 = recent['Close'].ewm(span=12).mean()
                    ema_26 = recent['Close'].ewm(span=26).mean()
                    macd = float((ema_12 - ema_26).iloc[-1])
                    features.append(np.tanh(macd / close) if close > 0 else 0)
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
            
            # Fill remaining technical indicators with safe defaults
            features.extend([0.0] * 9)  # Placeholder for other indicators
            
            # 5. Statistical features (5 features) - ✅ FIXED
            try:
                returns = recent['Close'].pct_change().dropna()
                if len(returns) > 0:
                    features.extend([
                        float(np.std(returns)),  # Volatility
                        float(np.mean(returns)),  # Mean return
                        float(stats.skew(returns)) if len(returns) > 2 else 0.0,  # ✅ FIXED: scipy.stats.skew
                        float(stats.kurtosis(returns)) if len(returns) > 3 else 0.0,  # ✅ FIXED: scipy.stats.kurtosis
                        float((close - recent['Close'].min()) / (recent['Close'].max() - recent['Close'].min())) 
                        if recent['Close'].max() != recent['Close'].min() else 0.5
                    ])
                else:
                    features.extend([0.001, 0.0, 0.0, 0.0, 0.5])  # Safe defaults
            except Exception as e:
                # Fallback values if calculation fails
                features.extend([0.001, 0.0, 0.0, 0.0, 0.5])
            
            # 6. Portfolio features (10 features) - FIXED
            try:
                backtester = self.backtester
                features.extend([
                    float(backtester.position),  # Current position
                    float(backtester.position_value / backtester.config.initial_capital) if backtester.position_value > 0 else 0.0,
                    float((backtester.capital - backtester.config.initial_capital) / backtester.config.initial_capital),
                    float(backtester.max_drawdown),
                    float(backtester.total_trades) / 100.0,
                    float(backtester.winning_trades / max(backtester.total_trades, 1)),
                    float(backtester.total_costs / backtester.config.initial_capital),
                    float(index) / len(data),  # Progress through data
                    0.0,  # Reserved
                    0.0   # Reserved
                ])
            except:
                features.extend([0.0] * 10)
            
            # CRITICAL: Ensure exactly 50 features - FIXED
            if len(features) > 50:
                features = features[:50]
            elif len(features) < 50:
                features.extend([0.0] * (50 - len(features)))
            
            # Convert to numpy array and clean - FIXED
            observation = np.array(features, dtype=np.float32)
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Final validation
            if observation.shape != (50,):
                logger.error(f"CRITICAL: Observation shape {observation.shape} != (50,)")
                return None
            
            return observation
            
        except Exception as e:
            logger.error(f"Error creating observation at index {index}: {e}")
            return None
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with basic indicators - FIXED"""
        try:
            df = data.copy()
            
            # Ensure basic columns exist and are numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill missing values safely
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Add returns
            if 'Returns' not in df.columns:
                df['Returns'] = df['Close'].pct_change().fillna(0)
            
            # Final cleaning
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            logger.info(f"Prepared data: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return data.fillna(0)
    
    def backtest_model(self, model, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Backtest RL model on data - FIXED"""
        logger.info(f"Starting backtest for {symbol}")
        
        # Reset backtester
        self.backtester.reset()
        
        # Prepare data
        clean_data = self.prepare_data(data)
        
        if len(clean_data) < 100:
            logger.error(f"Insufficient data for {symbol}: {len(clean_data)} rows")
            return {
                'symbol': symbol,
                'metrics': self.backtester.get_performance_metrics(),
                'equity_curve': [],
                'trades': [],
                'actions': [],
                'successful_predictions': 0
            }
        
        # Track actions and observations
        actions = []
        successful_predictions = 0
        
        # Run backtest - FIXED loop
        for i in range(50, len(clean_data)):
            try:
                # Create observation
                obs = self.create_observation(clean_data, i)
                if obs is None:
                    continue
                
                # Get model prediction
                action, _ = model.predict(obs, deterministic=True)
                actions.append(float(action[0]) if isinstance(action, np.ndarray) else float(action))
                
                # Execute trade
                current_price = float(clean_data.iloc[i]['Close'])
                current_time = clean_data.index[i] if hasattr(clean_data.index[i], 'to_pydatetime') else datetime.now()
                
                self.backtester.execute_trade(action, current_price, current_time)
                self.backtester.update_equity(current_price, current_time)
                
                successful_predictions += 1
                
                # Progress logging
                if i % 1000 == 0:
                    logger.info(f"Processed {i}/{len(clean_data)} steps")
                
            except Exception as e:
                logger.warning(f"Error at step {i}: {e}")
                continue
        
        # Close any open position
        if self.backtester.position != 0:
            try:
                final_price = float(clean_data.iloc[-1]['Close'])
                final_time = clean_data.index[-1] if hasattr(clean_data.index[-1], 'to_pydatetime') else datetime.now()
                self.backtester._close_position(final_price, final_time)
            except Exception as e:
                logger.warning(f"Error closing final position: {e}")
        
        # Get results
        metrics = self.backtester.get_performance_metrics()
        
        logger.info(f"✅ Backtest completed: {successful_predictions} predictions, {metrics['total_trades']} trades")
        
        return {
            'symbol': symbol,
            'metrics': metrics,
            'equity_curve': self.backtester.equity_curve,
            'trades': self.backtester.trades,
            'actions': actions,
            'successful_predictions': successful_predictions
        }

def load_models(models_dir: str = "./models/") -> Dict[str, Any]:
    """Load trained RL models"""
    models = {}
    models_path = Path(models_dir)
    
    if not models_path.exists():
        logger.error(f"Models directory {models_dir} not found")
        return models
    
    for model_file in models_path.glob("*.zip"):
        try:
            model_name = model_file.stem
            
            if 'PPO' in model_name.upper():
                model = PPO.load(str(model_file))
            elif 'SAC' in model_name.upper():
                model = SAC.load(str(model_file))
            else:
                logger.warning(f"Unknown model type: {model_name}")
                continue
            
            models[model_name] = model
            logger.info(f"✅ Loaded model: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_file}: {e}")
    
    return models

def load_data() -> Dict[str, pd.DataFrame]:
    """Load historical data for backtesting"""
    data_dict = {}
    symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']
    
    try:
        from enhanced_datahandler import EnhancedDataHandler
        import sys
        sys.path.insert(0, 'configs')
        import config as user_config
        
        data_handler = EnhancedDataHandler(user_config)
        if data_handler.connect():
            for symbol in symbols:
                start_date = pd.to_datetime('2021-01-01')
                end_date = pd.to_datetime('2024-08-01')
                
                data = data_handler.get_historical_data(symbol, 'H1', start_date, end_date)
                if not data.empty:
                    data_dict[symbol] = data
                    logger.info(f"✅ Loaded {len(data)} bars for {symbol}")
                else:
                    logger.warning(f"❌ No data for {symbol}")
        
        data_handler.disconnect()
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        
        # Create synthetic data for testing if real data fails
        logger.info("Creating synthetic data for testing...")
        np.random.seed(42)
        dates = pd.date_range('2021-01-01', '2024-08-01', freq='H')[:5000]
        
        for symbol in symbols:
            price = 1.0 + np.cumsum(np.random.randn(len(dates)) * 0.0001)
            data_dict[symbol] = pd.DataFrame({
                'Open': price * (1 + np.random.randn(len(dates)) * 0.0001),
                'High': price * (1 + np.abs(np.random.randn(len(dates))) * 0.0002),
                'Low': price * (1 - np.abs(np.random.randn(len(dates))) * 0.0002),
                'Close': price,
                'Volume': 1000000 + np.random.randn(len(dates)) * 100000
            }, index=dates)
            logger.info(f"✅ Created synthetic data for {symbol}: {len(data_dict[symbol])} bars")
    
    return data_dict

def run_validation() -> Dict[str, Any]:
    """Run complete RL model validation - FIXED"""
    logger.info("🚀 Starting FIXED RL Model Validation")
    
    # Install scipy if needed
    try:
        import scipy.stats
        logger.info("✅ Scipy available")
    except ImportError:
        logger.error("❌ Please install scipy: pip install scipy")
        return {}
    
    # Load models
    models = load_models()
    if not models:
        logger.error("No models found!")
        return {}
    
    # Load data
    data_dict = load_data()
    if not data_dict:
        logger.error("No data loaded!")
        return {}
    
    # Run backtests
    config = BacktestConfig()
    tester = RLModelTester(config)
    results = {}
    
    for model_name, model in models.items():
        # Test on first available symbol
        symbol = list(data_dict.keys())[0]
        data = data_dict[symbol]
        
        logger.info(f"Testing {model_name} on {symbol}")
        result = tester.backtest_model(model, data, symbol)
        results[model_name] = result
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Create simple report
    report_lines = ["# RL Model Backtesting Results\n"]
    for name, result in results.items():
        metrics = result['metrics']
        symbol = result['symbol']
        report_lines.extend([
            f"## {name} - {symbol}\n",
            f"- **Total Return:** {metrics['total_return']:.2%}\n",
            f"- **Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}\n",
            f"- **Max Drawdown:** {metrics['max_drawdown']:.2%}\n",
            f"- **Win Rate:** {metrics['win_rate']:.2%}\n",
            f"- **Total Trades:** {int(metrics['total_trades'])}\n",
            f"- **Total Costs:** ${metrics['total_costs']:.2f}\n\n"
        ])
    
    with open(results_dir / 'backtest_report.md', 'w') as f:
        f.writelines(report_lines)
    
    # Save detailed results
    summary_results = {}
    for name, result in results.items():
        summary_results[name] = {
            'symbol': result['symbol'],
            'metrics': result['metrics'],
            'total_predictions': result['successful_predictions'],
            'total_trades': len(result['trades'])
        }
    
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(summary_results, f, indent=2, default=str)
    
    logger.info(f"✅ Results saved to {results_dir}/")
    
    return results

def print_results_summary(results: Dict[str, Any]):
    """Print clean results summary"""
    print("\n" + "="*70)
    print("🎯 RL MODEL BACKTESTING RESULTS - FIXED VERSION")
    print("="*70)
    
    if not results:
        print("❌ No results to display")
        return
    
    for model_name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            symbol = result['symbol']
            
            print(f"\n🤖 {model_name} ({symbol}):")
            print(f"   💰 Total Return:    {metrics['total_return']:>8.2%}")
            print(f"   📊 Sharpe Ratio:    {metrics['sharpe_ratio']:>8.2f}")
            print(f"   📉 Max Drawdown:    {metrics['max_drawdown']:>8.2%}")
            print(f"   🎯 Win Rate:        {metrics['win_rate']:>8.2%}")
            print(f"   🏪 Total Trades:    {int(metrics['total_trades']):>8}")
            print(f"   💸 Total Costs:     ${metrics['total_costs']:>7.2f}")
            print(f"   📈 Predictions:     {result.get('successful_predictions', 0):>8}")
    
    print("\n📁 Detailed results saved to: ./results/")
    print("📄 Read backtest_report.md for full analysis")
    print("="*70)

if __name__ == "__main__":
    # Install scipy first if needed
    try:
        import scipy.stats
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scipy'])
    
    # Run the validation
    results = run_validation()
    
    if results:
        print_results_summary(results)
    else:
        print("❌ Validation failed - check logs for errors")

"""
Fixed RL Model Evaluation Script
"""
import pandas as pd
from datetime import datetime
import pytz
from stable_baselines3 import A2C
import numpy as np
import matplotlib.pyplot as plt

import config
from data_handler import DataHandler
from market_intelligence import MarketIntelligence
from trading_environment import TradingEnvironment

# Configuration
SYMBOL = 'EURUSD'
EVALUATION_START_DATE = '2023-01-01'
EVALUATION_END_DATE = '2024-01-01'
TIMEFRAME = 'H1'
MODEL_PATH = f"model_rl_{SYMBOL}_final_fixed.zip"

def prepare_evaluation_data():
    """Prepare data for evaluation (same process as training)"""
    print("--- Preparing Evaluation Data ---")
    data_handler = DataHandler(config)
    market_intel = MarketIntelligence(data_handler, config)
    
    data_handler.connect()
    timezone = pytz.timezone("Etc/UTC")
    start_date_dt = datetime.strptime(EVALUATION_START_DATE, '%Y-%m-%d').replace(tzinfo=timezone)
    end_date_dt = datetime.strptime(EVALUATION_END_DATE, '%Y-%m-%d').replace(tzinfo=timezone)
    
    df = data_handler.get_data_by_range(SYMBOL, TIMEFRAME, start_date_dt, end_date_dt)
    data_handler.disconnect()
    
    if df is None or df.empty:
        print("Could not fetch evaluation data.")
        return None

    df_features = market_intel._analyze_data(df).copy()
    
    # Same feature engineering as training
    features_to_drop = ['Open', 'High', 'Low', 'Volume', 'hurst', 
                       'fib_0.236', 'fib_0.382', 'fib_0.500', 'fib_0.618']
    df_features.drop(columns=features_to_drop, inplace=True, errors='ignore')
    
    # Add same additional features
    df_features['price_change'] = df_features['Close'].pct_change()
    df_features['volatility'] = df_features['price_change'].rolling(20).std()
    df_features['momentum'] = df_features['Close'].rolling(10).apply(lambda x: (x[-1] - x[0]) / x[0])
    
    df_features = df_features.fillna(method='ffill').fillna(method='bfill')
    df_features.dropna(inplace=True)
    
    print(f"Evaluation data prepared: {len(df_features)} bars")
    return df_features

def evaluate_agent(df):
    """FIXED: Comprehensive evaluation with proper metrics"""
    print(f"\n--- Evaluating RL Agent from {MODEL_PATH} ---")
    
    try:
        model = A2C.load(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    env = TradingEnvironment(df)
    obs, _ = env.reset()
    
    # Tracking variables
    actions_taken = []
    rewards_received = []
    equity_curve = [env.initial_balance]
    positions_history = []
    trade_log = []
    
    print("Running evaluation...")
    for i in range(len(df) - 1):
        action, _states = model.predict(obs, deterministic=True)
        prev_position = env.position
        prev_equity = env.equity
        
        obs, reward, done, truncated, info = env.step(action)
        
        # Track everything
        actions_taken.append(int(action))
        rewards_received.append(reward)
        equity_curve.append(env.equity)
        positions_history.append(env.position)
        
        # Log trades
        if prev_position == 0 and env.position != 0:  # Position opened
            trade_log.append({
                'open_time': df.index[i] if hasattr(df, 'index') else i,
                'open_price': env.entry_price,
                'direction': 'LONG' if env.position == 1 else 'SHORT',
                'position_size': env.position_size,
                'status': 'OPEN'
            })
        elif prev_position != 0 and env.position == 0:  # Position closed
            if trade_log and trade_log[-1]['status'] == 'OPEN':
                trade_log[-1].update({
                    'close_time': df.index[i] if hasattr(df, 'index') else i,
                    'close_price': df['Close'].iloc[i],
                    'pnl': env.equity - prev_equity,
                    'status': 'CLOSED'
                })
        
        if done:
            break
    
    # Calculate comprehensive metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Basic stats
    total_return = (env.equity - env.initial_balance) / env.initial_balance * 100
    closed_trades = [t for t in trade_log if t['status'] == 'CLOSED']
    
    print(f"Initial Balance:     ${env.initial_balance:,.2f}")
    print(f"Final Equity:        ${env.equity:,.2f}")
    print(f"Total Return:        {total_return:.2f}%")
    print(f"Total Trades:        {len(closed_trades)}")
    
    if closed_trades:
        profitable_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] <= 0]
        
        win_rate = len(profitable_trades) / len(closed_trades) * 100
        avg_win = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t['pnl'] for t in profitable_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
        
        print(f"Win Rate:            {win_rate:.1f}%")
        print(f"Profitable Trades:   {len(profitable_trades)}")
        print(f"Losing Trades:       {len(losing_trades)}")
        print(f"Average Win:         ${avg_win:,.2f}")
        print(f"Average Loss:        ${avg_loss:,.2f}")
        print(f"Profit Factor:       {profit_factor:.2f}")
        
        # Risk metrics
        equity_curve_series = pd.Series(equity_curve)
        drawdowns = (equity_curve_series / equity_curve_series.cummax() - 1) * 100
        max_drawdown = drawdowns.min()
        
        print(f"Max Drawdown:        {max_drawdown:.2f}%")
        
        # Sharpe-like ratio (simplified)
        returns = equity_curve_series.pct_change().dropna()
        if len(returns) > 1 and returns.std() != 0:
            sharpe_like = returns.mean() / returns.std() * np.sqrt(252 * 24)  # Annualized for hourly data
            print(f"Risk-Adjusted Return: {sharpe_like:.2f}")
    
    # Action distribution
    action_counts = pd.Series(actions_taken).value_counts().sort_index()
    print(f"\nAction Distribution:")
    action_names = {0: 'Hold/Close', 1: 'Buy/Long', 2: 'Sell/Short'}
    for action, count in action_counts.items():
        percentage = count / len(actions_taken) * 100
        print(f"  {action_names.get(action, f'Action {action}')}: {count} ({percentage:.1f}%)")
    
    # Trading activity check
    if len(closed_trades) == 0:
        print("\n⚠️  WARNING: NO TRADES WERE EXECUTED!")
        print("This suggests the model learned to never trade.")
        print("Consider:")
        print("1. Adjusting the reward function")
        print("2. Increasing exploration during training")
        print("3. Checking the action space implementation")
    else:
        print(f"\n✅ Model executed {len(closed_trades)} trades successfully.")
    
    # Save results
    results = {
        'equity_curve': equity_curve,
        'actions': actions_taken,
        'positions': positions_history,
        'trades': closed_trades,
        'total_return': total_return,
        'win_rate': win_rate if closed_trades else 0,
        'max_drawdown': max_drawdown if closed_trades else 0
    }
    
    return results

if __name__ == '__main__':
    eval_data = prepare_evaluation_data()
    if eval_data is not None:
        results = evaluate_agent(eval_data)
        print("\nEvaluation completed!")
    else:
        print("Evaluation failed - could not prepare data.")

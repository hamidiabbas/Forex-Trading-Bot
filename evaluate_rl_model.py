"""
/******************************************************************************
 *
 * FILE NAME:           evaluate_rl_model.py (Complete & Verified)
 *
 * PURPOSE:
 *
 * This script evaluates the performance of a pre-trained Reinforcement
 * Learning agent by running it through a simulated trading environment
 * and generating a detailed performance report. This is the complete and
 * correct version of the script.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 28, 2025
 *
 * VERSION:             62.2 (Complete & Verified)
 *
 ******************************************************************************/
"""
import pandas as pd
from datetime import datetime
import pytz
from stable_baselines3 import PPO

import config
from data_handler import DataHandler
from market_intelligence import MarketIntelligence
from trading_environment import TradingEnvironment
from performance_analyzer import PerformanceAnalyzer

# --- 1. CONFIGURATION ---
SYMBOL = 'EURUSD'
EVALUATION_START_DATE = '2023-01-01' # Use a different period than training
EVALUATION_END_DATE = '2024-01-01'
TIMEFRAME = 'H1'
MODEL_PATH = f"model_rl_{SYMBOL}.zip"

def prepare_data():
    """ Fetches and prepares the market data for the evaluation environment. """
    print("--- Preparing Data for RL Evaluation ---")
    data_handler = DataHandler(config)
    market_intel = MarketIntelligence(data_handler, config)
    
    data_handler.connect()
    timezone = pytz.timezone("Etc/UTC")
    start_date_dt = datetime.strptime(EVALUATION_START_DATE, '%Y-%m-%d').replace(tzinfo=timezone)
    end_date_dt = datetime.strptime(EVALUATION_END_DATE, '%Y-%m-%d').replace(tzinfo=timezone)
    
    df = data_handler.get_data_by_range(SYMBOL, TIMEFRAME, start_date_dt, end_date_dt)
    data_handler.disconnect()
    
    if df is None or df.empty:
        print("Could not fetch data. Aborting.")
        return None

    print("Calculating features...")
    df_features = market_intel._analyze_data(df.copy())
    
    # Drop columns that are not useful as direct observations for the RL agent
    features_to_drop = ['Open', 'High', 'Low', 'Volume', 'hurst']
    df_features.drop(columns=features_to_drop, inplace=True, errors='ignore')
    
    df_features.dropna(inplace=True)
    
    print(f"Feature preparation complete. Dataset has {len(df_features)} bars.")
    return df_features

def evaluate_agent(df):
    """ Loads the RL agent and evaluates its performance. """
    print(f"\n--- Evaluating Reinforcement Learning Agent from {MODEL_PATH} ---")
    
    # 1. Load the trained model
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Create the environment with the evaluation data
    env = TradingEnvironment(df)
    obs, _ = env.reset()
    
    equity_curve = [{'timestamp': df.index[0], 'equity': env.initial_balance}]
    trade_log = []
    open_trade = None

    # 3. Loop through the evaluation data
    for i in range(len(df) - 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        equity_curve.append({'timestamp': df.index[i+1], 'equity': env.equity})

        # --- Improved Trade Logging ---
        position_just_closed = open_trade and info.get('current_position', 0) == 0
        position_just_opened = not open_trade and info.get('current_position', 0) != 0

        # Log the closing of a trade
        if position_just_closed:
            open_trade['exit_price'] = df['Close'].iloc[i]
            open_trade['exit_time'] = df.index[i]
            if open_trade['direction'] == 'BUY':
                open_trade['profit'] = (open_trade['exit_price'] - open_trade['entry_price'])
            else: # SELL
                open_trade['profit'] = (open_trade['entry_price'] - open_trade['exit_price'])
            trade_log.append(open_trade)
            open_trade = None
        
        # Log the opening of a new trade
        if position_just_opened:
            open_trade = {
                'symbol': SYMBOL,
                'strategy': 'Reinforcement-Learning',
                'direction': 'BUY' if info['current_position'] == 1 else 'SELL',
                'entry_price': df['Close'].iloc[i],
                'entry_time': df.index[i],
            }

        if done:
            break

    # 4. Analyze the performance
    if trade_log:
        trade_log_df = pd.DataFrame(trade_log)
        equity_curve_df = pd.DataFrame(equity_curve).set_index('timestamp')
        
        analyzer = PerformanceAnalyzer(trade_log_df=trade_log_df, equity_curve_df=equity_curve_df)
        report = analyzer.analyze()
        
        print("\n--- RL Agent Performance Report ---")
        for key, value in report.items():
            if isinstance(value, (int, float)):
                 print(f"  {key.replace('_', ' ').title()}: {value:,.2f}")
            else:
                 print(f"  {key.replace('_', ' ').title()}: {value}")
    else:
        print("\nNo trades were executed by the RL agent.")


if __name__ == '__main__':
    market_data = prepare_data()
    if market_data is not None:
        evaluate_agent(market_data)
"""
/******************************************************************************
 *
 * FILE NAME:           evaluate_rl_model.py (Final Corrected)
 *
 * PURPOSE:
 *
 * This version corrects a FileNotFoundError by updating the MODEL_PATH to
 * point to the correct saved model file. It also resolves the pandas
 * SettingWithCopyWarning.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 29, 2025
 *
 * VERSION:             78.0 (Final Corrected)
 *
 ******************************************************************************/
"""
import pandas as pd
from datetime import datetime
import pytz
from stable_baselines3 import PPO

import configs.config as config
from core.data_handler import DataHandler
from core.market_intelligence import MarketIntelligence
from trading_environment import TradingEnvironment
from utils.performance_analyzer import PerformanceAnalyzer

# --- 1. CONFIGURATION ---
SYMBOL = 'EURUSD'
EVALUATION_START_DATE = '2023-01-01'
EVALUATION_END_DATE = '2024-01-01'
TIMEFRAME = 'H1'
# --- THIS IS THE FIX ---
# The filename now correctly points to the model saved by the training script.
MODEL_PATH = f"model_rl_{SYMBOL}_final.zip"

def prepare_data():
    """ Fetches and prepares the market data for the evaluation environment. """
    print("--- Preparing Evaluation Data ---")
    data_handler = DataHandler(configmanager)
    market_intel = MarketIntelligence(data_handler, configmanager)
    
    data_handler.connect()
    timezone = pytz.timezone("Etc/UTC")
    start_date_dt = datetime.strptime(EVALUATION_START_DATE, '%Y-%m-%d').replace(tzinfo=timezone)
    end_date_dt = datetime.strptime(EVALUATION_END_DATE, '%Y-%m-%d').replace(tzinfo=timezone)
    
    df = data_handler.get_data_by_range(SYMBOL, TIMEFRAME, start_date_dt, end_date_dt)
    data_handler.disconnect()
    
    if df is None or df.empty:
        print("Could not fetch data. Aborting.")
        return None

    # --- FIX for SettingWithCopyWarning ---
    # We create a new dataframe explicitly to avoid warnings.
    df_features = market_intel._analyze_data(df).copy()
    
    features_to_drop = ['Open', 'High', 'Low', 'Volume', 'hurst', 'fib_0.236', 'fib_0.382', 'fib_0.500', 'fib_0.618']
    df_final = df_features.drop(columns=features_to_drop, errors='ignore').dropna()
    
    print(f"Evaluation data prepared: {len(df_final)} bars")
    return df_final

def evaluate_agent(df):
    """ Loads the RL agent and evaluates its performance. """
    print(f"\n--- Evaluating RL Agent from {MODEL_PATH} ---")
    
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    env = TradingEnvironment(df)
    obs, _ = env.reset()
    
    equity_curve = [{'timestamp': df.index[0], 'equity': env.initial_balance}]
    trade_log = []
    open_trade = None

    for i in range(len(df) - 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        equity_curve.append({'timestamp': df.index[i+1], 'equity': env.equity})

        position_just_closed = open_trade and info.get('current_position', 0) == 0
        position_just_opened = not open_trade and info.get('current_position', 0) != 0

        if position_just_closed:
            open_trade['exit_price'] = df['Close'].iloc[i]
            open_trade['exit_time'] = df.index[i]
            if open_trade['direction'] == 'BUY':
                open_trade['profit'] = (open_trade['exit_price'] - open_trade['entry_price'])
            else:
                open_trade['profit'] = (open_trade['entry_price'] - open_trade['exit_price'])
            trade_log.append(open_trade)
            open_trade = None
        
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
        print("\nEvaluation completed!")
    else:
        print("\nNo trades were executed by the RL agent.")
        print("\nEvaluation completed!")


if __name__ == '__main__':
    market_data = prepare_data()
    if market_data is not None:
        evaluate_agent(market_data)
"""
/******************************************************************************
 *
 * FILE NAME:           train_rl_model.py (Final Version)
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 29, 2025
 *
 * VERSION:             77.0 (Final)
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

# --- 1. CONFIGURATION ---
SYMBOL = 'EURUSD'
START_DATE = '2022-01-01'
END_DATE = '2024-01-01'
TIMEFRAME = 'H1'
TRAINING_STEPS = 250000

def prepare_data():
    """ Fetches and prepares the market data for the environment. """
    print("--- Preparing Data for RL Environment ---")
    data_handler = DataHandler(config)
    market_intel = MarketIntelligence(data_handler, config)
    
    data_handler.connect()
    timezone = pytz.timezone("Etc/UTC")
    start_date_dt = datetime.strptime(START_DATE, '%Y-%m-%d').replace(tzinfo=timezone)
    end_date_dt = datetime.strptime(END_DATE, '%Y-%m-%d').replace(tzinfo=timezone)
    
    df = data_handler.get_data_by_range(SYMBOL, TIMEFRAME, start_date_dt, end_date_dt)
    data_handler.disconnect()
    
    if df is None or df.empty:
        print("Could not fetch data. Aborting.")
        return None

    print("Calculating features...")
    df_features = market_intel._analyze_data(df).copy()
    
    # --- THIS IS THE FIX ---
    # The list of columns to drop now perfectly matches what the environment expects
    features_to_drop = ['Open', 'High', 'Low', 'Volume', 'hurst', 'fib_0.236', 'fib_0.382', 'fib_0.500', 'fib_0.618']
    df_features.drop(columns=features_to_drop, inplace=True, errors='ignore')
    
    df_features.dropna(inplace=True)
    
    print(f"Feature preparation complete. Dataset has {len(df_features)} bars.")
    return df_features

def train_agent(df):
    """ Creates the environment and trains the PPO agent. """
    print("\n--- Training Reinforcement Learning Agent ---")
    
    env = TradingEnvironment(df)
    
    model = PPO("MlpPolicy", env, verbose=1)
    
    print(f"Starting training for {TRAINING_STEPS} steps...")
    model.learn(total_timesteps=TRAINING_STEPS)
    
    model.save(f"model_rl_{SYMBOL}_final.zip")
    print(f"\nTraining complete. Final RL model saved to 'model_rl_{SYMBOL}_final.zip'")

if __name__ == '__main__':
    market_data = prepare_data()
    if market_data is not None:
        train_agent(market_data)
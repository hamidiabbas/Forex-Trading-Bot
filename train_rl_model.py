"""
/******************************************************************************
 *
 * FILE NAME:           train_rl_model.py
 *
 * PURPOSE:
 *
 * This script trains a Reinforcement Learning agent to trade in a custom
 * simulated forex environment using the Stable-Baselines3 library. This is
 * the complete and correct version of this file.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 26, 2025
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

# --- 1. CONFIGURATION ---
SYMBOL = 'EURUSD'
START_DATE = '2022-01-01'
END_DATE = '2024-01-01'
TIMEFRAME = 'H1'
TRAINING_STEPS = 200000

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
    # Analyze data to get all indicators
    df_features = market_intel._analyze_data(df.copy())
    
    # Drop columns that are not useful as direct observations for the RL agent
    # The 'Close' price is kept because the environment needs it to calculate equity
    features_to_drop = ['Open', 'High', 'Low', 'Volume', 'hurst']
    df_features.drop(columns=features_to_drop, inplace=True, errors='ignore')
    
    df_features.dropna(inplace=True)
    
    print(f"Feature preparation complete. Dataset has {len(df_features)} bars.")
    return df_features

def train_agent(df):
    """ Creates the environment and trains the PPO agent. """
    print("\n--- Training Reinforcement Learning Agent ---")
    
    # Create the custom trading environment
    env = TradingEnvironment(df)
    
    # Create the PPO agent, a type of Actor-Critic model
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the agent
    print(f"Starting training for {TRAINING_STEPS} steps...")
    model.learn(total_timesteps=TRAINING_STEPS)
    
    # Save the trained model
    model.save(f"model_rl_{SYMBOL}.zip")
    print(f"\nTraining complete. RL model saved to 'model_rl_{SYMBOL}.zip'")

if __name__ == '__main__':
    market_data = prepare_data()
    if market_data is not None:
        train_agent(market_data)
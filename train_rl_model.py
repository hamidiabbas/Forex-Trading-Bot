"""
Fixed RL Training Script - Addresses algorithm choice and training configuration
"""
import pandas as pd
from datetime import datetime
import pytz
from stable_baselines3 import A2C  # FIXED: Better algorithm for trading
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import numpy as np

import config
from data_handler import DataHandler
from market_intelligence import MarketIntelligence
from trading_environment import TradingEnvironment

# FIXED: Proper configuration
SYMBOL = 'EURUSD'
START_DATE = '2020-01-01'  # More data for better training
END_DATE = '2023-01-01'
TIMEFRAME = 'H1'
TRAINING_STEPS = 2000000  # FIXED: Increased training steps
EVAL_FREQ = 50000
EVAL_EPISODES = 10

def prepare_data():
    """Fetches and prepares market data with proper feature engineering"""
    print("--- Preparing Enhanced Data for RL Environment ---")
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
        return None, None

    print("Calculating enhanced feature set...")
    df_features = market_intel._analyze_data(df).copy()
    
    # FIXED: Keep only relevant features and handle NaN properly
    features_to_drop = ['Open', 'High', 'Low', 'Volume', 'hurst', 
                       'fib_0.236', 'fib_0.382', 'fib_0.500', 'fib_0.618']
    df_features.drop(columns=features_to_drop, inplace=True, errors='ignore')
    
    # Add momentum and volatility features
    df_features['price_change'] = df_features['Close'].pct_change()
    df_features['volatility'] = df_features['price_change'].rolling(20).std()
    df_features['momentum'] = df_features['Close'].rolling(10).apply(lambda x: (x[-1] - x[0]) / x[0])
    
    # Fill NaN values and remove any remaining NaN rows
    df_features = df_features.fillna(method='ffill').fillna(method='bfill')
    df_features.dropna(inplace=True)
    
    # Split data for training and validation
    split_idx = int(len(df_features) * 0.8)
    train_data = df_features.iloc[:split_idx].copy()
    val_data = df_features.iloc[split_idx:].copy()
    
    print(f"Data preparation complete:")
    print(f"  Training set: {len(train_data)} bars")
    print(f"  Validation set: {len(val_data)} bars")
    print(f"  Features: {list(train_data.columns)}")
    
    return train_data, val_data

def train_agent(train_data, val_data):
    """FIXED: Enhanced training with proper validation and callbacks"""
    print(f"\n--- Training Enhanced RL Agent ---")
    
    # Create environments
    train_env = TradingEnvironment(train_data)
    val_env = TradingEnvironment(val_data)
    
    # Verify environment
    check_env(train_env)
    print("Environment check passed!")
    
    # FIXED: A2C with proper hyperparameters for trading
    model = A2C(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=0.0003,  # Conservative learning rate
        n_steps=2048,          # Longer rollouts for better samples
        gamma=0.99,            # High discount for long-term rewards
        gae_lambda=0.95,       # Good GAE parameter
        ent_coef=0.01,         # Slight exploration bonus
        vf_coef=0.25,          # Value function coefficient
        max_grad_norm=0.5,     # Gradient clipping
        policy_kwargs=dict(
            net_arch=[512, 512, 256],  # Larger network for complex patterns
            activation_fn=torch.nn.ReLU
        )
    )
    
    # Set up evaluation callback
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=f'./best_model_{SYMBOL}/',
        log_path=f'./logs_{SYMBOL}/',
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False
    )
    
    # Reward threshold callback (stop if we achieve good performance)
    reward_threshold_callback = StopTrainingOnRewardThreshold(
        reward_threshold=0.5,  # Stop if average reward > 0.5
        verbose=1
    )
    
    print(f"Starting training for {TRAINING_STEPS:,} steps...")
    print("Training progress will be logged. This may take several hours.")
    
    try:
        model.learn(
            total_timesteps=TRAINING_STEPS,
            callback=[eval_callback, reward_threshold_callback],
            progress_bar=True
        )
        
        # Save final model
        model.save(f"model_rl_{SYMBOL}_final_fixed.zip")
        print(f"\nTraining complete! Models saved:")
        print(f"  Final model: model_rl_{SYMBOL}_final_fixed.zip")
        print(f"  Best model: ./best_model_{SYMBOL}/best_model.zip")
        
        return model
        
    except Exception as e:
        print(f"Training error: {e}")
        # Save whatever we have
        model.save(f"model_rl_{SYMBOL}_interrupted.zip")
        return None

if __name__ == '__main__':
    import torch  # Import here to avoid issues if not needed
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    train_data, val_data = prepare_data()
    if train_data is not None and val_data is not None:
        trained_model = train_agent(train_data, val_data)
        if trained_model:
            print("Training completed successfully!")
        else:
            print("Training encountered issues. Check logs.")
    else:
        print("Data preparation failed. Cannot proceed with training.")

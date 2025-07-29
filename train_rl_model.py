import pandas as pd
import numpy as np
import torch
from datetime import datetime
import pytz
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os

import configs.config as config
from core.data_handler import DataHandler
from core.market_intelligence import MarketIntelligence
from core.trading_environment import TradingEnvironment

# FIXED: SAC Configuration optimized for trading
SYMBOL = 'EURUSD'
START_DATE = '2020-01-01'
END_DATE = '2023-01-01'
TIMEFRAME = 'H1'
TRAINING_STEPS = 1_500_000  # SAC typically needs fewer steps than A2C
EVAL_FREQ = 25_000          # More frequent evaluation for SAC
EVAL_EPISODES = 5           # Fewer episodes but more frequent

def prepare_data():
    """Fetches and prepares market data with enhanced feature engineering"""
    print("--- Preparing Enhanced Data for SAC RL Environment ---")
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
    
    # Keep only numeric columns and handle NaN properly
    features_to_drop = ['Open', 'High', 'Low', 'Volume', 'hurst', 
                       'fib_0.236', 'fib_0.382', 'fib_0.500', 'fib_0.618']
    df_features.drop(columns=features_to_drop, inplace=True, errors='ignore')
    
    # Enhanced feature engineering for SAC
    df_features['price_change'] = df_features['Close'].pct_change()
    df_features['volatility'] = df_features['price_change'].rolling(20).std()
    df_features['momentum'] = df_features['Close'].pct_change(periods=10)
    
    # Additional features that help SAC
    df_features['rsi_momentum'] = df_features['RSI_14'].diff()
    df_features['macd_signal_diff'] = df_features['MACD_12_26_9'] - df_features['MACDs_12_26_9']
    df_features['bb_position'] = (df_features['Close'] - df_features['BBL_20_2.0']) / (df_features['BBU_20_2.0'] - df_features['BBL_20_2.0'])
    
    # Use modern pandas methods
    df_features = df_features.ffill().bfill()
    df_features.dropna(inplace=True)
    
    # Split data with more validation data for SAC
    split_idx = int(len(df_features) * 0.75)  # 75/25 split for SAC
    train_data = df_features.iloc[:split_idx].copy()
    val_data = df_features.iloc[split_idx:].copy()
    
    print(f"Data preparation complete:")
    print(f"  Training set: {len(train_data)} bars")
    print(f"  Validation set: {len(val_data)} bars")
    print(f"  Features: {len(df_features.columns)} total")
    
    # Save data for analysis
    train_data.to_csv('sac_train_data.csv', index=False)
    val_data.to_csv('sac_val_data.csv', index=False)
    
    return train_data, val_data

def create_sac_model(train_env):
    """Create SAC model optimized for trading"""
    print("Creating SAC model optimized for trading environments...")
    
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=0.0001,           # Conservative learning rate
        buffer_size=200_000,            # Large replay buffer for diverse experiences
        batch_size=256,                 # Good batch size for stable learning
        tau=0.005,                      # Soft update coefficient
        gamma=0.99,                     # High discount for long-term rewards
        train_freq=1,                   # Train after every step
        gradient_steps=1,               # One gradient step per train call
        ent_coef='auto',                # Automatic entropy tuning - crucial for exploration
        target_update_interval=1,       # Frequent target updates
        learning_starts=10_000,         # Start learning after collecting experience
        use_sde=False,                  # State-dependent exploration
        sde_sample_freq=-1,             # No SDE sampling
        use_sde_at_warmup=False,
        policy_kwargs=dict(
            net_arch=[400, 300],        # Larger networks for complex patterns
            activation_fn=torch.nn.ReLU,
            use_expln=False,            # Don't use expln for continuous actions
            clip_mean=2.0,              # Clip mean for stability
            features_extractor_kwargs=dict(features_dim=128)
        ),
        tensorboard_log=f"./sac_tensorboard_{SYMBOL}/",
        seed=42  # For reproducibility
    )
    
    print("SAC model created successfully!")
    return model

def train_sac_agent(train_data, val_data):
    """Enhanced SAC training with proper validation and callbacks"""
    print(f"\n--- Training SAC RL Agent ---")
    
    # Create environments with Monitor wrapper for better logging
    train_env = TradingEnvironment(train_data)
    train_env = Monitor(train_env, filename=f"./sac_logs_{SYMBOL}/train_monitor.csv")
    
    val_env = TradingEnvironment(val_data)
    val_env = Monitor(val_env, filename=f"./sac_logs_{SYMBOL}/val_monitor.csv")
    
    # Create directories
    os.makedirs(f"./sac_logs_{SYMBOL}", exist_ok=True)
    os.makedirs(f"./best_sac_model_{SYMBOL}", exist_ok=True)
    
    # Verify environment
    check_env(train_env)
    print("Environment check passed!")
    
    # Create SAC model
    model = create_sac_model(train_env)
    
    # SAC-specific callbacks
    stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=0.3, verbose=1)
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=f'./best_sac_model_{SYMBOL}/',
        log_path=f'./sac_logs_{SYMBOL}/',
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        callback_on_new_best=stop_train_callback,
        verbose=1,
        warn=False  # Suppress monitor warnings
    )
    
    print(f"Starting SAC training for {TRAINING_STEPS:,} steps...")
    print("SAC will explore more actively than A2C - expect to see more trading activity!")
    
    try:
        # Pre-training diagnostic
        print("\n=== PRE-TRAINING DIAGNOSTIC ===")
        obs, _ = train_env.reset()
        for i in range(10):
            action = train_env.action_space.sample()  # Random actions
            obs, reward, done, _, info = train_env.step(action)
            print(f"Random Action {action}: Reward={reward:.4f}, Position={info.get('current_position', 0)}")
            if done:
                obs, _ = train_env.reset()
        print("============================\n")
        
        # Start training
        model.learn(
            total_timesteps=TRAINING_STEPS,
            callback=eval_callback,
            progress_bar=True,
            log_interval=100  # Log every 100 episodes
        )
        
        # Save final model
        model.save(f"model_sac_{SYMBOL}_final.zip")
        print(f"\nSAC Training complete! Models saved:")
        print(f"  Final model: model_sac_{SYMBOL}_final.zip")
        print(f"  Best model: ./best_sac_model_{SYMBOL}/best_model.zip")
        
        # Post-training diagnostic
        print("\n=== POST-TRAINING DIAGNOSTIC ===")
        test_trading_behavior(model, val_env)
        
        return model
        
    except Exception as e:
        print(f"Training error: {e}")
        # Save whatever we have
        model.save(f"model_sac_{SYMBOL}_interrupted.zip")
        return None

def test_trading_behavior(model, test_env):
    """Test the trained model's trading behavior"""
    print("Testing SAC model trading behavior...")
    
    obs, _ = test_env.reset()
    actions_taken = []
    rewards_received = []
    positions_held = []
    
    for i in range(200):  # Test 200 steps
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(int(action))
        
        obs, reward, done, _, info = test_env.step(action)
        rewards_received.append(reward)
        positions_held.append(info.get('current_position', 0))
        
        if done:
            obs, _ = test_env.reset()
    
    # Analysis
    from collections import Counter
    action_dist = Counter(actions_taken)
    position_changes = sum(1 for i in range(1, len(positions_held)) if positions_held[i] != positions_held[i-1])
    
    print(f"Action Distribution (200 samples):")
    print(f"  Hold/Close (0): {action_dist[0]} ({action_dist[0]/2:.1f}%)")
    print(f"  Buy (1):       {action_dist[1]} ({action_dist[1]/2:.1f}%)")
    print(f"  Sell (2):      {action_dist[2]} ({action_dist[2]/2:.1f}%)")
    print(f"Position Changes: {position_changes}")
    print(f"Average Reward: {np.mean(rewards_received):.6f}")
    print(f"Reward Std:     {np.std(rewards_received):.6f}")
    
    # Assessment
    trading_activity = ((action_dist[1] + action_dist[2]) / 200) * 100
    if trading_activity > 15:
        print(f"✅ EXCELLENT: High trading activity ({trading_activity:.1f}%)")
    elif trading_activity > 5:
        print(f"✅ GOOD: Moderate trading activity ({trading_activity:.1f}%)")
    else:
        print(f"⚠️ WARNING: Low trading activity ({trading_activity:.1f}%)")
    
    print("================================")

if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("="*60)
    print("SAC (Soft Actor-Critic) RL Trading Bot Training")
    print("="*60)
    print("SAC Advantages:")
    print("✅ Better exploration through entropy regularization")
    print("✅ Handles sparse rewards more effectively")
    print("✅ More stable than policy gradient methods")
    print("✅ Automatic entropy tuning")
    print("="*60)
    
    train_data, val_data = prepare_data()
    if train_data is not None and val_data is not None:
        trained_model = train_sac_agent(train_data, val_data)
        if trained_model:
            print("SAC Training completed successfully!")
            print("\nNext steps:")
            print("1. Run evaluate_sac_model.py to test performance")
            print("2. Integrate SAC model into main.py")
            print("3. Monitor live trading performance")
        else:
            print("SAC Training encountered issues. Check logs.")
    else:
        print("Data preparation failed. Cannot proceed with training.")

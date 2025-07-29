# ✅ FIXED: Complete SAC Training Script
"""
SAC Model Training Script - Enhanced and Fixed
"""
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from configs.config_manager import config_manager
from core.trading_environment import TradingEnvironment
import os

def train_sac_model():
    """Train SAC model with proper configuration"""
    try:
        # Load configuration
        config = config_manager
        symbols = config.get_trading_symbols()
        primary_symbol = symbols[0] if symbols else 'EURUSD'
        
        print(f"🚀 Starting SAC model training for {primary_symbol}")
        
        # Create training environment
        env = TradingEnvironment(symbol=primary_symbol, config=config.config)
        env = DummyVecEnv([lambda: env])
        
        # Create evaluation environment
        eval_env = TradingEnvironment(symbol=primary_symbol, config=config.config)
        eval_env = DummyVecEnv([lambda: eval_env])
        
        # ✅ FIXED: Configure SAC with optimal hyperparameters
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=1000,
            verbose=1,
            tensorboard_log="./sac_tensorboard/"
        )
        
        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f'./best_sac_model_{primary_symbol}/',
            log_path=f'./sac_logs_{primary_symbol}/',
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=1000,
            verbose=1
        )
        
        # Train model
        print("🏋️ Training SAC model...")
        model.learn(
            total_timesteps=100000,
            callback=[eval_callback, stop_callback],
            log_interval=100
        )
        
        # Save final model
        final_model_path = f'./model_sac_{primary_symbol}_final.zip'
        model.save(final_model_path)
        print(f"✅ Final SAC model saved to {final_model_path}")
        
        # Update config with new model path
        config_manager.config['rl_model']['model_path'] = f'./best_sac_model_{primary_symbol}/best_model.zip'
        
        print("✅ SAC model training completed successfully!")
        
    except Exception as e:
        print(f"❌ SAC training failed: {e}")

if __name__ == "__main__":
    train_sac_model()

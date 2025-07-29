# ✅ FIXED: Complete RL Model Evaluation Script
"""
SAC Model Evaluation Script - Updated for SAC Algorithm
"""
import os
import numpy as np
import pandas as pd
from stable_baselines3 import SAC  # ✅ FIXED: Changed from PPO to SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import MetaTrader5 as mt5
from configs.config_manager import config_manager
from core.trading_environment import TradingEnvironment

class SACModelEvaluator:
    """
    ✅ FIXED: SAC Model Evaluator (previously PPO)
    """
    
    def __init__(self, model_path: str = None):
        self.config = config_manager
        
        # ✅ FIXED: Use SAC model path from config
        if model_path is None:
            model_path = self.config.get('rl_model.model_path', './best_sac_model_EURUSD/best_model.zip')
        
        self.model_path = model_path
        self.model = None
        self.env = None
        
        print(f"✅ SAC Model Evaluator initialized")
        print(f"   Model Path: {model_path}")
    
    def load_model(self) -> bool:
        """Load SAC model with validation"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            # ✅ FIXED: Load SAC model instead of PPO
            self.model = SAC.load(self.model_path)
            print(f"✅ SAC model loaded successfully from {self.model_path}")
            
            # Validate model
            if hasattr(self.model, 'policy'):
                print(f"✅ Model validation successful")
                print(f"   Policy Type: {type(self.model.policy)}")
                return True
            else:
                print(f"❌ Invalid model structure")
                return False
                
        except Exception as e:
            print(f"❌ Error loading SAC model: {e}")
            return False
    
    def create_evaluation_environment(self) -> bool:
        """Create environment for evaluation"""
        try:
            # Get trading symbols from config
            symbols = self.config.get_trading_symbols()
            primary_symbol = symbols[0] if symbols else 'EURUSD'
            
            # Create trading environment
            self.env = TradingEnvironment(
                symbol=primary_symbol,
                config=self.config.config
            )
            
            # Wrap in DummyVecEnv for SAC compatibility
            self.env = DummyVecEnv([lambda: self.env])
            
            print(f"✅ Evaluation environment created for {primary_symbol}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating evaluation environment: {e}")
            return False
    
    def evaluate_model(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        ✅ FIXED: Evaluate SAC model performance
        """
        if not self.model or not self.env:
            print("❌ Model or environment not initialized")
            return {}
        
        try:
            episode_rewards = []
            episode_lengths = []
            win_rate = 0
            total_trades = 0
            profitable_trades = 0
            
            print(f"🔄 Starting SAC model evaluation ({num_episodes} episodes)...")
            
            for episode in range(num_episodes):
                obs = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_trades = 0
                episode_wins = 0
                
                done = False
                while not done:
                    # ✅ FIXED: Use SAC model for prediction
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.env.step(action)
                    
                    episode_reward += reward[0]  # DummyVecEnv returns array
                    episode_length += 1
                    
                    # Track trades from info
                    if info and len(info) > 0:
                        episode_info = info[0]
                        if episode_info.get('trade_executed', False):
                            episode_trades += 1
                            total_trades += 1
                            if episode_info.get('trade_profit', 0) > 0:
                                episode_wins += 1
                                profitable_trades += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                print(f"   Episode {episode + 1}: Reward={episode_reward:.2f}, "
                      f"Length={episode_length}, Trades={episode_trades}, Wins={episode_wins}")
            
            # Calculate statistics
            avg_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            avg_length = np.mean(episode_lengths)
            win_rate = (profitable_trades / max(1, total_trades)) * 100
            
            results = {
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'avg_episode_length': avg_length,
                'total_episodes': num_episodes,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': win_rate,
                'sharpe_ratio': avg_reward / max(std_reward, 0.01)
            }
            
            print(f"\n✅ SAC Model Evaluation Results:")
            print(f"   Average Reward: {avg_reward:.4f} ± {std_reward:.4f}")
            print(f"   Average Episode Length: {avg_length:.1f}")
            print(f"   Total Trades: {total_trades}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Sharpe Ratio: {results['sharpe_ratio']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error during model evaluation: {e}")
            return {}
    
    def compare_with_baseline(self, baseline_results: Dict[str, float] = None) -> None:
        """Compare SAC model with baseline performance"""
        try:
            if not baseline_results:
                # Default baseline (buy and hold or random strategy)
                baseline_results = {
                    'avg_reward': 0.0,
                    'win_rate': 50.0,
                    'sharpe_ratio': 0.0
                }
            
            # Run evaluation
            sac_results = self.evaluate_model()
            
            if sac_results:
                print(f"\n📊 SAC vs Baseline Comparison:")
                print(f"   Reward: SAC {sac_results['avg_reward']:.4f} vs Baseline {baseline_results['avg_reward']:.4f}")
                print(f"   Win Rate: SAC {sac_results['win_rate']:.1f}% vs Baseline {baseline_results['win_rate']:.1f}%")
                print(f"   Sharpe: SAC {sac_results['sharpe_ratio']:.4f} vs Baseline {baseline_results['sharpe_ratio']:.4f}")
                
                # Determine if SAC is better
                improvements = 0
                if sac_results['avg_reward'] > baseline_results['avg_reward']:
                    improvements += 1
                if sac_results['win_rate'] > baseline_results['win_rate']:
                    improvements += 1
                if sac_results['sharpe_ratio'] > baseline_results['sharpe_ratio']:
                    improvements += 1
                
                if improvements >= 2:
                    print(f"✅ SAC model outperforms baseline in {improvements}/3 metrics")
                else:
                    print(f"⚠️ SAC model needs improvement ({improvements}/3 metrics better)")
        
        except Exception as e:
            print(f"❌ Error in baseline comparison: {e}")

def main():
    """Main evaluation function"""
    try:
        # Initialize evaluator
        evaluator = SACModelEvaluator()
        
        # Load model
        if not evaluator.load_model():
            print("❌ Failed to load SAC model")
            return
        
        # Create environment
        if not evaluator.create_evaluation_environment():
            print("❌ Failed to create evaluation environment")
            return
        
        # Run evaluation
        results = evaluator.evaluate_model(num_episodes=20)
        
        if results:
            # Compare with baseline
            evaluator.compare_with_baseline()
            
            # Save results
            results_df = pd.DataFrame([results])
            results_df.to_csv('sac_evaluation_results.csv', index=False)
            print(f"✅ Results saved to sac_evaluation_results.csv")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main()

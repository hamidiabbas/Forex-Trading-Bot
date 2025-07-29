"""
Complete Production RL Model Manager with stable-baselines3
Full SAC implementation with advanced features
"""
import os
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import pickle
import json

from stable_baselines3 import SAC, PPO, A2C, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import torch
import torch.nn as nn

from core.trading_environment import TradingEnvironment

logger = logging.getLogger(__name__)

class TensorboardCallback(BaseCallback):
    """Custom callback for tensorboard logging"""
    
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log custom metrics
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'total_profit' in info:
                self.logger.record('custom/total_profit', info['total_profit'])
                self.logger.record('custom/win_rate', info.get('win_rate', 0))
                self.logger.record('custom/successful_trades', info.get('successful_trades', 0))
                self.logger.record('custom/max_drawdown', info.get('max_drawdown', 0))
        
        return True

class RLModelManager:
    """
    Complete Production RL Model Manager with full ML capabilities
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_path = None
        self.model_type = None
        self.env = None
        self.eval_env = None
        self.is_loaded = False
        
        # Model configuration
        self.model_config = config.get('rl_model', {})
        self.observation_size = self.model_config.get('observation_size', 64)
        self.confidence_threshold = self.model_config.get('prediction_confidence_threshold', 0.6)
        
        # Training configuration
        self.training_config = {
            'total_timesteps': 100000,
            'eval_freq': 5000,
            'n_eval_episodes': 10,
            'learning_rate': 3e-4,
            'batch_size': 256,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'train_freq': 1,
            'gradient_steps': 1,
            'target_update_interval': 1000,
            'gamma': 0.99,
            'tau': 0.005,
            'use_sde': False,
            'sde_sample_freq': -1,
            'use_sde_at_warmup': False
        }
        
        # Performance tracking
        self.signals_generated = 0
        self.successful_predictions = 0
        self.model_performance_history = []
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("✅ Production RL Model Manager initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Observation Size: {self.observation_size}")
    
    def load_model(self, model_path: str = None) -> bool:
        """Load production SAC model"""
        try:
            # Get model configuration
            self.model_type = self.model_config.get('model_type', 'SAC')
            if model_path:
                self.model_path = model_path
            else:
                self.model_path = self.model_config.get('model_path', './best_sac_model_EURUSD/best_model.zip')
            
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Creating new model for training...")
                return self._create_new_model()
            
            # Load model based on type
            if self.model_type.upper() == 'SAC':
                self.model = SAC.load(self.model_path, device=self.device)
                logger.info(f"✅ SAC model loaded from: {self.model_path}")
            elif self.model_type.upper() == 'PPO':
                self.model = PPO.load(self.model_path, device=self.device)
                logger.info(f"✅ PPO model loaded from: {self.model_path}")
            elif self.model_type.upper() == 'TD3':
                self.model = TD3.load(self.model_path, device=self.device)
                logger.info(f"✅ TD3 model loaded from: {self.model_path}")
            elif self.model_type.upper() == 'A2C':
                self.model = A2C.load(self.model_path, device=self.device)
                logger.info(f"✅ A2C model loaded from: {self.model_path}")
            elif self.model_type.upper() == 'DDPG':
                self.model = DDPG.load(self.model_path, device=self.device)
                logger.info(f"✅ DDPG model loaded from: {self.model_path}")
            else:
                logger.error(f"❌ Unsupported model type: {self.model_type}")
                return False
            
            # Validate model
            if not self._validate_model():
                return False
            
            self.is_loaded = True
            logger.info("✅ RL model validation successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading RL model: {e}")
            return False
    
    def _create_new_model(self) -> bool:
        """Create new model for training"""
        try:
            # Create dummy environment for model creation
            symbols = self.config.get_trading_symbols()
            primary_symbol = symbols[0] if symbols else 'EURUSD'
            
            env = TradingEnvironment(symbol=primary_symbol, config=self.config.config)
            env = DummyVecEnv([lambda: env])
            
            # Create model based on type
            if self.model_type.upper() == 'SAC':
                self.model = SAC(
                    'MlpPolicy',
                    env,
                    learning_rate=self.training_config['learning_rate'],
                    buffer_size=self.training_config['buffer_size'],
                    learning_starts=self.training_config['learning_starts'],
                    batch_size=self.training_config['batch_size'],
                    tau=self.training_config['tau'],
                    gamma=self.training_config['gamma'],
                    train_freq=self.training_config['train_freq'],
                    gradient_steps=self.training_config['gradient_steps'],
                    target_update_interval=self.training_config['target_update_interval'],
                    use_sde=self.training_config['use_sde'],
                    device=self.device,
                    verbose=1,
                    tensorboard_log="./tensorboard_logs/"
                )
            elif self.model_type.upper() == 'PPO':
                self.model = PPO(
                    'MlpPolicy',
                    env,
                    learning_rate=self.training_config['learning_rate'],
                    gamma=self.training_config['gamma'],
                    device=self.device,
                    verbose=1,
                    tensorboard_log="./tensorboard_logs/"
                )
            # Add other model types as needed
            
            self.is_loaded = True
            logger.info(f"✅ New {self.model_type} model created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating new model: {e}")
            return False
    
    def _validate_model(self) -> bool:
        """Validate loaded model"""
        try:
            if not self.model:
                return False
            
            # Check if model has required attributes
            required_attrs = ['policy', 'predict']
            for attr in required_attrs:
                if not hasattr(self.model, attr):
                    logger.error(f"❌ Model missing required attribute: {attr}")
                    return False
            
            # Test prediction with dummy observation
            dummy_obs = np.zeros((1, self.observation_size), dtype=np.float32)
            action, _ = self.model.predict(dummy_obs, deterministic=True)
            
            if action is None:
                logger.error("❌ Model prediction test failed")
                return False
            
            logger.debug("✅ Model structure validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model validation error: {e}")
            return False
    
    def create_environment(self, symbol: str) -> bool:
        """Create trading environment for the model"""
        try:
            # Create main environment
            self.env = TradingEnvironment(
                symbol=symbol,
                config=self.config.config
            )
            
            # Wrap in DummyVecEnv for compatibility
            self.env = DummyVecEnv([lambda: Monitor(self.env)])
            
            # Create evaluation environment
            self.eval_env = TradingEnvironment(
                symbol=symbol,
                config=self.config.config
            )
            self.eval_env = DummyVecEnv([lambda: Monitor(self.eval_env)])
            
            logger.debug(f"✅ Trading environments created for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating environment for {symbol}: {e}")
            return False
    
    def train_model(self, symbol: str, total_timesteps: int = None) -> bool:
        """Train the RL model"""
        try:
            if not self.model:
                logger.error("No model loaded for training")
                return False
            
            if not self.create_environment(symbol):
                logger.error("Failed to create training environment")
                return False
            
            # Set model environment
            self.model.set_env(self.env)
            
            # Configure training parameters
            if total_timesteps is None:
                total_timesteps = self.training_config['total_timesteps']
            
            # Setup callbacks
            callbacks = []
            
            # Evaluation callback
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=f'./best_{self.model_type.lower()}_model_{symbol}/',
                log_path=f'./logs/{self.model_type.lower()}_{symbol}/',
                eval_freq=self.training_config['eval_freq'],
                n_eval_episodes=self.training_config['n_eval_episodes'],
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
            
            # Stop training callback
            stop_callback = StopTrainingOnRewardThreshold(
                reward_threshold=1000,
                verbose=1
            )
            callbacks.append(stop_callback)
            
            # Custom tensorboard callback
            tb_callback = TensorboardCallback()
            callbacks.append(tb_callback)
            
            # Start training
            logger.info(f"🏋️ Starting {self.model_type} model training...")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Total Timesteps: {total_timesteps}")
            logger.info(f"   Device: {self.device}")
            
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=100,
                progress_bar=True
            )
            
            # Save final model
            final_model_path = f'./model_{self.model_type.lower()}_{symbol}_final.zip'
            self.model.save(final_model_path)
            logger.info(f"✅ Final {self.model_type} model saved to {final_model_path}")
            
            # Update config with new model path
            best_model_path = f'./best_{self.model_type.lower()}_model_{symbol}/best_model.zip'
            if os.path.exists(best_model_path):
                self.model_path = best_model_path
                logger.info(f"✅ Best model available at: {best_model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return False
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signal using trained model"""
        if not self.is_loaded or not self.model:
            logger.warning("RL model not loaded")
            return None
        
        try:
            self.signals_generated += 1
            
            # Create observation from market data
            observation = self._create_observation_from_market_data(market_data)
            
            if observation is None:
                logger.warning(f"Failed to create observation for {symbol}")
                return None
            
            # Get model prediction
            action, action_prob = self._get_model_prediction(observation)
            
            if action is None:
                return None
            
            # Convert action to trading signal
            signal = self._convert_action_to_signal(action, action_prob, symbol, market_data)
            
            # Track performance
            if signal and signal.get('confidence', 0) >= self.confidence_threshold:
                logger.debug(f"Production RL signal: {symbol} {signal.get('direction')} (confidence: {signal.get('confidence', 0):.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating RL signal for {symbol}: {e}")
            return None
    
    def _create_observation_from_market_data(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Create comprehensive model observation from market data"""
        try:
            # Extract features (matching TradingEnvironment observation structure)
            current_price = market_data.get('current_price', market_data.get('close', 1.0))
            
            if current_price <= 0:
                return None
            
            # Price features
            open_price = market_data.get('open', current_price)
            high_price = market_data.get('high', current_price)
            low_price = market_data.get('low', current_price)
            
            # Technical indicators
            rsi = market_data.get('rsi', market_data.get('rsi_14', 50))
            macd = market_data.get('macd', 0)
            macd_signal = market_data.get('macd_signal', 0)
            macd_histogram = market_data.get('macd_histogram', 0)
            bb_position = market_data.get('bb_position', 0.5)
            atr = market_data.get('atr', current_price * 0.01)
            adx = market_data.get('adx', 25)
            
            # Volume
            volume_ratio = market_data.get('volume_ratio', 1.0)
            
            # Moving averages
            sma_20 = market_data.get('sma_20', current_price)
            sma_50 = market_data.get('sma_50', current_price)
            ema_12 = market_data.get('ema_12', current_price)
            ema_26 = market_data.get('ema_26', current_price)
            
            # Create observation array (64 features to match environment)
            observation = np.array([
                # Price features (8)
                (current_price / open_price) - 1,  # Daily return
                (high_price - low_price) / current_price,  # Daily range
                (current_price / sma_20) - 1,  # Price vs SMA20
                (current_price / sma_50) - 1,  # Price vs SMA50
                bb_position - 0.5,  # BB position centered
                (high_price / current_price) - 1,  # High ratio
                (current_price / low_price) - 1,  # Low ratio
                (current_price / open_price) - 1,  # Open-close ratio
                
                # Technical indicators (16)
                (rsi - 50) / 50,  # RSI normalized
                macd / current_price,  # MACD normalized
                macd_histogram / current_price,  # MACD histogram normalized
                market_data.get('stoch_k', 50) / 100 - 0.5,  # Stoch K normalized
                market_data.get('stoch_d', 50) / 100 - 0.5,  # Stoch D normalized
                (market_data.get('williams_r', -50) + 50) / 100,  # Williams R normalized
                market_data.get('cci', 0) / 100,  # CCI normalized
                market_data.get('momentum', 0) / current_price,  # Momentum normalized
                (adx - 25) / 25,  # ADX normalized
                atr / current_price,  # ATR normalized
                market_data.get('roc', 0),  # Rate of change
                market_data.get('mfi', 50) / 100 - 0.5,  # MFI normalized
                market_data.get('plus_di', 25) / 50 - 0.5,  # +DI normalized
                market_data.get('minus_di', 25) / 50 - 0.5,  # -DI normalized
                market_data.get('volatility', atr/current_price),  # Volatility
                market_data.get('bb_width', 0.1),  # BB width
                
                # Volume features (4)
                volume_ratio - 1,  # Volume ratio
                market_data.get('obv', 0) / 1000000,  # OBV scaled
                market_data.get('ad', 0) / 1000000,  # A/D scaled
                market_data.get('adosc', 0) / 1000,  # ADOSC scaled
                
                # Trend features (8)
                1 if sma_20 > sma_50 else -1,  # MA trend
                1 if current_price > sma_20 else -1,  # Price above SMA20
                1 if ema_12 > ema_26 else -1,  # EMA crossover
                1 if macd > macd_signal else -1,  # MACD signal
                0,  # Higher highs (placeholder)
                0,  # Lower lows (placeholder)
                0,  # Support distance (placeholder)
                0,  # Resistance distance (placeholder)
                
                # Portfolio features (8) - Set to neutral for signal generation
                0,  # Current position
                0,  # Position size
                0,  # Portfolio return
                0,  # Realized return
                0,  # Max drawdown
                0,  # Win rate
                0.5,  # Time progress (middle)
                0,  # Trade frequency
                
                # Market microstructure features (8)
                market_data.get('price_change_pct', 0),  # Price change
                market_data.get('volatility', 0.01),  # Return volatility
                0,  # Recent high (placeholder)
                0,  # Recent low (placeholder)
                volume_ratio - 1,  # Volume trend
                0,  # Bullish bars ratio (placeholder)
                0,  # ATR trend (placeholder)
                0,  # Price trend correlation (placeholder)
                
                # Time features (12)
                np.sin(2 * np.pi * datetime.now().hour / 24),  # Hour sin
                np.cos(2 * np.pi * datetime.now().hour / 24),  # Hour cos
                np.sin(2 * np.pi * datetime.now().weekday() / 7),  # Day sin
                np.cos(2 * np.pi * datetime.now().weekday() / 7),  # Day cos
                1 if 8 <= datetime.now().hour <= 17 else 0,  # London/NY session
                1 if 13 <= datetime.now().hour <= 17 else 0,  # Overlap session
                1 if datetime.now().hour < 6 or datetime.now().hour > 22 else 0,  # Low liquidity
                1 if datetime.now().weekday() < 5 else 0,  # Weekday
                1 if datetime.now().weekday() == 4 else 0,  # Friday
                1 if datetime.now().weekday() == 0 else 0,  # Monday
                datetime.now().hour / 24,  # Hour normalized
                datetime.now().weekday() / 7  # Day normalized
            ], dtype=np.float32)
            
            # Ensure exactly 64 features
            if len(observation) > 64:
                observation = observation[:64]
            elif len(observation) < 64:
                observation = np.pad(observation, (0, 64 - len(observation)), 'constant')
            
            # Handle NaN/inf values
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
            observation = np.clip(observation, -10.0, 10.0)  # Clip extreme values
            
            return observation.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            return None
    
    def _get_model_prediction(self, observation: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Get prediction from the model with confidence estimation"""
        try:
            # Get action from model
            action, _ = self.model.predict(observation, deterministic=True)
            
            # Get action probabilities/confidence for supported models
            confidence = 0.7  # Default confidence
            
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'get_distribution'):
                # For policy-based models, try to get action probabilities
                try:
                    obs_tensor = torch.FloatTensor(observation).to(self.device)
                    with torch.no_grad():
                        if self.model_type.upper() == 'SAC':
                            # For SAC, use actor network
                            actions, log_prob, _ = self.model.policy.actor.action_dist.sample(
                                self.model.policy.actor(obs_tensor)
                            )
                            confidence = min(0.9, 0.5 + torch.exp(log_prob).item() * 0.5)
                        else:
                            # For other models, use different confidence estimation
                            confidence = 0.7
                except Exception:
                    confidence = 0.7
            
            return int(action), float(confidence)
            
        except Exception as e:
            logger.error(f"Error getting model prediction: {e}")
            return None, None
    
    def _convert_action_to_signal(self, action: int, confidence: float, 
                                 symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert model action to trading signal"""
        try:
            # Action mapping: 0=hold, 1=buy, 2=sell
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            direction = action_map.get(action, 'HOLD')
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                direction = 'HOLD'
                confidence = 0.5
            
            # Get current price for entry
            current_price = market_data.get('current_price', market_data.get('close', 0))
            
            # Calculate stop loss and take profit using ATR
            atr = market_data.get('atr', current_price * 0.01)
            
            if direction == 'BUY':
                stop_loss = current_price - (atr * 2.0)
                take_profit = current_price + (atr * 3.0)
            elif direction == 'SELL':
                stop_loss = current_price + (atr * 2.0)
                take_profit = current_price - (atr * 3.0)
            else:
                stop_loss = 0
                take_profit = 0
            
            signal = {
                'symbol': symbol,
                'direction': direction,
                'confidence': confidence,
                'strategy': f'RL-{self.model_type}',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': market_data.get('timestamp', datetime.now()),
                'model_action': action,
                'model_type': self.model_type,
                'device': str(self.device)
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error converting action to signal: {e}")
            return {
                'symbol': symbol,
                'direction': 'HOLD',
                'confidence': 0.0,
                'strategy': f'RL-{self.model_type}-ERROR'
            }
    
    def evaluate_model(self, symbol: str, n_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            if not self.create_environment(symbol):
                return {'error': 'Failed to create evaluation environment'}
            
            logger.info(f"🔍 Evaluating {self.model_type} model for {symbol}...")
            
            episode_rewards = []
            episode_profits = []
            episode_trade_counts = []
            episode_win_rates = []
            
            for episode in range(n_episodes):
                obs = self.eval_env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    episode_reward += reward[0]
                
                # Extract episode statistics
                if info and len(info) > 0:
                    episode_info = info[0]
                    episode_profits.append(episode_info.get('total_profit', 0))
                    episode_trade_counts.append(episode_info.get('total_trades', 0))
                    episode_win_rates.append(episode_info.get('win_rate', 0))
                
                episode_rewards.append(episode_reward)
                
                logger.debug(f"Episode {episode + 1}: Reward={episode_reward:.2f}")
            
            # Calculate statistics
            results = {
                'symbol': symbol,
                'model_type': self.model_type,
                'n_episodes': n_episodes,
                'avg_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'avg_profit': np.mean(episode_profits) if episode_profits else 0,
                'avg_trades': np.mean(episode_trade_counts) if episode_trade_counts else 0,
                'avg_win_rate': np.mean(episode_win_rates) if episode_win_rates else 0,
                'sharpe_ratio': np.mean(episode_rewards) / max(np.std(episode_rewards), 0.01),
                'max_profit': max(episode_profits) if episode_profits else 0,
                'min_profit': min(episode_profits) if episode_profits else 0,
                'evaluation_date': datetime.now()
            }
            
            logger.info(f"✅ {self.model_type} Evaluation Results:")
            logger.info(f"   Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
            logger.info(f"   Average Profit: ${results['avg_profit']:.2f}")
            logger.info(f"   Average Win Rate: {results['avg_win_rate']:.1f}%")
            logger.info(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during model evaluation: {e}")
            return {'error': str(e)}
    
    def update_model_performance(self, signal: Dict[str, Any], trade_result: Dict[str, Any]) -> None:
        """Update model performance tracking"""
        try:
            success = trade_result.get('success', False)
            
            if success:
                self.successful_predictions += 1
            
            # Create performance record
            performance_record = {
                'timestamp': datetime.now(),
                'symbol': signal.get('symbol'),
                'direction': signal.get('direction'),
                'confidence': signal.get('confidence', 0),
                'success': success,
                'profit': trade_result.get('profit', 0),
                'model_type': self.model_type
            }
            
            self.model_performance_history.append(performance_record)
            
            # Keep history manageable
            if len(self.model_performance_history) > 1000:
                self.model_performance_history = self.model_performance_history[-500:]
            
            # Log periodic performance
            if self.signals_generated % 20 == 0:
                success_rate = (self.successful_predictions / self.signals_generated) * 100
                logger.info(f"📊 {self.model_type} Performance: {success_rate:.1f}% success rate ({self.successful_predictions}/{self.signals_generated})")
                
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    def save_performance_report(self, filename: str = None) -> str:
        """Save detailed performance report"""
        try:
            if filename is None:
                filename = f"rl_performance_report_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Calculate performance metrics
            if self.model_performance_history:
                recent_performance = self.model_performance_history[-100:]  # Last 100 signals
                
                total_profit = sum(p['profit'] for p in recent_performance)
                successful_trades = sum(1 for p in recent_performance if p['success'])
                win_rate = (successful_trades / len(recent_performance)) * 100
                
                avg_confidence = np.mean([p['confidence'] for p in recent_performance])
                
                performance_by_symbol = {}
                for record in recent_performance:
                    symbol = record['symbol']
                    if symbol not in performance_by_symbol:
                        performance_by_symbol[symbol] = {'total': 0, 'successful': 0, 'profit': 0}
                    
                    performance_by_symbol[symbol]['total'] += 1
                    if record['success']:
                        performance_by_symbol[symbol]['successful'] += 1
                    performance_by_symbol[symbol]['profit'] += record['profit']
            else:
                total_profit = 0
                win_rate = 0
                avg_confidence = 0
                performance_by_symbol = {}
            
            report = {
                'model_type': self.model_type,
                'device': str(self.device),
                'total_signals_generated': self.signals_generated,
                'successful_predictions': self.successful_predictions,
                'overall_success_rate': (self.successful_predictions / max(1, self.signals_generated)) * 100,
                'recent_performance': {
                    'total_profit': total_profit,
                    'win_rate': win_rate,
                    'avg_confidence': avg_confidence,
                    'by_symbol': performance_by_symbol
                },
                'model_info': self.get_model_info(),
                'generated_at': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"✅ Performance report saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")
            return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        model_info = {
            'loaded': self.is_loaded,
            'model_type': self.model_type,
            'model_path': self.model_path,
            'device': str(self.device),
            'observation_size': self.observation_size,
            'confidence_threshold': self.confidence_threshold,
            'signals_generated': self.signals_generated,
            'successful_predictions': self.successful_predictions,
            'success_rate': (self.successful_predictions / max(1, self.signals_generated)) * 100,
            'training_config': self.training_config,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        # Add model-specific information
        if self.model and hasattr(self.model, 'policy'):
            try:
                model_info['policy_architecture'] = str(self.model.policy)
                if hasattr(self.model, 'learning_rate'):
                    model_info['learning_rate'] = self.model.learning_rate
                if hasattr(self.model, 'gamma'):
                    model_info['gamma'] = self.model.gamma
            except Exception:
                pass
        
        return model_info
    
    def shutdown(self) -> None:
        """Enhanced shutdown with performance reporting"""
        try:
            logger.info("📊 Production RL Model Manager Shutdown Report:")
            logger.info("=" * 60)
            
            # Final performance summary
            if self.signals_generated > 0:
                success_rate = (self.successful_predictions / self.signals_generated) * 100
                logger.info(f"Model Type: {self.model_type}")
                logger.info(f"Device: {self.device}")
                logger.info(f"Total Signals Generated: {self.signals_generated}")
                logger.info(f"Successful Predictions: {self.successful_predictions}")
                logger.info(f"Overall Success Rate: {success_rate:.1f}%")
                
                # Save final performance report
                report_file = self.save_performance_report()
                if report_file:
                    logger.info(f"Performance Report: {report_file}")
            
            # Cleanup
            if self.env:
                self.env.close()
                self.env = None
            
            if self.eval_env:
                self.eval_env.close()
                self.eval_env = None
            
            self.model = None
            self.is_loaded = False
            
            logger.info("=" * 60)
            logger.info("✅ Production RL Model Manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during RL model shutdown: {e}")

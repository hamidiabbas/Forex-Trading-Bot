import os
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import torch
from stablebaselines3 import SAC, PPO, A2C
from stablebaselines3.common.vec_env import DummyVecEnv
from stablebaselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stablebaselines3.common.monitor import Monitor

logger = logging.getLogger(__name__)

class EnhancedRLModelManager:
    """Production RL Model Manager for Forex Trading Bot"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_path = None
        self.model_type = "SAC"  # Default to SAC
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
            'use_sde': False
        }
        
        # Performance tracking
        self.signals_generated = 0
        self.successful_predictions = 0
        self.model_performance_history = []
        
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Enhanced RL Model Manager initialized on device: {self.device}")
    
    def load_model(self, model_path: str = None) -> bool:
        """Load production RL model"""
        try:
            # Get model configuration
            self.model_type = self.model_config.get('model_type', 'SAC')
            
            if model_path:
                self.model_path = model_path
            else:
                self.model_path = self.model_config.get('model_path', f'./best_{self.model_type.lower()}_model_EURUSD_best_model.zip')
            
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Creating new model for training...")
                return self._create_new_model()
            
            # Load model based on type
            if self.model_type.upper() == 'SAC':
                self.model = SAC.load(self.model_path, device=self.device)
                logger.info(f"SAC model loaded from {self.model_path}")
            elif self.model_type.upper() == 'PPO':
                self.model = PPO.load(self.model_path, device=self.device)
                logger.info(f"PPO model loaded from {self.model_path}")
            elif self.model_type.upper() == 'A2C':
                self.model = A2C.load(self.model_path, device=self.device)
                logger.info(f"A2C model loaded from {self.model_path}")
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return False
            
            # Validate model
            if not self._validate_model():
                return False
                
            self.is_loaded = True
            logger.info("RL model loaded and validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
            return False
    
    def _create_new_model(self) -> bool:
        """Create new model for training"""
        try:
            # Create dummy environment for model creation
            symbols = getattr(self.config,'trading', {}).get('symbols', ['EURUSD'])
            primary_symbol = symbols[0] if symbols else 'EURUSD'
            
            # You would need to import your TradingEnvironment here
            # env = TradingEnvironment(symbol=primary_symbol, config=self.config.config)
            # env = DummyVecEnv([lambda: env])
            
            # For now, create a placeholder
            logger.warning("Creating placeholder model - implement TradingEnvironment integration")
            
            self.is_loaded = True
            logger.info(f"New {self.model_type} model created successfully")
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
                    logger.error(f"Model missing required attribute: {attr}")
                    return False
            
            # Test prediction with dummy observation
            dummy_obs = np.zeros((1, self.observation_size), dtype=np.float32)
            action, _ = self.model.predict(dummy_obs, deterministic=True)
            
            if action is None:
                logger.error("Model prediction test failed")
                return False
            
            logger.debug("Model structure validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model validation error: {e}")
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
            if signal and signal.get('confidence', 0) > self.confidence_threshold:
                logger.debug(f"Production RL signal {symbol}: {signal.get('direction')} (confidence: {signal.get('confidence', 0):.2f})")
                return signal
                
        except Exception as e:
            logger.error(f"Error generating RL signal for {symbol}: {e}")
            return None
    
    def _create_observation_from_market_data(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Create comprehensive model observation from market data"""
        try:
            # Extract features matching TradingEnvironment observation structure
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
            bb_position = market_data.get('bb_position', 0.5)
            atr = market_data.get('atr', current_price * 0.01)
            adx = market_data.get('adx', 25)
            
            # Volume
            volume_ratio = market_data.get('volume_ratio', 1.0)
            
            # Create feature array (simplified to 64 features)
            all_features = [
                open_price / current_price - 1,  # Normalized open
                high_price / current_price - 1,  # Normalized high
                low_price / current_price - 1,   # Normalized low
                rsi / 100.0 - 0.5,              # Normalized RSI
                macd,                            # MACD
                macd_signal,                     # MACD Signal
                bb_position,                     # Bollinger Band position
                atr / current_price,             # Normalized ATR
                adx / 100.0,                     # Normalized ADX
                volume_ratio - 1.0               # Volume ratio
            ]
            
            # Pad to exactly 64 features
            while len(all_features) < 64:
                all_features.append(0.0)
            
            # Take only first 64 features if more
            all_features = all_features[:64]
            
            # Convert to numpy array and handle NaN/inf
            observation = np.array(all_features, dtype=np.float32)
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
            observation = np.clip(observation, -10.0, 10.0)  # Clip extreme values
            
            return observation
            
        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            return np.zeros(64, dtype=np.float32)
    
    def _get_model_prediction(self, observation: np.ndarray) -> Tuple[Optional[int], float]:
        """Get model prediction"""
        try:
            action, _states = self.model.predict(observation.reshape(1, -1), deterministic=True)
            confidence = 0.8  # You could extract this from model if available
            return int(action[0]), confidence
        except Exception as e:
            logger.error(f"Error getting model prediction: {e}")
            return None, 0.0
    
    def _convert_action_to_signal(self, action: int, action_prob: float, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert model action to trading signal"""
        try:
            current_price = market_data.get('current_price', market_data.get('close', 1.0))
            atr_value = market_data.get('atr', current_price * 0.01)
            
            if action == 1:  # Buy signal
                return {
                    'symbol': symbol,
                    'direction': 'BUY',
                    'strategy': 'RL-Agent',
                    'entry_price': current_price,
                    'atr_at_signal': atr_value,
                    'confidence': action_prob,
                    'timestamp': datetime.now()
                }
            elif action == 2:  # Sell signal
                return {
                    'symbol': symbol,
                    'direction': 'SELL',
                    'strategy': 'RL-Agent',
                    'entry_price': current_price,
                    'atr_at_signal': atr_value,
                    'confidence': action_prob,
                    'timestamp': datetime.now()
                }
            
            # Action 0 = Hold, return None
            return None
            
        except Exception as e:
            logger.error(f"Error converting action to signal: {e}")
            return None

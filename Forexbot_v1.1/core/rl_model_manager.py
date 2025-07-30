"""
Professional RL Model Manager
Handles model loading, validation, and prediction with error recovery
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import json
from datetime import datetime

try:
    from stable_baselines3 import SAC, A2C, PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

class RLModelManager:
    """
    Enhanced RL model management with validation and error recovery
    """
    
    def __init__(self, feature_manager):
        self.logger = logging.getLogger(__name__)
        self.feature_manager = feature_manager
        self.model = None
        self.model_type = None
        self.model_info = {}
        self.prediction_cache = {}
        self.performance_tracker = []
        
        # Model paths with priority order
        self.model_paths = [
            ("./best_sac_model_EURUSD/best_model.zip", "SAC"),
            ("./model_sac_EURUSD_final.zip", "SAC"),
            ("./best_model_EURUSD/best_model.zip", "A2C"),
            ("./model_rl_EURUSD_final_fixed.zip", "A2C"),
            ("./model_a2c_EURUSD_final.zip", "A2C"),
            ("./model_ppo_EURUSD_final.zip", "PPO")
        ]
        
        if RL_AVAILABLE:
            self._load_best_model()
        else:
            self.logger.error("RL libraries not available")

    def _load_best_model(self) -> bool:
        """Load the best available RL model"""
        try:
            for model_path, model_type in self.model_paths:
                if Path(model_path).exists():
                    try:
                        if model_type == "SAC":
                            self.model = SAC.load(model_path)
                        elif model_type == "A2C":
                            self.model = A2C.load(model_path)
                        elif model_type == "PPO":
                            self.model = PPO.load(model_path)
                        else:
                            continue
                        
                        self.model_type = model_type
                        self.model_info = {
                            'path': model_path,
                            'type': model_type,
                            'loaded_at': datetime.now(),
                            'observation_space': self.model.observation_space.shape[0] if hasattr(self.model, 'observation_space') else None
                        }
                        
                        self.logger.info(f"✅ {model_type} model loaded from {model_path}")
                        
                        # Validate model compatibility
                        if self._validate_model():
                            return True
                        else:
                            self.logger.warning(f"Model validation failed for {model_path}")
                            continue
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to load {model_type} from {model_path}: {e}")
                        continue
            
            self.logger.error("No valid RL model could be loaded")
            return False
            
        except Exception as e:
            self.logger.error(f"Error in model loading process: {e}")
            return False

    def _validate_model(self) -> bool:
        """Validate model compatibility with current feature setup"""
        try:
            if self.model is None:
                return False
            
            # Test prediction with dummy observation
            dummy_obs = np.random.randn(self.feature_manager.observation_size).astype(np.float32)
            
            try:
                action, _ = self.model.predict(dummy_obs, deterministic=True)
                self.logger.info(f"✅ Model validation successful. Test action: {action}")
                return True
            except Exception as pred_error:
                self.logger.error(f"Model prediction test failed: {pred_error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating model: {e}")
            return False

    def predict_with_validation(self, observation: np.ndarray, deterministic: bool = True) -> Optional[Tuple[int, float]]:
        """
        Predict with automatic observation validation and error recovery
        
        Args:
            observation: Input observation array
            deterministic: Whether to use deterministic prediction
            
        Returns:
            (action, confidence) or None if prediction fails
        """
        try:
            if self.model is None:
                return None
            
            # Validate observation
            if not self._validate_observation(observation):
                return None
            
            # Cache check (optional performance optimization)
            obs_key = hash(observation.tobytes())
            if obs_key in self.prediction_cache:
                cached_result = self.prediction_cache[obs_key]
                # Use cache if recent (within last 5 predictions)
                if len(self.prediction_cache) - list(self.prediction_cache.keys()).index(obs_key) < 5:
                    return cached_result
            
            # Make prediction
            action, _states = self.model.predict(observation, deterministic=deterministic)
            
            # Calculate confidence based on model type
            confidence = self._calculate_confidence(action, observation)
            
            # Cache result
            result = (int(action), confidence)
            self.prediction_cache[obs_key] = result
            
            # Limit cache size
            if len(self.prediction_cache) > 100:
                # Remove oldest entries
                oldest_keys = list(self.prediction_cache.keys())[:50]
                for key in oldest_keys:
                    del self.prediction_cache[key]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in model prediction: {e}")
            return None

    def _validate_observation(self, observation: np.ndarray) -> bool:
        """Validate observation array"""
        try:
            if observation is None:
                return False
            
            if not isinstance(observation, np.ndarray):
                return False
            
            if observation.shape != (self.feature_manager.observation_size,):
                self.logger.error(f"Observation shape mismatch: {observation.shape} != ({self.feature_manager.observation_size},)")
                return False
            
            if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
                self.logger.warning("Observation contains NaN or Inf values")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating observation: {e}")
            return False

    def _calculate_confidence(self, action: int, observation: np.ndarray) -> float:
        """Calculate prediction confidence based on model type and performance"""
        try:
            base_confidence = 0.7
            
            # Model-specific confidence adjustments
            if self.model_type == "SAC":
                base_confidence = 0.8  # SAC typically more confident
            elif self.model_type == "PPO":
                base_confidence = 0.75
            
            # Adjust based on recent performance
            if len(self.performance_tracker) > 0:
                recent_performance = np.mean(self.performance_tracker[-10:])
                performance_adjustment = min(0.2, max(-0.2, recent_performance))
                base_confidence += performance_adjustment
            
            # Observation-based confidence (simple heuristic)
            obs_std = np.std(observation)
            if obs_std > 2.0:  # High volatility in features
                base_confidence *= 0.9
            elif obs_std < 0.1:  # Very stable features
                base_confidence *= 0.95
            
            return max(0.3, min(0.95, base_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.7

    def update_performance(self, trade_result: float):
        """Update model performance tracking"""
        try:
            self.performance_tracker.append(trade_result)
            
            # Keep only last 100 results
            if len(self.performance_tracker) > 100:
                self.performance_tracker = self.performance_tracker[-100:]
            
            # Log performance summary periodically
            if len(self.performance_tracker) % 10 == 0:
                avg_performance = np.mean(self.performance_tracker[-10:])
                self.logger.info(f"RL Model Recent Performance (last 10): {avg_performance:.3f}")
                
        except Exception as e:
            self.logger.error(f"Error updating performance: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            **self.model_info,
            'is_loaded': self.model is not None,
            'predictions_made': len(self.prediction_cache),
            'performance_samples': len(self.performance_tracker),
            'avg_recent_performance': np.mean(self.performance_tracker[-10:]) if len(self.performance_tracker) >= 10 else 0.0,
            'feature_manager_version': self.feature_manager.feature_config.version
        }

    def is_available(self) -> bool:
        """Check if RL model is available and ready"""
        return RL_AVAILABLE and self.model is not None

    def get_action_meanings(self) -> Dict[int, str]:
        """Get human-readable action meanings"""
        return {
            0: "HOLD/CLOSE",
            1: "BUY/LONG", 
            2: "SELL/SHORT"
        }

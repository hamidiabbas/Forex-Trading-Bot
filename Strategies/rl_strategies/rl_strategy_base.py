# strategies/rl_strategies/rl_strategy_base.py
import numpy as np
from stable_baselines3 import SAC, A2C
from strategies.base_strategy import BaseStrategy, TradingSignal
from typing import Dict, Any, Optional

class RLStrategyBase(BaseStrategy):
    """Base class for RL-based trading strategies"""
    
    def __init__(self, config: Dict[str, Any], model_path: str):
        super().__init__(config)
        self.model_path = model_path
        self.model = None
        self.observation_size = config.get('observation_size', 32)
        self.confidence_threshold = config.get('rl_confidence_threshold', 0.6)
        self.load_model()
    
    def load_model(self):
        """Load trained RL model"""
        try:
            if 'sac' in self.model_path.lower():
                self.model = SAC.load(self.model_path)
                self.model_type = 'SAC'
            elif 'a2c' in self.model_path.lower():
                self.model = A2C.load(self.model_path)
                self.model_type = 'A2C'
            else:
                raise ValueError(f"Unsupported model type in path: {self.model_path}")
        except Exception as e:
            self.model = None
            print(f"Failed to load RL model: {e}")
    
    def prepare_observation(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare observation vector for RL model"""
        try:
            # Extract numeric features
            df = market_data.get('EXECUTION')
            if df is None or len(df) < 20:
                return None
            
            # Get latest technical indicators
            features = []
            feature_cols = ['Close', 'Volume', 'RSI14', 'MACD_12_26_9', 
                          'BBU_20_2.0', 'BBL_20_2.0', 'ATRr_14']
            
            for col in feature_cols:
                if col in df.columns:
                    features.append(df[col].iloc[-1])
                else:
                    features.append(0.0)
            
            # Add derived features
            if len(df) >= 10:
                price_change = df['Close'].pct_change().iloc[-1]
                volatility = df['Close'].pct_change().rolling(10).std().iloc[-1]
                momentum = df['Close'].pct_change(periods=5).iloc[-1]
            else:
                price_change = volatility = momentum = 0.0
            
            features.extend([price_change, volatility, momentum])
            
            # Pad or truncate to observation_size
            obs = np.array(features, dtype=np.float32)
            if len(obs) < self.observation_size:
                obs = np.pad(obs, (0, self.observation_size - len(obs)))
            else:
                obs = obs[:self.observation_size]
            
            # Replace NaN/inf values
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return obs
            
        except Exception as e:
            print(f"Error preparing RL observation: {e}")
            return None
    
    def calculate_confidence(self, action: int, observation: np.ndarray) -> float:
        """Calculate confidence score for RL prediction"""
        try:
            # Use model's action probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(observation.reshape(1, -1))[0]
                confidence = np.max(probs)
            else:
                # For deterministic policies, use a heuristic
                # Based on observation magnitude and model consistency
                obs_magnitude = np.linalg.norm(observation)
                confidence = min(0.9, max(0.5, obs_magnitude * 0.1 + 0.6))
            
            return float(confidence)
        except Exception:
            return 0.5  # Default moderate confidence
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate RL-based trading signal"""
        if not self.model or not self.is_active:
            return None
        
        try:
            observation = self.prepare_observation(market_data)
            if observation is None:
                return None
            
            # Get model prediction
            action, _ = self.model.predict(observation, deterministic=True)
            confidence = self.calculate_confidence(action, observation)
            
            # Skip if confidence too low
            if confidence < self.confidence_threshold:
                return None
            
            # Convert action to signal
            df = market_data.get('EXECUTION')
            current_price = df['Close'].iloc[-1]
            atr = df.get('ATRr_14', pd.Series([0.01])).iloc[-1]
            
            if action == 1:  # BUY
                direction = 'BUY'
                stop_loss = current_price - (2.0 * atr)
                take_profit = current_price + (3.0 * atr)
            elif action == 2:  # SELL
                direction = 'SELL'
                stop_loss = current_price + (2.0 * atr)
                take_profit = current_price - (3.0 * atr)
            else:  # HOLD
                return None
            
            return TradingSignal(
                symbol=market_data.get('symbol', 'UNKNOWN'),
                direction=direction,
                strategy_type=f'RL-{self.model_type}',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=1.5,
                metadata={
                    'model_type': self.model_type,
                    'observation_size': len(observation),
                    'action_raw': int(action)
                }
            )
            
        except Exception as e:
            print(f"Error generating RL signal: {e}")
            return None

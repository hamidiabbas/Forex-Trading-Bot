"""
Professional Feature Engineering Manager
Ensures consistency between training and live trading with robust data handling
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas_ta as ta
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class FeatureConfig:
    """Feature configuration schema"""
    version: str
    features: List[str]
    observation_size: int
    feature_params: Dict

class FeatureManager:
    """
    Centralized feature engineering with robust data type handling
    """
    
    def __init__(self, config_path: str = "configs/features.json"):
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path)
        self.feature_config = self._load_feature_config()
        self.expected_features = self.feature_config.features
        self.observation_size = self.feature_config.observation_size
        
        self.logger.info(f"FeatureManager initialized - Version: {self.feature_config.version}")
        self.logger.info(f"Expected features: {len(self.expected_features)}")
        self.logger.info(f"Observation size: {self.observation_size}")

    def _load_feature_config(self) -> FeatureConfig:
        """Load feature configuration from JSON file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                return FeatureConfig(**config_data)
            else:
                return self._create_default_config()
        except Exception as e:
            self.logger.error(f"Error loading feature config: {e}")
            return self._create_default_config()

    def _create_default_config(self) -> FeatureConfig:
        """Create default feature configuration"""
        default_features = [
            'Close', 'ADX_14', 'DMP_14', 'DMN_14', 'RSI_14', 'ATRr_14',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'EMA_20', 'EMA_50',
            'OBV', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ISA_9', 'ISB_26',
            'ITS_9', 'IKS_26', 'ICS_26', 'BBW', 'ATRr_14_median',
            'price_change', 'volatility', 'momentum',
            'position_info_1', 'position_info_2', 'position_info_3'
        ]
        
        return FeatureConfig(
            version="1.0.1",
            features=default_features,
            observation_size=32,
            feature_params={
                "rsi_period": 14,
                "atr_period": 14,
                "bb_period": 20,
                "bb_std": 2.0,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "ema_periods": [20, 50],
                "stoch_k": 14,
                "stoch_d": 3
            }
        )

    def engineer_features(self, df_input: Union[pd.DataFrame, tuple, list, np.ndarray]) -> pd.DataFrame:
        """
        🔧 FIXED: Robust feature engineering with data type validation
        
        Args:
            df_input: Input data (DataFrame, tuple, list, or array)
            
        Returns:
            DataFrame with engineered features
        """
        try:
            # 🛡️ CRITICAL FIX: Handle different input data types
            df = self._ensure_dataframe(df_input)
            
            if df is None or len(df) < 50:
                raise ValueError("Insufficient data for feature engineering")
            
            # Validate required columns
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            df_features = df.copy()
            params = self.feature_config.feature_params
            
            self.logger.debug(f"Starting feature engineering on DataFrame with shape: {df_features.shape}")
            
            # 1. Basic Moving Averages (Robust Implementation)
            df_features['EMA_20'] = self._safe_ema(df_features['Close'], length=params["ema_periods"][0])
            df_features['EMA_50'] = self._safe_ema(df_features['Close'], length=params["ema_periods"][1])
            
            # 2. RSI with error handling
            df_features['RSI_14'] = self._safe_rsi(df_features['Close'], length=params["rsi_period"])
            
            # 3. ATR with robust calculation
            df_features = self._calculate_atr(df_features, params["atr_period"])
            
            # 4. Bollinger Bands with validation
            df_features = self._calculate_bollinger_bands(df_features, params)
            
            # 5. MACD with error handling
            df_features = self._calculate_macd(df_features, params)
            
            # 6. ADX and Directional Movement
            df_features = self._calculate_adx(df_features, params["rsi_period"])
            
            # 7. Stochastic Oscillator
            df_features = self._calculate_stochastic(df_features, params)
            
            # 8. Ichimoku (simplified and safe)
            df_features = self._calculate_ichimoku_safe(df_features)
            
            # 9. Volume indicators (safe implementation)
            df_features = self._calculate_volume_indicators(df_features)
            
            # 10. Price momentum features
            df_features = self._calculate_momentum_features(df_features)
            
            # 11. Clean and validate all features
            df_features = self._clean_and_validate_features(df_features)
            
            self.logger.debug(f"Feature engineering completed. Final shape: {df_features.shape}")
            self.logger.debug(f"Features created: {list(df_features.columns)}")
            
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            self.logger.error(f"Input type: {type(df_input)}")
            if hasattr(df_input, 'shape'):
                self.logger.error(f"Input shape: {df_input.shape}")
            
            # Return a basic DataFrame to prevent total failure
            return self._create_fallback_dataframe(df_input)

    def _ensure_dataframe(self, data_input: Union[pd.DataFrame, tuple, list, np.ndarray]) -> Optional[pd.DataFrame]:
        """🔧 CRITICAL FIX: Convert any input type to DataFrame"""
        try:
            if isinstance(data_input, pd.DataFrame):
                return data_input
            
            elif isinstance(data_input, tuple):
                self.logger.warning(f"Received tuple input, attempting to convert: {type(data_input)}")
                
                # Handle different tuple structures
                if len(data_input) > 0:
                    # If first element is a DataFrame
                    if isinstance(data_input[0], pd.DataFrame):
                        return data_input[0]
                    
                    # If tuple contains arrays
                    elif isinstance(data_input[0], (list, np.ndarray)):
                        try:
                            # Try to create DataFrame from tuple data
                            if len(data_input) >= 5:  # OHLCV data
                                df = pd.DataFrame({
                                    'Open': data_input[0],
                                    'High': data_input[1],
                                    'Low': data_input[2],
                                    'Close': data_input[3],
                                    'Volume': data_input[4] if len(data_input) > 4 else [0] * len(data_input[0])
                                })
                                return df
                        except Exception as e:
                            self.logger.error(f"Failed to convert tuple to DataFrame: {e}")
                
                return None
            
            elif isinstance(data_input, (list, np.ndarray)):
                self.logger.warning(f"Received {type(data_input)} input, attempting to convert")
                try:
                    # Convert to DataFrame
                    if isinstance(data_input, np.ndarray) and len(data_input.shape) == 2:
                        columns = ['Open', 'High', 'Low', 'Close', 'Volume'][:data_input.shape[1]]
                        return pd.DataFrame(data_input, columns=columns)
                    elif isinstance(data_input, list) and len(data_input) > 0:
                        if isinstance(data_input[0], dict):
                            return pd.DataFrame(data_input)
                except Exception as e:
                    self.logger.error(f"Failed to convert {type(data_input)} to DataFrame: {e}")
                
                return None
            
            else:
                self.logger.error(f"Unsupported input type: {type(data_input)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error ensuring DataFrame: {e}")
            return None

    def _safe_ema(self, series: pd.Series, length: int) -> pd.Series:
        """Safe EMA calculation with fallback"""
        try:
            if len(series) < length:
                return pd.Series(index=series.index, data=series.iloc[0], dtype=float)
            return ta.ema(series, length=length).fillna(series.iloc[0])
        except Exception as e:
            self.logger.warning(f"EMA calculation failed, using fallback: {e}")
            return series.ewm(span=length).mean().fillna(series.iloc[0])

    def _safe_rsi(self, series: pd.Series, length: int) -> pd.Series:
        """Safe RSI calculation with fallback"""
        try:
            if len(series) < length + 1:
                return pd.Series(index=series.index, data=50.0, dtype=float)
            
            result = ta.rsi(series, length=length)
            if result is None:
                raise ValueError("RSI returned None")
            return result.fillna(50.0)
        except Exception as e:
            self.logger.warning(f"RSI calculation failed, using manual calculation: {e}")
            return self._manual_rsi(series, length)

    def _manual_rsi(self, series: pd.Series, length: int) -> pd.Series:
        """Manual RSI calculation as fallback"""
        try:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50.0)
        except Exception:
            return pd.Series(index=series.index, data=50.0, dtype=float)

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Safe ATR calculation"""
        try:
            # Manual ATR calculation for reliability
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            df['ATR_14'] = true_range.rolling(period).mean().fillna(true_range.iloc[0])
            df['ATRr_14'] = (df['ATR_14'] / df['Close']).fillna(0.001)
            df['ATRr_14_median'] = df['ATRr_14'].rolling(20).median().fillna(df['ATRr_14'])
            
            return df
        except Exception as e:
            self.logger.warning(f"ATR calculation failed: {e}")
            df['ATR_14'] = df['Close'] * 0.001  # 0.1% of price as fallback
            df['ATRr_14'] = 0.001
            df['ATRr_14_median'] = 0.001
            return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Safe Bollinger Bands calculation"""
        try:
            period = params["bb_period"]
            std_dev = params["bb_std"]
            
            # Manual calculation for reliability
            sma = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            
            df['BBM_20_2.0'] = sma.fillna(df['Close'])
            df['BBU_20_2.0'] = (sma + (std * std_dev)).fillna(df['Close'] * 1.02)
            df['BBL_20_2.0'] = (sma - (std * std_dev)).fillna(df['Close'] * 0.98)
            
            # Calculate additional BB indicators
            df['BBB_20_2.0'] = ((df['Close'] - df['BBL_20_2.0']) / 
                              (df['BBU_20_2.0'] - df['BBL_20_2.0'])).fillna(0.5)
            df['BBP_20_2.0'] = df['BBB_20_2.0']  # Same as BBB
            df['BBW'] = ((df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']).fillna(0.02)
            
            return df
        except Exception as e:
            self.logger.warning(f"Bollinger Bands calculation failed: {e}")
            # Fallback values
            df['BBM_20_2.0'] = df['Close']
            df['BBU_20_2.0'] = df['Close'] * 1.02
            df['BBL_20_2.0'] = df['Close'] * 0.98
            df['BBB_20_2.0'] = 0.5
            df['BBP_20_2.0'] = 0.5
            df['BBW'] = 0.02
            return df

    def _calculate_macd(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Safe MACD calculation"""
        try:
            fast = params["macd_fast"]
            slow = params["macd_slow"]
            signal = params["macd_signal"]
            
            # Manual MACD calculation
            ema_fast = df['Close'].ewm(span=fast).mean()
            ema_slow = df['Close'].ewm(span=slow).mean()
            
            df['MACD_12_26_9'] = (ema_fast - ema_slow).fillna(0)
            df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=signal).mean().fillna(0)
            df['MACDh_12_26_9'] = (df['MACD_12_26_9'] - df['MACDs_12_26_9']).fillna(0)
            
            return df
        except Exception as e:
            self.logger.warning(f"MACD calculation failed: {e}")
            df['MACD_12_26_9'] = 0
            df['MACDs_12_26_9'] = 0
            df['MACDh_12_26_9'] = 0
            return df

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Safe ADX calculation"""
        try:
            # Simplified ADX calculation
            high_diff = df['High'].diff()
            low_diff = df['Low'].diff()
            
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff < 0), 0)
            
            # True Range
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift())
            tr3 = abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Smoothed values
            plus_dm_smooth = plus_dm.rolling(period).mean()
            minus_dm_smooth = minus_dm.rolling(period).mean()
            tr_smooth = true_range.rolling(period).mean()
            
            # Directional indicators
            df['DMP_14'] = (100 * plus_dm_smooth / tr_smooth).fillna(25)
            df['DMN_14'] = (100 * minus_dm_smooth / tr_smooth).fillna(25)
            
            # ADX calculation (simplified)
            dx = abs((df['DMP_14'] - df['DMN_14']) / (df['DMP_14'] + df['DMN_14'] + 1e-10)) * 100
            df['ADX_14'] = dx.rolling(period).mean().fillna(25)
            
            return df
        except Exception as e:
            self.logger.warning(f"ADX calculation failed: {e}")
            df['ADX_14'] = 25
            df['DMP_14'] = 25
            df['DMN_14'] = 25
            return df

    def _calculate_stochastic(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Safe Stochastic calculation"""
        try:
            k_period = params["stoch_k"]
            d_period = params["stoch_d"]
            
            # Manual Stochastic calculation
            lowest_low = df['Low'].rolling(k_period).min()
            highest_high = df['High'].rolling(k_period).max()
            
            k_percent = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low + 1e-10))
            
            df['STOCHk_14_3_3'] = k_percent.fillna(50)
            df['STOCHd_14_3_3'] = df['STOCHk_14_3_3'].rolling(d_period).mean().fillna(50)
            
            return df
        except Exception as e:
            self.logger.warning(f"Stochastic calculation failed: {e}")
            df['STOCHk_14_3_3'] = 50
            df['STOCHd_14_3_3'] = 50
            return df

    def _calculate_ichimoku_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Safe Ichimoku calculation with fallbacks"""
        try:
            # Simplified Ichimoku calculation
            high_9 = df['High'].rolling(9).max()
            low_9 = df['Low'].rolling(9).min()
            df['ISA_9'] = ((high_9 + low_9) / 2).fillna(df['Close'])
            
            high_26 = df['High'].rolling(26).max()
            low_26 = df['Low'].rolling(26).min()
            df['ISB_26'] = ((high_26 + low_26) / 2).fillna(df['Close'])
            
            df['ITS_9'] = ((df['ISA_9'] + df['ISB_26']) / 2).fillna(df['Close'])
            
            high_52 = df['High'].rolling(52).max()
            low_52 = df['Low'].rolling(52).min()
            df['IKS_26'] = ((high_52 + low_52) / 2).fillna(df['Close'])
            
            df['ICS_26'] = df['Close'].shift(-26).fillna(df['Close'])
            
            return df
        except Exception as e:
            self.logger.warning(f"Ichimoku calculation failed: {e}")
            df['ISA_9'] = df['Close']
            df['ISB_26'] = df['Close']
            df['ITS_9'] = df['Close']
            df['IKS_26'] = df['Close']
            df['ICS_26'] = df['Close']
            return df

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Safe volume indicators"""
        try:
            if 'Volume' in df.columns and df['Volume'].sum() > 0:
                # Simple OBV calculation
                obv = [0]
                for i in range(1, len(df)):
                    if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                        obv.append(obv[-1] + df['Volume'].iloc[i])
                    elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                        obv.append(obv[-1] - df['Volume'].iloc[i])
                    else:
                        obv.append(obv[-1])
                
                df['OBV'] = obv
            else:
                df['OBV'] = 0
                
            return df
        except Exception as e:
            self.logger.warning(f"Volume indicators calculation failed: {e}")
            df['OBV'] = 0
            return df

    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Safe momentum features"""
        try:
            df['price_change'] = df['Close'].pct_change().fillna(0)
            df['volatility'] = df['price_change'].rolling(20).std().fillna(0.001)
            df['momentum'] = df['Close'].pct_change(periods=10).fillna(0)
            
            return df
        except Exception as e:
            self.logger.warning(f"Momentum features calculation failed: {e}")
            df['price_change'] = 0
            df['volatility'] = 0.001
            df['momentum'] = 0
            return df

    def _clean_and_validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive feature cleaning and validation"""
        try:
            # Replace infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with appropriate defaults
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if df[col].isna().any():
                    if 'price' in col.lower() or 'close' in col.lower():
                        df[col] = df[col].fillna(method='ffill').fillna(df['Close'].iloc[0] if 'Close' in df.columns else 1.0)
                    elif 'rsi' in col.lower() or 'stoch' in col.lower():
                        df[col] = df[col].fillna(50)
                    elif 'volume' in col.lower() or 'obv' in col.lower():
                        df[col] = df[col].fillna(0)
                    elif 'macd' in col.lower():
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(method='ffill').fillna(0)
            
            # Ensure no remaining NaN values
            df = df.fillna(0)
            
            self.logger.debug(f"Feature cleaning completed. NaN count: {df.isna().sum().sum()}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning features: {e}")
            return df.fillna(0)

    def _create_fallback_dataframe(self, original_input) -> pd.DataFrame:
        """Create a fallback DataFrame when feature engineering fails"""
        try:
            self.logger.warning("Creating fallback DataFrame with basic features")
            
            # Try to extract basic price data
            if isinstance(original_input, pd.DataFrame) and 'Close' in original_input.columns:
                close_price = original_input['Close'].iloc[-1] if not original_input.empty else 1.0
            else:
                close_price = 1.0
            
            # Create minimal feature set
            basic_data = {
                'Close': [close_price],
                'Open': [close_price],
                'High': [close_price],
                'Low': [close_price],
                'Volume': [0]
            }
            
            # Add all expected features with default values
            for feature in self.expected_features[:-3]:  # Exclude position info
                if feature not in basic_data:
                    if 'rsi' in feature.lower() or 'stoch' in feature.lower():
                        basic_data[feature] = [50.0]
                    elif 'price' in feature.lower() or 'close' in feature.lower():
                        basic_data[feature] = [close_price]
                    else:
                        basic_data[feature] = [0.0]
            
            return pd.DataFrame(basic_data)
            
        except Exception as e:
            self.logger.error(f"Error creating fallback DataFrame: {e}")
            # Absolute fallback
            return pd.DataFrame({'Close': [1.0]})

    def validate_features(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """Validate features against expected schema"""
        try:
            if df is None or df.empty:
                return False, self.expected_features[:-3], []
            
            current_features = set(df.select_dtypes(include=[np.number]).columns)
            expected_features = set(self.expected_features[:-3])  # Exclude position info
            
            missing_features = list(expected_features - current_features)
            extra_features = list(current_features - expected_features)
            
            is_valid = len(missing_features) == 0
            
            if not is_valid:
                self.logger.debug(f"Feature validation: missing {len(missing_features)}, extra {len(extra_features)}")
            
            return is_valid, missing_features, extra_features
            
        except Exception as e:
            self.logger.error(f"Error validating features: {e}")
            return False, [], []

    def prepare_observation(self, df: pd.DataFrame, position_info: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Prepare observation vector for RL model with exact size matching"""
        try:
            if df is None or len(df) == 0:
                self.logger.warning("Cannot prepare observation: empty DataFrame")
                return None
            
            # Get latest row of numeric features
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                self.logger.warning("No numeric features found in DataFrame")
                return None
            
            latest_data = numeric_df.iloc[-1]
            
            # Extract features in exact expected order
            feature_values = []
            for feature_name in self.expected_features[:-3]:  # Exclude position info
                if feature_name in latest_data.index:
                    value = latest_data[feature_name]
                    # Ensure value is valid
                    if pd.isna(value) or np.isinf(value):
                        if 'rsi' in feature_name.lower() or 'stoch' in feature_name.lower():
                            value = 50.0
                        elif 'price' in feature_name.lower() or 'close' in feature_name.lower():
                            value = float(latest_data.get('Close', 1.0))
                        else:
                            value = 0.0
                    feature_values.append(float(value))
                else:
                    # Missing feature - use appropriate default
                    if 'rsi' in feature_name.lower() or 'stoch' in feature_name.lower():
                        feature_values.append(50.0)
                    elif 'price' in feature_name.lower() or 'close' in feature_name.lower():
                        feature_values.append(float(latest_data.get('Close', 1.0)))
                    else:
                        feature_values.append(0.0)
            
            # Add position information
            if position_info is None:
                position_info = np.array([0.0, 0.0, 0.0])  # No position
            elif len(position_info) != 3:
                position_info = np.array([0.0, 0.0, 0.0])
            
            # Combine all features
            observation = np.array(feature_values + position_info.tolist(), dtype=np.float32)
            
            # Ensure exact observation size
            if len(observation) < self.observation_size:
                # Pad with zeros
                padding = np.zeros(self.observation_size - len(observation), dtype=np.float32)
                observation = np.concatenate([observation, padding])
            elif len(observation) > self.observation_size:
                # Truncate
                observation = observation[:self.observation_size]
            
            # Final validation and cleaning
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if len(observation) != self.observation_size:
                self.logger.error(f"CRITICAL: Observation size mismatch: {len(observation)} != {self.observation_size}")
                return None
            
            self.logger.debug(f"✅ Observation prepared successfully: shape {observation.shape}")
            return observation
            
        except Exception as e:
            self.logger.error(f"Error preparing observation: {e}")
            return None

    def save_feature_config(self):
        """Save current feature configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = {
                "version": self.feature_config.version,
                "features": self.feature_config.features,
                "observation_size": self.feature_config.observation_size,
                "feature_params": self.feature_config.feature_params
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"Feature configuration saved to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving feature config: {e}")

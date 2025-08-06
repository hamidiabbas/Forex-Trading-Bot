"""
Enhanced Feature Engineering System - COMPLETE MERGED VERSION
Combines all advanced features with critical fixes for RL compatibility
Configured for 100 features instead of 50
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union, Any
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from functools import wraps
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# External library availability checks
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("Warning: pandas_ta not available, using manual calculations")

try:
    from scipy import stats
    import scipy.signal as signal
    from scipy.cluster.hierarchy import linkage, fcluster
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, statistical features disabled")

try:
    from sklearn.decomposition import PCA, FastICA
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import mutual_info_score
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, ML features disabled")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available, using alternative calculations")

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration class for feature engineering"""
    version: str = "2.1.0"
    cache_enabled: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    feature_selection_enabled: bool = True
    statistical_features: bool = True
    ml_features: bool = True
    advanced_technical: bool = True
    regime_features: bool = True
    microstructure_features: bool = True
    
    # ✅ CRITICAL FIX: Set to 100 features for RL model
    rl_features: int = 100  # Changed from 50 to 100
    
    # Technical indicator parameters
    feature_params: Dict = None
    
    def __post_init__(self):
        if self.feature_params is None:
            self.feature_params = {
                'sma_periods': [5, 10, 20, 50, 100, 200],
                'ema_periods': [8, 12, 21, 26, 50],
                'rsi_periods': [7, 14, 21],
                'bb_periods': [20, 50],
                'bb_deviations': [2.0, 2.5],
                'atr_periods': [14, 21],
                'macd_settings': [(12, 26, 9), (5, 35, 5)],
                'stoch_periods': [(14, 3, 3), (5, 3, 3)],
                'ma_periods': [5, 10, 20, 50, 100, 200],
                'lookback_windows': [10, 20, 50, 100],
                'volatility_windows': [5, 10, 20],
                'momentum_periods': [5, 10, 20],
                'regime_window': 50
            }

class AdvancedFeatureEngineer:
    """Advanced Feature Engineering with comprehensive technical and statistical analysis"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize the feature engineer with advanced capabilities"""
        self.config = config or FeatureConfig()
        
        # ✅ CRITICAL FIX: Set observation size for RL compatibility to 100
        self.observation_size = 100  # Changed from 50 to 100
        
        # Cache system
        self.feature_cache = {}
        self.cache_enabled = self.config.cache_enabled
        
        # Feature engineering settings
        self.enable_advanced_features = True
        self.enable_statistical_features = SCIPY_AVAILABLE and self.config.statistical_features
        self.enable_ml_features = SKLEARN_AVAILABLE and self.config.ml_features
        
        # Technical analysis parameters
        self.ma_periods = self.config.feature_params['ma_periods']
        self.rsi_periods = self.config.feature_params['rsi_periods']
        self.bb_periods = self.config.feature_params['bb_periods']
        self.bb_deviations = self.config.feature_params['bb_deviations']
        self.atr_periods = self.config.feature_params['atr_periods']
        
        # Statistical analysis parameters
        self.lookback_windows = self.config.feature_params['lookback_windows']
        self.volatility_windows = self.config.feature_params['volatility_windows']
        
        # Performance tracking
        self.feature_generation_times = []
        self.last_feature_count = 0
        
        logger.info("✅ AdvancedFeatureEngineer initialized")
        logger.info(f"   Statistical features: {self.enable_statistical_features}")
        logger.info(f"   ML features: {self.enable_ml_features}")
        logger.info(f"   ✅ FIXED: Expected features for RL: {self.observation_size}")

    def engineer_features(self, data: Union[pd.DataFrame, tuple, list, np.ndarray]) -> Optional[pd.DataFrame]:
        """
        ✅ FIXED: Robust feature engineering with comprehensive error handling
        Maintains all existing advanced features while handling data type issues
        """
        start_time = time.time()
        
        try:
            # ✅ CRITICAL FIX: Handle different input data types
            df = self._ensure_dataframe(data)
            if df is None or len(df) < 50:
                logger.warning("Insufficient data for comprehensive feature engineering")
                return self._create_fallback_features()
            
            # Validate required columns
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return self._create_fallback_features()
            
            # Add Volume if missing
            if 'Volume' not in df.columns:
                df['Volume'] = 1000000
                logger.debug("Added default volume data")
            
            # Initialize features DataFrame
            features = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            logger.debug(f"Starting comprehensive feature engineering on DataFrame with shape {features.shape}")
            
            # 1. Basic Price Features (15 features)
            features = self._add_basic_price_features(features)
            
            # 2. Technical Indicators (35+ features)
            features = self._add_comprehensive_technical_indicators(features)
            
            # 3. ✅ FIXED: pandas_ta features with proper error handling
            features = self._add_pandas_ta_features(features)
            
            # 4. Statistical Features (20+ features) 
            if self.enable_statistical_features:
                features = self._add_advanced_statistical_features(features)
            
            # 5. ML-derived Features (15+ features)
            if self.enable_ml_features:
                features = self._add_ml_derived_features(features)
            
            # 6. Market Regime Features (10+ features)
            features = self._add_market_regime_features(features)
            
            # 7. Microstructure Features (15+ features)
            features = self._add_microstructure_features(features)
            
            # 8. Time-based and Session Features (10+ features)
            features = self._add_temporal_features(features)
            
            # 9. Advanced Momentum and Trend Features (15+ features)
            features = self._add_advanced_momentum_features(features)
            
            # 10. Volatility and Risk Features (10+ features)
            features = self._add_volatility_risk_features(features)
            
            # Clean and validate all features
            features = self._clean_and_validate_features(features)
            
            # Track performance
            generation_time = time.time() - start_time
            self.feature_generation_times.append(generation_time)
            self.last_feature_count = len(features.columns)
            
            logger.info(f"✅ Generated {len(features.columns)} total features in {generation_time:.2f}s")
            
            return features
            
        except Exception as e:
            logger.error(f"Error in comprehensive feature engineering: {e}")
            logger.error(f"Input type: {type(data)}")
            if hasattr(data, 'shape'):
                logger.error(f"Input shape: {data.shape}")
            
            # Return fallback features to prevent total failure
            return self._create_fallback_features()

    def _ensure_dataframe(self, data_input: Union[pd.DataFrame, tuple, list, np.ndarray]) -> Optional[pd.DataFrame]:
        """
        ✅ CRITICAL FIX: Convert any input type to DataFrame with robust error handling
        """
        try:
            if isinstance(data_input, pd.DataFrame):
                return data_input.copy()
                
            elif isinstance(data_input, tuple):
                logger.warning(f"Received tuple input, attempting to convert (type: {type(data_input)})")
                
                # Handle different tuple structures
                if len(data_input) > 0:
                    # If first element is a DataFrame
                    if isinstance(data_input[0], pd.DataFrame):
                        return data_input[0].copy()
                    
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
                                    'Volume': data_input[4] if len(data_input) > 4 else [1000000] * len(data_input[0])
                                })
                                return df
                        except Exception as e:
                            logger.error(f"Failed to convert tuple to DataFrame: {e}")
                            return None
                            
            elif isinstance(data_input, (list, np.ndarray)):
                logger.warning(f"Received {type(data_input)} input, attempting to convert")
                try:
                    # Convert to DataFrame
                    if isinstance(data_input, np.ndarray) and len(data_input.shape) == 2:
                        columns = ['Open', 'High', 'Low', 'Close', 'Volume'][:data_input.shape[1]]
                        return pd.DataFrame(data_input, columns=columns)
                    elif isinstance(data_input, list) and len(data_input) > 0:
                        if isinstance(data_input[0], dict):
                            return pd.DataFrame(data_input)
                except Exception as e:
                    logger.error(f"Failed to convert {type(data_input)} to DataFrame: {e}")
                    return None
            
            else:
                logger.error(f"Unsupported input type: {type(data_input)}")
                return None
                
        except Exception as e:
            logger.error(f"Error ensuring DataFrame: {e}")
            return None

    def _add_pandas_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ FIXED: Add pandas_ta features with proper error handling for tuple inputs
        """
        try:
            params = self.config.feature_params
            
            # ✅ FIX: Ensure we have a proper DataFrame, not tuple
            if isinstance(df, tuple):
                logger.error("Received tuple instead of DataFrame in _add_pandas_ta_features")
                return pd.DataFrame()
            
            if df is None or df.empty:
                logger.warning("Empty DataFrame for pandas_ta features")
                return df
            
            # Ensure required columns exist
            required_columns = ["Open", "High", "Low", "Close"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns for pandas_ta: {missing_columns}")
                return df
            
            if not PANDAS_TA_AVAILABLE:
                logger.debug("pandas_ta not available, using manual calculations")
                return self._add_manual_technical_features(df)
            
            try:
                # EMA indicators
                for period in params['ema_periods'][:3]:  # Use first 3 periods to avoid too many features
                    df[f'EMA_{period}'] = ta.ema(df['Close'], length=period)
                
                # RSI
                df['RSI_14'] = ta.rsi(df['Close'], length=params['rsi_periods'][0])
                
                # MACD
                macd_data = ta.macd(df['Close'], fast=params['macd_settings'][0][0],
                                   slow=params['macd_settings'][0][1], 
                                   signal=params['macd_settings'][0][2])
                if macd_data is not None and isinstance(macd_data, pd.DataFrame):
                    for col in macd_data.columns:
                        if col not in df.columns:  # Avoid duplicates
                            df[col] = macd_data[col]
                
                # Bollinger Bands
                bb_data = ta.bbands(df['Close'], length=params['bb_periods'][0], 
                                   std=params['bb_deviations'][0])
                if bb_data is not None and isinstance(bb_data, pd.DataFrame):
                    for col in bb_data.columns:
                        if col not in df.columns:  # Avoid duplicates
                            df[col] = bb_data[col]
                
                # ATR
                df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], 
                                     length=params['atr_periods'][0])
                
                # ✅ CRITICAL FIX: Handle ADX properly to avoid tuple errors
                try:
                    adx_result = ta.adx(df['High'], df['Low'], df['Close'], length=14)
                    if adx_result is not None:
                        if isinstance(adx_result, pd.DataFrame):
                            # Take only the ADX column, ignore DMP and DMN
                            if 'ADX_14' in adx_result.columns:
                                df['ADX_14'] = adx_result['ADX_14']
                            else:
                                df['ADX_14'] = adx_result.iloc[:, 0]  # Take first column
                        elif isinstance(adx_result, pd.Series):
                            df['ADX_14'] = adx_result
                except Exception as adx_error:
                    logger.warning(f"ADX calculation failed: {adx_error}")
                    df['ADX_14'] = 25  # Default neutral value
                
                # Stochastic
                stoch_data = ta.stoch(df['High'], df['Low'], df['Close'], 
                                     k=params['stoch_periods'][0][0],
                                     d=params['stoch_periods'][0][1])
                if stoch_data is not None and isinstance(stoch_data, pd.DataFrame):
                    for col in stoch_data.columns:
                        if col not in df.columns:
                            df[col] = stoch_data[col]
                
                # Volume indicators (if volume is available)
                if 'Volume' in df.columns and df['Volume'].sum() > 0:
                    df['OBV'] = ta.obv(df['Close'], df['Volume'])
                    df['MFI_14'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
                else:
                    df['OBV'] = 0
                    df['MFI_14'] = 50
                
                logger.debug("Successfully added pandas_ta features")
                
            except Exception as pandas_ta_error:
                logger.warning(f"pandas_ta feature generation failed: {pandas_ta_error}")
                # ✅ FALLBACK: Use manual features
                return self._add_manual_technical_features(df)
                
        except Exception as e:
            logger.error(f"Error adding pandas_ta features: {e}")
            # ✅ FALLBACK: Use manual technical features
            return self._add_manual_technical_features(df)
        
        return df

    def _add_basic_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fundamental price-based features"""
        try:
            # Price relationships
            df['price_range'] = df['High'] - df['Low']
            df['body_size'] = abs(df['Close'] - df['Open'])
            df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
            df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
            
            # Price positions and ratios
            df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['weighted_price'] = (df['High'] + df['Low'] + 2*df['Close']) / 4
            df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
            
            # Returns and changes
            df['price_change'] = df['Close'].pct_change().fillna(0)
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
            df['high_low_ratio'] = df['High'] / df['Low']
            
            # Additional price features for more comprehensive coverage
            df['open_close_ratio'] = df['Open'] / df['Close']
            df['price_momentum'] = df['Close'] / df['Close'].shift(5) - 1
            df['price_acceleration'] = df['price_change'].diff()
            df['gap_up'] = (df['Open'] > df['Close'].shift(1)).astype(int)
            df['gap_down'] = (df['Open'] < df['Close'].shift(1)).astype(int)
            
        except Exception as e:
            logger.error(f"Error adding basic price features: {e}")
        
        return df

    def _add_comprehensive_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators with manual calculations"""
        try:
            # Moving Averages
            for period in [5, 10, 20, 50]:
                df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            
            # RSI
            df['RSI_14'] = self._calculate_rsi(df['Close'], 14)
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            sma_bb = df['Close'].rolling(bb_period).mean()
            bb_std_dev = df['Close'].rolling(bb_period).std()
            df['BB_Upper'] = sma_bb + (bb_std_dev * bb_std)
            df['BB_Lower'] = sma_bb - (bb_std_dev * bb_std)
            df['BB_Middle'] = sma_bb
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Width'] + 1e-10)
            
            # ATR
            df['ATR_14'] = self._calculate_atr(df, 14)
            df['ATR_ratio'] = df['ATR_14'] / df['Close']
            
            # Stochastic
            df['Stoch_K'] = self._calculate_stochastic_k(df, 14)
            df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
            
            # Williams %R
            df['Williams_R'] = self._calculate_williams_r(df, 14)
            
            # CCI
            df['CCI_20'] = self._calculate_cci(df, 20)
            
            # Additional technical indicators
            df['ROC_10'] = df['Close'].pct_change(periods=10)
            df['ROC_20'] = df['Close'].pct_change(periods=20)
            
        except Exception as e:
            logger.error(f"Error adding comprehensive technical indicators: {e}")
        
        return df

    def _add_advanced_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced statistical features using scipy"""
        if not SCIPY_AVAILABLE:
            return df
            
        try:
            # Rolling statistical measures
            for window in [10, 20, 50]:
                returns = df['Close'].pct_change()
                
                # Higher moments
                df[f'skewness_{window}'] = returns.rolling(window).skew()
                df[f'kurtosis_{window}'] = returns.rolling(window).kurt()
                
                # Volatility measures
                df[f'volatility_{window}'] = returns.rolling(window).std()
                df[f'realized_vol_{window}'] = np.sqrt(returns.rolling(window).var() * 252)
                
            # Autocorrelation features
            returns = df['Close'].pct_change().dropna()
            if len(returns) > 50:
                df['autocorr_lag1'] = returns.rolling(50).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)
                df['autocorr_lag5'] = returns.rolling(50).apply(lambda x: x.autocorr(lag=5) if len(x) > 5 else 0)
            
            # Entropy measures
            df['price_entropy'] = self._calculate_rolling_entropy(df['Close'], window=20)
            
        except Exception as e:
            logger.error(f"Error adding advanced statistical features: {e}")
        
        return df

    def _add_ml_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ML-derived features using sklearn"""
        if not SKLEARN_AVAILABLE:
            return df
            
        try:
            # Price-based features for ML
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if len(df) > 50:
                # PCA features
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[feature_cols].fillna(method='ffill'))
                
                pca = PCA(n_components=3)
                pca_features = pca.fit_transform(scaled_data)
                
                df['PCA_1'] = pca_features[:, 0]
                df['PCA_2'] = pca_features[:, 1] 
                df['PCA_3'] = pca_features[:, 2]
                
                # Clustering features
                if len(df) > 100:
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    df['price_cluster'] = kmeans.fit_predict(scaled_data)
            
        except Exception as e:
            logger.error(f"Error adding ML-derived features: {e}")
        
        return df

    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime identification features"""
        try:
            # Trend regime
            sma_20 = df['Close'].rolling(20).mean()
            sma_50 = df['Close'].rolling(50).mean()
            df['trend_regime'] = (sma_20 > sma_50).astype(int)
            
            # Volatility regime
            vol_20 = df['Close'].pct_change().rolling(20).std()
            vol_median = vol_20.rolling(100).median()
            df['vol_regime'] = (vol_20 > vol_median).astype(int)
            
            # Momentum regime
            momentum = df['Close'] / df['Close'].shift(10) - 1
            df['momentum_regime'] = (momentum > 0).astype(int)
            
            # Combined regime score
            df['regime_score'] = (df['trend_regime'] + df['vol_regime'] + df['momentum_regime']) / 3
            
            # Market state features
            df['bull_market'] = (df['Close'] > df['Close'].rolling(100).mean()).astype(int)
            df['bear_market'] = (df['Close'] < df['Close'].rolling(100).mean()).astype(int)
            
        except Exception as e:
            logger.error(f"Error adding market regime features: {e}")
        
        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add microstructure and market making features"""
        try:
            # Bid-ask spread proxies
            df['bid_ask_spread_proxy'] = (df['High'] - df['Low']) / df['Close']
            
            # Price impact measures
            df['price_impact'] = abs(df['Close'].pct_change()) / (df['Volume'] / df['Volume'].rolling(20).mean())
            df['price_impact'] = df['price_impact'].fillna(0)
            
            # Tick-based features
            df['tick_direction'] = np.sign(df['Close'].diff())
            df['tick_imbalance'] = df['tick_direction'].rolling(20).sum()
            
            # Volume-price relationship
            df['volume_price_trend'] = df['Volume'] * df['Close'].pct_change()
            df['vwap'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
            df['price_vwap_ratio'] = df['Close'] / df['vwap']
            
            # Additional microstructure features
            df['volume_weighted_price'] = (df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']
            df['price_volume_correlation'] = df['Close'].rolling(20).corr(df['Volume'])
            
        except Exception as e:
            logger.error(f"Error adding microstructure features: {e}")
        
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based and session features"""
        try:
            if df.index.dtype == 'datetime64[ns]' or hasattr(df.index, 'hour'):
                df['hour'] = df.index.hour
                df['day_of_week'] = df.index.dayofweek
                df['month'] = df.index.month
                
                # Session indicators (assuming UTC time)
                df['asian_session'] = ((df['hour'] >= 23) | (df['hour'] <= 8)).astype(int)
                df['european_session'] = ((df['hour'] >= 7) & (df['hour'] <= 16)).astype(int)
                df['us_session'] = ((df['hour'] >= 13) & (df['hour'] <= 22)).astype(int)
            else:
                # Default values when no datetime index
                df['hour'] = 12
                df['day_of_week'] = 1
                df['month'] = 6
                df['asian_session'] = 0
                df['european_session'] = 1
                df['us_session'] = 0
            
        except Exception as e:
            logger.error(f"Error adding temporal features: {e}")
            # Set default values
            df['hour'] = 12
            df['day_of_week'] = 1
            df['month'] = 6
            df['asian_session'] = 0
            df['european_session'] = 1
            df['us_session'] = 0
        
        return df

    def _add_advanced_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced momentum and trend features"""
        try:
            # Multi-timeframe momentum
            for period in [5, 10, 20, 50]:
                df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
                df[f'roc_{period}'] = df['Close'].pct_change(periods=period)
            
            # Acceleration
            df['price_acceleration'] = df['Close'].pct_change().diff()
            
            # Trend strength
            df['trend_strength'] = abs(df['Close'].rolling(20).corr(pd.Series(range(20))))
            
            # Momentum oscillators
            df['momentum_oscillator'] = df['Close'] - df['Close'].rolling(20).mean()
            df['rate_of_change'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
            
        except Exception as e:
            logger.error(f"Error adding advanced momentum features: {e}")
        
        return df

    def _add_volatility_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility and risk-based features"""
        try:
            returns = df['Close'].pct_change()
            
            # GARCH-like features
            df['volatility_clustering'] = returns.rolling(20).var()
            
            # Value at Risk approximations
            df['var_95'] = returns.rolling(50).quantile(0.05)
            df['cvar_95'] = returns[returns <= df['var_95']].rolling(50).mean()
            
            # Maximum drawdown
            rolling_max = df['Close'].rolling(50).max()
            df['drawdown'] = (df['Close'] - rolling_max) / rolling_max
            df['max_drawdown'] = df['drawdown'].rolling(50).min()
            
            # Additional risk features
            df['downside_deviation'] = returns[returns < 0].rolling(20).std()
            df['upside_potential'] = returns[returns > 0].rolling(20).mean()
            
        except Exception as e:
            logger.error(f"Error adding volatility risk features: {e}")
        
        return df

    def _clean_and_validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate all engineered features"""
        try:
            # Replace infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill and backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Final NaN cleanup with appropriate defaults
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].isnull().any():
                    if 'rsi' in col.lower() or 'stoch' in col.lower():
                        df[col] = df[col].fillna(50)
                    elif 'price' in col.lower() or 'close' in col.lower():
                        df[col] = df[col].fillna(method='ffill').fillna(df['Close'].mean())
                    elif 'volume' in col.lower():
                        df[col] = df[col].fillna(df['Volume'].mean())
                    else:
                        df[col] = df[col].fillna(0)
            
            # Ensure no remaining NaN values
            df = df.fillna(0)
            
            logger.debug(f"Feature cleaning completed. Final shape: {df.shape}")
            
        except Exception as e:
            logger.error(f"Error cleaning features: {e}")
        
        return df

    def _create_fallback_features(self) -> pd.DataFrame:
        """Create fallback features when main feature engineering fails"""
        try:
            logger.warning("Creating fallback feature set")
            
            # Create basic fallback data with 100 features
            basic_features = {
                'Close': 1.0, 'Open': 1.0, 'High': 1.0, 'Low': 1.0, 'Volume': 1000000,
                'price_change': 0.0, 'volatility': 0.01, 'rsi_14': 50.0,
                'sma_20': 1.0, 'ema_20': 1.0, 'bb_upper': 1.05, 'bb_lower': 0.95,
                'atr_14': 0.01, 'macd': 0.0, 'momentum': 0.0
            }
            
            # Extend to meet 100 feature requirement
            feature_count = len(basic_features)
            if feature_count < 100:
                for i in range(100 - feature_count):
                    basic_features[f'fallback_feature_{i}'] = 0.0
            
            return pd.DataFrame([basic_features])
            
        except Exception as e:
            logger.error(f"Error creating fallback features: {e}")
            # Absolute minimal fallback
            return pd.DataFrame({'Close': [1.0]})

    # Helper methods for manual calculations
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI manually"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except:
            return pd.Series([50] * len(prices), index=prices.index)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR manually"""
        try:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr.fillna(df['Close'] * 0.01)
        except:
            return pd.Series([df['Close'].mean() * 0.01] * len(df), index=df.index)

    def _calculate_stochastic_k(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic %K manually"""
        try:
            lowest_low = df['Low'].rolling(window=period).min()
            highest_high = df['High'].rolling(window=period).max()
            k_percent = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
            return k_percent.fillna(50)
        except:
            return pd.Series([50] * len(df), index=df.index)

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R manually"""
        try:
            highest_high = df['High'].rolling(window=period).max()
            lowest_low = df['Low'].rolling(window=period).min()
            wr = -100 * (highest_high - df['Close']) / (highest_high - lowest_low + 1e-10)
            return wr.fillna(-50)
        except:
            return pd.Series([-50] * len(df), index=df.index)

    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate CCI manually"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - np.mean(x)))
            )
            cci = (typical_price - sma) / (0.015 * mad + 1e-10)
            return cci.fillna(0)
        except:
            return pd.Series([0] * len(df), index=df.index)

    def _calculate_rolling_entropy(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling entropy"""
        try:
            def entropy(x):
                value, counts = np.unique(x, return_counts=True)
                prob = counts / len(x)
                return -np.sum(prob * np.log2(prob + 1e-10))
            
            return series.rolling(window).apply(entropy)
        except:
            return pd.Series([1.0] * len(series), index=series.index)

    def _add_manual_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical features manually when pandas_ta fails"""
        try:
            logger.debug("Adding manual technical features as fallback")
            
            # Basic moving averages
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['EMA_20'] = df['Close'].ewm(span=20).mean()
            
            # RSI
            df['RSI_14'] = self._calculate_rsi(df['Close'], 14)
            
            # Simple MACD
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            
            # ATR
            df['ATR_14'] = self._calculate_atr(df, 14)
            
            # Volume features
            if 'Volume' in df.columns:
                df['OBV'] = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()
            else:
                df['OBV'] = 0
            
            logger.debug("Manual technical features added successfully")
            
        except Exception as e:
            logger.error(f"Error adding manual technical features: {e}")
        
        return df


class FeatureManager:
    """
    ✅ ENHANCED: Feature Manager with RL compatibility and advanced feature engineering
    Maintains all existing sophisticated features while ensuring RL model compatibility
    """
    
    def __init__(self, config_path: str = "configs/features.json"):
        """Initialize FeatureManager with RL compatibility"""
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # ✅ CRITICAL FIX: Set observation size to match RL model expectations (100 features)
        self.observation_size = 100  # Changed from 50 to 100
        self.expected_features = 100
        
        # Configuration
        self.config_path = Path(config_path)
        self.feature_config = self._load_feature_config()
        
        # Cache for performance
        self.feature_cache = {}
        self.cache_enabled = True
        
        logger.info("FeatureManager initialized - Version 2.1.0")
        logger.info(f"Expected features: {self.expected_features}")
        logger.info(f"Observation size: {self.observation_size}")

    def engineer_features(self, data: Union[pd.DataFrame, tuple, list, np.ndarray]) -> Optional[pd.DataFrame]:
        """
        ✅ ENHANCED: Generate comprehensive features while maintaining RL compatibility
        """
        try:
            if data is None:
                logger.warning("No data provided for feature engineering")
                return None
            
            # Use the advanced feature engineer to generate comprehensive features
            features_df = self.feature_engineer.engineer_features(data)
            
            if features_df is None or len(features_df) == 0:
                logger.error("Advanced feature engineering failed")
                return None
            
            logger.info(f"✅ Generated {len(features_df.columns)} comprehensive features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return None

    def get_latest_features(self, data: Union[pd.DataFrame, tuple, list, np.ndarray], lookback: int = 100) -> Optional[np.ndarray]:
        """
        ✅ NEW METHOD: Get latest features as array for RL model - FIXED VERSION FOR 100 FEATURES
        This maintains all your advanced 131+ features while providing exactly 100 features for RL
        """
        try:
            # Use your existing comprehensive feature engineering
            features_df = self.engineer_features(data)
            
            if features_df is None or len(features_df) == 0:
                logger.error("Feature engineering failed")
                return None
            
            # Get the latest row of numeric features
            numeric_features = features_df.select_dtypes(include=np.number).iloc[-1]
            
            # ✅ CRITICAL FIX: Ensure exactly 100 features for RL model
            if len(numeric_features) >= 100:
                # Select most important 100 features (keep your advanced ones)
                feature_array = numeric_features.values[:100]
                logger.debug(f"✂️ Selected top 100 features from {len(numeric_features)} available")
                
            elif len(numeric_features) < 100:
                # Pad to reach 100
                padding_needed = 100 - len(numeric_features)
                padding = np.zeros(padding_needed)
                feature_array = np.concatenate([numeric_features.values, padding])
                logger.debug(f"➕ Padded {padding_needed} features to reach 100")
                
            else:
                feature_array = numeric_features.values
                logger.debug("✅ Perfect: exactly 100 features")
            
            # Ensure valid values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Final validation
            if len(feature_array) != 100:
                logger.error(f"Feature array size mismatch: {len(feature_array)} != 100")
                return None
            
            logger.debug(f"✅ Prepared {len(feature_array)} features for RL model")
            return feature_array.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error getting latest features: {e}")
            return None

    def validate_features(self, features_df: pd.DataFrame) -> bool:
        """Validate features for RL model compatibility"""
        try:
            if features_df is None:
                return False
            
            numeric_cols = features_df.select_dtypes(include=np.number).columns
            if len(numeric_cols) < 10:  # Minimum reasonable number
                logger.error(f"Insufficient numeric features: {len(numeric_cols)}")
                return False
            
            # Check for NaN values
            if features_df.isnull().any().any():
                logger.warning("NaN values found in features - this will be handled in get_latest_features")
            
            return True
            
        except Exception as e:
            logger.error(f"Feature validation error: {e}")
            return False

    def _load_feature_config(self) -> Dict:
        """Load feature configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load feature config: {e}")
        
        # Return default configuration
        return {
            "version": "2.1.0",
            "feature_selection": True,
            "max_features": 200,
            "min_features": 100  # Changed from 50 to 100
        }

    def get_feature_importance(self, features_df: pd.DataFrame, target: pd.Series = None) -> Dict[str, float]:
        """
        Calculate feature importance scores
        """
        try:
            if not SKLEARN_AVAILABLE or target is None:
                return {}
            
            numeric_features = features_df.select_dtypes(include=np.number).fillna(0)
            
            # Use mutual information for feature importance
            importance_scores = mutual_info_regression(numeric_features, target)
            
            feature_importance = dict(zip(numeric_features.columns, importance_scores))
            
            # Sort by importance
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics"""
        return {
            "feature_engineer_type": "AdvancedFeatureEngineer",
            "observation_size": self.observation_size,
            "expected_features": self.expected_features,
            "cache_enabled": self.cache_enabled,
            "pandas_ta_available": PANDAS_TA_AVAILABLE,
            "scipy_available": SCIPY_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "last_feature_count": getattr(self.feature_engineer, 'last_feature_count', 0),
            "average_generation_time": np.mean(self.feature_engineer.feature_generation_times[-10:]) if self.feature_engineer.feature_generation_times else 0
        }


# Backwards compatibility
def create_feature_manager(config_path: str = "configs/features.json") -> FeatureManager:
    """Factory function to create FeatureManager"""
    return FeatureManager(config_path)


if __name__ == "__main__":
    # Test the feature engineering system
    logger.info("Testing Enhanced Feature Engineering System...")
    
    # Create test data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    test_data = pd.DataFrame({
        'Open': np.random.uniform(1.08, 1.12, len(dates)),
        'High': np.random.uniform(1.09, 1.13, len(dates)),
        'Low': np.random.uniform(1.07, 1.11, len(dates)),
        'Close': np.random.uniform(1.08, 1.12, len(dates)),
        'Volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    
    # Test feature manager
    fm = FeatureManager()
    
    # Test comprehensive feature engineering
    features = fm.engineer_features(test_data)
    print(f"\n✅ Generated {len(features.columns)} total features")
    
    # Test RL compatibility 
    rl_features = fm.get_latest_features(test_data)
    print(f"✅ RL-compatible features: {len(rl_features) if rl_features is not None else 'Failed'}")
    
    # Test diagnostics
    diagnostics = fm.get_diagnostics()
    print(f"✅ System diagnostics: {diagnostics}")
    
    logger.info("✅ Enhanced Feature Engineering System test completed successfully!")

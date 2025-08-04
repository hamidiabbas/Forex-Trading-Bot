# enhanced_feature_engineering.py
"""
Enhanced Feature Engineering Module for Forex Trading Bot
Compatible with existing MT5 architecture and RL training system
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced mathematical imports with fallbacks
try:
    from scipy import stats, signal
    from scipy.fft import fft, ifft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Advanced Feature Engineering class for Forex Trading RL
    
    Integrates with existing MT5 data pipeline and creates sophisticated
    features for reinforcement learning training[97][98].
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.feature_cache = {}
        
        # Feature engineering settings
        self.enable_advanced_features = True
        self.enable_statistical_features = SCIPY_AVAILABLE
        self.enable_ml_features = SKLEARN_AVAILABLE
        
        # Technical analysis parameters
        self.ma_periods = [5, 10, 20, 50, 100, 200]
        self.rsi_periods = [7, 14, 21]
        self.bb_periods = [20, 50]
        self.atr_periods = [14, 21]
        
        # Statistical analysis parameters
        self.lookback_windows = [10, 20, 50, 100]
        self.volatility_windows = [5, 10, 20]
        
        logger.info("✅ AdvancedFeatureEngineer initialized")
        logger.info(f"   Statistical features: {self.enable_statistical_features}")
        logger.info(f"   ML features: {self.enable_ml_features}")
    
    def engineer_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        
        Args:
            data: OHLCV DataFrame from MT5
            symbol: Trading symbol (for caching)
            
        Returns:
            Enhanced DataFrame with engineered features
        """
        try:
            if data.empty:
                logger.warning("Empty input data for feature engineering")
                return data
            
            logger.info(f"🔧 Starting feature engineering for {symbol or 'unknown symbol'}")
            start_time = datetime.now()
            
            df = data.copy()
            original_features = len(df.columns)
            
            # Phase 1: Basic Technical Indicators
            df = self._add_basic_technical_indicators(df)
            
            # Phase 2: Advanced Technical Features
            df = self._add_advanced_technical_features(df)
            
            # Phase 3: Statistical Features
            if self.enable_statistical_features:
                df = self._add_statistical_features(df)
            
            # Phase 4: Price Action Features
            df = self._add_price_action_features(df)
            
            # Phase 5: Volume Features
            df = self._add_volume_features(df)
            
            # Phase 6: Temporal Features
            df = self._add_temporal_features(df)
            
            # Phase 7: Market Structure Features
            df = self._add_market_structure_features(df)
            
            # Phase 8: ML-derived Features
            if self.enable_ml_features:
                df = self._add_ml_features(df)
            
            # Final cleanup
            df = self._cleanup_features(df)
            
            # Performance summary
            processing_time = (datetime.now() - start_time).total_seconds()
            new_features = len(df.columns) - original_features
            
            logger.info(f"✅ Feature engineering completed!")
            logger.info(f"   📊 Original features: {original_features}")
            logger.info(f"   ➕ Added features: {new_features}")
            logger.info(f"   📈 Total features: {len(df.columns)}")
            logger.info(f"   ⏱️ Processing time: {processing_time:.2f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            return data
    
    def _add_basic_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators[91][92]"""
        try:
            # Moving Averages
            for period in self.ma_periods:
                if len(df) >= period:
                    df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
                    df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            
            # RSI
            for period in self.rsi_periods:
                if len(df) >= period:
                    df[f'RSI_{period}'] = self._calculate_rsi(df['Close'], period)
            
            # MACD
            if len(df) >= 26:
                macd_line, macd_signal = self._calculate_macd(df['Close'])
                df['MACD'] = macd_line
                df['MACD_Signal'] = macd_signal
                df['MACD_Histogram'] = macd_line - macd_signal
            
            # Bollinger Bands
            for period in self.bb_periods:
                if len(df) >= period:
                    bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(df['Close'], period)
                    df[f'BB_Upper_{period}'] = bb_upper
                    df[f'BB_Lower_{period}'] = bb_lower
                    df[f'BB_Middle_{period}'] = bb_middle
                    df[f'BB_Width_{period}'] = (bb_upper - bb_lower) / bb_middle
                    df[f'BB_Position_{period}'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
            
            # ATR
            for period in self.atr_periods:
                if len(df) >= period:
                    df[f'ATR_{period}'] = self._calculate_atr(df, period)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding basic technical indicators: {e}")
            return df
    
    def _add_advanced_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical features"""
        try:
            # Momentum indicators
            momentum_periods = [5, 10, 20]
            for period in momentum_periods:
                if len(df) >= period:
                    df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
                    df[f'ROC_{period}'] = df['Close'].pct_change(period) * 100
            
            # Stochastic Oscillator
            if len(df) >= 14:
                k_period = 14
                d_period = 3
                lowest_low = df['Low'].rolling(k_period).min()
                highest_high = df['High'].rolling(k_period).max()
                
                k_percent = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
                df['Stoch_K'] = k_percent.rolling(d_period).mean()
                df['Stoch_D'] = df['Stoch_K'].rolling(d_period).mean()
            
            # Williams %R
            if len(df) >= 14:
                period = 14
                highest_high = df['High'].rolling(period).max()
                lowest_low = df['Low'].rolling(period).min()
                df['Williams_R'] = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
            
            # Commodity Channel Index (CCI)
            if len(df) >= 20:
                period = 20
                typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                sma_tp = typical_price.rolling(period).mean()
                mad = typical_price.rolling(period).apply(lambda x: abs(x - x.mean()).mean())
                df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding advanced technical features: {e}")
            return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features using scipy[98]"""
        try:
            if not SCIPY_AVAILABLE:
                return df
            
            # Rolling statistics
            for window in self.lookback_windows:
                if len(df) >= window:
                    returns = df['Close'].pct_change()
                    
                    # Statistical moments
                    df[f'Mean_{window}'] = returns.rolling(window).mean()
                    df[f'Std_{window}'] = returns.rolling(window).std()
                    df[f'Skew_{window}'] = returns.rolling(window).skew()
                    df[f'Kurt_{window}'] = returns.rolling(window).kurt()
                    
                    # Quantiles
                    df[f'Q25_{window}'] = returns.rolling(window).quantile(0.25)
                    df[f'Q75_{window}'] = returns.rolling(window).quantile(0.75)
                    df[f'IQR_{window}'] = df[f'Q75_{window}'] - df[f'Q25_{window}']
            
            # Z-score (standardized returns)
            if len(df) >= 20:
                returns = df['Close'].pct_change()
                rolling_mean = returns.rolling(20).mean()
                rolling_std = returns.rolling(20).std()
                df['Z_Score_20'] = (returns - rolling_mean) / rolling_std
            
            # Correlation with lagged prices
            for lag in [1, 2, 5, 10]:
                if len(df) >= lag + 20:
                    df[f'Corr_Lag_{lag}'] = df['Close'].rolling(20).corr(df['Close'].shift(lag))
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding statistical features: {e}")
            return df
    
    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action and candlestick features"""
        try:
            # Basic price relationships
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Close_Open_Ratio'] = df['Close'] / df['Open']
            
            # Candlestick body and shadow analysis
            df['Body_Size'] = abs(df['Close'] - df['Open'])
            df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
            df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
            df['Total_Range'] = df['High'] - df['Low']
            
            # Normalized ratios
            df['Body_Ratio'] = df['Body_Size'] / (df['Total_Range'] + 1e-10)
            df['Upper_Shadow_Ratio'] = df['Upper_Shadow'] / (df['Total_Range'] + 1e-10)
            df['Lower_Shadow_Ratio'] = df['Lower_Shadow'] / (df['Total_Range'] + 1e-10)
            
            # Price position within range
            df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
            
            # Gap analysis
            df['Gap_Up'] = (df['Open'] > df['High'].shift(1)).astype(int)
            df['Gap_Down'] = (df['Open'] < df['Low'].shift(1)).astype(int)
            df['Gap_Size'] = df['Open'] - df['Close'].shift(1)
            
            # Price velocity and acceleration
            df['Price_Velocity'] = df['Close'].diff()
            df['Price_Acceleration'] = df['Price_Velocity'].diff()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding price action features: {e}")
            return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        try:
            # Ensure Volume column exists
            if 'Volume' not in df.columns:
                df['Volume'] = 1000000  # Default volume
            
            # Volume moving averages
            volume_periods = [10, 20, 50]
            for period in volume_periods:
                if len(df) >= period:
                    df[f'Volume_MA_{period}'] = df['Volume'].rolling(period).mean()
                    df[f'Volume_Ratio_{period}'] = df['Volume'] / df[f'Volume_MA_{period}']
            
            # Price-Volume relationships
            df['Price_Volume'] = df['Close'] * df['Volume']
            df['Volume_Price_Trend'] = df['Close'].pct_change() * df['Volume']
            
            # On Balance Volume (OBV)
            price_change = df['Close'].diff()
            obv_values = np.where(price_change > 0, df['Volume'], 
                         np.where(price_change < 0, -df['Volume'], 0))
            df['OBV'] = obv_values.cumsum()
            
            # Volume Rate of Change
            df['Volume_ROC'] = df['Volume'].pct_change() * 100
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding volume features: {e}")
            return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
            
            # Time components
            df['Hour'] = df.index.hour
            df['DayOfWeek'] = df.index.dayofweek
            df['DayOfMonth'] = df.index.day
            df['Month'] = df.index.month
            df['Quarter'] = df.index.quarter
            
            # Market session indicators (assuming UTC)
            df['Asian_Session'] = ((df.index.hour >= 0) & (df.index.hour < 9)).astype(int)
            df['European_Session'] = ((df.index.hour >= 8) & (df.index.hour < 17)).astype(int)
            df['American_Session'] = ((df.index.hour >= 13) & (df.index.hour < 22)).astype(int)
            df['Session_Overlap'] = ((df['European_Session'] & df['American_Session']) | 
                                   (df['Asian_Session'] & df['European_Session'])).astype(int)
            
            # Cyclical encoding for time features
            df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding temporal features: {e}")
            return df
    
    def _add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure and regime features"""
        try:
            # Volatility regimes
            for window in self.volatility_windows:
                if len(df) >= window:
                    returns = df['Close'].pct_change()
                    volatility = returns.rolling(window).std()
                    vol_percentile = volatility.rolling(100).rank(pct=True)
                    
                    df[f'Vol_Regime_{window}'] = np.where(vol_percentile > 0.8, 2,  # High vol
                                                np.where(vol_percentile < 0.2, 0, 1))  # Low, Medium vol
            
            # Trend strength
            if len(df) >= 50:
                df['Trend_Strength'] = abs(df['EMA_20'] - df['EMA_50']) / df['Close']
            
            # Support and resistance levels
            if len(df) >= 20:
                df['Resistance_Level'] = df['High'].rolling(20).max()
                df['Support_Level'] = df['Low'].rolling(20).min()
                df['Support_Resistance_Position'] = (df['Close'] - df['Support_Level']) / (df['Resistance_Level'] - df['Support_Level'] + 1e-10)
            
            # Fractal patterns (simplified)
            if len(df) >= 5:
                df['Fractal_High'] = ((df['High'] > df['High'].shift(1)) & 
                                    (df['High'] > df['High'].shift(2)) & 
                                    (df['High'] > df['High'].shift(-1)) & 
                                    (df['High'] > df['High'].shift(-2))).astype(int)
                
                df['Fractal_Low'] = ((df['Low'] < df['Low'].shift(1)) & 
                                   (df['Low'] < df['Low'].shift(2)) & 
                                   (df['Low'] < df['Low'].shift(-1)) & 
                                   (df['Low'] < df['Low'].shift(-2))).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding market structure features: {e}")
            return df
    
    def _add_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add machine learning derived features[98]"""
        try:
            if not SKLEARN_AVAILABLE or len(df) < 100:
                return df
            
            # Prepare feature matrix for ML operations
            price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
            feature_data = df[price_features].fillna(method='ffill').fillna(0)
            
            if feature_data.empty:
                return df
            
            # Principal Component Analysis
            try:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(feature_data)
                
                pca = PCA(n_components=3)
                pca_result = pca.fit_transform(scaled_data)
                
                df['PCA_1'] = pca_result[:, 0]
                df['PCA_2'] = pca_result[:, 1]
                df['PCA_3'] = pca_result[:, 2]
                
            except Exception as e:
                logger.warning(f"PCA failed: {e}")
            
            # Clustering-based features
            try:
                if len(df) >= 200:
                    # Use recent data for clustering
                    recent_data = scaled_data[-200:]
                    
                    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(recent_data)
                    
                    # Extend cluster labels to full dataset
                    cluster_labels = np.zeros(len(df))
                    cluster_labels[-200:] = clusters
                    df['Market_Cluster'] = cluster_labels
                    
            except Exception as e:
                logger.warning(f"Clustering failed: {e}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding ML features: {e}")
            return df
    
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup and validation of features"""
        try:
            # Replace infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Intelligent NaN handling
            for col in df.columns:
                if col in ['Open', 'High', 'Low', 'Close']:
                    # Forward fill price data only
                    df[col] = df[col].fillna(method='ffill')
                elif 'Volume' in col:
                    # Fill volume with median or default
                    median_vol = df[col].median()
                    df[col] = df[col].fillna(median_vol if not pd.isna(median_vol) else 1000000)
                elif any(indicator in col for indicator in ['RSI', 'Stoch', 'Williams', 'CCI']):
                    # Fill oscillators with neutral values
                    if 'RSI' in col or 'Stoch' in col:
                        df[col] = df[col].fillna(50)
                    elif 'Williams' in col:
                        df[col] = df[col].fillna(-50)
                    elif 'CCI' in col:
                        df[col] = df[col].fillna(0)
                elif any(pattern in col for pattern in ['_Ratio', '_Position', '_Percent']):
                    # Fill ratios and percentages
                    df[col] = df[col].fillna(0)
                else:
                    # Forward fill other indicators
                    df[col] = df[col].fillna(method='ffill')
            
            # Final cleanup
            df = df.fillna(method='bfill').fillna(0)
            
            # Ensure all values are finite
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[col] = np.where(np.isfinite(df[col]), df[col], 0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature cleanup: {e}")
            return df
    
    # Helper methods for calculations
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with Wilder's smoothing"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd.fillna(0), macd_signal.fillna(0)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper.fillna(prices), lower.fillna(prices), sma.fillna(prices)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        
        return atr.fillna(0.001)
    
    def get_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance scores for analysis"""
        try:
            if not SKLEARN_AVAILABLE:
                return {}
            
            # Simple variance-based importance
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_importance = {}
            
            for col in numeric_cols:
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    variance = df[col].var()
                    feature_importance[col] = variance if pd.notna(variance) else 0.0
            
            # Normalize scores
            total_variance = sum(feature_importance.values())
            if total_variance > 0:
                feature_importance = {k: v/total_variance for k, v in feature_importance.items()}
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}

# Compatibility functions for existing code
def create_advanced_features(data: pd.DataFrame, config=None) -> pd.DataFrame:
    """Compatibility function for existing code"""
    engineer = AdvancedFeatureEngineer(config)
    return engineer.engineer_features(data)

def get_feature_names() -> List[str]:
    """Get list of feature names that will be created"""
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Add common feature patterns
    ma_features = [f'SMA_{p}' for p in [5, 10, 20, 50, 100, 200]]
    ma_features.extend([f'EMA_{p}' for p in [5, 10, 20, 50, 100, 200]])
    
    technical_features = [
        'RSI_7', 'RSI_14', 'RSI_21',
        'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Upper_20', 'BB_Lower_20', 'BB_Middle_20',
        'ATR_14', 'ATR_21',
        'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI'
    ]
    
    price_action_features = [
        'Body_Size', 'Upper_Shadow', 'Lower_Shadow',
        'Body_Ratio', 'Price_Position', 'Price_Velocity'
    ]
    
    return base_features + ma_features + technical_features + price_action_features

if __name__ == "__main__":
    # Test the module
    print("✅ Enhanced Feature Engineering module loaded successfully")
    print(f"📊 Available feature count: ~{len(get_feature_names())}")
    
    # Create test instance
    engineer = AdvancedFeatureEngineer()
    print(f"🔧 Feature engineering capabilities:")
    print(f"   Advanced features: {engineer.enable_advanced_features}")
    print(f"   Statistical features: {engineer.enable_statistical_features}")
    print(f"   ML features: {engineer.enable_ml_features}")

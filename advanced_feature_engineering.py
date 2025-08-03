# advanced_feature_engineering.py - Complete Professional Implementation

import numpy as np
import pandas as pd
import talib
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering system for trading algorithms
    Implements fractal analysis, market microstructure, and alternative data integration
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.lookback_periods = [5, 10, 20, 50, 100, 200]
        self.volatility_windows = [5, 10, 20, 50]
        self.fractal_dimensions = {}
        self.market_regimes = {}
        
        # Scalers for different feature types
        self.scalers = {
            'price': RobustScaler(),
            'volume': StandardScaler(),
            'technical': StandardScaler(),
            'fractal': StandardScaler()
        }
        
        logger.info("Advanced Feature Engineer initialized")

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        
        logger.info("Starting advanced feature engineering...")
        
        try:
            df_enhanced = df.copy()
            
            # Core market features
            df_enhanced = self.add_price_features(df_enhanced)
            df_enhanced = self.add_volume_features(df_enhanced)
            df_enhanced = self.add_volatility_features(df_enhanced)
            
            # Technical indicators
            df_enhanced = self.add_technical_indicators(df_enhanced)
            df_enhanced = self.add_momentum_indicators(df_enhanced)
            df_enhanced = self.add_trend_indicators(df_enhanced)
            
            # Advanced mathematical features
            df_enhanced = self.add_fractal_features(df_enhanced)
            df_enhanced = self.add_entropy_features(df_enhanced)
            df_enhanced = self.add_wavelet_features(df_enhanced)
            
            # Market microstructure
            df_enhanced = self.add_microstructure_features(df_enhanced)
            df_enhanced = self.add_liquidity_features(df_enhanced)
            
            # Market regime detection
            df_enhanced = self.add_regime_features(df_enhanced)
            
            # Alternative data features
            df_enhanced = self.add_temporal_features(df_enhanced)
            df_enhanced = self.add_cyclical_features(df_enhanced)
            
            # Feature interactions
            df_enhanced = self.add_interaction_features(df_enhanced)
            
            # Clean and normalize
            df_enhanced = self.clean_and_normalize_features(df_enhanced)
            
            logger.info(f"Feature engineering complete. Total features: {len(df_enhanced.columns)}")
            return df_enhanced
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return df

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        
        # Basic returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Multi-period returns
        for period in self.lookback_periods:
            df[f'returns_{period}'] = df['Close'].pct_change(period)
            df[f'log_returns_{period}'] = np.log(df['Close'] / df['Close'].shift(period))
        
        # Price relatives
        df['hl_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['oc_ratio'] = (df['Close'] - df['Open']) / df['Open']
        df['ho_ratio'] = (df['High'] - df['Open']) / df['Open']
        df['lo_ratio'] = (df['Low'] - df['Open']) / df['Open']
        
        # Gap analysis
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['gap_filled'] = (df['gap'].abs() < 0.001).astype(int)
        
        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        
        if 'Volume' not in df.columns:
            # If no volume data, create synthetic volume proxy
            df['Volume'] = np.abs(df['returns']) * 1000000  # Proxy based on returns
        
        # Volume indicators
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        # Volume-price relationship
        df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['price_volume_trend'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']).rolling(20).sum()
        
        # On-Balance Volume
        df['obv'] = (np.sign(df['returns']) * df['Volume']).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        
        return df

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        
        for window in self.volatility_windows:
            # Historical volatility
            df[f'volatility_{window}'] = df['returns'].rolling(window).std() * np.sqrt(252)
            
            # Garman-Klass volatility
            df[f'gk_vol_{window}'] = np.sqrt(
                ((np.log(df['High'] / df['Close']) * np.log(df['High'] / df['Open'])) +
                 (np.log(df['Low'] / df['Close']) * np.log(df['Low'] / df['Open']))).rolling(window).mean()
            )
            
            # Parkinson volatility
            df[f'parkinson_vol_{window}'] = np.sqrt(
                (0.25 * np.log(df['High'] / df['Low'])**2).rolling(window).mean()
            ) * np.sqrt(252)
        
        # Volatility regime
        df['vol_regime'] = pd.qcut(df['volatility_20'].fillna(0), q=3, labels=['low', 'medium', 'high'])
        df['vol_regime_encoded'] = df['vol_regime'].cat.codes
        
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        
        # Moving averages
        ma_periods = [5, 10, 20, 50, 100, 200]
        for period in ma_periods:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
        
        # Bollinger Bands
        for period in [10, 20]:
            sma = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
        
        # RSI variations
        for period in [14, 21]:
            df[f'rsi_{period}'] = talib.RSI(df['Close'].values, timeperiod=period)
            df[f'rsi_{period}_sma'] = df[f'rsi_{period}'].rolling(5).mean()
        
        # MACD variations
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['Close'].values)
        df['macd_slope'] = df['macd'].diff()
        df['macd_signal_slope'] = df['macd_signal'].diff()
        
        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators"""
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = talib.ROC(df['Close'].values, timeperiod=period)
        
        # Williams %R
        for period in [14, 21]:
            df[f'williams_r_{period}'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=period)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values)
        
        # Commodity Channel Index
        df['cci'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values)
        
        # Money Flow Index
        if 'Volume' in df.columns:
            df['mfi'] = talib.MFI(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values)
        
        return df

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators"""
        
        # ADX
        df['adx'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values)
        df['plus_di'] = talib.PLUS_DI(df['High'].values, df['Low'].values, df['Close'].values)
        df['minus_di'] = talib.MINUS_DI(df['High'].values, df['Low'].values, df['Close'].values)
        
        # Parabolic SAR
        df['sar'] = talib.SAR(df['High'].values, df['Low'].values)
        df['sar_signal'] = (df['Close'] > df['sar']).astype(int)
        
        # Aroon
        df['aroon_up'], df['aroon_down'] = talib.AROON(df['High'].values, df['Low'].values)
        df['aroon_osc'] = df['aroon_up'] - df['aroon_down']
        
        return df

    def add_fractal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fractal and chaos theory features"""[1]
        
        # Hurst Exponent
        for window in [50, 100]:
            df[f'hurst_{window}'] = df['Close'].rolling(window).apply(
                lambda x: self.calculate_hurst_exponent(x.values), raw=False
            )
        
        # Fractal dimension
        df['fractal_dimension'] = df['Close'].rolling(50).apply(
            lambda x: self.calculate_fractal_dimension(x.values), raw=False
        )
        
        # Detrended Fluctuation Analysis
        df['dfa'] = df['returns'].rolling(50).apply(
            lambda x: self.calculate_dfa(x.values), raw=False
        )
        
        return df

    def calculate_hurst_exponent(self, price_series: np.ndarray, max_lag: int = 20) -> float:
        """Calculate Hurst exponent for fractal analysis"""[2]
        
        if len(price_series) < max_lag + 1:
            return 0.5
        
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag]))) 
               for lag in lags]
        
        # Avoid log(0)
        tau = [max(t, 1e-10) for t in tau]
        
        try:
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0] * 2.0
        except:
            return 0.5

    def calculate_fractal_dimension(self, price_series: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        
        if len(price_series) < 10:
            return 1.5
        
        try:
            # Normalize series
            normalized = (price_series - np.min(price_series)) / (np.max(price_series) - np.min(price_series))
            
            # Box counting
            scales = np.logspace(0.01, 1, 10)
            counts = []
            
            for scale in scales:
                # Simple box counting approximation
                boxes = int(1.0 / scale)
                if boxes > 0:
                    count = len(np.unique(np.floor(normalized * boxes)))
                    counts.append(count)
                else:
                    counts.append(1)
            
            # Calculate dimension
            coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
            return -coeffs[0]
        except:
            return 1.5

    def calculate_dfa(self, series: np.ndarray) -> float:
        """Calculate Detrended Fluctuation Analysis"""
        
        if len(series) < 20:
            return 0.5
        
        try:
            # Integrate the series
            y = np.cumsum(series - np.mean(series))
            
            # Define scales
            scales = np.unique(np.logspace(0.5, 2, 10).astype(int))
            fluctuations = []
            
            for scale in scales:
                if scale >= len(y):
                    continue
                    
                # Divide series into segments
                segments = int(len(y) // scale)
                if segments == 0:
                    continue
                
                segment_flucs = []
                for i in range(segments):
                    start = i * scale
                    end = start + scale
                    segment = y[start:end]
                    
                    # Detrend
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = coeffs[0] * x + coeffs[1]
                    detrended = segment - trend
                    
                    # Calculate fluctuation
                    fluc = np.sqrt(np.mean(detrended**2))
                    segment_flucs.append(fluc)
                
                if segment_flucs:
                    fluctuations.append(np.mean(segment_flucs))
            
            if len(fluctuations) > 2:
                log_scales = np.log(scales[:len(fluctuations)])
                log_flucs = np.log(fluctuations)
                alpha = np.polyfit(log_scales, log_flucs, 1)[0]
                return alpha
            else:
                return 0.5
        except:
            return 0.5

    def add_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add entropy and information theory features"""
        
        # Shannon entropy
        for window in [20, 50]:
            df[f'entropy_{window}'] = df['returns'].rolling(window).apply(
                lambda x: self.calculate_shannon_entropy(x.values), raw=False
            )
        
        # Approximate entropy
        df['approx_entropy'] = df['returns'].rolling(50).apply(
            lambda x: self.calculate_approximate_entropy(x.values), raw=False
        )
        
        return df

    def calculate_shannon_entropy(self, series: np.ndarray, bins: int = 10) -> float:
        """Calculate Shannon entropy"""
        
        try:
            # Create histogram
            counts, _ = np.histogram(series, bins=bins)
            counts = counts[counts > 0]  # Remove zero counts
            
            if len(counts) == 0:
                return 0.0
            
            # Calculate probabilities
            probs = counts / np.sum(counts)
            
            # Calculate entropy
            entropy = -np.sum(probs * np.log2(probs))
            return entropy
        except:
            return 0.0

    def calculate_approximate_entropy(self, series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate Approximate Entropy (ApEn)"""
        
        try:
            N = len(series)
            
            if N < m + 1:
                return 0.0
            
            def _maxdist(xi, xj, N, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([series[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    template_i = patterns[i]
                    for j in range(N - m + 1):
                        if _maxdist(template_i, patterns[j], N, m) <= r * np.std(series):
                            C[i] += 1.0
                
                phi = (N - m + 1.0) ** (-1) * np.sum(np.log(C / (N - m + 1.0)))
                return phi
            
            return _phi(m) - _phi(m + 1)
        except:
            return 0.0

    def add_wavelet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add wavelet transform features"""
        
        try:
            import pywt
            
            # Wavelet decomposition
            for scale in [2, 4, 8]:
                coeffs = pywt.dwt(df['Close'].fillna(method='ffill').values, 'db4', mode='symmetric')
                
                # Pad coefficients to match original length
                cA, cD = coeffs
                cA_padded = np.pad(cA, (0, len(df) - len(cA)), mode='edge')
                cD_padded = np.pad(cD, (0, len(df) - len(cD)), mode='edge')
                
                df[f'wavelet_approx_{scale}'] = cA_padded
                df[f'wavelet_detail_{scale}'] = cD_padded
                
        except ImportError:
            logger.warning("PyWavelets not available, skipping wavelet features")
        
        return df

    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        
        # Bid-ask spread proxy
        df['spread_proxy'] = (df['High'] - df['Low']) / df['Close']
        
        # Price impact
        df['price_impact'] = np.abs(df['returns']) / (df['Volume'] + 1e-10)
        
        # Tick direction
        df['tick_rule'] = np.sign(df['Close'] - df['Close'].shift(1))
        
        # Kyle's lambda (price impact coefficient)
        for window in [20, 50]:
            df[f'kyle_lambda_{window}'] = df['returns'].abs().rolling(window).mean() / (df['Volume'].rolling(window).mean() + 1e-10)
        
        return df

    def add_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity-related features"""
        
        # Amihud illiquidity
        for window in [20, 50]:
            df[f'amihud_{window}'] = (np.abs(df['returns']) / (df['Volume'] + 1e-10)).rolling(window).mean()
        
        # Roll spread estimator
        df['roll_spread'] = 2 * np.sqrt(-df['returns'].rolling(20).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else 0))
        
        return df

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features"""
        
        # Volatility regime (using HMM-like approach)
        vol = df['volatility_20'].fillna(method='ffill')
        
        # Simple regime classification
        vol_quantiles = vol.quantile([0.33, 0.67])
        df['vol_regime'] = pd.cut(vol, bins=[-np.inf, vol_quantiles[0.33], vol_quantiles[0.67], np.inf], 
                                 labels=['low_vol', 'medium_vol', 'high_vol'])
        df['vol_regime_encoded'] = pd.Categorical(df['vol_regime']).codes
        
        # Trend regime
        trend_strength = df['Close'].rolling(50).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0)
        trend_quantiles = trend_strength.quantile([0.33, 0.67])
        df['trend_regime'] = pd.cut(trend_strength, bins=[-np.inf, trend_quantiles[0.33], trend_quantiles[0.67], np.inf],
                                   labels=['downtrend', 'sideways', 'uptrend'])
        df['trend_regime_encoded'] = pd.Categorical(df['trend_regime']).codes
        
        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        # Extract time components
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Market session features
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['tokyo_session'] = ((df['hour'] >= 23) | (df['hour'] < 8)).astype(int)
        df['overlap_london_ny'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        
        return df

    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical encoding of temporal features"""
        
        if 'hour' in df.columns:
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df

    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions and cross-products"""
        
        # Volume-volatility interaction
        if 'Volume' in df.columns and 'volatility_20' in df.columns:
            df['vol_volume_interaction'] = df['volatility_20'] * df['volume_ratio']
        
        # RSI-MACD interaction
        if 'rsi_14' in df.columns and 'macd' in df.columns:
            df['rsi_macd_interaction'] = df['rsi_14'] * df['macd']
        
        # Price-volume trend interaction
        if 'price_sma_20_ratio' in df.columns and 'volume_ratio' in df.columns:
            df['price_vol_trend'] = df['price_sma_20_ratio'] * df['volume_ratio']
        
        return df

    def clean_and_normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize all features"""
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill and backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Fill any remaining NaN with median
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
        
        # Feature selection based on correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_threshold = 0.95
        
        # Remove highly correlated features
        corr_matrix = df[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
        
        df = df.drop(columns=high_corr_features)
        
        logger.info(f"Removed {len(high_corr_features)} highly correlated features")
        logger.info(f"Final feature count: {len(df.columns)}")
        
        return df

    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'returns') -> pd.DataFrame:
        """Calculate feature importance using multiple methods"""
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import mutual_info_regression
        
        # Prepare data
        X = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
        y = df[target_col].shift(-1).fillna(0)  # Next period return
        
        # Remove NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        importance_df = pd.DataFrame({'feature': X_clean.columns})
        
        # Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_clean, y_clean)
        importance_df['rf_importance'] = rf.feature_importances_
        
        # Mutual information
        mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
        importance_df['mutual_info'] = mi_scores
        
        # Sort by average importance
        importance_df['avg_importance'] = (importance_df['rf_importance'] + importance_df['mutual_info']) / 2
        importance_df = importance_df.sort_values('avg_importance', ascending=False)
        
        return importance_df


# Usage example for integration
def enhance_trading_data(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Main function to enhance trading data with advanced features"""
    
    feature_engineer = AdvancedFeatureEngineer(config)
    enhanced_df = feature_engineer.engineer_all_features(df)
    
    return enhanced_df


if __name__ == "__main__":
    # Test with sample data
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='H')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(len(dates)).cumsum() + 1.1000,
        'High': np.random.randn(len(dates)).cumsum() + 1.1020,
        'Low': np.random.randn(len(dates)).cumsum() + 1.0980,
        'Close': np.random.randn(len(dates)).cumsum() + 1.1000,
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    enhanced_data = enhance_trading_data(sample_data)
    print(f"Enhanced data shape: {enhanced_data.shape}")
    print(f"New features: {len(enhanced_data.columns) - len(sample_data.columns)}")

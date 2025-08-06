# enhanced_marketintelligence.py - Production Ready Market Intelligence
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class EnhancedMarketIntelligence:
    """Enhanced Market Intelligence with comprehensive technical analysis"""
    
    def __init__(self, data_handler, user_config):
        self.data_handler = data_handler
        self.user_config = user_config
        self.feature_cache = {}
        
        logger.info("Enhanced Market Intelligence initialized")
    
    def analyze_comprehensive(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Comprehensive market analysis with advanced features"""
        try:
            if data.empty:
                logger.warning("Empty input data")
                return data
            
            df = data.copy()
            original_features = len(df.columns)
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Add statistical features
            df = self.add_statistical_features(df)
            
            # Add price action features
            df = self.add_price_action_features(df)
            
            # Final cleanup
            df = self.cleanup_features(df)
            
            new_features = len(df.columns) - original_features
            logger.info(f"✅ Analysis complete: {original_features} -> {len(df.columns)} features (+{new_features})")
            
            return df
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return data
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        try:
            # Moving Averages
            ma_periods = [5, 10, 20, 50, 100]
            for period in ma_periods:
                if len(df) >= period:
                    df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
                    df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            
            # RSI
            for period in [14, 21]:
                if len(df) >= period:
                    df[f'RSI_{period}'] = self.calculate_rsi(df['Close'], period)
            
            # MACD
            if len(df) >= 26:
                macd_line, macd_signal = self.calculate_macd(df['Close'])
                df['MACD'] = macd_line
                df['MACD_Signal'] = macd_signal
                df['MACD_Histogram'] = macd_line - macd_signal
            
            # Bollinger Bands
            for period in [20, 50]:
                if len(df) >= period:
                    bb_upper, bb_lower, bb_middle = self.calculate_bollinger_bands(df['Close'], period)
                    df[f'BB_Upper_{period}'] = bb_upper
                    df[f'BB_Lower_{period}'] = bb_lower
                    df[f'BB_Middle_{period}'] = bb_middle
                    df[f'BB_Width_{period}'] = (bb_upper - bb_lower) / bb_middle
            
            # ATR
            for period in [14, 21]:
                if len(df) >= period:
                    df[f'ATR_{period}'] = self.calculate_atr(df, period)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        try:
            # Returns and volatility
            df['Returns'] = df['Close'].pct_change()
            
            # Rolling statistics
            for window in [10, 20, 50]:
                if len(df) >= window:
                    df[f'Volatility_{window}'] = df['Returns'].rolling(window).std()
                    df[f'Skew_{window}'] = df['Returns'].rolling(window).skew()
                    df[f'Kurt_{window}'] = df['Returns'].rolling(window).kurt()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding statistical features: {e}")
            return df
    
    def add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action features"""
        try:
            # Basic price relationships
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Close_Open_Ratio'] = df['Close'] / df['Open']
            
            # Candlestick analysis
            df['Body_Size'] = abs(df['Close'] - df['Open'])
            df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
            df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
            df['Total_Range'] = df['High'] - df['Low']
            
            # Normalized ratios
            df['Body_Ratio'] = df['Body_Size'] / (df['Total_Range'] + 1e-10)
            df['Upper_Shadow_Ratio'] = df['Upper_Shadow'] / (df['Total_Range'] + 1e-10)
            df['Lower_Shadow_Ratio'] = df['Lower_Shadow'] / (df['Total_Range'] + 1e-10)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding price action features: {e}")
            return df
    
    def cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup of features"""
        try:
            # Replace infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
            return df
    
    # Helper calculation methods
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd.fillna(0), macd_signal.fillna(0)
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper.fillna(prices), lower.fillna(prices), sma.fillna(prices)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
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


# marketintelligence.py - Complete Market Intelligence
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class MarketIntelligence:
    def __init__(self, data_handler, config=None):
        self.data_handler = data_handler
        self.config = config or {}
        logger.info("MarketIntelligence initialized")
    
    def analyze_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()
            required_cols = ['Open', 'High', 'Low', 'Close']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return df
            
            # Technical indicators
            df['RSI_14'] = self._calculate_rsi(df['Close'])
            df['MACD_12_26_9'], df['MACDs_12_26_9'] = self._calculate_macd(df['Close'])
            df['BB_upper'], df['BB_lower'], df['BB_middle'] = self._calculate_bollinger_bands(df['Close'])
            df['ATR_14'] = self._calculate_atr(df)
            df['EMA_20'] = df['Close'].ewm(span=20).mean()
            df['EMA_50'] = df['Close'].ewm(span=50).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_30'] = df['Close'].rolling(window=30).mean()
            
            # Price action features
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Close_Open_Ratio'] = df['Close'] / df['Open']
            df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
            
            # Volume features
            if 'Volume' not in df.columns:
                df['Volume'] = 1000000
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Volatility features
            df['Returns'] = df['Close'].pct_change()
            df['Volatility_20'] = df['Returns'].rolling(20).std()
            df['Momentum_10'] = df['Close'].pct_change(10)
            
            # Clean NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            logger.info(f"Analysis complete: {len(df.columns)} features generated")
            return df
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return data
    
    def identify_regime(self, data: pd.DataFrame) -> str:
        try:
            if len(data) < 50:
                return 'insufficient_data'
            
            recent_data = data.tail(50)
            if 'Returns' in recent_data.columns:
                returns = recent_data['Returns'].dropna()
            else:
                returns = recent_data['Close'].pct_change().dropna()
            
            if len(returns) == 0:
                return 'unknown'
                
            trend_strength = abs(returns.mean()) / (returns.std() + 1e-10)
            volatility = returns.std()
            
            if trend_strength > 0.15 and volatility < 0.025:
                return 'strong_trending'
            elif trend_strength > 0.08:
                return 'trending'
            elif volatility > 0.03:
                return 'high_volatility'
            elif trend_strength < 0.05 and volatility < 0.015:
                return 'ranging'
            else:
                return 'normal'
        except Exception as e:
            logger.error(f"Error identifying regime: {e}")
            return 'unknown'
    
    def generate_enhanced_signal(self, symbol: str, data_dict: Dict, regime: str, sentiment_data: Dict = None) -> Optional[Dict[str, Any]]:
        try:
            execution_data = data_dict.get('EXECUTION')
            if execution_data is None or len(execution_data) < 50:
                return None
            
            recent_data = execution_data.tail(20)
            current_price = recent_data['Close'].iloc[-1]
            
            rsi = recent_data.get('RSI_14', pd.Series([50] * len(recent_data))).iloc[-1]
            macd = recent_data.get('MACD_12_26_9', pd.Series([0] * len(recent_data))).iloc[-1]
            macd_signal = recent_data.get('MACDs_12_26_9', pd.Series([0] * len(recent_data))).iloc[-1]
            
            signal_strength = 0
            direction = 'HOLD'
            
            # RSI signals
            if rsi < 30:
                signal_strength += 0.4
                direction = 'BUY'
            elif rsi > 70:
                signal_strength += 0.4
                direction = 'SELL'
            
            # MACD signals
            if macd > macd_signal:
                signal_strength += 0.3
                if direction == 'HOLD':
                    direction = 'BUY'
            elif macd < macd_signal:
                signal_strength += 0.3
                if direction == 'HOLD':
                    direction = 'SELL'
            
            # Regime adjustment
            if regime == 'trending':
                signal_strength *= 1.2
            elif regime == 'ranging':
                signal_strength *= 0.8
            
            if signal_strength < 0.5 or direction == 'HOLD':
                return None
            
            return {
                'symbol': symbol,
                'direction': direction,
                'strategy': 'EnhancedTraditional',
                'confidence': min(signal_strength, 1.0),
                'entry_price': current_price,
                'rsi': rsi,
                'macd': macd,
                'regime': regime,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd.fillna(0), macd_signal.fillna(0)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> tuple:
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band.fillna(prices), lower_band.fillna(prices), sma.fillna(prices)
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        high, low, close = data['High'], data['Low'], data['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window).mean()
        return atr.fillna(0.001)

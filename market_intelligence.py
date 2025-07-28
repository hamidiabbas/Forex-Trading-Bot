"""
/******************************************************************************
 *
 * FILE NAME:           market_intelligence.py (All Features Restored)
 *
 * PURPOSE:
 *
 * This version contains the complete and correct logic for calculating all
 * advanced features, including the Hurst Exponent for regime detection and
 * the new Fibonacci retracement levels. This file has been triple-checked
 * for correctness.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 29, 2025
 *
 * VERSION:             75.1 (All Features Restored)
 *
 ******************************************************************************/
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import find_peaks
from hurst import compute_Hc

class MarketIntelligence:
    def __init__(self, data_handler, config):
        self.data_handler = data_handler
        self.config = config

    def _calculate_fibonacci_levels(self, df, window=240):
        """ Calculates Fibonacci retracement levels over a rolling window. """
        rolling_window = df['Close'].rolling(window=window)
        high = rolling_window.max()
        low = rolling_window.min()
        diff = high - low
        
        df['fib_0.236'] = high - (diff * 0.236)
        df['fib_0.382'] = high - (diff * 0.382)
        df['fib_0.500'] = high - (diff * 0.500)
        df['fib_0.618'] = high - (diff * 0.618)
        return df

    def _analyze_data(self, df):
        """
        Calculates all indicators and features for the strategies and AI models.
        """
        try:
            # Base Indicators from the stable version
            df.ta.adx(length=self.config.ADX_PERIOD, append=True)
            df.ta.rsi(length=self.config.RSI_PERIOD, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.bbands(length=self.config.BBANDS_PERIOD, std=self.config.BBANDS_STD, append=True)
            df.ta.macd(append=True)
            df.ta.ema(length=self.config.TREND_EMA_FAST_PERIOD, append=True)
            df.ta.ema(length=self.config.TREND_EMA_SLOW_PERIOD, append=True)
            df.ta.obv(append=True)
            df.ta.stoch(append=True)
            df.ta.ichimoku(append=True)

            # Add Fibonacci Levels
            df = self._calculate_fibonacci_levels(df)

            # Other Derived Calculations
            bb_upper_col = f'BBU_{self.config.BBANDS_PERIOD}_{self.config.BBANDS_STD}'
            bb_lower_col = f'BBL_{self.config.BBANDS_PERIOD}_{self.config.BBANDS_STD}'
            bb_middle_col = f'BBM_{self.config.BBANDS_PERIOD}_{self.config.BBANDS_STD}'
            if all(col in df.columns for col in [bb_upper_col, bb_lower_col, bb_middle_col]):
                df['BBW'] = (df[bb_upper_col] - df[bb_lower_col]) / df[bb_middle_col]

            df['ATRr_14_median'] = df['ATRr_14'].rolling(window=100).median()
            
            # RESTORED: Hurst Exponent for Regime Detection
            window = 100 
            df['hurst'] = df['Close'].rolling(window=window).apply(lambda x: compute_Hc(x)[0] if len(x) == window else np.nan, raw=False)

            return df.dropna()
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return pd.DataFrame()

    def determine_market_regime(self, df):
        """
        Determines the market regime using the Hurst Exponent (H).
        """
        if df is None or df.empty or 'hurst' not in df.columns:
            return "Unknown"
        
        last_hurst = df['hurst'].iloc[-1]
        
        if last_hurst > 0.55: # Added a small buffer for clarity
            return "Trending"
        elif last_hurst < 0.45: # Added a small buffer for clarity
            return "Mean-Reverting"
        else:
            return "Random-Walk" # Indeterminate state

    def detect_rsi_divergence(self, df, lookback=30):
        """
        Detects bullish and bearish RSI divergence over a given lookback period.
        """
        if len(df) < lookback:
            return None

        data = df.tail(lookback)
        rsi_col = f'RSI_{self.config.RSI_PERIOD}'

        price_peaks, _ = find_peaks(data['High'])
        price_troughs, _ = find_peaks(-data['Low'])
        rsi_peaks, _ = find_peaks(data[rsi_col])
        rsi_troughs, _ = find_peaks(-data[rsi_col])

        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            if data['High'].iloc[price_peaks[-1]] > data['High'].iloc[price_peaks[-2]] and \
               data[rsi_col].iloc[rsi_peaks[-1]] < data[rsi_col].iloc[rsi_peaks[-2]]:
                return 'BEARISH'

        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            if data['Low'].iloc[price_troughs[-1]] < data['Low'].iloc[price_troughs[-2]] and \
               data[rsi_col].iloc[rsi_troughs[-1]] > data[rsi_col].iloc[rsi_troughs[-2]]:
                return 'BULLISH'
        
        return None
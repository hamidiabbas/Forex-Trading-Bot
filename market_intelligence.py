"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           market_intelligence.py (Hurst Exponent - Corrected)
 *
 * PURPOSE:
 *
 * This version uses the Hurst Exponent for advanced market regime
 * classification and includes a corrected _analyze_data function.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 24, 2025
 *
 * VERSION:             35.1 (Hurst Exponent - Corrected)
 *
 ******************************************************************************/
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
from hurst import compute_Hc # Import the Hurst Exponent function

class MarketIntelligence:
    def __init__(self, data_handler, config):
        self.data_handler = data_handler
        self.config = config

    def _analyze_data(self, df):
        """
        Calculates all indicators, including the Hurst Exponent.
        """
        try:
            # --- Standard Indicators ---
            df.ta.adx(length=self.config.ADX_PERIOD, append=True)
            df.ta.rsi(length=self.config.RSI_PERIOD, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.bbands(length=self.config.BBANDS_PERIOD, std=self.config.BBANDS_STD, append=True)
            df.ta.macd(append=True)
            df.ta.ema(length=self.config.TREND_EMA_FAST_PERIOD, append=True)
            df.ta.ema(length=self.config.TREND_EMA_SLOW_PERIOD, append=True)
            df.ta.obv(append=True)
            # --- Hurst Exponent Calculation ---
            window = 100 
            df['hurst'] = df['Close'].rolling(window=window).apply(lambda x: compute_Hc(x)[0], raw=False)

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

        # Classify regime based on the Hurst Exponent value
        if last_hurst > 0.5:
            return "Trending"
        elif last_hurst < 0.5:
            return "Mean-Reverting"
        else:
            return "Random-Walk"
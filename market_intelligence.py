"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           market_intelligence.py (High-Frequency Crossover)
 *
 * PURPOSE:
 *
 * This version is streamlined to calculate only the indicators needed
 * for the high-frequency crossover strategy.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             6.0 (Crossover)
 *
 ******************************************************************************/
"""
import pandas as pd
import pandas_ta as ta

class MarketIntelligence:
    def __init__(self, data_handler, config):
        self.data_handler = data_handler
        self.config = config

    def _analyze_data(self, df):
        """
        Calculates only the indicators needed for the Crossover strategy.
        """
        try:
            # --- Moving Averages for Crossover Strategy ---
            df.ta.ema(length=self.config.CROSSOVER_FAST_EMA_PERIOD, append=True)
            df.ta.ema(length=self.config.CROSSOVER_SLOW_EMA_PERIOD, append=True)
            df.ta.sma(length=self.config.CROSSOVER_TREND_FILTER_PERIOD, append=True)

            # ATR for volatility (Stop Loss calculation)
            df.ta.atr(length=14, append=True)

            return df.dropna()
            
        except Exception as e:
            print(f"Error calculating indicators in MarketIntelligence: {e}")
            return pd.DataFrame()

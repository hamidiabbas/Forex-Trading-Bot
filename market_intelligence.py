"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           market_intelligence.py (Confluence Scoring Version)
 *
 * PURPOSE:
 *
 * This module serves as the analytical engine of the trading bot. This
 * version has been significantly upgraded to calculate a wide array of
 * technical indicators required for the confluence scoring system.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             5.0 (Confluence)
 *
 ******************************************************************************/
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import find_peaks

class MarketIntelligence:
    def __init__(self, data_handler, config):
        self.data_handler = data_handler
        self.config = config

    def analyze_all_timeframes(self, symbol):
        analyzed_data = {}
        for tf_name, tf_value in self.config.TIMEFRAMES.items():
            df = self.data_handler.get_price_data(symbol, tf_value, 500)
            if df is not None:
                analyzed_data[tf_value] = self._analyze_data(df)
        return analyzed_data

    def _analyze_data(self, df):
        """
        Calculates all technical indicators needed for the confluence strategy.
        """
        try:
            # --- Moving Averages ---
            for period in self.config.EMA_PERIODS:
                df.ta.ema(length=period, append=True)
            for period in self.config.SMA_PERIODS:
                df.ta.sma(length=period, append=True)

            # --- Oscillators & Other Indicators ---
            # RSI
            df.ta.rsi(length=self.config.RSI_PERIOD, append=True)
            # Stochastic Oscillator
            df.ta.stoch(k=self.config.STOCH_K, d=self.config.STOCH_D, append=True)
            # CCI
            df.ta.cci(length=self.config.CCI_PERIOD, append=True)
            # ADX
            df.ta.adx(length=self.config.ADX_PERIOD, append=True)
            # Momentum
            df.ta.mom(length=self.config.MOMENTUM_PERIOD, append=True)
            # MACD
            df.ta.macd(fast=self.config.MACD_FAST, slow=self.config.MACD_SLOW, signal=self.config.MACD_SIGNAL, append=True)
            # ATR for volatility
            df.ta.atr(length=14, append=True)

            # Drop rows with NaN values that are created during indicator calculation
            return df.dropna()
            
        except Exception as e:
            print(f"Error calculating indicators in MarketIntelligence: {e}")
            return pd.DataFrame() # Return empty dataframe on error

    # The rest of the functions remain the same as they are not used by the new strategy
    # but are kept for potential future use.
    def identify_support_resistance(self, df, lookback=60, tolerance=0.01):
        if df.empty or len(df) < lookback:
            return {'support': [], 'resistance': []}
        highs = df['High'][-lookback:]
        lows = df['Low'][-lookback:]
        resistance_peaks, _ = find_peaks(highs, distance=5, prominence=0.001)
        support_troughs, _ = find_peaks(-lows, distance=5, prominence=0.001)
        resistance_levels = highs.iloc[resistance_peaks].tolist()
        support_levels = lows.iloc[support_troughs].tolist()
        resistance_levels = self._cluster_levels(resistance_levels, tolerance)
        support_levels = self._cluster_levels(support_levels, tolerance)
        return {'support': support_levels, 'resistance': resistance_levels}

    def _cluster_levels(self, levels, tolerance):
        if not levels: return []
        levels.sort()
        current_cluster = [levels[0]]
        clustered_levels = []
        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                current_cluster.append(level)
            else:
                clustered_levels.append(np.mean(current_cluster))
                current_cluster = [level]
        clustered_levels.append(np.mean(current_cluster))
        return clustered_levels

    def determine_market_regime(self, df):
        # This function is no longer the primary driver of the strategy,
        # but can be used for context or logging.
        if df.empty or len(df) < 100:
            return "Unknown"
        adx_col = f'ADX_{self.config.ADX_PERIOD}'
        if df[adx_col].iloc[-1] > 25:
            return "Trend"
        else:
            return "Ranging"

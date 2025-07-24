"""
/******************************************************************************
 *
 * FILE NAME:           strategy_manager.py (Regime Name Corrected)
 *
 * PURPOSE:
 *
 * This version corrects a KeyError by updating the regime names in the
 * dispatcher to match the output of the Hurst Exponent logic.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 24, 2025
 *
 * VERSION:             44.1 (Regime Name Corrected)
 *
 ******************************************************************************/
"""
import pandas as pd

class StrategyManager:
    def __init__(self, config, market_intelligence):
        self.config = config
        self.market_intelligence = market_intelligence

    def evaluate_signals(self, symbol, data_dict, i, regime):
        """
        The Master Dispatcher. Calls the correct strategy based on the regime.
        """
        df = data_dict['EXECUTION']

        if regime == "Trending":
            return self._evaluate_trend_following_strategy(symbol, df, i)
        # --- THIS IS THE FIX ---
        # Changed "Ranging" to "Mean-Reverting" to match the Hurst Exponent output
        elif regime == "Mean-Reverting":
            return self._evaluate_mean_reversion_strategy(symbol, df, i)
        elif regime == "High-Volatility":
            return self._evaluate_breakout_strategy(symbol, df, i)
        return None

    def _evaluate_trend_following_strategy(self, symbol, df, i):
        """
        Generates signals based on a confluence of EMA Crossover, MACD, and RSI.
        """
        if i < 1: return None
        fast_ema_col = f"EMA_{self.config.TREND_EMA_FAST_PERIOD}"
        slow_ema_col = f"EMA_{self.config.TREND_EMA_SLOW_PERIOD}"
        rsi_col = f"RSI_{self.config.RSI_PERIOD}"
        macd_col = 'MACD_12_26_9'
        macdsignal_col = 'MACDs_12_26_9'
        prev_fast_ema = df[fast_ema_col].iloc[i-1]
        prev_slow_ema = df[slow_ema_col].iloc[i-1]
        last_fast_ema = df[fast_ema_col].iloc[i]
        last_slow_ema = df[slow_ema_col].iloc[i]
        last_macd = df[macd_col].iloc[i]
        last_macdsignal = df[macdsignal_col].iloc[i]
        last_rsi = df[rsi_col].iloc[i]
        is_bullish_cross = prev_fast_ema <= prev_slow_ema and last_fast_ema > last_slow_ema
        if is_bullish_cross:
            has_momentum = last_macd > last_macdsignal
            has_strength = last_rsi > 50
            if has_momentum and has_strength:
                return self._generate_signal(symbol, 'BUY', 'Confluence-Trend', df, i)
        is_bearish_cross = prev_fast_ema >= prev_slow_ema and last_fast_ema < last_slow_ema
        if is_bearish_cross:
            has_momentum = last_macd < last_macdsignal
            has_strength = last_rsi < 50
            if has_momentum and has_strength:
                return self._generate_signal(symbol, 'SELL', 'Confluence-Trend', df, i)
        return None

    def _evaluate_mean_reversion_strategy(self, symbol, df, i):
        """ Bollinger Band Mean-Reversion with RSI filter. """
        if i < 1: return None
        bb_upper_col = f'BBU_{self.config.BBANDS_PERIOD}_{self.config.BBANDS_STD}'
        bb_lower_col = f'BBL_{self.config.BBANDS_PERIOD}_{self.config.BBANDS_STD}'
        rsi_col = f'RSI_{self.config.RSI_PERIOD}'
        last_high = df['High'].iloc[i]
        prev_high = df['High'].iloc[i-1]
        last_low = df['Low'].iloc[i]
        prev_low = df['Low'].iloc[i-1]
        last_rsi = df[rsi_col].iloc[i]
        upper_band = df[bb_upper_col].iloc[i]
        lower_band = df[bb_lower_col].iloc[i]
        if prev_high < df[bb_upper_col].iloc[i-1] and last_high >= upper_band:
            if last_rsi > self.config.RANGE_RSI_OVERBOUGHT:
                return self._generate_signal(symbol, 'SELL', 'Mean-Reversion', df, i)
        if prev_low > df[bb_lower_col].iloc[i-1] and last_low <= lower_band:
            if last_rsi < self.config.RANGE_RSI_OVERSOLD:
                return self._generate_signal(symbol, 'BUY', 'Mean-Reversion', df, i)
        return None
        
    def _evaluate_breakout_strategy(self, symbol, df, i):
        """ Generates signals on a price breakout confirmed by a volatility spike. """
        if i < 1: return None
        last_close = df['Close'].iloc[i]
        upper_band = df[f'BBU_{self.config.BBANDS_PERIOD}_{self.config.BBANDS_STD}'].iloc[i]
        lower_band = df[f'BBL_{self.config.BBANDS_PERIOD}_{self.config.BBANDS_STD}'].iloc[i]
        current_atr = df['ATRr_14'].iloc[i]
        median_atr = df['ATRr_14_median'].iloc[i]
        is_volatility_spike = current_atr > (median_atr * self.config.BREAKOUT_ATR_SPIKE_FACTOR)
        if not is_volatility_spike:
            return None
        if last_close > upper_band:
            return self._generate_signal(symbol, 'BUY', 'Volatility-Breakout', df, i)
        if last_close < lower_band:
            return self._generate_signal(symbol, 'SELL', 'Volatility-Breakout', df, i)
        return None

    def _generate_signal(self, symbol, direction, strategy, df, i):
        """ Generates a signal using data from the current index `i`. """
        return {
            'symbol': symbol, 'direction': direction, 'strategy': strategy,
            'entry_price': df['Close'].iloc[i],
            'atr_at_signal': df['ATRr_14'].iloc[i]
        }
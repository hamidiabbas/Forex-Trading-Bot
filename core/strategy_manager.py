import pandas as pd
import pandas_ta as ta

class StrategyManager:
    def __init__(self, config, market_intelligence):
        self.config = config
        self.market_intelligence = market_intelligence

    def evaluate_signals(self, symbol, data_dict, i, regime):
        """
        The Master Dispatcher. Dispatches strategy selection based on current regime.
        Includes divergence-reversal (RSI divergence), trend, and mean-reversion logic.
        """
        df = data_dict.get('EXECUTION')
        if df is None:
            return None

        if regime == "Mean-Reverting":
            # Check for high-probability divergence reversal first
            divergence_signal = self.market_intelligence.detect_rsi_divergence(df.iloc[:i+1])
            if divergence_signal:
                return self._evaluate_divergence_reversal_strategy(symbol, df, i, divergence_signal)
            # Otherwise, try standard mean-reversion
            return self._evaluate_mean_reversion_strategy(symbol, df, i)

        elif regime == "Trending":
            return self._evaluate_trend_following_strategy(symbol, df, i)

        # No valid regime found
        return None

    def _evaluate_divergence_reversal_strategy(self, symbol, df, i, divergence_signal):
        """
        Generates a trade signal based on the detected RSI divergence.
        """
        # For debugging, you may uncomment:
        # print(f"--- Divergence Signal Detected: {divergence_signal} ---")
        if divergence_signal == 'BULLISH':
            return self._generate_signal(symbol, 'BUY', 'Divergence-Reversal', df, i)
        elif divergence_signal == 'BEARISH':
            return self._generate_signal(symbol, 'SELL', 'Divergence-Reversal', df, i)
        return None

    def _evaluate_trend_following_strategy(self, symbol, df, i):
        """
        Generates signals based on a confluence of EMA Crossover, MACD, and RSI.
        """
        if i < self.config.TREND_EMA_SLOW_PERIOD:
            return None
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
        if is_bullish_cross and last_macd > last_macdsignal and last_rsi > 50:
            return self._generate_signal(symbol, 'BUY', 'Confluence-Trend', df, i)

        is_bearish_cross = prev_fast_ema >= prev_slow_ema and last_fast_ema < last_slow_ema
        if is_bearish_cross and last_macd < last_macdsignal and last_rsi < 50:
            return self._generate_signal(symbol, 'SELL', 'Confluence-Trend', df, i)

        return None

    def _evaluate_mean_reversion_strategy(self, symbol, df, i):
        """ Bollinger Band Mean-Reversion with RSI filter. """
        if i < 1:
            return None
        bb_upper_col = f"BBU_{self.config.BBANDS_PERIOD}_{self.config.BBANDS_STD}"
        bb_lower_col = f"BBL_{self.config.BBANDS_PERIOD}_{self.config.BBANDS_STD}"
        rsi_col = f'RSI_{self.config.RSI_PERIOD}'
        last_high = df['High'].iloc[i]
        prev_high = df['High'].iloc[i-1]
        last_low = df['Low'].iloc[i]
        prev_low = df['Low'].iloc[i-1]
        last_rsi = df[rsi_col].iloc[i]
        upper_band = df[bb_upper_col].iloc[i]
        lower_band = df[bb_lower_col].iloc[i]

        if prev_high < df[bb_upper_col].iloc[i-1] and last_high >= upper_band and last_rsi > self.config.RANGE_RSI_OVERBOUGHT:
            return self._generate_signal(symbol, 'SELL', 'Mean-Reversion', df, i)

        if prev_low > df[bb_lower_col].iloc[i-1] and last_low <= lower_band and last_rsi < self.config.RANGE_RSI_OVERSOLD:
            return self._generate_signal(symbol, 'BUY', 'Mean-Reversion', df, i)

        return None

    def _generate_signal(self, symbol, direction, strategy, df, i):
        """ Generates a signal using data from the current index `i`. """
        return {
            'symbol': symbol,
            'direction': direction,
            'strategy': strategy,
            'entry_price': df['Close'].iloc[i],
            'atr_at_signal': df['ATRr_14'].iloc[i]
        }

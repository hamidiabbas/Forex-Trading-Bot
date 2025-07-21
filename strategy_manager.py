"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           strategy_manager.py (High-Frequency Crossover)
 *
 * PURPOSE:
 *
 * This module has been redesigned to implement a simple and fast
 * Dual Moving Average Crossover strategy, designed for high-frequency trading.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             6.0 (Crossover)
 *
 ******************************************************************************/
"""

class StrategyManager:
    def __init__(self, config):
        self.config = config

    def evaluate_signals(self, symbol, analyzed_data):
        """
        Evaluates the Crossover strategy on the execution timeframe.
        """
        df = analyzed_data.get(self.config.TIMEFRAMES['EXECUTION'])

        if df is None or df.empty or len(df) < 2:
            return None

        # Define column names for clarity
        fast_ema_col = f"EMA_{self.config.CROSSOVER_FAST_EMA_PERIOD}"
        slow_ema_col = f"EMA_{self.config.CROSSOVER_SLOW_EMA_PERIOD}"
        trend_filter_col = f"SMA_{self.config.CROSSOVER_TREND_FILTER_PERIOD}"

        # Get the last two values for comparison
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        # --- Crossover Logic ---
        # Bullish crossover: Fast EMA crosses above Slow EMA
        bullish_crossover = prev_row[fast_ema_col] < prev_row[slow_ema_col] and \
                            last_row[fast_ema_col] > last_row[slow_ema_col]
        
        # Bearish crossover: Fast EMA crosses below Slow EMA
        bearish_crossover = prev_row[fast_ema_col] > prev_row[slow_ema_col] and \
                            last_row[fast_ema_col] < last_row[slow_ema_col]

        # --- Trend Filter Logic ---
        # Is the overall trend up?
        is_uptrend = last_row['Close'] > last_row[trend_filter_col]
        # Is the overall trend down?
        is_downtrend = last_row['Close'] < last_row[trend_filter_col]

        # --- Final Signal Generation ---
        if bullish_crossover and is_uptrend:
            print(f"DEBUG: High-Frequency BUY signal for {symbol}")
            return self._generate_signal(symbol, 'BUY', 'Crossover', df)
        
        elif bearish_crossover and is_downtrend:
            print(f"DEBUG: High-Frequency SELL signal for {symbol}")
            return self._generate_signal(symbol, 'SELL', 'Crossover', df)
        
        return None

    def _generate_signal(self, symbol, direction, strategy, df):
        """
        Generates a detailed signal dictionary.
        """
        return {
            'symbol': symbol,
            'direction': direction,
            'strategy': strategy,
            'entry_price': df['Close'].iloc[-1],
            'signal_candle_high': df['High'].iloc[-1],
            'signal_candle_low': df['Low'].iloc[-1],
            'atr_at_signal': df['ATRr_14'].iloc[-1]
        }
    
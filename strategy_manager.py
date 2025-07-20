"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           strategy_manager.py
 *
 * PURPOSE:
 *
 * This module acts as the "brain" of the trading bot. It contains the
 * explicit logic for each of the trading strategies. It evaluates the
 * market intelligence provided by the MarketIntelligence module and,
 * based on a strict set of rules, generates trading signals. This
 * module is responsible for identifying high-probability trading
 * opportunities that are aligned with the current market regime.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             4.0
 *
 ******************************************************************************/
"""

import pandas_ta as ta

class StrategyManager:
    def __init__(self, config):
        self.config = config

    def evaluate_signals(self, symbol, analyzed_data_mta, regime):
        if regime in ["Strong Trend", "Weak Trend"] and "TREND" in self.config.ACTIVE_STRATEGIES:
            return self._evaluate_trend_strategy(symbol, analyzed_data_mta)
        elif regime == "Ranging" and "RANGE" in self.config.ACTIVE_STRATEGIES:
            return self._evaluate_range_strategy(symbol, analyzed_data_mta)
        elif regime == "Breakout-Pending" and "BREAKOUT" in self.config.ACTIVE_STRATEGIES:
            return self._evaluate_breakout_strategy(symbol, analyzed_data_mta)
        return None

    def _evaluate_trend_strategy(self, symbol, analyzed_data_mta):
        structural_tf = self.config.TIMEFRAMES['STRUCTURAL']
        positional_tf = self.config.TIMEFRAMES['POSITIONAL']
        execution_tf = self.config.TIMEFRAMES['EXECUTION']

        structural_df = analyzed_data_mta.get(structural_tf)
        positional_df = analyzed_data_mta.get(positional_tf)
        execution_df = analyzed_data_mta.get(execution_tf)

        if any(df is None or df.empty for df in [structural_df, positional_df, execution_df]):
            return None
        
        sma_col = f'SMA_{self.config.TREND_STRATEGY_MA_PERIOD}'

        trend = 'NONE'
        if structural_df['Close'].iloc[-1] > structural_df[sma_col].iloc[-1] and \
           positional_df['Close'].iloc[-1] > positional_df[sma_col].iloc[-1]:
            trend = 'UP'
        elif structural_df['Close'].iloc[-1] < structural_df[sma_col].iloc[-1] and \
             positional_df['Close'].iloc[-1] < positional_df[sma_col].iloc[-1]:
            trend = 'DOWN'
        else:
            return None

        if trend == 'UP' and execution_df['Low'].iloc[-1] <= execution_df[sma_col].iloc[-1]:
            if (execution_df['Close'].iloc[-1] > execution_df['Open'].iloc[-1] and execution_df['Open'].iloc[-1] < execution_df['Close'].iloc[-2]) or \
               (execution_df['Close'].iloc[-1] > execution_df['Open'].iloc[-1] and (execution_df['High'].iloc[-1] - execution_df['Close'].iloc[-1]) > 2 * abs(execution_df['Close'].iloc[-1] - execution_df['Open'].iloc[-1])):
                if positional_df['ADX_14'].iloc[-1] > 20:
                    return self._generate_signal(symbol, 'BUY', 'TREND', execution_df)

        elif trend == 'DOWN' and execution_df['High'].iloc[-1] >= execution_df[sma_col].iloc[-1]:
            if (execution_df['Close'].iloc[-1] < execution_df['Open'].iloc[-1] and execution_df['Open'].iloc[-1] > execution_df['Close'].iloc[-2]) or \
               (execution_df['Close'].iloc[-1] < execution_df['Open'].iloc[-1] and (execution_df['High'].iloc[-1] - execution_df['Open'].iloc[-1]) > 2 * abs(execution_df['Close'].iloc[-1] - execution_df['Open'].iloc[-1])):
                if positional_df['ADX_14'].iloc[-1] > 20:
                    return self._generate_signal(symbol, 'SELL', 'TREND', execution_df)
        return None

    def _evaluate_range_strategy(self, symbol, analyzed_data_mta):
        execution_tf = self.config.TIMEFRAMES['EXECUTION']
        execution_df = analyzed_data_mta.get(execution_tf)
        if execution_df is None or execution_df.empty: return None

        if execution_df['Close'].iloc[-1] > execution_df['BBU_20_2.0'].iloc[-1] and execution_df['RSI_14'].iloc[-1] > self.config.RANGE_STRATEGY_RSI_OVERBOUGHT:
            if execution_df['Close'].iloc[-1] < execution_df['Open'].iloc[-1]:
                return self._generate_signal(symbol, 'SELL', 'RANGE', execution_df)
        elif execution_df['Close'].iloc[-1] < execution_df['BBL_20_2.0'].iloc[-1] and execution_df['RSI_14'].iloc[-1] < self.config.RANGE_STRATEGY_RSI_OVERSOLD:
            if execution_df['Close'].iloc[-1] > execution_df['Open'].iloc[-1]:
                return self._generate_signal(symbol, 'BUY', 'RANGE', execution_df)
        return None

    def _evaluate_breakout_strategy(self, symbol, analyzed_data_mta):
        positional_tf = self.config.TIMEFRAMES['POSITIONAL']
        execution_tf = self.config.TIMEFRAMES['EXECUTION']
        positional_df = analyzed_data_mta.get(positional_tf)
        execution_df = analyzed_data_mta.get(execution_tf)
        if any(df is None or df.empty for df in [positional_df, execution_df]): return None
        
        # Note: The breakout strategy logic was incomplete in the blueprint.
        # This is a simplified version.
        if execution_df['Close'].iloc[-1] > execution_df['BBU_20_2.0'].iloc[-1]:
            if execution_df['ATRr_14'].iloc[-1] > execution_df['ATRr_14'].rolling(20).mean().iloc[-1] * self.config.BREAKOUT_STRATEGY_ATR_SPIKE_FACTOR:
                return self._generate_signal(symbol, 'BUY', 'BREAKOUT', execution_df)
        elif execution_df['Close'].iloc[-1] < execution_df['BBL_20_2.0'].iloc[-1]:
            if execution_df['ATRr_14'].iloc[-1] > execution_df['ATRr_14'].rolling(20).mean().iloc[-1] * self.config.BREAKOUT_STRATEGY_ATR_SPIKE_FACTOR:
                return self._generate_signal(symbol, 'SELL', 'BREAKOUT', execution_df)
        return None
    
    def _generate_signal(self, symbol, direction, strategy, df):
        """
        Generates a detailed signal dictionary.
        """
        # We can now remove the debug print line
        # print("\nDEBUG: Columns available in _generate_signal:", df.columns.tolist())
        
        return {
            'symbol': symbol,
            'direction': direction,
            'strategy': strategy,
            'entry_price': df['Close'].iloc[-1],
            'signal_candle_high': df['High'].iloc[-1],
            'signal_candle_low': df['Low'].iloc[-1],
            # --- THIS IS THE CORRECTED LINE ---
            'atr_at_signal': df['ATRr_14'].iloc[-1]
        }
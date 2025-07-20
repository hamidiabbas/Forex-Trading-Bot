"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           strategy_manager.py (Confluence Scoring Version)
 *
 * PURPOSE:
 *
 * This module acts as the "brain" of the trading bot. This version has
 * been completely redesigned to use a sophisticated "Confluence Scoring"
 * system, which evaluates multiple indicators to generate a trade signal,
 * replacing the previous rigid strategy logic.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             5.0 (Confluence)
 *
 ******************************************************************************/
"""

class StrategyManager:
    def __init__(self, config):
        self.config = config

    def evaluate_signals(self, symbol, analyzed_data_mta):
        """
        Evaluates the confluence score on the execution timeframe and generates
        a signal if the score meets the minimum threshold.
        """
        execution_tf = self.config.TIMEFRAMES['EXECUTION']
        df = analyzed_data_mta.get(execution_tf)

        if df is None or df.empty or len(df) < 2:
            return None

        bullish_score, bearish_score = self._calculate_confluence_score(df)

        print(f"DEBUG - {symbol} Scores: Bullish={bullish_score}, Bearish={bearish_score}")

        if bullish_score >= self.config.MINIMUM_CONFLUENCE_SCORE:
            return self._generate_signal(symbol, 'BUY', 'Confluence', df)
        elif bearish_score >= self.config.MINIMUM_CONFLUENCE_SCORE:
            return self._generate_signal(symbol, 'SELL', 'Confluence', df)
        
        return None

    def _calculate_confluence_score(self, df):
        """
        Calculates a bullish and bearish score based on a confluence of indicators.
        Returns: (bullish_score, bearish_score)
        """
        bullish_score = 0
        bearish_score = 0
        
        # --- Rule 1: Moving Average Trend (Weight: 2) ---
        # Is the price above the long-term moving averages?
        if df['Close'].iloc[-1] > df['SMA_200'].iloc[-1] and df['Close'].iloc[-1] > df['SMA_100'].iloc[-1]:
            bullish_score += 2
        # Is the price below the long-term moving averages?
        if df['Close'].iloc[-1] < df['SMA_200'].iloc[-1] and df['Close'].iloc[-1] < df['SMA_100'].iloc[-1]:
            bearish_score += 2

        # --- Rule 2: Moving Average Crossover (Weight: 1) ---
        # Has the short-term MA crossed above the medium-term MA?
        if df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1] and df['EMA_20'].iloc[-2] < df['EMA_50'].iloc[-2]:
            bullish_score += 1
        # Has the short-term MA crossed below the medium-term MA?
        if df['EMA_20'].iloc[-1] < df['EMA_50'].iloc[-1] and df['EMA_20'].iloc[-2] > df['EMA_50'].iloc[-2]:
            bearish_score += 1

        # --- Rule 3: MACD (Weight: 1) ---
        # Is the MACD line above the signal line?
        if df['MACD_12_26_9'].iloc[-1] > df['MACDs_12_26_9'].iloc[-1]:
            bullish_score += 1
        # Is the MACD line below the signal line?
        if df['MACD_12_26_9'].iloc[-1] < df['MACDs_12_26_9'].iloc[-1]:
            bearish_score += 1
            
        # --- Rule 4: RSI (Weight: 1) ---
        # Is RSI showing upward momentum?
        if df['RSI_14'].iloc[-1] > 55:
            bullish_score += 1
        # Is RSI showing downward momentum?
        if df['RSI_14'].iloc[-1] < 45:
            bearish_score += 1
            
        # --- Rule 5: Stochastic Oscillator (Weight: 1) ---
        # Is the Stochastic in an uptrend (not overbought)?
        stoch_k = df[f'STOCHk_{self.config.STOCH_K}_{self.config.STOCH_D}_{self.config.STOCH_D}'].iloc[-1]
        if stoch_k > 20 and stoch_k < 80 and stoch_k > df[f'STOCHd_{self.config.STOCH_K}_{self.config.STOCH_D}_{self.config.STOCH_D}'].iloc[-1]:
            bullish_score += 1
        # Is the Stochastic in a downtrend (not oversold)?
        if stoch_k < 80 and stoch_k > 20 and stoch_k < df[f'STOCHd_{self.config.STOCH_K}_{self.config.STOCH_D}_{self.config.STOCH_D}'].iloc[-1]:
            bearish_score += 1

        # --- Rule 6: ADX Trend Strength (Weight: 1) ---
        # Is the market trending? (This point is added to whichever side has more points)
        if df[f'ADX_{self.config.ADX_PERIOD}'].iloc[-1] > 25:
            if bullish_score > bearish_score:
                bullish_score += 1
            elif bearish_score > bullish_score:
                bearish_score += 1

        return bullish_score, bearish_score

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

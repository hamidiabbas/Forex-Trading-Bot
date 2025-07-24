"""
/******************************************************************************
 *
 * FILE NAME:           performance_analyzer.py (Final Diagnostic Version)
 *
 * PURPOSE:
 *
 * This version includes a diagnostic print statement to confirm it is being
 * run correctly and ensures all metrics are returned as raw numbers.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 24, 2025
 *
 * VERSION:             38.2 (Final Diagnostic)
 *
 ******************************************************************************/
"""

import pandas as pd
import numpy as np
import os

class PerformanceAnalyzer:
    def __init__(self, trade_log_df=None, equity_curve_df=None, trade_log_path="backtest_results.csv", equity_curve_path="equity_curve.csv"):
        self.trade_log_df = trade_log_df
        self.equity_curve_df = equity_curve_df
        self.trade_log_path = trade_log_path
        self.equity_curve_path = equity_curve_path

    def analyze(self):
        # --- NEW DIAGNOSTIC PRINT ---
        print("\n>>> RUNNING LATEST PERFORMANCE ANALYZER (VERSION 38.2) <<<\n")

        trades_df = self.trade_log_df
        equity_df = self.equity_curve_df

        if trades_df is None and os.path.exists(self.trade_log_path):
            trades_df = pd.read_csv(self.trade_log_path)
        if equity_df is None and os.path.exists(self.equity_curve_path):
            equity_df = pd.read_csv(self.equity_curve_path, index_col='timestamp', parse_dates=True)

        if trades_df is None or trades_df.empty:
            return {'total_trades': 0}

        # --- ALL METRICS ARE CALCULATED AS RAW NUMBERS ---
        total_trades = len(trades_df)
        wins = trades_df[trades_df['profit'] > 0]
        losses = trades_df[trades_df['profit'] <= 0]
        
        total_net_profit = trades_df['profit'].sum()
        gross_profit = wins['profit'].sum()
        gross_loss = losses['profit'].sum()
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf
        expected_payoff = trades_df['profit'].mean()

        long_trades = trades_df[trades_df['direction'] == 'BUY']
        short_trades = trades_df[trades_df['direction'] == 'SELL']
        long_wins = long_trades[long_trades['profit'] > 0]
        short_wins = short_trades[short_trades['profit'] > 0]
        long_win_rate = (len(long_wins) / len(long_trades)) * 100 if not long_trades.empty else 0
        short_win_rate = (len(short_wins) / len(short_trades)) * 100 if not short_trades.empty else 0

        largest_profit = trades_df['profit'].max() if not trades_df.empty else 0
        largest_loss = trades_df['profit'].min() if not trades_df.empty else 0

        is_win = (trades_df['profit'] > 0).astype(int)
        win_streaks = is_win.groupby((is_win != is_win.shift()).cumsum()).cumsum()
        loss_streaks = (1 - is_win).groupby(((1 - is_win) != (1 - is_win).shift()).cumsum()).cumsum()
        max_consecutive_wins = win_streaks.max() if not win_streaks.empty else 0
        max_consecutive_losses = loss_streaks.max() if not loss_streaks.empty else 0

        metrics = {
            'total_trades': total_trades, 'net_profit': total_net_profit, 'profit_factor': profit_factor,
            'win_rate_percent': win_rate, 'expected_payoff': expected_payoff, 'total_long_trades': len(long_trades),
            'long_win_rate_percent': long_win_rate, 'total_short_trades': len(short_trades),
            'short_win_rate_percent': short_win_rate, 'largest_profit': largest_profit, 'largest_loss': largest_loss,
            'max_consecutive_wins': int(max_consecutive_wins), 'max_consecutive_losses': int(max_consecutive_losses),
            'max_drawdown_percent': self._calculate_max_drawdown(equity_df['equity']) * 100,
            'sharpe_ratio': self._calculate_sharpe_ratio(equity_df['equity']),
            'average_win': wins['profit'].mean() if not wins.empty else 0,
            'average_loss': losses['profit'].mean() if not losses.empty else 0,
        }
        return metrics

    def _calculate_max_drawdown(self, equity_series):
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        return abs(drawdown.min())

    def _calculate_sharpe_ratio(self, equity_series, risk_free_rate=0.0):
        daily_returns = equity_series.resample('D').last().pct_change().dropna()
        if daily_returns.std() == 0 or daily_returns.empty: return 0.0
        sharpe = (daily_returns.mean() - risk_free_rate) / daily_returns.std()
        return sharpe * np.sqrt(252)
"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           backtester.py (Final Version)
 *
 * PURPOSE:
 *
 * This module provides a robust framework for backtesting by simulating
 * the bot's full multi-timeframe analysis (MTA) logic and correctly
 * handling offline calculations for risk management.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             5.4 (Final)
 *
 ******************************************************************************/
"""

import pandas as pd
from datetime import datetime
import pytz

# Import our existing modules
import config
from data_handler import DataHandler
from market_intelligence import MarketIntelligence
from risk_manager import RiskManager
from strategy_manager import StrategyManager

class Backtester:
    def __init__(self, config_obj):
        self.config = config_obj
        self.data_handler = DataHandler(self.config)
        self.market_intelligence = MarketIntelligence(self.data_handler, self.config)
        self.risk_manager = RiskManager(self.data_handler, self.config)
        self.strategy_manager = StrategyManager(self.config)

    def run(self, symbol, start_date_str, end_date_str):
        print(f"--- Starting Advanced MTA Backtest for {symbol} from {start_date_str} to {end_date_str} ---")
        self.data_handler.connect()
        timezone = pytz.timezone("Etc/UTC")
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone)
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone)

        df_dict = {}
        for tf_name, tf_value in self.config.TIMEFRAMES.items():
            print(f"Loading {tf_value} data...")
            df = self.data_handler.get_data_by_range(symbol, tf_value, start_date, end_date)
            if df is None or df.empty:
                print(f"Could not load {tf_value} data. Aborting.")
                self.data_handler.disconnect()
                return
            df_dict[tf_value] = self.market_intelligence._analyze_data(df.copy())
        
        self.data_handler.disconnect()
        print("All data loaded and analyzed.")

        balance = 100000.0
        open_positions = []
        trade_log = []
        equity_curve = [balance]
        
        print("Step 3: Starting bar-by-bar simulation...")
        execution_df = df_dict[self.config.TIMEFRAMES['EXECUTION']]

        for i in range(100, len(execution_df)):
            current_time = execution_df.index[i]
            current_price = execution_df['Close'].iloc[i]
            
            structural_slice = df_dict[self.config.TIMEFRAMES['STRUCTURAL']][:current_time]
            positional_slice = df_dict[self.config.TIMEFRAMES['POSITIONAL']][:current_time]
            execution_slice = execution_df.iloc[0:i+1]
            
            if len(structural_slice) < 50 or len(positional_slice) < 50:
                continue

            true_mta_data = {
                self.config.TIMEFRAMES['STRUCTURAL']: structural_slice,
                self.config.TIMEFRAMES['POSITIONAL']: positional_slice,
                self.config.TIMEFRAMES['EXECUTION']: execution_slice
            }
            
            if not open_positions:
                signal = self.strategy_manager.evaluate_signals(symbol, true_mta_data)
                if signal:
                    support_resistance = self.market_intelligence.identify_support_resistance(execution_slice)
                    sl_price, tp_prices, sl_pips = self.risk_manager.calculate_sl_tp(signal, signal['atr_at_signal'], support_resistance)
                    lot_size = self.risk_manager.calculate_position_size(balance, sl_pips, symbol, current_price=current_price)
                    
                    if lot_size > 0:
                        trade = {'symbol': symbol, 'direction': signal['direction'], 'entry_price': current_price, 'sl': sl_price, 'tp': tp_prices[0], 'lot_size': lot_size, 'entry_time': current_time}
                        open_positions.append(trade)
                        print(f"{current_time}: Opened {trade['direction']} trade for {symbol} at {trade['entry_price']:.5f}")

            elif open_positions:
                trade_to_close = None
                for trade in open_positions:
                    if (trade['direction'] == 'BUY' and (current_price <= trade['sl'] or current_price >= trade['tp'])) or \
                       (trade['direction'] == 'SELL' and (current_price >= trade['sl'] or current_price <= trade['tp'])):
                        profit = ((current_price - trade['entry_price']) if trade['direction'] == 'BUY' else (trade['entry_price'] - current_price)) * trade['lot_size'] * 100000
                        balance += profit
                        trade.update({'exit_price': current_price, 'profit': profit, 'exit_time': current_time})
                        trade_log.append(trade)
                        trade_to_close = trade
                        print(f"{current_time}: Closed {trade['direction']} trade for {symbol} for a profit of ${profit:.2f}")
                        break
                if trade_to_close:
                    open_positions.remove(trade_to_close)

            equity_curve.append(balance)

        # --- THIS IS THE NEW, CORRECTED SECTION ---
        # After the loop, close any trade that was left open
        if open_positions:
            print("\nForce-closing any remaining open positions at the end of the backtest...")
            last_price = execution_df['Close'].iloc[-1]
            last_time = execution_df.index[-1]
            for trade in open_positions:
                profit = ((last_price - trade['entry_price']) if trade['direction'] == 'BUY' else (trade['entry_price'] - last_price)) * trade['lot_size'] * 100000
                balance += profit
                trade.update({'exit_price': last_price, 'profit': profit, 'exit_time': last_time})
                trade_log.append(trade)
                print(f"{last_time}: Force-closed {trade['direction']} trade for {symbol} for a profit of ${profit:.2f}")
        # ---------------------------------------------

        print("\n--- Backtest Complete ---")
        if trade_log:
            results_df = pd.DataFrame(trade_log)
            total_trades = len(results_df)
            wins = results_df[results_df['profit'] > 0]
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            total_profit = results_df['profit'].sum()
            print(f"Total Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Total Profit: ${total_profit:.2f}")
            results_df.to_csv("backtest_results.csv", index=False)
            pd.DataFrame(equity_curve, columns=['Equity']).to_csv("equity_curve.csv", index=False)
            print("Saved detailed results to 'backtest_results.csv' and 'equity_curve.csv'")
        else:
            print("No trades were executed during the backtest.")

if __name__ == '__main__':
    backtester = Backtester(config)
    backtester.run(symbol='EURUSD', start_date_str='2023-01-01', end_date_str='2024-01-01')
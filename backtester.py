"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           backtester.py
 *
 * PURPOSE:
 *
 * This module provides the framework for backtesting trading strategies
 * against historical market data. It simulates the trading process bar-by-bar,
 * allowing for the evaluation of a strategy's performance over a specific
 * period without risking real capital.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             4.0
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
    """
    A class to run a backtest of a trading strategy.
    """
    def __init__(self, config_obj):
        self.config = config_obj
        self.data_handler = DataHandler(self.config)
        self.market_intelligence = MarketIntelligence(self.data_handler, self.config)
        self.risk_manager = RiskManager(self.data_handler, self.config)
        self.strategy_manager = StrategyManager(self.config)

    def run(self, symbol, timeframe, start_date_str, end_date_str):
        """
        Runs the backtest for a given symbol and date range.
        """
        print(f"--- Starting Backtest for {symbol} on {timeframe} from {start_date_str} to {end_date_str} ---")

        # 1. Load Historical Data
        print("Step 1: Loading historical data...")
        self.data_handler.connect()
        
        # We need to make sure the start and end dates are timezone-aware for MT5
        timezone = pytz.timezone("Etc/UTC")
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone)
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone)

        # Use the new method in DataHandler to fetch data by date range
        historical_df = self.data_handler.get_data_by_range(symbol, timeframe, start_date, end_date)
        self.data_handler.disconnect()

        if historical_df is None or historical_df.empty:
            print("Could not load historical data. Aborting backtest.")
            return

        print(f"Loaded {len(historical_df)} data points.")

        # 2. Setup Simulation Environment
        print("Step 2: Setting up simulation...")
        balance = 100000.0
        equity = 100000.0
        open_positions = []
        trade_log = []
        equity_curve = []

        # 3. Main Backtesting Loop (Iterate through each bar)
        print("Step 3: Starting bar-by-bar simulation...")
        for i in range(50, len(historical_df)): # Start after 50 bars for indicators to warm up
            
            # Create a "current view" of the market data up to the current bar
            current_market_slice = historical_df.iloc[0:i]
            
            # --- This simulates the logic from main.py ---
            
            analyzed_data = self.market_intelligence._analyze_data(current_market_slice.copy())
            
            if analyzed_data is None or analyzed_data.empty:
                continue

            regime = self.market_intelligence.determine_market_regime(analyzed_data)
            
            # Use a fake MTA dictionary for the strategy manager
            fake_mta = {
                self.config.TIMEFRAMES['STRUCTURAL']: analyzed_data,
                self.config.TIMEFRAMES['POSITIONAL']: analyzed_data,
                self.config.TIMEFRAMES['EXECUTION']: analyzed_data,
            }
            
            # Only check for new signals if there are no open positions
            if not open_positions:
                signal = self.strategy_manager.evaluate_signals(symbol, fake_mta, regime)
                
                if signal:
                    support_resistance = self.market_intelligence.identify_support_resistance(analyzed_data)
                    sl_price, tp_prices, sl_pips = self.risk_manager.calculate_sl_tp(signal, signal['entry_price'], signal['atr_at_signal'], support_resistance)
                    lot_size = self.risk_manager.calculate_position_size(equity, sl_pips, symbol)
                    
                    if lot_size > 0:
                        # Simulate opening a trade
                        trade = {
                            'symbol': symbol,
                            'direction': signal['direction'],
                            'entry_price': signal['entry_price'],
                            'sl': sl_price,
                            'tp': tp_prices[0],
                            'lot_size': lot_size,
                            'entry_time': analyzed_data.index[-1]
                        }
                        open_positions.append(trade)
                        print(f"{trade['entry_time']}: Opened {trade['direction']} trade for {symbol} at {trade['entry_price']:.5f}")

            # Check to close any open positions
            elif open_positions:
                current_price = analyzed_data['Close'].iloc[-1]
                trade_to_close = None
                for trade in open_positions:
                    if trade['direction'] == 'BUY' and (current_price <= trade['sl'] or current_price >= trade['tp']):
                        profit = (current_price - trade['entry_price']) * trade['lot_size'] * 100000 # Simplified profit calc
                        balance += profit
                        equity = balance
                        trade['exit_price'] = current_price
                        trade['profit'] = profit
                        trade_log.append(trade)
                        trade_to_close = trade
                        print(f"{analyzed_data.index[-1]}: Closed {trade['direction']} trade for {symbol} for a profit of ${profit:.2f}")
                        break
                    elif trade['direction'] == 'SELL' and (current_price >= trade['sl'] or current_price <= trade['tp']):
                        profit = (trade['entry_price'] - current_price) * trade['lot_size'] * 100000 # Simplified profit calc
                        balance += profit
                        equity = balance
                        trade['exit_price'] = current_price
                        trade['profit'] = profit
                        trade_log.append(trade)
                        trade_to_close = trade
                        print(f"{analyzed_data.index[-1]}: Closed {trade['direction']} trade for {symbol} for a profit of ${profit:.2f}")
                        break
                
                if trade_to_close:
                    open_positions.remove(trade_to_close)

            equity_curve.append(equity)

        # 4. Generate Performance Report
        print("\n--- Backtest Complete ---")
        print("Step 4: Generating performance report...")
        
        results_df = pd.DataFrame(trade_log)
        if not results_df.empty:
            total_trades = len(results_df)
            wins = results_df[results_df['profit'] > 0]
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            total_profit = results_df['profit'].sum()

            print(f"Total Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Total Profit: ${total_profit:.2f}")

            # Save results to CSV for the dashboard
            results_df.to_csv("backtest_results.csv", index=False)
            pd.DataFrame(equity_curve, columns=['Equity']).to_csv("equity_curve.csv", index=False)
            print("Saved detailed results to 'backtest_results.csv' and 'equity_curve.csv'")
        else:
            print("No trades were executed during the backtest.")

if __name__ == '__main__':
    backtester = Backtester(config)
    backtester.run(
        symbol='EURUSD', 
        timeframe='H1', 
        start_date_str='2023-01-01', 
        end_date_str='2024-01-01'
    )
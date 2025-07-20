"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           backtester.py (Disciplined Crossover)
 *
 * PURPOSE:
 *
 * This version of the backtester is upgraded to simulate advanced
 * psychological discipline rules, including a maximum daily drawdown
 * and a breakeven stop loss mechanism.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             7.0 (Disciplined)
 *
 ******************************************************************************/
"""
import pandas as pd
from datetime import datetime
import pytz

import config
from data_handler import DataHandler
from market_intelligence import MarketIntelligence
from risk_manager import RiskManager
from strategy_manager import StrategyManager

class Backtester:
    def __init__(self, config_obj):
        self.config = config_obj
        # Note: We only need a few modules for this simpler backtester
        self.data_handler = DataHandler(self.config)
        self.market_intelligence = MarketIntelligence(self.data_handler, self.config)
        self.risk_manager = RiskManager(self.data_handler, self.config)
        self.strategy_manager = StrategyManager(self.config)

    def run(self, symbol, start_date_str, end_date_str):
        print(f"--- Starting Disciplined Crossover Backtest for {symbol} from {start_date_str} to {end_date_str} ---")
        
        # 1. Load and Prepare Data
        self.data_handler.connect()
        timezone = pytz.timezone("Etc/UTC")
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone)
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone)
        execution_tf = self.config.TIMEFRAMES['EXECUTION']
        df = self.data_handler.get_data_by_range(symbol, execution_tf, start_date, end_date)
        if df is None or df.empty:
            print("Could not load data. Aborting.")
            self.data_handler.disconnect()
            return
        analyzed_df = self.market_intelligence._analyze_data(df.copy())
        self.data_handler.disconnect()
        print(f"Data loaded and analyzed for {len(analyzed_df)} bars.")

        # 2. Setup Simulation
        balance = 100000.0
        equity = 100000.0
        open_positions = []
        trade_log = []
        equity_curve = [equity]
        
        # NEW: Variables for daily drawdown tracking
        max_daily_drawdown = self.config.MAX_DAILY_DRAWDOWN_PERCENT / 100.0
        current_day = None
        equity_at_day_start = equity
        trading_halted_for_the_day = False

        # 3. Main Loop
        print("Step 3: Starting bar-by-bar simulation...")
        for i in range(1, len(analyzed_df)):
            current_time = analyzed_df.index[i]
            current_price = analyzed_df['Close'].iloc[i]
            current_slice = analyzed_df.iloc[0:i+1]
            
            # --- Daily Drawdown Logic ---
            if current_time.date() != current_day:
                current_day = current_time.date()
                equity_at_day_start = equity
                trading_halted_for_the_day = False
                print(f"--- New Day: {current_day} | Starting Equity: ${equity:,.2f} ---")

            if (equity_at_day_start - equity) / equity_at_day_start >= max_daily_drawdown:
                if not trading_halted_for_the_day:
                    print(f"!!! MAX DAILY DRAWDOWN LIMIT HIT on {current_day}. Halting trading. !!!")
                    trading_halted_for_the_day = True
            
            # --- Position Management (Breakeven and Closing) ---
            if open_positions:
                trade_to_close = None
                for trade in open_positions:
                    # Breakeven Logic
                    if self.config.ENABLE_BREAKEVEN_STOP and trade['sl'] != trade['entry_price']:
                        initial_risk = abs(trade['entry_price'] - trade['initial_sl'])
                        trigger_price = trade['entry_price'] + initial_risk if trade['direction'] == 'BUY' else trade['entry_price'] - initial_risk
                        if (trade['direction'] == 'BUY' and current_price >= trigger_price) or \
                           (trade['direction'] == 'SELL' and current_price <= trigger_price):
                            trade['sl'] = trade['entry_price']
                            print(f"$$$ Moved SL to Breakeven for {trade['direction']} trade at {current_time} $$$")
                    
                    # Check for SL/TP Hit
                    if (trade['direction'] == 'BUY' and (current_price <= trade['sl'] or current_price >= trade['tp'])) or \
                       (trade['direction'] == 'SELL' and (current_price >= trade['sl'] or current_price <= trade['tp'])):
                        profit = ((current_price - trade['entry_price']) if trade['direction'] == 'BUY' else (trade['entry_price'] - current_price)) * trade['lot_size'] * 100000
                        balance += profit
                        equity += profit
                        trade.update({'exit_price': current_price, 'profit': profit, 'exit_time': current_time})
                        trade_log.append(trade)
                        trade_to_close = trade
                        print(f"{current_time}: Closed {trade['direction']} trade for a profit of ${profit:.2f}")
                        break
                if trade_to_close:
                    open_positions.remove(trade_to_close)

            # --- New Signal Evaluation ---
            if not open_positions and not trading_halted_for_the_day:
                data_for_strategy = {execution_tf: current_slice}
                signal = self.strategy_manager.evaluate_signals(symbol, data_for_strategy)
                if signal:
                    sl_price, tp_prices, sl_pips = self.risk_manager.calculate_sl_tp(signal, signal['atr_at_signal'], {})
                    lot_size = self.risk_manager.calculate_position_size(equity, sl_pips, symbol, current_price=current_price)
                    
                    if lot_size > 0:
                        trade = {
                            'symbol': symbol, 'direction': signal['direction'],
                            'entry_price': current_price, 'sl': sl_price, 'tp': tp_prices[0],
                            'lot_size': lot_size, 'entry_time': current_time,
                            'initial_sl': sl_price # Store initial SL for breakeven calc
                        }
                        open_positions.append(trade)
                        print(f"{current_time}: Opened {trade['direction']} trade at {trade['entry_price']:.5f}")

            equity_curve.append(equity)

        # 4. Final Reporting
        if open_positions:
            last_price = analyzed_df['Close'].iloc[-1]
            last_time = analyzed_df.index[-1]
            for trade in open_positions:
                profit = ((last_price - trade['entry_price']) if trade['direction'] == 'BUY' else (trade['entry_price'] - last_price)) * trade['lot_size'] * 100000
                balance += profit
                trade.update({'exit_price': last_price, 'profit': profit, 'exit_time': last_time})
                trade_log.append(trade)
        
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

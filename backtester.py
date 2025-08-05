"""
/******************************************************************************
 *
 * FILE NAME:           backtester.py (Corrected & Final)
 *
 * PURPOSE:
 *
 * This version restores the main execution block and contains the complete,
 * fully-functional backtesting and signal generation logic.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 28, 2025
 *
 * VERSION:             58.2 (Corrected & Final)
 *
 ******************************************************************************/
"""
import pandas as pd
from datetime import datetime
import pytz
import random
import MetaTrader5 as mt5

import config.configmanager as configmanager
from core.data_handler import DataHandler
from core.marketintelligence import MarketIntelligence
from core.riskmanager import RiskManager
from core.strategy_manager import StrategyManager
from utils.performance_analyzer import PerformanceAnalyzer

class Backtester:
    def __init__(self, config_obj):
        self.config = config_obj
        self.data_handler = DataHandler(self.config)
        self.market_intelligence = MarketIntelligence(self.data_handler, self.config)
        self.risk_manager = RiskManager(self.data_handler, self.config)
        self.strategy_manager = StrategyManager(self.config, self.market_intelligence)

    def _get_commission_cost(self, symbol):
        if not hasattr(self.config, 'USE_MT5_COMMISSION') or not self.config.USE_MT5_COMMISSION: 
            return getattr(self.config, 'FALLBACK_COMMISSION_PER_LOT', 0.0)
        if not self.data_handler.connect(): 
            return getattr(self.config, 'FALLBACK_COMMISSION_PER_LOT', 0.0)
        symbol_info = mt5.symbol_info(symbol)
        self.data_handler.disconnect() 
        if symbol_info is None: 
            return getattr(self.config, 'FALLBACK_COMMISSION_PER_LOT', 0.0)
        if hasattr(symbol_info, 'trade_commission_by_money'): 
            return symbol_info.trade_commission_by_money * 2
        else: 
            return 0.0

    def run(self, symbol, start_date_str, end_date_str, verbose=True):
        """ This is the original, fully functional backtesting method. """
        if verbose: print(f"--- Running Standard Backtest for {symbol} ---")
        
        self.data_handler.connect()
        timezone = pytz.timezone("Etc/UTC")
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone)
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone)
        
        execution_tf = self.config.TIMEFRAMES.get('EXECUTION', 'H1')
        df = self.data_handler.get_data_by_range(symbol, execution_tf, start_date, end_date)
        self.data_handler.disconnect()
        if df is None or df.empty: return None

        analyzed_df = self.market_intelligence._analyze_data(df.copy())
        if verbose: print(f"Data loaded and analyzed for {len(analyzed_df)} bars.")

        equity = 100000.0
        open_position, trade_log, equity_curve = None, [], [{'timestamp': analyzed_df.index[0], 'equity': equity}]
        start_index = 101
        commission_per_lot = self._get_commission_cost(symbol)
        pip_size = 0.01 if "JPY" in symbol else 0.0001
        
        for i in range(start_index, len(analyzed_df)):
            current_bar = analyzed_df.iloc[i]
            
            historical_slice = analyzed_df.iloc[0:i+1]
            regime = self.market_intelligence.determine_market_regime(historical_slice)
            data_for_strategy = { 'EXECUTION': analyzed_df }
            signal = self.strategy_manager.evaluate_signals(symbol, data_for_strategy, i, regime)
            
            if open_position:
                trade_closed = False
                close_price = 0
                current_high = current_bar['High']
                current_low = current_bar['Low']
                if open_position['direction'] == 'BUY':
                    if current_low <= open_position['sl']: trade_closed = True; close_price = open_position['sl']
                    elif current_high >= open_position['tp']: trade_closed = True; close_price = open_position['tp']
                elif open_position['direction'] == 'SELL':
                    if current_high >= open_position['sl']: trade_closed = True; close_price = open_position['sl']
                    elif current_low <= open_position['tp']: trade_closed = True; close_price = open_position['tp']
                if not trade_closed and signal and signal['direction'] != open_position['direction']:
                    if verbose: print(f"--- Reversal signal! Closing {open_position['direction']} trade. ---")
                    trade_closed = True
                    close_price = current_bar['Close']
                if trade_closed:
                    spread_cost = random.uniform(self.config.MIN_SPREAD_PIPS, self.config.MAX_SPREAD_PIPS) * pip_size
                    if open_position['direction'] == 'BUY':
                        exit_price = close_price - spread_cost
                        gross_profit = (exit_price - open_position['entry_price']) * open_position['lot_size'] * 100000
                    else:
                        exit_price = close_price + spread_cost
                        gross_profit = (open_position['entry_price'] - exit_price) * open_position['lot_size'] * 100000
                    commission_cost_for_trade = open_position['lot_size'] * commission_per_lot
                    net_profit = gross_profit - commission_cost_for_trade
                    equity += net_profit
                    open_position.update({
                        'exit_price': close_price, 'profit': net_profit,
                        'commission': commission_cost_for_trade, 'exit_time': current_bar.name
                    })
                    trade_log.append(open_position)
                    equity_curve.append({'timestamp': current_bar.name, 'equity': equity})
                    open_position = None
            
            if not open_position and signal:
                current_price = signal['entry_price']
                spread_cost = random.uniform(self.config.MIN_SPREAD_PIPS, self.config.MAX_SPREAD_PIPS) * pip_size
                slippage_cost = random.uniform(0, self.config.MAX_SLIPPAGE_PIPS) * pip_size
                if signal['direction'] == 'BUY': entry_price = current_price + spread_cost + slippage_cost
                else: entry_price = current_price - slippage_cost
                signal['entry_price'] = entry_price
                sl_price, tp_prices, _ = self.risk_manager.calculate_sl_tp(signal, signal['atr_at_signal'], {})
                lot_size = self.risk_manager.calculate_position_size(equity, symbol, signal['atr_at_signal'], current_price=entry_price)
                if lot_size > 0:
                    open_position = {
                        'symbol': symbol, 'direction': signal['direction'], 'strategy': signal['strategy'],
                        'entry_price': entry_price, 'sl': sl_price, 'tp': tp_prices[0],
                        'lot_size': lot_size, 'entry_time': current_bar.name
                    }
        
        if verbose: print("\n--- Backtest Complete ---")
        if trade_log:
            results_df = pd.DataFrame(trade_log)
            equity_curve_df = pd.DataFrame(equity_curve)
            equity_curve_df.set_index('timestamp', inplace=True)
            if verbose:
                results_df.to_csv("backtest_results.csv", index=False)
                equity_curve_df.to_csv("equity_curve.csv")
                print("Saved results to 'backtest_results.csv' and 'equity_curve.csv'")
            analyzer = PerformanceAnalyzer(trade_log_df=results_df, equity_curve_df=equity_curve_df)
            return analyzer.analyze()
        else:
            if verbose: print("No trades were executed during the backtest.")
            return None

    def generate_signals(self, symbol, start_date_str, end_date_str):
        """
        Runs in a special mode to find and save all potential trade signals.
        """
        print(f"--- Generating Primary Signals for {symbol} ---")
        
        self.data_handler.connect()
        timezone = pytz.timezone("Etc/UTC")
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone)
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone)
        
        execution_tf = self.config.TIMEFRAMES.get('EXECUTION', 'H1')
        df = self.data_handler.get_data_by_range(symbol, execution_tf, start_date, end_date)
        self.data_handler.disconnect()
        if df is None or df.empty: return None

        analyzed_df = self.market_intelligence._analyze_data(df.copy())
        print(f"Data loaded and analyzed for {len(analyzed_df)} bars.")

        all_signals = []
        start_index = 101
        
        for i in range(start_index, len(analyzed_df)):
            historical_slice = analyzed_df.iloc[0:i+1]
            regime = self.market_intelligence.determine_market_regime(historical_slice)
            data_for_strategy = { 'EXECUTION': analyzed_df }
            
            signal = self.strategy_manager.evaluate_signals(symbol, data_for_strategy, i, regime)
            
            if signal:
                signal['timestamp'] = analyzed_df.index[i]
                all_signals.append(signal)
        
        if all_signals:
            signals_df = pd.DataFrame(all_signals)
            signals_df.set_index('timestamp', inplace=True)
            signals_df.to_csv(f"primary_signals_{symbol}.csv")
            print(f"\nFound and saved {len(signals_df)} potential signals to 'primary_signals_{symbol}.csv'")
        else:
            print("\nNo potential signals found for the given period.")

if __name__ == '__main__':
    # --- THIS IS THE FIX ---
    # To run a backtest that executes trades, run this section.
    # To generate signals for the AI, comment this out and uncomment the section below.
    
    print("--- Running in Standard Backtest Mode ---")
    backtester = Backtester(configmanager)
    backtester.run(symbol='EURUSD', start_date_str='2023-01-01', end_date_str='2024-01-01')

    # --- To generate signals for the AI model ---
    # print("--- Running in Signal Generation Mode ---")
    # backtester = Backtester(config)
    # backtester.generate_signals(symbol='EURUSD', start_date_str='2019-01-01', end_date_str='2024-01-01')
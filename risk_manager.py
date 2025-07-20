"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           risk_manager.py (Backtest Compatible)
 *
 * PURPOSE:
 *
 * This module is the guardian of the bot's capital. This version is
 * updated to allow for pip value calculation using historical prices
 * during a backtest, removing the dependency on a live connection.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             4.2 (Backtest Compatible)
 *
 ******************************************************************************/
"""

import MetaTrader5 as mt5

class RiskManager:
    def __init__(self, data_handler, config):
        self.data_handler = data_handler
        self.config = config

    def get_pip_value(self, symbol, current_price=None, account_currency="USD"):
        """
        Calculates pip value. Uses live tick data if connected, otherwise uses
        the provided current_price for backtesting.
        """
        pip_size = 0.0001
        if "JPY" in symbol:
            pip_size = 0.01

        # Base currency is the first in the pair (e.g., EUR in EURUSD)
        base_currency = symbol[:3]
        # Quote currency is the second (e.g., USD in EURUSD)
        quote_currency = symbol[3:]

        # If quote is the account currency (e.g., EURUSD with USD account)
        if quote_currency == account_currency:
            return 100000 * pip_size

        # If base is the account currency (e.g., USDCAD with USD account)
        if base_currency == account_currency:
            if current_price:
                return (100000 * pip_size) / current_price
            # Try live connection if available
            elif self.data_handler.connection_status:
                tick = mt5.symbol_info_tick(symbol)
                if tick: return (100000 * pip_size) / tick.ask
            return 0.0

        # For cross-pairs (e.g., EURJPY with USD account)
        else:
            conversion_pair = f"{quote_currency}{account_currency}"
            # In a real backtester, you'd need the historical price for the conversion pair.
            # For simplicity, we'll assume a direct conversion rate of 1 for backtesting
            # if a live tick isn't available. This is an approximation.
            conversion_rate = 1.0 
            
            if self.data_handler.connection_status:
                tick = mt5.symbol_info_tick(conversion_pair)
                if tick: conversion_rate = tick.ask
            
            return 100000 * pip_size * conversion_rate

    def calculate_position_size(self, account_equity, stop_loss_pips, symbol, current_price=None):
        """
        Calculates trade volume. Now accepts current_price for backtesting.
        """
        risk_amount = account_equity * (self.config.GLOBAL_RISK_PER_TRADE_PERCENTAGE / 100)
        pip_value = self.get_pip_value(symbol, current_price=current_price)

        if pip_value is None or pip_value == 0 or stop_loss_pips == 0:
            return 0.0

        lot_size = risk_amount / (stop_loss_pips * pip_value)
        
        # In a real system, you'd check broker limits here via symbol_info
        # For backtesting, we'll just round
        return round(lot_size, 2)

    def calculate_sl_tp(self, signal, atr, support_resistance_levels):
        """
        Calculates Stop Loss and Take Profit levels.
        """
        entry_price = signal['entry_price']
        sl_distance_points = atr * self.config.ATR_SL_MULTIPLIER
        
        # JPY pairs have different pip decimal places
        pip_decimal_place = 0.0001
        if "JPY" in signal['symbol']:
            pip_decimal_place = 0.01
        
        sl_distance_pips = sl_distance_points / pip_decimal_place

        if signal['direction'] == 'BUY':
            sl_price = entry_price - sl_distance_points
            for support in sorted(support_resistance_levels.get('support', []), reverse=True):
                if sl_price > support:
                    sl_price = support - (atr * 0.1)
                    break
            tp_price = entry_price + (sl_distance_points * self.config.MINIMUM_RR_RATIO)
        else: # SELL
            sl_price = entry_price + sl_distance_points
            for resistance in sorted(support_resistance_levels.get('resistance', [])):
                if sl_price < resistance:
                    sl_price = resistance + (atr * 0.1)
                    break
            tp_price = entry_price - (sl_distance_points * self.config.MINIMUM_RR_RATIO)

        return sl_price, [tp_price], sl_distance_pips
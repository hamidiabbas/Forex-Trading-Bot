"""
/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           risk_manager.py (ATR Sizing - Debug)
 *
 * PURPOSE:
 *
 * This version includes debugging print statements to diagnose why the
 * position size is being calculated as zero.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 23, 2025
 *
 * VERSION:             27.1 (ATR Sizing - Debug)
 *
 ******************************************************************************/
"""

import MetaTrader5 as mt5

class RiskManager:
    def __init__(self, data_handler, config):
        self.data_handler = data_handler
        self.config = config

    def get_pip_value(self, symbol, current_price=None, account_currency="USD"):
        # ... (this function is unchanged)
        pip_size = 0.0001
        if "JPY" in symbol:
            pip_size = 0.01

        base_currency = symbol[:3]
        quote_currency = symbol[3:]

        if quote_currency == account_currency:
            return 100000 * pip_size

        if base_currency == account_currency:
            if current_price:
                return (100000 * pip_size) / current_price
            elif self.data_handler.connection_status:
                tick = mt5.symbol_info_tick(symbol)
                if tick: return (100000 * pip_size) / tick.ask
            return 0.0

        else:
            conversion_pair = f"{quote_currency}{account_currency}"
            conversion_rate = 1.0 
            
            if self.data_handler.connection_status:
                tick = mt5.symbol_info_tick(conversion_pair)
                if tick: conversion_rate = tick.ask
            
            return 100000 * pip_size * conversion_rate

    def calculate_position_size(self, account_equity, symbol, atr_at_signal, current_price=None):
        """
        Calculates trade volume using ATR-based sizing with debugging.
        """
        print("\n--- Calculating Position Size ---")
        
        # --- CLEANUP: The 'sl_pips' argument was removed as it's no longer used ---
        risk_amount_per_trade = account_equity * (self.config.GLOBAL_RISK_PER_TRADE_PERCENTAGE / 100)
        pip_value = self.get_pip_value(symbol, current_price=current_price)
        
        print(f"Equity: ${account_equity:,.2f}, Risk %: {self.config.GLOBAL_RISK_PER_TRADE_PERCENTAGE}%")
        print(f"Risk Amount: ${risk_amount_per_trade:,.2f}")
        print(f"ATR at Signal: {atr_at_signal:.5f}")
        print(f"Pip Value: ${pip_value:.2f}")

        if pip_value is None or pip_value == 0 or atr_at_signal == 0:
            print("--- Calculation aborted: Pip value or ATR is zero. ---")
            return 0.0

        pip_decimal_place = 0.01 if "JPY" in symbol else 0.0001
        sl_distance_in_price = atr_at_signal * self.config.ATR_SL_MULTIPLIER
        sl_pips_from_atr = sl_distance_in_price / pip_decimal_place
        
        # This is the dollar risk for trading 1 standard lot
        unit_risk = sl_pips_from_atr * pip_value
        print(f"Stop Loss Distance: {sl_pips_from_atr:.2f} pips")
        print(f"Risk per Lot (Unit Risk): ${unit_risk:,.2f}")

        if unit_risk == 0:
            print("--- Calculation aborted: Unit risk is zero. ---")
            return 0.0
        
        lot_size = risk_amount_per_trade / unit_risk
        print(f"Calculated Lot Size (pre-rounding): {lot_size:.4f}")
        print("---------------------------------")
        
        return round(lot_size, 2)

    def calculate_sl_tp(self, signal, atr, support_resistance_levels):
        # ... (this function is unchanged)
        entry_price = signal['entry_price']
        sl_distance_points = atr * self.config.ATR_SL_MULTIPLIER
        
        pip_decimal_place = 0.0001
        if "JPY" in signal['symbol']:
            pip_decimal_place = 0.01
        
        sl_distance_pips = sl_distance_points / pip_decimal_place

        if signal['direction'] == 'BUY':
            sl_price = entry_price - sl_distance_points
            tp_price = entry_price + (sl_distance_points * self.config.MINIMUM_RR_RATIO)
        else: # SELL
            sl_price = entry_price + sl_distance_points
            tp_price = entry_price - (sl_distance_points * self.config.MINIMUM_RR_RATIO)

        return sl_price, [tp_price], sl_distance_pips
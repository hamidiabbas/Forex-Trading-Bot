"""/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           execution_manager.py
 *
 * PURPOSE:
 *
 * This module is the "arm" of the bot. It is the final checkpoint
 * before a trade is sent to the broker. Its responsibilities include
 * executing trades based on the signals and risk calculations provided,
 * actively managing open positions (e.g., trailing stops, breakeven),
 * and diligently journaling every action for later performance analysis.
 * This module ensures that the bot's actions are precise, controlled,
 * and fully documented.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             4.0
 *
 ******************************************************************************/"""

import MetaTrader5 as mt5
import csv
import logging
from datetime import datetime

class ExecutionManager:
    """
    Manages trade execution and position management.
    """

    def __init__(self, data_handler, config):
        """
        Initializes the ExecutionManager.

        Args:
            data_handler: An instance of the DataHandler class.
            config: The configuration object.
        """
        self.data_handler = data_handler
        self.config = config

    def execute_trade(self, signal, lot_size, sl_price, tp_prices):
        """
        Sends a trade order to the MT5 terminal.

        Args:
            signal (dict): The trade signal.
            lot_size (float): The calculated lot size.
            sl_price (float): The stop loss price.
            tp_prices (list): A list of take profit prices.

        Returns:
            bool: True if the trade was executed successfully, False otherwise.
        """
        if not self.data_handler.connection_status:
            logging.error("Cannot execute trade, not connected to MT5.")
            return False

        trade_type = mt5.ORDER_TYPE_BUY if signal['direction'] == 'BUY' else mt5.ORDER_TYPE_SELL
        symbol = signal['symbol']

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": trade_type,
            "price": mt5.symbol_info_tick(symbol).ask if trade_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid,
            "sl": sl_price,
            "tp": tp_prices[0], # Using the first TP for now
            "deviation": 20,
            "magic": 234000,
            "comment": f"{signal['strategy']} {signal['direction']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order send failed, retcode={result.retcode}")
            return False

        logging.info(f"Trade executed: {signal['direction']} {lot_size} lots of {symbol}")
        self._journal_trade(result, signal, lot_size)
        return True

    def _journal_trade(self, result, signal, lot_size):
        """
        Records the details of a trade to a CSV file.

        Args:
            result: The result of the order_send command.
            signal (dict): The trade signal.
            lot_size (float): The trade's lot size.
        """
        with open(self.config.TRADE_JOURNAL_FILE, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now(),
                result.order,
                signal['symbol'],
                signal['direction'],
                lot_size,
                result.price,
                result.sl,
                result.tp,
                signal['strategy']
            ])

    def manage_open_positions(self):
        """
        Manages all open positions (breakeven, trailing stop).
        """
        if not self.data_handler.connection_status:
            return

        positions = mt5.positions_get()
        if positions is None:
            return

        for position in positions:
            if self.config.ENABLE_BREAKEVEN_STOP:
                self._check_and_move_to_breakeven(position)

            if self.config.ENABLE_TRAILING_STOP:
                self._trail_stop_loss(position)

    def _check_and_move_to_breakeven(self, position):
        """
        Moves the SL to breakeven if the trade is in sufficient profit.
        """
        if position.type == mt5.ORDER_TYPE_BUY:
            if position.price_current >= position.price_open + (position.price_open - position.sl) * self.config.BREAKEVEN_TRIGGER_RR:
                if position.sl != position.price_open:
                    self._modify_position(position.ticket, position.price_open, position.tp)
        else: # SELL
            if position.price_current <= position.price_open - (position.sl - position.price_open) * self.config.BREAKEVEN_TRIGGER_RR:
                if position.sl != position.price_open:
                    self._modify_position(position.ticket, position.price_open, position.tp)

    def _trail_stop_loss(self, position):
        """
        Trails the stop loss based on ATR.
        """
        # We need the ATR of the execution timeframe for this
        # This is a simplification, a more robust implementation would store the ATR at the time of the trade
        df = self.data_handler.get_price_data(position.symbol, self.config.TIMEFRAMES['EXECUTION'], 1)
        if df is None:
            return
        atr = df['ATR_14'].iloc[-1]

        if position.type == mt5.ORDER_TYPE_BUY:
            new_sl = position.price_current - (atr * self.config.TRAILING_STOP_ATR_MULTIPLIER)
            if new_sl > position.sl:
                self._modify_position(position.ticket, new_sl, position.tp)
        else: # SELL
            new_sl = position.price_current + (atr * self.config.TRAILING_STOP_ATR_MULTIPLIER)
            if new_sl < position.sl:
                self._modify_position(position.ticket, new_sl, position.tp)

    def _modify_position(self, ticket, sl, tp):
        """
        Modifies an open position's SL and TP.
        """
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Position {ticket} modified.")
        else:
            logging.error(f"Failed to modify position {ticket}. Error: {result.retcode}")
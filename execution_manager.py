"""
/******************************************************************************
 *
 * FILE NAME:           execution_manager.py (Advanced Trade Management)
 *
 * PURPOSE:
 *
 * This version includes advanced trade management features, including an
 * automatic breakeven stop and a dynamic ATR-based trailing stop. This is
 * the complete and correct version of this file.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 29, 2025
 *
 * VERSION:             76.0 (Advanced Trade Management)
 *
 ******************************************************************************/
"""
import MetaTrader5 as mt5
import time
import logging
import pandas_ta as ta

class ExecutionManager:
    def __init__(self, data_handler, config, market_intelligence):
        self.data_handler = data_handler
        self.config = config
        self.market_intelligence = market_intelligence

    def execute_trade(self, signal, lot_size, sl_price, tp_prices):
        """
        Constructs and sends a trade request to the broker.
        """
        if not self.data_handler.connection_status:
            logging.error("Cannot execute trade, not connected to MT5.")
            return None

        symbol = signal['symbol']
        direction = signal['direction']
        
        order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logging.error(f"Could not get tick for {symbol} to execute trade.")
            return None
        
        price = tick.ask if direction == 'BUY' else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_prices[0],
            "deviation": 20,
            "magic": self.config.MAGIC_NUMBER,
            "comment": f"{signal['strategy']} Signal",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order send failed for {symbol}: {result.comment}")
            return None
        
        logging.info(f"Order sent successfully for {symbol}. Ticket: {result.order}")
        return result

    def manage_open_positions(self):
        """
        Manages all open positions, applying breakeven, trailing stops,
        and divergence-based exits.
        """
        if not self.data_handler.connection_status:
            return

        positions = mt5.positions_get(magic=self.config.MAGIC_NUMBER)
        if positions is None or len(positions) == 0:
            return

        for position in positions:
            df = self.data_handler.get_price_data(position.symbol, self.config.TIMEFRAMES['EXECUTION'], 100)
            if df is not None:
                df.ta.rsi(length=self.config.RSI_PERIOD, append=True)
                divergence_signal = self.market_intelligence.detect_rsi_divergence(df)
                if (position.type == mt5.ORDER_TYPE_BUY and divergence_signal == 'BEARISH') or \
                   (position.type == mt5.ORDER_TYPE_SELL and divergence_signal == 'BULLISH'):
                    logging.info(f"Divergence detected. Closing ticket {position.ticket}.")
                    self.close_position(position)
                    continue

            self._apply_breakeven(position)
            self._apply_trailing_stop(position)
    
    def close_position(self, position):
        """ Closes a position based on its ticket number. """
        tick = mt5.symbol_info_tick(position.symbol)
        if not tick:
            logging.error(f"Could not get tick for {position.symbol} to close position.")
            return

        price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": price,
            "deviation": 20,
            "magic": position.magic,
            "comment": "Closed by Manager",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to close position {position.ticket}: {result.comment}")

    def _apply_breakeven(self, position):
        """ Moves the stop loss to the entry price if the trigger is met. """
        if not self.config.ENABLE_BREAKEVEN_STOP or position.sl == position.price_open:
            return

        sl_distance = abs(position.price_open - position.sl)
        if sl_distance == 0: return

        trigger_price_buy = position.price_open + (sl_distance * self.config.BREAKEVEN_TRIGGER_RR)
        trigger_price_sell = position.price_open - (sl_distance * self.config.BREAKEVEN_TRIGGER_RR)
        
        current_tick = mt5.symbol_info_tick(position.symbol)
        if not current_tick: return

        if position.type == mt5.ORDER_TYPE_BUY and current_tick.bid >= trigger_price_buy:
            self._modify_sl(position, position.price_open)
            logging.info(f"Breakeven triggered for ticket {position.ticket}.")
        elif position.type == mt5.ORDER_TYPE_SELL and current_tick.ask <= trigger_price_sell:
            self._modify_sl(position, position.price_open)
            logging.info(f"Breakeven triggered for ticket {position.ticket}.")

    def _apply_trailing_stop(self, position):
        """ Adjusts the stop loss dynamically based on ATR. """
        if not self.config.ENABLE_TRAILING_STOP:
            return

        if (position.type == mt5.ORDER_TYPE_BUY and position.sl < position.price_open) or \
           (position.type == mt5.ORDER_TYPE_SELL and position.sl > position.price_open):
            return

        df = self.data_handler.get_price_data(position.symbol, self.config.TIMEFRAMES['EXECUTION'], 20)
        if df is None: return
        df.ta.atr(length=14, append=True)
        current_atr = df['ATRr_14'].iloc[-1]
        
        trailing_distance = current_atr * self.config.TRAILING_STOP_ATR_MULTIPLIER
        current_tick = mt5.symbol_info_tick(position.symbol)
        if not current_tick: return

        new_sl = 0
        if position.type == mt5.ORDER_TYPE_BUY:
            potential_new_sl = current_tick.bid - trailing_distance
            if potential_new_sl > position.sl:
                new_sl = potential_new_sl
        elif position.type == mt5.ORDER_TYPE_SELL:
            potential_new_sl = current_tick.ask + trailing_distance
            if potential_new_sl < position.sl:
                new_sl = potential_new_sl

        if new_sl != 0:
            self._modify_sl(position, new_sl)
            logging.info(f"Trailing stop updated for ticket {position.ticket} to {new_sl:.5f}")

    def _modify_sl(self, position, new_sl):
        """ Helper function to send an SL modification request. """
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "sl": new_sl,
            "tp": position.tp,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to modify SL for ticket {position.ticket}: {result.comment}")
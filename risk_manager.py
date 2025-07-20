"""/******************************************************************************
 *
 * PROJECT NAME:        Algorithmic Forex Trading Bot
 *
 * FILE NAME:           risk_manager.py
 *
 * PURPOSE:
 *
 * This module is the guardian of the bot's capital. It is responsible
 * for all critical risk calculations that must be performed before any
 * trade is executed. Its primary functions include calculating the
 * dynamic value of a pip, determining the precise position size based
 * on a predefined risk percentage, and calculating intelligent Stop
 * Loss and Take Profit levels that consider both market volatility and
 * key price structures. This module enforces the bot's supreme
 * directive: capital preservation.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 20, 2025
 *
 * VERSION:             4.0
 *
 ******************************************************************************/"""

import MetaTrader5 as mt5

class RiskManager:
    """
    Manages risk for all trading operations.
    """

    def __init__(self, data_handler, config):
        """
        Initializes the RiskManager.

        Args:
            data_handler: An instance of the DataHandler class.
            config: The configuration object.
        """
        self.data_handler = data_handler
        self.config = config

    def get_pip_value(self, symbol, account_currency="USD"):
        """
        Calculates the monetary value of one pip for a standard lot.

        Args:
            symbol (str): The currency pair.
            account_currency (str): The account's base currency.

        Returns:
            float: The value of one pip in the account currency.
        """
        if not self.data_handler.connection_status:
            return 0.0

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.0

        pip_size = symbol_info.point * 10 if "JPY" not in symbol else symbol_info.point * 1000

        # Direct pairs (e.g., EURUSD, GBPUSD)
        if symbol.endswith(account_currency):
            return pip_size * 100000

        # Indirect pairs (e.g., USDCHF, USDCAD)
        elif symbol.startswith(account_currency):
            quote_currency = symbol[3:]
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return (pip_size * 100000) / tick.ask
            else:
                return 0.0
        # Cross pairs (e.g., EURJPY, GBPCHF)
        else:
            quote_currency = symbol[3:]
            conversion_pair = f"{quote_currency}{account_currency}"
            tick = mt5.symbol_info_tick(conversion_pair)
            if tick:
                return (pip_size * 100000) * tick.ask
            else:
                # Try the inverse
                conversion_pair = f"{account_currency}{quote_currency}"
                tick = mt5.symbol_info_tick(conversion_pair)
                if tick:
                    return (pip_size * 100000) / tick.ask
                else:
                    return 0.0


    def calculate_position_size(self, account_equity, stop_loss_pips, symbol):
        """
        Calculates the trade volume (lot size).

        Args:
            account_equity (float): The current account equity.
            stop_loss_pips (float): The stop loss in pips.
            symbol (str): The currency pair.

        Returns:
            float: The calculated lot size, or 0 if invalid.
        """
        risk_amount = account_equity * (self.config.GLOBAL_RISK_PER_TRADE_PERCENTAGE / 100)
        pip_value = self.get_pip_value(symbol)

        if pip_value == 0 or stop_loss_pips == 0:
            return 0.0

        lot_size = risk_amount / (stop_loss_pips * pip_value)

        # Round down to two decimal places
        lot_size = round(lot_size, 2)

        # Check against broker limits
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            if lot_size < min_lot:
                return 0.0
            if lot_size > max_lot:
                return max_lot

        return lot_size

    def calculate_sl_tp(self, signal, entry_price, atr, support_resistance_levels):
        """
        Calculates Stop Loss and Take Profit levels.

        Args:
            signal (dict): The trade signal dictionary.
            entry_price (float): The entry price of the trade.
            atr (float): The Average True Range value.
            support_resistance_levels (dict): A dictionary of support and resistance levels.

        Returns:
            tuple: A tuple containing the SL price, a list of TP prices, and the SL distance in pips.
        """
        direction = signal['direction']
        sl_distance_pips = atr * self.config.ATR_SL_MULTIPLIER

        if direction == 'BUY':
            sl_price = entry_price - (sl_distance_pips / 10000)
            # Adjust SL based on support levels
            for support in sorted(support_resistance_levels['support'], reverse=True):
                if sl_price > support:
                    sl_price = support - (atr * 0.1 / 10000) # Place SL just below support
                    break

            tp1_price = entry_price + (sl_distance_pips * self.config.MINIMUM_RR_RATIO / 10000)
            tp2_price = entry_price + (sl_distance_pips * (self.config.MINIMUM_RR_RATIO + 1) / 10000)


        else: # SELL
            sl_price = entry_price + (sl_distance_pips / 10000)
            # Adjust SL based on resistance levels
            for resistance in sorted(support_resistance_levels['resistance']):
                if sl_price < resistance:
                    sl_price = resistance + (atr * 0.1 / 10000) # Place SL just above resistance
                    break

            tp1_price = entry_price - (sl_distance_pips * self.config.MINIMUM_RR_RATIO / 10000)
            tp2_price = entry_price - (sl_distance_pips * (self.config.MINIMUM_RR_RATIO + 1) / 10000)

        # Adjust TP based on support/resistance
        if direction == 'BUY':
            for resistance in sorted(support_resistance_levels['resistance']):
                if tp1_price > resistance:
                    tp1_price = resistance
                    break
        else: # SELL
            for support in sorted(support_resistance_levels['support'], reverse=True):
                if tp1_price < support:
                    tp1_price = support
                    break


        return sl_price, [tp1_price, tp2_price], sl_distance_pips
# check_symbol_info.py (Corrected)

import MetaTrader5 as mt5
<<<<<<< HEAD
import config as config  # Your existing config file
=======
import config.configmanager as configmanager  # Your existing config file
>>>>>>> bed1ce25508448d83ab04287b1b5ea2b8271dd7a

def inspect_symbol(symbol):
    """
    Connects to MT5 and prints all attributes of a symbol's info object
    using a more robust inspection method.
    """
    # Use credentials from your config file
    if not mt5.initialize(path=configmanager.MT5_TERMINAL_PATH,
                          login=configmanager.MT5_ACCOUNT_NUMBER,
                          password=configmanager.MT5_PASSWORD,
                          server=configmanager.MT5_SERVER_NAME):
        print(f"MT5 initialization failed. Error: {mt5.last_error()}")
        return

    print(f"\nSuccessfully connected to MT5. Fetching info for {symbol}...")

    # Get information about the specified symbol
    symbol_info = mt5.symbol_info(symbol)

    if symbol_info is None:
        print(f"Failed to get info for {symbol}. Error: {mt5.last_error()}")
    else:
        print("\n--- SymbolInfo Object Attributes ---")
        # Use dir() to get all attribute names and getattr() to get their values.
        # This works on all object types.
        for attr_name in dir(symbol_info):
            # We exclude built-in "dunder" methods to keep the output clean
            if not attr_name.startswith('__'):
                try:
                    attr_value = getattr(symbol_info, attr_name)
                    print(f"{attr_name}: {attr_value}")
                except Exception as e:
                    # This helps catch any other potential issues
                    print(f"Could not retrieve value for {attr_name}: {e}")
        print("\n--- End of Attributes ---")

    # Shut down the connection to MT5
    mt5.shutdown()
    print("\nDisconnected from MT5.")

if __name__ == "__main__":
    inspect_symbol('EURUSD')
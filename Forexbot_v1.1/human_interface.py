"""
/******************************************************************************
 *
 * FILE NAME:           human_interface.py
 *
 * PURPOSE:
 *
 * This module provides a command-line interface for real-time monitoring and
 * control of the trading bot, running in a separate thread to allow for
 * non-blocking user input.
 *
 * AUTHOR:              Gemini Al
 *
 * DATE:                July 24, 2025
 *
 * VERSION:             37.0
 *
 ******************************************************************************/
"""
import threading
import time

class BotInterface(threading.Thread):
    def __init__(self, trading_bot_instance):
        super().__init__()
        self.bot = trading_bot_instance
        self.daemon = True # Allows main program to exit even if this thread is running

    def run(self):
        """ The main loop that listens for user commands. """
        print("\nHuman Interface active. Type 'help' for a list of commands.")
        while not self.bot.stop_event.is_set():
            try:
                command = input("> ").lower().strip()
                if command:
                    self.handle_command(command)
            except EOFError:
                # This can happen if the input stream is closed
                time.sleep(1)

    def handle_command(self, command):
        """ Processes the user's command. """
        parts = command.split()
        cmd = parts[0]

        if cmd == 'help':
            print("Available commands:")
            print("  status        - Show current bot status and account info.")
            print("  positions     - List all open positions.")
            print("  performance   - [Placeholder] Show performance report.")
            print("  force_close   - [Placeholder] Close a position by ticket.")
            print("  stop          - Gracefully shut down the bot.")
        
        elif cmd == 'status':
            self.bot.show_status()

        elif cmd == 'positions':
            self.bot.show_open_positions()

        elif cmd == 'stop':
            print("Shutdown signal received. The bot will stop after the current cycle.")
            self.bot.stop()

        else:
            print(f"Unknown command: '{cmd}'. Type 'help' for options.")
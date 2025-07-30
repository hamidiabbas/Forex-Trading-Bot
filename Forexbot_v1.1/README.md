Algorithmic Forex Trading Bot
Version: 4.0
Author: Gemini Al

A modular, multi-strategy, and risk-managed automated Forex trading robot developed in Python for the MetaTrader 5 platform.

⚠️ Disclaimer
Trading foreign exchange on margin carries a high level of risk and may not be suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to trade foreign exchange you should carefully consider your investment objectives, level of experience, and risk appetite. The possibility exists that you could sustain a loss of some or all of your initial investment and therefore you should not invest money that you cannot afford to lose.

This project is for educational purposes only. The author is not responsible for any financial losses incurred by using this software. Always test thoroughly on a demo account before considering a live account.

Features
Multi-Strategy Framework: Implements a core logic to switch between Trend-Following, Mean-Reversion (Range), and Breakout strategies based on market conditions.

Advanced Risk Management: Enforces strict capital preservation rules, including dynamic position sizing based on a fixed percentage risk, ATR-based Stop Loss, and a minimum Risk-to-Reward ratio.

Multi-Timeframe Analysis (MTA): Makes decisions based on a holistic view of the market, analyzing Structural (D1), Positional (H4), and Execution (H1) timeframes.

Automated Trade Management: Includes options for automatically moving Stop Loss to breakeven and enabling a dynamic trailing stop to lock in profits.

Modular & Testable Architecture: Each component of the bot (data handling, strategy logic, risk, execution) is separated into its own module for clarity, maintainability, and independent testing.

Detailed Logging & Journaling: Keeps a comprehensive trade_journal.csv for every trade and logs all major actions and errors for easy debugging and performance review.

System Architecture
The bot is built with a modular design, where each file has a specific responsibility:

main.py: The central orchestrator. Initializes all modules and runs the main trading loop.

config.py: The control panel. A centralized file for all user-adjustable settings, credentials, and strategy parameters.

data_handler.py: The bridge to the broker. Manages the connection and all data retrieval from the MetaTrader 5 platform.

market_intelligence.py: The analytical engine. Performs all technical analysis, identifies market regimes, and calculates currency strength.

strategy_manager.py: The brain of the bot. Contains the specific rule sets for each trading strategy to generate signals.

risk_manager.py: The guardian of capital. Calculates position sizes, Stop Loss, and Take Profit levels based on predefined risk rules.

execution_manager.py: The arm of the bot. Executes and manages all trade orders on the MT5 platform.

performance_analyzer.py: The accountant. Reads the trade journal to calculate and report key performance metrics.

Setup and Installation
Follow these steps to get the trading bot running on your local machine.

Step 1: Prerequisites
Python 3.8+

MetaTrader 5 Terminal installed and logged into a demo or live account.

Step 2: Clone the Project
If this were a Git repository, you would clone it. For now, create a folder for the project and place all the .py files inside it.

mkdir ForexBot
cd ForexBot
# Add all .py files to this directory

Step 3: Create and Activate a Virtual Environment
It is highly recommended to run this project in an isolated Python virtual environment.

# Create the virtual environment
python -m venv .venv

# Activate the environment
# On Windows:
.\.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

You will know the environment is active when you see (.venv) at the beginning of your terminal prompt.

Step 4: Install Dependencies
Create a file named requirements.txt in your project folder and paste the following content into it:

MetaTrader5
pandas
pandas-ta
numpy
scipy

Now, install all the required libraries by running:

pip install -r requirements.txt

Configuration
Before running the bot, you must configure it properly.

Configure MetaTrader 5:

Open your MT5 terminal.

Go to Tools -> Options -> Expert Advisors.

Check the box for "Allow algorithmic trading".

Edit config.py:

Open the config.py file.

Fill in your MT5 credentials and file path:

MT5_ACCOUNT_NUMBER = 12345678  # Your account number
MT5_PASSWORD = "YOUR_PASSWORD"  # Your account password
MT5_SERVER_NAME = "YOUR_BROKER_SERVER"  # Your broker's server
MT5_TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe" # IMPORTANT: Use the correct path

Review and adjust all other parameters, such as SYMBOLS_TO_TRADE, GLOBAL_RISK_PER_TRADE_PERCENTAGE, and ACTIVE_STRATEGIES, to fit your trading plan.

How to Run the Bot
Once the setup and configuration are complete, you can start the bot from your terminal (make sure your virtual environment is still active).

python main.py

The bot will initialize, connect to MT5, and begin its analysis and trading loop. You will see log messages in your terminal detailing its actions. To stop the bot gracefully, press Ctrl+C in the terminal.
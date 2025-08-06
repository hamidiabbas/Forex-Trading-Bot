"""
Run Enhanced Trading Bot in Full Simulation Mode
Perfect for testing all features without MT5 connection
"""

import logging
from main import EnhancedTradingBot

# Override MT5 availability for simulation
import sys
sys.modules['MetaTrader5'] = None

if __name__ == "__main__":
    print("🚀 Starting Enhanced Trading Bot in SIMULATION MODE")
    print("✅ All features active: RL, Sentiment Analysis, Kelly Sizing")
    print("✅ Using realistic fallback data generation")
    print("=" * 70)
    
    try:
        # Initialize and run bot
        bot = EnhancedTradingBot()
        bot.run_enhanced()
    except KeyboardInterrupt:
        print("\n🔄 Simulation stopped by user")
    except Exception as e:
        logging.error(f"Simulation error: {e}")

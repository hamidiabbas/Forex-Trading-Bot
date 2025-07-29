"""
Professional Trading Bot - Unicode Safe Version
Complete implementation without Unicode characters that cause encoding issues
"""
import os
import sys
import logging
import signal
import time
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import trading bot components
try:
    from config.config_manager import ConfigManager
    from core.market_intelligence import EnhancedMarketIntelligence
    from core.risk_manager import EnhancedRiskManager
    from core.execution_engine import EnhancedExecutionEngine
    from core.rl_model_manager import RLModelManager
    from utils.logger import setup_enhanced_logging
    
    IMPORTS_SUCCESSFUL = True
    
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print("Some components may not be available. Creating minimal implementation...")
    IMPORTS_SUCCESSFUL = False

class TradingBot:
    """
    Professional Trading Bot - Production Implementation
    """
    
    def __init__(self):
        # Initialize core components
        self.config = None
        self.market_intelligence = None
        self.risk_manager = None  
        self.execution_engine = None
        self.rl_model_manager = None
        
        # Bot state
        self.is_running = False
        self.start_time = None
        self.total_trades = 0
        self.successful_trades = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("Trading Bot initialized successfully")
    
    def initialize(self) -> bool:
        """Initialize all trading bot components"""
        try:
            # Load configuration
            self.config = ConfigManager('config/config.yaml')
            print(f"Config loaded with symbols: {self.config.get_trading_symbols()}")
            
            if not IMPORTS_SUCCESSFUL:
                print("Running in minimal mode due to missing components")
                return True
            
            # Initialize market intelligence
            self.market_intelligence = EnhancedMarketIntelligence(self.config.config)
            print("Market Intelligence initialized")
            
            # Initialize risk manager
            self.risk_manager = EnhancedRiskManager(self.config.config)
            print("Risk Manager initialized")
            
            # Initialize execution engine
            self.execution_engine = EnhancedExecutionEngine(
                self.config.config, self.risk_manager
            )
            print("Execution Engine initialized")
            
            # Initialize RL model manager
            self.rl_model_manager = RLModelManager(self.config)
            if self.rl_model_manager.load_model():
                print("RL Model Manager loaded successfully")
            else:
                print("RL Model Manager using rule-based approach")
            
            return True
            
        except Exception as e:
            print(f"ERROR during initialization: {e}")
            return False
    
    def start_trading(self) -> None:
        """Start the main trading loop"""
        if not self.initialize():
            print("FAILED to initialize trading bot")
            return
        
        print("Starting trading bot...")
        print("=" * 60)
        
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            if IMPORTS_SUCCESSFUL:
                self._advanced_trading_loop()
            else:
                self._basic_trading_loop()
                
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
        except Exception as e:
            print(f"CRITICAL ERROR in trading loop: {e}")
        finally:
            self._shutdown()
    
    def _advanced_trading_loop(self) -> None:
        """Advanced trading loop with all components"""
        print("Running advanced trading loop...")
        
        symbols = self.config.get_trading_symbols()
        loop_count = 0
        
        while self.is_running:
            try:
                loop_count += 1
                current_time = datetime.now()
                
                print(f"\n--- Trading Loop {loop_count} at {current_time.strftime('%H:%M:%S')} ---")
                
                for symbol in symbols:
                    if not self.is_running:
                        break
                    
                    # Get market data and analysis
                    market_data = self.market_intelligence.get_market_data(symbol)
                    
                    if not market_data:
                        print(f"No market data available for {symbol}")
                        continue
                    
                    print(f"Processing {symbol}: Price {market_data.get('current_price', 'N/A')}")
                    
                    # Generate trading signal
                    signal = self.rl_model_manager.generate_signal(symbol, market_data)
                    
                    if signal and signal.get('direction') != 'HOLD':
                        print(f"Signal: {signal['direction']} {symbol} (confidence: {signal.get('confidence', 0):.2f})")
                        
                        # Execute trade
                        trade_result = self.execution_engine.execute_trade(signal)
                        
                        if trade_result and trade_result.get('success'):
                            self.total_trades += 1
                            if trade_result.get('profit', 0) > 0:
                                self.successful_trades += 1
                            print(f"Trade executed: {trade_result.get('trade_id', 'N/A')}")
                        else:
                            print(f"Trade execution failed for {symbol}")
                    
                    # Small delay between symbols
                    time.sleep(1)
                
                # Display performance statistics
                if loop_count % 10 == 0:
                    self._display_performance()
                
                # Main loop delay
                time.sleep(30)  # 30 second intervals
                
            except Exception as e:
                print(f"ERROR in trading loop: {e}")
                time.sleep(5)
    
    def _basic_trading_loop(self) -> None:
        """Basic trading loop for minimal mode"""
        print("Running basic trading loop (minimal mode)...")
        
        symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']
        loop_count = 0
        
        while self.is_running:
            try:
                loop_count += 1
                current_time = datetime.now()
                
                print(f"\n--- Basic Loop {loop_count} at {current_time.strftime('%H:%M:%S')} ---")
                
                for symbol in symbols:
                    if not self.is_running:
                        break
                    
                    # Simulate basic market analysis
                    import random
                    price = round(random.uniform(1.0, 2.0), 5)
                    
                    print(f"Monitoring {symbol}: Simulated price {price}")
                    
                    # Simulate occasional signals
                    if random.random() < 0.1:  # 10% chance
                        direction = random.choice(['BUY', 'SELL'])
                        print(f"Simulated signal: {direction} {symbol}")
                        self.total_trades += 1
                        
                        if random.random() < 0.6:  # 60% success rate
                            self.successful_trades += 1
                
                # Display performance
                if loop_count % 5 == 0:
                    self._display_performance()
                
                time.sleep(30)
                
            except Exception as e:
                print(f"ERROR in basic loop: {e}")
                time.sleep(5)
    
    def _display_performance(self) -> None:
        """Display current performance statistics"""
        try:
            if self.start_time:
                runtime = datetime.now() - self.start_time
                runtime_str = str(runtime).split('.')[0]  # Remove microseconds
            else:
                runtime_str = "Unknown"
            
            win_rate = (self.successful_trades / max(1, self.total_trades)) * 100
            
            print("\n" + "=" * 40)
            print("PERFORMANCE SUMMARY")
            print("=" * 40)
            print(f"Runtime: {runtime_str}")
            print(f"Total Trades: {self.total_trades}")
            print(f"Successful Trades: {self.successful_trades}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Bot Status: {'RUNNING' if self.is_running else 'STOPPED'}")
            print("=" * 40)
            
        except Exception as e:
            print(f"Error displaying performance: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        self.is_running = False
    
    def _shutdown(self) -> None:
        """Graceful shutdown of trading bot"""
        print("\nShutting down trading bot...")
        self.is_running = False
        
        try:
            # Shutdown components
            if self.rl_model_manager:
                self.rl_model_manager.shutdown()
                print("RL Model Manager shutdown")
            
            if self.execution_engine:
                self.execution_engine.shutdown()
                print("Execution Engine shutdown")
            
            if self.market_intelligence:
                print("Market Intelligence shutdown")
            
            # Final performance report
            self._display_performance()
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
        
        print("Trading bot shutdown completed")

def main():
    """Main entry point"""
    print("=" * 60)
    print("PROFESSIONAL TRADING BOT v2.0")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create and start trading bot
    bot = TradingBot()
    bot.start_trading()

if __name__ == "__main__":
    main()

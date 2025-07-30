# core/enhanced_trading_engine.py
import asyncio
import threading
import time
import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sqlite3
from contextlib import contextmanager
import signal
import sys

# Import our previously built components
from .technical_analysis import ComprehensiveTechnicalAnalyzer, TechnicalSignal, TrendDirection
from .dynamic_position_manager import CompleteDynamicPositionManager, DynamicPosition, PositionStatus
from .symbol_manager import EnhancedSymbolManager
from .risk_manager import AdvancedRiskManager
from .data_manager import RealTimeDataManager
from .broker_interface import BrokerInterface
from .news_manager import EconomicCalendarManager

class SystemStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class TradingMode(Enum):
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    ANALYSIS_ONLY = "analysis_only"

class SignalQuality(Enum):
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    POOR = 2
    VERY_POOR = 1

@dataclass
class TradingSignal:
    symbol: str
    direction: str  # 'long' or 'short'
    signal_quality: SignalQuality
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    timeframe: str
    timestamp: datetime
    technical_signal: TechnicalSignal
    risk_assessment: Dict
    market_conditions: Dict
    news_impact: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'signal_quality': self.signal_quality.name,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat(),
            'risk_assessment': self.risk_assessment,
            'market_conditions': self.market_conditions,
            'news_impact': self.news_impact
        }

@dataclass
class SystemMetrics:
    total_positions_opened: int = 0
    total_positions_closed: int = 0
    active_positions: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    signals_generated: int = 0
    signals_executed: int = 0
    execution_rate: float = 0.0
    average_hold_time: float = 0.0
    risk_events: int = 0
    emergency_exits: int = 0
    system_uptime: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class EnhancedTradingEngine:
    """
    Complete Enhanced Trading Engine that orchestrates all components
    """
    
    def __init__(self, config_path: str = "config/trading_config.yaml"):
        # Core configuration
        self.config = self._load_configuration(config_path)
        self.trading_mode = TradingMode(self.config.get('trading_mode', 'paper'))
        
        # System state
        self.status = SystemStatus.STOPPED
        self.start_time = None
        self.shutdown_requested = False
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize database
        self.db_path = self.config.get('database_path', 'data/trading_system.db')
        self._initialize_database()
        
        # Initialize core components
        self._initialize_components()
        
        # Threading and async
        self.main_loop_thread = None
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.loop = None
        
        # System metrics and monitoring
        self.metrics = SystemMetrics()
        self.performance_history = []
        self.active_signals = {}
        self.signal_history = []
        
        # Risk and portfolio management
        self.portfolio_value = self.config.get('initial_balance', 10000.0)
        self.max_daily_loss = self.config.get('max_daily_loss', 500.0)
        self.max_total_positions = self.config.get('max_total_positions', 5)
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
        
        self.logger.info(f"Enhanced Trading Engine initialized in {self.trading_mode.value} mode")
    
    def _load_configuration(self, config_path: str) -> Dict:
        """Load system configuration from YAML file"""
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Validate critical configuration
            required_keys = ['broker_config', 'risk_management', 'symbols', 'timeframes']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required configuration key: {key}")
            
            return config
            
        except FileNotFoundError:
            # Create default configuration
            default_config = self._create_default_config()
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as file:
                yaml.dump(default_config, file, default_flow_style=False)
            
            return default_config
        
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def _create_default_config(self) -> Dict:
        """Create default configuration"""
        
        return {
            'trading_mode': 'paper',
            'initial_balance': 10000.0,
            'max_daily_loss': 500.0,
            'max_total_positions': 5,
            'analysis_interval_seconds': 5,
            'database_path': 'data/trading_system.db',
            
            'broker_config': {
                'broker_type': 'mt5',
                'server': 'demo_server',
                'login': 'demo_login',
                'password': 'demo_password',
                'timeout': 10000
            },
            
            'risk_management': {
                'max_risk_per_trade': 0.02,
                'max_portfolio_risk': 0.06,
                'correlation_limit': 0.7,
                'news_risk_filter': True,
                'emergency_exit_threshold': -0.05
            },
            
            'symbols': {
                'active_pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD'],
                'major_pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'],
                'minor_pairs': ['EURJPY', 'EURGBP', 'EURCHF', 'GBPJPY', 'GBPCHF', 'AUDJPY'],
                'exotic_pairs': ['EURTRY', 'USDTRY', 'USDZAR', 'USDMXN']
            },
            
            'timeframes': {
                'analysis': ['M15', 'H1', 'H4', 'D1'],
                'primary': 'H1',
                'confirmation': 'H4',
                'execution': 'M15'
            },
            
            'technical_analysis': {
                'min_confidence': 0.6,
                'trend_strength_threshold': 3,
                'reversal_probability_threshold': 0.7,
                'volume_confirmation_required': False
            },
            
            'position_management': {
                'scaling_enabled': True,
                'max_scale_ins': 2,
                'profit_taking_levels': [1.5, 3.0, 5.0, 8.0],
                'trailing_stop_enabled': True,
                'breakeven_threshold': 1.0
            },
            
            'news_integration': {
                'enabled': True,
                'high_impact_filter': True,
                'news_buffer_minutes': 30,
                'sentiment_analysis': False
            },
            
            'logging': {
                'level': 'INFO',
                'file_path': 'logs/trading_system.log',
                'max_file_size': '10MB',
                'backup_count': 5
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system"""
        
        logger = logging.getLogger('EnhancedTradingEngine')
        logger.setLevel(getattr(logging, self.config.get('logging', {}).get('level', 'INFO')))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = self.config.get('logging', {}).get('file_path', 'logs/trading_system.log')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        # Trading-specific handler for trades and signals
        trade_log_file = log_file.replace('.log', '_trades.log')
        trade_handler = logging.FileHandler(trade_log_file)
        trade_handler.setLevel(logging.INFO)
        trade_handler.setFormatter(detailed_formatter)
        
        trade_logger = logging.getLogger('TradingActivity')
        trade_logger.addHandler(trade_handler)
        trade_logger.setLevel(logging.INFO)
        
        return logger
    
    def _initialize_database(self):
        """Initialize SQLite database for system data"""
        
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create tables
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS positions (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        size REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        open_time TEXT NOT NULL,
                        close_time TEXT,
                        pnl REAL,
                        status TEXT NOT NULL,
                        metadata TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        stop_loss REAL NOT NULL,
                        take_profit REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        executed BOOLEAN DEFAULT 0,
                        signal_data TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        timestamp TEXT PRIMARY KEY,
                        total_pnl REAL,
                        daily_pnl REAL,
                        active_positions INTEGER,
                        win_rate REAL,
                        profit_factor REAL,
                        max_drawdown REAL,
                        metrics_data TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS system_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        description TEXT,
                        data TEXT
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
                    CREATE INDEX IF NOT EXISTS idx_positions_open_time ON positions(open_time);
                    CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events(timestamp);
                """)
                
                conn.commit()
                
        except Exception as e:
            raise RuntimeError(f"Error initializing database: {e}")
    
    def _initialize_components(self):
        """Initialize all system components"""
        
        try:
            # Symbol Manager
            self.symbol_manager = EnhancedSymbolManager(self.config['symbols'])
            
            # Risk Manager
            self.risk_manager = AdvancedRiskManager(
                config=self.config['risk_management'],
                initial_balance=self.portfolio_value
            )
            
            # Data Manager
            self.data_manager = RealTimeDataManager(
                broker_config=self.config['broker_config'],
                symbols=self.symbol_manager.get_active_symbols(),
                timeframes=self.config['timeframes']['analysis']
            )
            
            # Broker Interface
            self.broker = BrokerInterface(
                config=self.config['broker_config'],
                trading_mode=self.trading_mode
            )
            
            # Technical Analyzer
            self.technical_analyzer = ComprehensiveTechnicalAnalyzer(
                logger=self.logger.getChild('TechnicalAnalysis')
            )
            
            # News Manager
            if self.config.get('news_integration', {}).get('enabled', False):
                self.news_manager = EconomicCalendarManager(
                    config=self.config['news_integration']
                )
            else:
                self.news_manager = None
            
            # Position Manager
            position_config = {
                'analysis_interval_seconds': self.config.get('analysis_interval_seconds', 5),
                'max_positions': self.max_total_positions,
                'emergency_exit_threshold': self.config['risk_management']['emergency_exit_threshold']
            }
            
            self.position_manager = CompleteDynamicPositionManager(
                trading_engine=self,
                technical_analyzer=self.technical_analyzer,
                risk_manager=self.risk_manager,
                config=position_config
            )
            
            self.logger.info("All system components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise RuntimeError(f"Component initialization failed: {e}")
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
            self.stop_system()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_system(self):
        """Start the complete trading system"""
        
        if self.status != SystemStatus.STOPPED:
            self.logger.warning(f"Cannot start system: Current status is {self.status.value}")
            return False
        
        try:
            self.logger.info("Starting Enhanced Trading Engine...")
            self.status = SystemStatus.STARTING
            self.start_time = datetime.now()
            
            # Initialize broker connection
            if not self.broker.connect():
                raise RuntimeError("Failed to connect to broker")
            
            # Start data feeds
            if not self.data_manager.start_data_feeds():
                raise RuntimeError("Failed to start data feeds")
            
            # Initialize portfolio state
            self._initialize_portfolio_state()
            
            # Start position manager
            self.position_manager.start_dynamic_monitoring()
            
            # Start main trading loop
            self.main_loop_thread = threading.Thread(
                target=self._main_trading_loop,
                daemon=False
            )
            self.main_loop_thread.start()
            
            # Start performance monitoring
            self._start_performance_monitoring()
            
            self.status = SystemStatus.RUNNING
            self.logger.info("Enhanced Trading Engine started successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting system: {e}")
            self.status = SystemStatus.ERROR
            return False
    
    def stop_system(self):
        """Stop the trading system gracefully"""
        
        if self.status == SystemStatus.STOPPED:
            self.logger.info("System is already stopped")
            return
        
        try:
            self.logger.info("Stopping Enhanced Trading Engine...")
            self.status = SystemStatus.STOPPING
            
            # Stop new signal generation
            self.shutdown_requested = True
            
            # Stop position manager
            if hasattr(self, 'position_manager'):
                self.position_manager.stop_dynamic_monitoring()
            
            # Handle open positions
            self._handle_system_shutdown()
            
            # Stop data feeds
            if hasattr(self, 'data_manager'):
                self.data_manager.stop_data_feeds()
            
            # Disconnect from broker
            if hasattr(self, 'broker'):
                self.broker.disconnect()
            
            # Wait for main loop to finish
            if self.main_loop_thread and self.main_loop_thread.is_alive():
                self.main_loop_thread.join(timeout=10)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Save final state
            self._save_system_state()
            
            self.status = SystemStatus.STOPPED
            self.logger.info("Enhanced Trading Engine stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
            self.status = SystemStatus.ERROR
    
    def _main_trading_loop(self):
        """Main trading loop that coordinates all activities"""
        
        self.logger.info("Starting main trading loop")
        
        loop_count = 0
        last_analysis_time = datetime.now()
        analysis_interval = timedelta(seconds=self.config.get('analysis_interval_seconds', 30))
        
        while not self.shutdown_requested:
            try:
                loop_start_time = datetime.now()
                loop_count += 1
                
                # Update system metrics
                self._update_system_metrics()
                
                # Check system health
                if not self._check_system_health():
                    self.logger.error("System health check failed, entering maintenance mode")
                    self.status = SystemStatus.MAINTENANCE
                    time.sleep(60)  # Wait before retry
                    continue
                
                # Check daily loss limits
                if self._check_daily_loss_limit():
                    self.logger.warning("Daily loss limit reached, pausing trading")
                    self.status = SystemStatus.PAUSED
                    time.sleep(300)  # Wait 5 minutes
                    continue
                
                # Periodic comprehensive analysis
                if datetime.now() - last_analysis_time >= analysis_interval:
                    self._perform_comprehensive_analysis()
                    last_analysis_time = datetime.now()
                
                # Generate and evaluate signals
                new_signals = self._generate_trading_signals()
                
                # Execute qualified signals
                if new_signals:
                    self._execute_trading_signals(new_signals)
                
                # Portfolio rebalancing check
                if loop_count % 60 == 0:  # Every ~5 minutes (assuming 5-sec intervals)
                    self._check_portfolio_rebalancing()
                
                # Performance logging
                if loop_count % 360 == 0:  # Every ~30 minutes
                    self._log_performance_summary()
                
                # Calculate sleep time to maintain consistent intervals
                loop_duration = (datetime.now() - loop_start_time).total_seconds()
                sleep_time = max(0, self.config.get('analysis_interval_seconds', 5) - loop_duration)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.logger.error(f"Error in main trading loop: {e}")
                time.sleep(10)  # Back-off on error
        
        self.logger.info("Main trading loop stopped")
    
    def _generate_trading_signals(self) -> List[TradingSignal]:
        """Generate trading signals for all active symbols"""
        
        signals = []
        active_symbols = self.symbol_manager.get_active_symbols()
        
        # Use thread pool for parallel analysis
        future_to_symbol = {}
        
        for symbol in active_symbols:
            # Skip if we already have maximum positions for this symbol
            if self._get_symbol_position_count(symbol) >= self.config.get('max_positions_per_symbol', 2):
                continue
            
            future = self.executor.submit(self._analyze_symbol_for_signal, symbol)
            future_to_symbol[future] = symbol
        
        # Collect results
        for future in as_completed(future_to_symbol, timeout=30):
            symbol = future_to_symbol[future]
            try:
                signal = future.result()
                if signal:
                    signals.append(signal)
                    self.metrics.signals_generated += 1
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol} for signals: {e}")
        
        return signals
    
    def _analyze_symbol_for_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Analyze a single symbol and generate trading signal if conditions are met"""
        
        try:
            # Get price data for multiple timeframes
            price_data = {}
            timeframes = self.config['timeframes']['analysis']
            
            for tf in timeframes:
                data = self.data_manager.get_recent_data(symbol, tf, 200)
                if data is not None and len(data) >= 100:
                    price_data[tf] = data
            
            if not price_data:
                return None
            
            # Primary analysis on main timeframe
            primary_tf = self.config['timeframes']['primary']
            if primary_tf not in price_data:
                return None
            
            # Perform technical analysis
            technical_analysis = self.technical_analyzer.analyze_symbol(
                symbol=symbol,
                timeframe=primary_tf,
                price_data=price_data[primary_tf]
            )
            
            technical_signal = technical_analysis['trading_signal']
            
            # Check minimum confidence threshold
            min_confidence = self.config['technical_analysis']['min_confidence']
            if technical_signal.confidence < min_confidence:
                return None
            
            # Multi-timeframe confirmation
            if not self._check_multi_timeframe_confirmation(symbol, price_data, technical_signal):
                return None
            
            # News risk assessment
            news_impact = {}
            if self.news_manager:
                news_risk, news_message = self.news_manager.check_news_risk(symbol)
                if news_risk:
                    self.logger.info(f"Signal blocked for {symbol}: {news_message}")
                    return None
                news_impact = {'risk_level': 'low', 'message': 'No high-impact news detected'}
            
            # Risk assessment
            risk_assessment = self.risk_manager.assess_trade_risk(
                symbol=symbol,
                direction=technical_signal.direction.name.lower(),
                entry_price=technical_signal.entry_price,
                stop_loss=technical_signal.stop_loss,
                take_profit=technical_signal.take_profit
            )
            
            if not risk_assessment['approved']:
                self.logger.info(f"Signal rejected for {symbol}: {risk_assessment['reason']}")
                return None
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                entry_price=technical_signal.entry_price,
                stop_loss=technical_signal.stop_loss,
                account_balance=self.portfolio_value
            )
            
            if position_size <= 0:
                return None
            
            # Determine signal quality
            signal_quality = self._assess_signal_quality(technical_analysis, risk_assessment)
            
            # Market conditions assessment
            market_conditions = {
                'volatility_regime': technical_analysis['indicators']['volatility']['volatility_regime'],
                'trend_strength': technical_analysis['signals']['trend']['strength'].value,
                'market_session': self.symbol_manager.get_current_session(),
                'correlation_risk': self.risk_manager.check_correlation_risk(symbol)
            }
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                direction='long' if technical_signal.direction.value > 0 else 'short',
                signal_quality=signal_quality,
                confidence=technical_signal.confidence,
                entry_price=technical_signal.entry_price,
                stop_loss=technical_signal.stop_loss,
                take_profit=technical_signal.take_profit,
                position_size=position_size,
                timeframe=primary_tf,
                timestamp=datetime.now(),
                technical_signal=technical_signal,
                risk_assessment=risk_assessment,
                market_conditions=market_conditions,
                news_impact=news_impact
            )
            
            # Log signal generation
            trade_logger = logging.getLogger('TradingActivity')
            trade_logger.info(f"Generated signal: {symbol} {signal.direction} @ {signal.entry_price:.5f} "
                            f"(Confidence: {signal.confidence:.2f}, Quality: {signal_quality.name})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol} for signals: {e}")
            return None
    
    def _check_multi_timeframe_confirmation(self, symbol: str, price_data: Dict, 
                                          primary_signal: TechnicalSignal) -> bool:
        """Check for multi-timeframe confirmation of the primary signal"""
        
        try:
            confirmation_tf = self.config['timeframes']['confirmation']
            if confirmation_tf not in price_data:
                return True  # Skip confirmation if data not available
            
            # Analyze confirmation timeframe
            confirmation_analysis = self.technical_analyzer.analyze_symbol(
                symbol=symbol,
                timeframe=confirmation_tf,
                price_data=price_data[confirmation_tf]
            )
            
            confirmation_signal = confirmation_analysis['trading_signal']
            
            # Check if directions align
            primary_direction = primary_signal.direction.value
            confirmation_direction = confirmation_signal.direction.value
            
            # Both should be in same direction (both positive or both negative)
            if (primary_direction > 0 and confirmation_direction > 0) or \
               (primary_direction < 0 and confirmation_direction < 0):
                return True
            
            # Allow neutral confirmation if primary signal is strong
            if confirmation_direction == 0 and abs(primary_direction) >= 1:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe confirmation for {symbol}: {e}")
            return True  # Don't block signal on error
    
    def _assess_signal_quality(self, technical_analysis: Dict, risk_assessment: Dict) -> SignalQuality:
        """Assess the overall quality of a trading signal"""
        
        try:
            quality_score = 0
            
            # Technical analysis strength (0-2 points)
            trend_strength = technical_analysis['signals']['trend']['strength'].value
            if trend_strength >= 4:
                quality_score += 2
            elif trend_strength >= 3:
                quality_score += 1
            
            # Signal confidence (0-2 points)
            confidence = technical_analysis['trading_signal'].confidence
            if confidence >= 0.8:
                quality_score += 2
            elif confidence >= 0.6:
                quality_score += 1
            
            # Risk-reward ratio (0-1 point)
            rr_ratio = risk_assessment.get('risk_reward_ratio', 0)
            if rr_ratio >= 2.0:
                quality_score += 1
            
            # Map score to quality enum
            if quality_score >= 4:
                return SignalQuality.EXCELLENT
            elif quality_score >= 3:
                return SignalQuality.GOOD
            elif quality_score >= 2:
                return SignalQuality.FAIR
            elif quality_score >= 1:
                return SignalQuality.POOR
            else:
                return SignalQuality.VERY_POOR
                
        except Exception as e:
            self.logger.error(f"Error assessing signal quality: {e}")
            return SignalQuality.FAIR
    
    def _execute_trading_signals(self, signals: List[TradingSignal]):
        """Execute qualified trading signals"""
        
        # Sort signals by quality and confidence
        qualified_signals = [s for s in signals if s.signal_quality.value >= 3]  # Fair or better
        qualified_signals.sort(key=lambda x: (x.signal_quality.value, x.confidence), reverse=True)
        
        # Limit concurrent executions
        max_concurrent_trades = self.config.get('max_concurrent_trades', 3)
        signals_to_execute = qualified_signals[:max_concurrent_trades]
        
        for signal in signals_to_execute:
            try:
                success = self._execute_single_signal(signal)
                if success:
                    self.metrics.signals_executed += 1
                    self.active_signals[signal.symbol] = signal
                    
                    # Save signal to database
                    self._save_signal_to_database(signal)
                    
            except Exception as e:
                self.logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    def _execute_single_signal(self, signal: TradingSignal) -> bool:
        """Execute a single trading signal"""
        
        try:
            trade_logger = logging.getLogger('TradingActivity')
            trade_logger.info(f"Executing signal: {signal.symbol} {signal.direction} "
                            f"Size: {signal.position_size:.3f} @ {signal.entry_price:.5f}")
            
            # Place order through broker
            order_result = self.broker.place_market_order(
                symbol=signal.symbol,
                direction=signal.direction,
                volume=signal.position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if not order_result['success']:
                trade_logger.error(f"Order execution failed for {signal.symbol}: {order_result['error']}")
                return False
            
            # Create position data for dynamic management
            position_data = {
                'position_id': order_result['position_id'],
                'symbol': signal.symbol,
                'direction': signal.direction,
                'size': signal.position_size,
                'entry_price': order_result['fill_price'],
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            }
            
            # Add to dynamic position manager
            self.position_manager.add_position(position_data)
            
            # Update metrics
            self.metrics.total_positions_opened += 1
            self.metrics.active_positions += 1
            
            # Save position to database
            self._save_position_to_database(position_data, signal)
            
            trade_logger.info(f"Position opened successfully: {order_result['position_id']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing signal for {signal.symbol}: {e}")
            return False
    
    def _update_system_metrics(self):
        """Update system performance metrics"""
        
        try:
            # Calculate current portfolio value
            current_portfolio_value = self._calculate_current_portfolio_value()
            
            # Update metrics
            self.metrics.total_pnl = current_portfolio_value - self.portfolio_value
            self.metrics.active_positions = len(self.position_manager.active_positions)
            
            # Calculate daily PnL
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            self.metrics.daily_pnl = self._calculate_daily_pnl(today_start)
            
            # Update drawdown
            if self.metrics.total_pnl > 0:
                if self.metrics.total_pnl > self.metrics.max_drawdown:
                    self.metrics.max_drawdown = self.metrics.total_pnl
                self.metrics.current_drawdown = self.metrics.max_drawdown - self.metrics.total_pnl
            
            # Calculate win rate and profit factor
            self._calculate_performance_ratios()
            
            # Update system uptime
            if self.start_time:
                self.metrics.system_uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Execution rate
            if self.metrics.signals_generated > 0:
                self.metrics.execution_rate = self.metrics.signals_executed / self.metrics.signals_generated
            
            self.metrics.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating system metrics: {e}")
    
    def _check_system_health(self) -> bool:
        """Check overall system health"""
        
        try:
            health_checks = []
            
            # Broker connection
            health_checks.append(self.broker.is_connected())
            
            # Data feed status
            health_checks.append(self.data_manager.is_healthy())
            
            # Position manager status
            health_checks.append(self.position_manager.monitoring_active)
            
            # Database connectivity
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("SELECT 1")
                health_checks.append(True)
            except:
                health_checks.append(False)
            
            # Memory and resource check
            health_checks.append(self._check_resource_usage())
            
            # All checks must pass
            system_healthy = all(health_checks)
            
            if not system_healthy:
                self.logger.warning(f"System health check failed. Results: {health_checks}")
            
            return system_healthy
            
        except Exception as e:
            self.logger.error(f"Error in system health check: {e}")
            return False
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached"""
        
        try:
            if abs(self.metrics.daily_pnl) >= self.max_daily_loss:
                self.logger.warning(f"Daily loss limit reached: {self.metrics.daily_pnl:.2f}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking daily loss limit: {e}")
            return False
    
    def _handle_system_shutdown(self):
        """Handle open positions during system shutdown"""
        
        try:
            active_positions = list(self.position_manager.active_positions.values())
            
            if not active_positions:
                return
            
            self.logger.info(f"Handling {len(active_positions)} open positions during shutdown")
            
            # Get user input for position handling
            if self.trading_mode == TradingMode.LIVE:
                print(f"\nSystem shutdown requested with {len(active_positions)} open positions:")
                for pos in active_positions:
                    pnl = pos.risk_metrics.unrealized_pnl if hasattr(pos, 'risk_metrics') else 0
                    print(f"  - {pos.symbol} {pos.direction} (PnL: {pnl:.2f})")
                
                choice = input("\nOptions:\n1. Close all positions\n2. Leave positions open\n3. Close only losing positions\nChoice (1-3): ")
                
                if choice == '1':
                    # Close all positions
                    for pos in active_positions:
                        self.broker.close_position(pos.position_id)
                        self.logger.info(f"Closed position {pos.position_id}")
                
                elif choice == '3':
                    # Close only losing positions
                    for pos in active_positions:
                        if hasattr(pos, 'risk_metrics') and pos.risk_metrics.unrealized_pnl < 0:
                            self.broker.close_position(pos.position_id)
                            self.logger.info(f"Closed losing position {pos.position_id}")
                
                # Option 2 (leave open) requires no action
            
            else:
                # In paper trading mode, close all positions
                for pos in active_positions:
                    self.broker.close_position(pos.position_id)
                    self.logger.info(f"Closed position {pos.position_id} (paper trading)")
                    
        except Exception as e:
            self.logger.error(f"Error handling system shutdown: {e}")
    
    # Database operations
    def _save_position_to_database(self, position_data: Dict, signal: TradingSignal):
        """Save position to database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO positions (
                        id, symbol, direction, size, entry_price, stop_loss, 
                        take_profit, open_time, status, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position_data['position_id'],
                    position_data['symbol'],
                    position_data['direction'],
                    position_data['size'],
                    position_data['entry_price'],
                    position_data['stop_loss'],
                    position_data['take_profit'],
                    datetime.now().isoformat(),
                    'open',
                    json.dumps({
                        'signal_quality': signal.signal_quality.name,
                        'confidence': signal.confidence,
                        'technical_signal': signal.technical_signal.signal_details
                    })
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving position to database: {e}")
    
    def _save_signal_to_database(self, signal: TradingSignal):
        """Save signal to database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO signals (
                        symbol, direction, confidence, entry_price, stop_loss,
                        take_profit, timestamp, executed, signal_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal.symbol,
                    signal.direction,
                    signal.confidence,
                    signal.entry_price,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.timestamp.isoformat(),
                    True,
                    json.dumps(signal.to_dict())
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving signal to database: {e}")
    
    def _save_system_state(self):
        """Save current system state and metrics"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO system_metrics (
                        timestamp, total_pnl, daily_pnl, active_positions,
                        win_rate, profit_factor, max_drawdown, metrics_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    self.metrics.total_pnl,
                    self.metrics.daily_pnl,
                    self.metrics.active_positions,
                    self.metrics.win_rate,
                    self.metrics.profit_factor,
                    self.metrics.max_drawdown,
                    json.dumps({
                        'signals_generated': self.metrics.signals_generated,
                        'signals_executed': self.metrics.signals_executed,
                        'execution_rate': self.metrics.execution_rate,
                        'system_uptime': self.metrics.system_uptime,
                        'risk_events': self.metrics.risk_events,
                        'emergency_exits': self.metrics.emergency_exits
                    })
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving system state: {e}")
    
    # System control methods
    def pause_trading(self):
        """Pause trading activities"""
        if self.status == SystemStatus.RUNNING:
            self.status = SystemStatus.PAUSED
            self.logger.info("Trading paused")
    
    def resume_trading(self):
        """Resume trading activities"""
        if self.status == SystemStatus.PAUSED:
            self.status = SystemStatus.RUNNING
            self.logger.info("Trading resumed")
    
    def get_system_status(self) -> Dict:
        """Get current system status and metrics"""
        
        return {
            'status': self.status.value,
            'trading_mode': self.trading_mode.value,
            'uptime_seconds': self.metrics.system_uptime,
            'metrics': {
                'total_pnl': self.metrics.total_pnl,
                'daily_pnl': self.metrics.daily_pnl,
                'active_positions': self.metrics.active_positions,
                'win_rate': self.metrics.win_rate,
                'profit_factor': self.metrics.profit_factor,
                'max_drawdown': self.metrics.max_drawdown,
                'signals_generated': self.metrics.signals_generated,
                'signals_executed': self.metrics.signals_executed,
                'execution_rate': self.metrics.execution_rate
            },
            'active_symbols': self.symbol_manager.get_active_symbols(),
            'position_manager_summary': self.position_manager.get_performance_summary(),
            'last_updated': self.metrics.last_updated.isoformat() if self.metrics.last_updated else None
        }
    
    # Helper methods (implementations would continue...)
    def _calculate_current_portfolio_value(self) -> float:
        """Calculate current portfolio value including open positions"""
        # Implementation would calculate based on current positions and account balance
        return self.portfolio_value + self.metrics.total_pnl
    
    def _calculate_daily_pnl(self, date: datetime) -> float:
        """Calculate PnL for a specific date"""
        # Implementation would query database for positions closed on date
        return 0.0
    
    def _calculate_performance_ratios(self):
        """Calculate win rate and profit factor"""
        # Implementation would query closed positions and calculate ratios
        pass
    
    def _check_resource_usage(self) -> bool:
        """Check system resource usage"""
        # Implementation would check memory, CPU, disk usage
        return True
    
    def _get_symbol_position_count(self, symbol: str) -> int:
        """Get number of active positions for symbol"""
        return len([p for p in self.position_manager.active_positions.values() 
                   if p.symbol == symbol])
    
    def _perform_comprehensive_analysis(self):
        """Perform comprehensive market analysis"""
        # Implementation would do deeper market analysis, correlation checks, etc.
        pass
    
    def _check_portfolio_rebalancing(self):
        """Check if portfolio rebalancing is needed"""
        # Implementation would check position sizes, correlations, exposure
        pass
    
    def _log_performance_summary(self):
        """Log periodic performance summary"""
        status = self.get_system_status()
        self.logger.info(f"Performance Summary - PnL: {status['metrics']['total_pnl']:.2f}, "
                        f"Active Positions: {status['metrics']['active_positions']}, "
                        f"Win Rate: {status['metrics']['win_rate']:.1%}")
    
    def _initialize_portfolio_state(self):
        """Initialize portfolio state on startup"""
        # Load existing positions from broker
        # Reconcile with database
        # Initialize position manager with existing positions
        pass
    
    def _start_performance_monitoring(self):
        """Start performance monitoring thread"""
        # Start thread for periodic performance calculations and reporting
        pass

# Command-line interface and main execution
def main():
    """Main function to run the Enhanced Trading Engine"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Forex Trading Engine')
    parser.add_argument('--config', default='config/trading_config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], 
                       default='paper', help='Trading mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    try:
        # Create and configure trading engine
        engine = EnhancedTradingEngine(config_path=args.config)
        engine.trading_mode = TradingMode(args.mode)
        
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        print(f"Starting Enhanced Trading Engine in {args.mode} mode...")
        print("Press Ctrl+C to stop the system gracefully\n")
        
        # Start the system
        if engine.start_system():
            # Keep main thread alive
            try:
                while engine.status in [SystemStatus.RUNNING, SystemStatus.PAUSED]:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutdown signal received...")
        
        # Stop the system
        engine.stop_system()
        print("Enhanced Trading Engine stopped.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

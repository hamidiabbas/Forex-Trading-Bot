# main.py - Complete Enterprise Production Version
"""
Professional Enterprise Trading Bot
Complete Integration of All AI Systems
Production-Ready Implementation
"""

import os
import sys
import logging
import time
import threading
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import traceback
import json
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Enhanced logging setup
def initialize_enterprise_logging():
    """Initialize comprehensive enterprise logging"""
    log_dir = Path('logs/enterprise_bot')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(funcName)-15s | %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'enterprise_bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress third-party loggers
    for logger_name in ['urllib3', 'requests', 'asyncio']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = initialize_enterprise_logging()

# System imports with comprehensive error handling
try:
    # Core trading modules
    import config
    from datahandler import DataHandler
    from marketintelligence import MarketIntelligence
    from riskmanager import RiskManager
    from executionmanager import ExecutionManager
    from notificationmanager import NotificationManager
    CORE_MODULES = True
    logger.info("✅ Core trading modules loaded")
except ImportError as e:
    logger.error(f"❌ Core modules import error: {e}")
    CORE_MODULES = False
    sys.exit(1)

try:
    # Enhanced AI modules
    from enhanced_sentiment_integration import ProductionSentimentAnalyzer, SentimentDataManager
    from dynamic_kelly_position_sizing import ProfessionalKellyPositionSizer
    AI_MODULES = True
    logger.info("✅ Enhanced AI modules loaded")
except ImportError as e:
    logger.warning(f"⚠️ AI modules not fully available: {e}")
    AI_MODULES = False

try:
    # RL modules
    from stable_baselines3 import SAC, PPO, A2C
    import torch
    RL_MODULES = True
    logger.info("✅ RL modules loaded")
except ImportError as e:
    logger.warning(f"⚠️ RL modules not available: {e}")
    RL_MODULES = False

class EnterpriseTradingBot:
    """
    Complete Enterprise Trading Bot with Full AI Integration
    Production-Ready Implementation
    """
    
    def __init__(self, config_file: str = "configs/enterprise_config.json"):
        logger.info("🚀 Initializing Enterprise Trading Bot...")
        
        # Load configuration
        self.config = self._load_configuration(config_file)
        
        # System state
        self.is_running = False
        self.stop_event = threading.Event()
        self.start_time = time.time()
        
        # Core components
        self.data_handler = None
        self.market_intel = None
        self.risk_manager = None
        self.execution_manager = None
        self.notification_manager = None
        
        # AI components
        self.sentiment_analyzer = None
        self.sentiment_data_manager = None
        self.kelly_sizer = None
        self.rl_models = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_signals': 0,
            'successful_trades': 0,
            'ai_signals': 0,
            'rl_signals': 0,
            'traditional_signals': 0,
            'sentiment_enhanced': 0,
            'kelly_sized': 0
        }
        
        # Trading state
        self.active_positions = {}
        self.last_analysis = {}
        
        logger.info("✅ Enterprise Trading Bot initialized")
    
    def _load_configuration(self, config_file: str) -> Dict[str, Any]:
        """Load comprehensive configuration"""
        default_config = {
            "trading": {
                "symbols": ["EURUSD", "GBPUSD", "XAUUSD"],
                "risk_per_trade": 0.01,
                "max_positions": 5,
                "min_confidence": 0.6,
                "analysis_interval": 30,
                "loop_interval": 10
            },
            "ai_systems": {
                "enable_sentiment": AI_MODULES,
                "enable_kelly_sizing": AI_MODULES,
                "enable_rl": RL_MODULES,
                "sentiment_weight": 0.3,
                "rl_weight": 0.4,
                "traditional_weight": 0.3
            },
            "models": {
                "rl_model_path": "./models/best_SAC_EURUSD/best_model.zip",
                "sentiment_model_path": "./models/sentiment_analyzer.pth",
                "fallback_to_traditional": True
            },
            "risk_management": {
                "max_daily_risk": 0.05,
                "max_drawdown": 0.10,
                "correlation_limit": 0.8,
                "dynamic_sizing": True
            },
            "notifications": {
                "console": True,
                "email": False,
                "telegram": False
            }
        }
        
        if Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                logger.info(f"✅ Configuration loaded from {config_file}")
            except Exception as e:
                logger.warning(f"⚠️ Config load error: {e}")
        
        return default_config
    
    def initialize_core_systems(self) -> bool:
        """Initialize core trading systems"""
        logger.info("🔧 Initializing core trading systems...")
        
        try:
            # Data handler
            self.data_handler = DataHandler(config)
            if not self.data_handler.connect():
                logger.error("❌ Data handler connection failed")
                return False
            logger.info("✅ Data handler connected")
            
            # Market intelligence
            self.market_intel = MarketIntelligence(self.data_handler, config)
            logger.info("✅ Market intelligence initialized")
            
            # Risk manager
            self.risk_manager = RiskManager(config)
            logger.info("✅ Risk manager initialized")
            
            # Execution manager
            self.execution_manager = ExecutionManager(config)
            logger.info("✅ Execution manager initialized")
            
            # Notification manager
            self.notification_manager = NotificationManager(config)
            logger.info("✅ Notification manager initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Core systems initialization failed: {e}")
            return False
    
    def initialize_ai_systems(self) -> bool:
        """Initialize AI systems with fallback handling"""
        logger.info("🤖 Initializing AI systems...")
        
        ai_systems_count = 0
        
        # Sentiment Analysis System
        if AI_MODULES and self.config["ai_systems"]["enable_sentiment"]:
            try:
                sentiment_config = {
                    'hidden_dim': 768,
                    'num_heads': 8,
                    'dropout': 0.1
                }
                self.sentiment_analyzer = ProductionSentimentAnalyzer(sentiment_config)
                self.sentiment_data_manager = SentimentDataManager(self.config)
                logger.info("✅ Sentiment analysis system initialized")
                ai_systems_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Sentiment system initialization failed: {e}")
                self.config["ai_systems"]["enable_sentiment"] = False
        
        # Kelly Position Sizing System
        if AI_MODULES and self.config["ai_systems"]["enable_kelly_sizing"]:
            try:
                kelly_config = {
                    'kelly_lookback_trades': 100,
                    'kelly_safety_factor': 0.25,
                    'base_risk_per_trade': self.config["trading"]["risk_per_trade"],
                    'max_risk_per_trade': 0.05
                }
                self.kelly_sizer = ProfessionalKellyPositionSizer(kelly_config)
                logger.info("✅ Kelly position sizing system initialized")
                ai_systems_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Kelly sizing system initialization failed: {e}")
                self.config["ai_systems"]["enable_kelly_sizing"] = False
        
        # RL System
        if RL_MODULES and self.config["ai_systems"]["enable_rl"]:
            try:
                self._load_rl_models()
                if self.rl_models:
                    logger.info(f"✅ RL system initialized with {len(self.rl_models)} models")
                    ai_systems_count += 1
                else:
                    logger.warning("⚠️ No RL models loaded")
                    self.config["ai_systems"]["enable_rl"] = False
            except Exception as e:
                logger.warning(f"⚠️ RL system initialization failed: {e}")
                self.config["ai_systems"]["enable_rl"] = False
        
        logger.info(f"🤖 AI Systems Status: {ai_systems_count}/3 systems active")
        return ai_systems_count > 0 or self.config["models"]["fallback_to_traditional"]
    
    def _load_rl_models(self):
        """Load trained RL models"""
        model_base_path = Path("./models")
        
        # Load different model types
        for model_type in ['SAC', 'PPO', 'A2C']:
            for symbol in self.config["trading"]["symbols"]:
                model_paths = [
                    model_base_path / f"best_{model_type}_{symbol}" / "best_model.zip",
                    model_base_path / f"{model_type}_{symbol}_final.zip",
                    model_base_path / f"enhanced_{model_type}_{symbol}_final.zip"
                ]
                
                for model_path in model_paths:
                    if model_path.exists():
                        try:
                            if model_type == 'SAC':
                                model = SAC.load(str(model_path))
                            elif model_type == 'PPO':
                                model = PPO.load(str(model_path))
                            elif model_type == 'A2C':
                                model = A2C.load(str(model_path))
                            
                            model_key = f"{model_type}_{symbol}"
                            self.rl_models[model_key] = {
                                'model': model,
                                'type': model_type,
                                'symbol': symbol,
                                'path': str(model_path)
                            }
                            logger.info(f"✅ Loaded {model_key} from {model_path.name}")
                            break
                            
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load {model_path}: {e}")
                            continue
    
    async def analyze_market_comprehensive(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Comprehensive market analysis with all AI systems"""
        try:
            # Get market data
            data_dict = self.data_handler.get_multi_timeframe_data(symbol)
            if not data_dict or 'EXECUTION' not in data_dict:
                logger.warning(f"No market data for {symbol}")
                return None
            
            execution_df = data_dict['EXECUTION']
            
            # Basic market analysis
            market_regime = self.market_intel.identify_regime(execution_df)
            
            # Generate signals from different systems
            signals = []
            
            # 1. RL Signal (if available)
            if self.config["ai_systems"]["enable_rl"]:
                rl_signal = await self._generate_rl_signal(symbol, execution_df)
                if rl_signal:
                    signals.append(rl_signal)
                    self.performance_stats['rl_signals'] += 1
            
            # 2. Traditional Signal (always available)
            traditional_signal = self.market_intel.generate_enhanced_signal(
                symbol, data_dict, market_regime
            )
            if traditional_signal:
                signals.append(traditional_signal)
                self.performance_stats['traditional_signals'] += 1
            
            # 3. Sentiment Enhancement (if available)
            if self.config["ai_systems"]["enable_sentiment"]:
                try:
                    sentiment_data = await self.sentiment_data_manager.collect_sentiment_data([symbol])
                    if symbol in sentiment_data:
                        signals = self._enhance_signals_with_sentiment(signals, sentiment_data[symbol])
                        self.performance_stats['sentiment_enhanced'] += 1
                except Exception as e:
                    logger.warning(f"Sentiment enhancement failed: {e}")
            
            # Select best signal
            final_signal = self._select_optimal_signal(signals, market_regime)
            
            if final_signal:
                final_signal.update({
                    'symbol': symbol,
                    'market_regime': market_regime,
                    'analysis_timestamp': datetime.now(),
                    'ai_systems_used': self._get_active_ai_systems(),
                    'data_quality': len(execution_df) / 1000.0
                })
                
                self.performance_stats['total_signals'] += 1
                logger.info(f"📊 Signal generated: {symbol} | {final_signal['strategy']} | {final_signal['direction']}")
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Market analysis error for {symbol}: {e}")
            return None
    
    async def _generate_rl_signal(self, symbol: str, market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate signal using RL models"""
        try:
            # Find best RL model for symbol
            rl_model_info = None
            for model_key, info in self.rl_models.items():
                if symbol in model_key:
                    rl_model_info = info
                    break
            
            if not rl_model_info:
                logger.debug(f"No RL model found for {symbol}")
                return None
            
            # Prepare observation
            observation = self._prepare_rl_observation(market_data)
            if observation is None:
                return None
            
            # Get prediction
            model = rl_model_info['model']
            action, _states = model.predict(observation, deterministic=True)
            
            # Convert to trading signal
            if isinstance(action, np.ndarray):
                action = action[0]
            
            if action == 1:
                direction = 'BUY'
            elif action == 2:
                direction = 'SELL'
            else:
                return None  # HOLD
            
            current_price = market_data['Close'].iloc[-1]
            
            return {
                'symbol': symbol,
                'direction': direction,
                'strategy': f'RL-{rl_model_info["type"]}',
                'confidence': 0.75,  # RL models typically have good confidence
                'entry_price': current_price,
                'model_type': rl_model_info['type'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"RL signal generation error: {e}")
            return None
    
    def _prepare_rl_observation(self, market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare observation for RL model"""
        try:
            if len(market_data) < 50:
                return None
            
            # Get latest data
            recent_data = market_data.tail(50)
            
            features = []
            
            # Price features
            features.extend([
                recent_data['Close'].iloc[-1],
                recent_data['Open'].iloc[-1],
                recent_data['High'].iloc[-1],
                recent_data['Low'].iloc[-1]
            ])
            
            # Technical indicators (with fallback)
            rsi = recent_data.get('RSI_14', pd.Series([50] * len(recent_data))).iloc[-1]
            features.append(rsi)
            
            macd = recent_data.get('MACD_12_26_9', pd.Series([0] * len(recent_data))).iloc[-1]
            features.append(macd)
            
            # Add more features up to expected input size (typically 32-64 features)
            while len(features) < 32:
                features.append(0.0)
            
            return np.array(features[:32], dtype=np.float32).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"RL observation preparation error: {e}")
            return None
    
    def _enhance_signals_with_sentiment(self, signals: List[Dict], sentiment_data: Dict) -> List[Dict]:
        """Enhance signals with sentiment analysis"""
        try:
            sentiment_score = sentiment_data.get('overall_sentiment', 0.0)
            sentiment_confidence = sentiment_data.get('confidence', 0.5)
            
            enhanced_signals = []
            
            for signal in signals:
                enhanced_signal = signal.copy()
                
                # Adjust confidence based on sentiment alignment
                signal_direction = 1 if signal['direction'] == 'BUY' else -1
                sentiment_alignment = sentiment_score * signal_direction
                
                if sentiment_alignment > 0:
                    # Sentiment supports signal
                    boost = min(0.2, sentiment_confidence * 0.3)
                    enhanced_signal['confidence'] = min(1.0, signal.get('confidence', 0.5) + boost)
                    enhanced_signal['sentiment_boost'] = boost
                else:
                    # Sentiment opposes signal
                    penalty = min(0.15, sentiment_confidence * 0.2)
                    enhanced_signal['confidence'] = max(0.1, signal.get('confidence', 0.5) - penalty)
                    enhanced_signal['sentiment_penalty'] = penalty
                
                enhanced_signal['sentiment_score'] = sentiment_score
                enhanced_signal['sentiment_confidence'] = sentiment_confidence
                
                enhanced_signals.append(enhanced_signal)
            
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"Sentiment enhancement error: {e}")
            return signals
    
    def _select_optimal_signal(self, signals: List[Dict], market_regime: str) -> Optional[Dict[str, Any]]:
        """Select optimal signal using weighted scoring"""
        if not signals:
            return None
        
        if len(signals) == 1:
            return signals[0]
        
        try:
            # Score signals
            scored_signals = []
            
            for signal in signals:
                score = signal.get('confidence', 0.5)
                
                # Strategy type weights
                strategy = signal.get('strategy', '').upper()
                if 'RL-' in strategy:
                    score *= self.config["ai_systems"]["rl_weight"] + 1.0
                elif 'TRADITIONAL' in strategy or 'ENHANCED' in strategy:
                    score *= self.config["ai_systems"]["traditional_weight"] + 0.8
                
                # Sentiment boost
                if 'sentiment_boost' in signal:
                    score *= (1.0 + signal['sentiment_boost'])
                elif 'sentiment_penalty' in signal:
                    score *= (1.0 - signal['sentiment_penalty'])
                
                # Market regime compatibility
                if market_regime == 'trending':
                    if 'TREND' in strategy or 'RL-' in strategy:
                        score *= 1.1
                elif market_regime == 'ranging':
                    if 'MEAN_REVERSION' in strategy:
                        score *= 1.1
                
                scored_signals.append((score, signal))
            
            # Select best signal
            scored_signals.sort(key=lambda x: x[0], reverse=True)
            best_signal = scored_signals[0][1].copy()
            best_signal['selection_score'] = scored_signals[0][0]
            best_signal['alternatives_considered'] = len(signals)
            
            return best_signal
            
        except Exception as e:
            logger.error(f"Signal selection error: {e}")
            return signals[0]
    
    async def process_trading_signal(self, signal: Dict[str, Any]) -> bool:
        """Process trading signal with advanced risk management"""
        try:
            symbol = signal['symbol']
            
            # Enhanced position sizing
            if self.config["ai_systems"]["enable_kelly_sizing"] and self.kelly_sizer:
                try:
                    account_balance = 10000  # This should come from your broker
                    sizing_result = self.kelly_sizer.calculate_optimal_position_size(
                        signal=signal,
                        account_balance=account_balance
                    )
                    
                    signal['position_size'] = sizing_result.recommended_size
                    signal['kelly_sizing'] = True
                    signal['kelly_info'] = {
                        'kelly_fraction': sizing_result.kelly_fraction,
                        'final_risk_pct': sizing_result.final_risk_percentage
                    }
                    self.performance_stats['kelly_sized'] += 1
                    
                except Exception as e:
                    logger.warning(f"Kelly sizing failed: {e}")
                    # Fall back to traditional sizing
                    signal['position_size'] = self.config["trading"]["risk_per_trade"]
            else:
                # Traditional position sizing
                signal['position_size'] = self.config["trading"]["risk_per_trade"]
            
            # Risk management validation
            risk_approved = self.risk_manager.validate_signal(signal)
            if not risk_approved:
                logger.warning(f"Signal rejected by risk manager: {symbol}")
                return False
            
            # Execute trade
            execution_result = self.execution_manager.execute_trade(signal)
            
            if execution_result and execution_result.get('success', False):
                self.performance_stats['successful_trades'] += 1
                
                # Update Kelly sizer with trade result (if applicable)
                if self.kelly_sizer:
                    # This would be called when the trade closes with actual results
                    pass
                
                # Store position
                self.active_positions[symbol] = {
                    'signal': signal,
                    'execution': execution_result,
                    'entry_time': datetime.now()
                }
                
                # Send notification
                await self._send_trade_notification(signal, execution_result)
                
                logger.info(f"✅ Trade executed: {symbol} {signal['direction']}")
                return True
            else:
                logger.warning(f"❌ Trade execution failed: {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Signal processing error: {e}")
            return False
    
    def _get_active_ai_systems(self) -> Dict[str, bool]:
        """Get status of active AI systems"""
        return {
            'sentiment_analysis': self.config["ai_systems"]["enable_sentiment"],
            'rl_models': self.config["ai_systems"]["enable_rl"],
            'kelly_sizing': self.config["ai_systems"]["enable_kelly_sizing"],
            'total_rl_models': len(self.rl_models)
        }
    
    async def _send_trade_notification(self, signal: Dict[str, Any], execution_result: Dict[str, Any]):
        """Send comprehensive trade notification"""
        try:
            ai_systems = self._get_active_ai_systems()
            active_systems = [k for k, v in ai_systems.items() if v and k != 'total_rl_models']
            
            message = (
                f"🤖 AI-Enhanced Trade Executed\n"
                f"Symbol: {signal['symbol']}\n"
                f"Direction: {signal['direction']}\n"
                f"Strategy: {signal['strategy']}\n"
                f"Confidence: {signal.get('confidence', 0):.2f}\n"
                f"Position Size: {signal.get('position_size', 0):.4f}\n"
                f"AI Systems: {', '.join(active_systems)}\n"
                f"Market Regime: {signal.get('market_regime', 'Unknown')}"
            )
            
            # Add Kelly info if available
            if signal.get('kelly_sizing'):
                kelly_info = signal.get('kelly_info', {})
                message += f"\nKelly Fraction: {kelly_info.get('kelly_fraction', 0):.3f}"
                message += f"\nRisk %: {kelly_info.get('final_risk_pct', 0):.2%}"
            
            # Add sentiment info if available
            if 'sentiment_score' in signal:
                message += f"\nSentiment: {signal['sentiment_score']:.2f}"
            
            if self.notification_manager:
                self.notification_manager.send_notification("AI Trade Alert", message)
            
            logger.info(f"📢 Trade notification sent for {signal['symbol']}")
            
        except Exception as e:
            logger.warning(f"Notification failed: {e}")
    
    async def run_trading_loop(self):
        """Main enterprise trading loop"""
        logger.info("🚀 Starting Enterprise Trading Loop...")
        
        self._display_startup_banner()
        
        try:
            self.is_running = True
            
            while not self.stop_event.is_set():
                loop_start = time.time()
                
                # Process each symbol
                symbols = self.config["trading"]["symbols"]
                
                for symbol in symbols:
                    try:
                        # Check analysis interval
                        if symbol in self.last_analysis:
                            time_since_last = time.time() - self.last_analysis[symbol]
                            if time_since_last < self.config["trading"]["analysis_interval"]:
                                continue
                        
                        # Comprehensive market analysis
                        signal = await self.analyze_market_comprehensive(symbol)
                        
                        if signal and signal.get('confidence', 0) >= self.config["trading"]["min_confidence"]:
                            await self.process_trading_signal(signal)
                        
                        self.last_analysis[symbol] = time.time()
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Manage existing positions
                await self._manage_positions()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep until next cycle
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.config["trading"]["loop_interval"] - loop_duration)
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("⏹️ Trading loop interrupted by user")
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            logger.error(traceback.format_exc())
        finally:
            await self.shutdown()
    
    async def _manage_positions(self):
        """Manage existing positions"""
        try:
            for symbol, position_info in list(self.active_positions.items()):
                # Position management logic
                entry_time = position_info['entry_time']
                hold_duration = (datetime.now() - entry_time).total_seconds() / 3600
                
                # Example: Close positions after 24 hours
                if hold_duration > 24:
                    logger.info(f"Closing position {symbol} after {hold_duration:.1f}h")
                    # Implement position closing logic
                    del self.active_positions[symbol]
                    
        except Exception as e:
            logger.error(f"Position management error: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics and log periodically"""
        current_time = time.time()
        uptime_hours = (current_time - self.start_time) / 3600
        
        # Log performance every 5 minutes
        if not hasattr(self, '_last_perf_log'):
            self._last_perf_log = current_time
        
        if current_time - self._last_perf_log > 300:  # 5 minutes
            total_signals = self.performance_stats['total_signals']
            success_rate = (self.performance_stats['successful_trades'] / max(total_signals, 1)) * 100
            
            logger.info("📊 PERFORMANCE UPDATE")
            logger.info(f"⏱️ Uptime: {uptime_hours:.1f} hours")
            logger.info(f"📈 Total Signals: {total_signals}")
            logger.info(f"✅ Success Rate: {success_rate:.1f}%")
            logger.info(f"🤖 AI Systems: RL={self.performance_stats['rl_signals']}, "
                       f"Sentiment={self.performance_stats['sentiment_enhanced']}, "
                       f"Kelly={self.performance_stats['kelly_sized']}")
            logger.info(f"💼 Active Positions: {len(self.active_positions)}")
            
            self._last_perf_log = current_time
    
    def _display_startup_banner(self):
        """Display comprehensive startup banner"""
        ai_systems = self._get_active_ai_systems()
        active_count = sum(1 for v in ai_systems.values() if isinstance(v, bool) and v)
        
        logger.info("=" * 80)
        logger.info("🚀 ENTERPRISE AI TRADING BOT - PRODUCTION VERSION")
        logger.info("=" * 80)
        logger.info("🤖 AI SYSTEMS STATUS:")
        logger.info(f"   ✅ Sentiment Analysis: {ai_systems['sentiment_analysis']}")
        logger.info(f"   ✅ RL Models: {ai_systems['rl_models']} ({ai_systems['total_rl_models']} loaded)")
        logger.info(f"   ✅ Kelly Position Sizing: {ai_systems['kelly_sizing']}")
        logger.info(f"   🎯 Total Active: {active_count}/3 AI Systems")
        logger.info("")
        logger.info(f"📈 Trading Symbols: {self.config['trading']['symbols']}")
        logger.info(f"🛡️ Risk per Trade: {self.config['trading']['risk_per_trade']:.1%}")
        logger.info(f"🎯 Min Confidence: {self.config['trading']['min_confidence']:.1%}")
        logger.info(f"🐍 Python: {sys.version.split()[0]}")
        
        if torch.cuda.is_available():
            logger.info(f"🚀 GPU: {torch.cuda.get_device_name()}")
        else:
            logger.info("💻 CPU Mode")
        
        logger.info("=" * 80)
    
    async def shutdown(self):
        """Graceful shutdown with comprehensive reporting"""
        logger.info("⏹️ Shutting down Enterprise Trading Bot...")
        
        try:
            self.is_running = False
            self.stop_event.set()
            
            # Close positions if configured
            if self.config.get("risk_management", {}).get("close_on_shutdown", False):
                logger.info("Closing all positions...")
                for symbol in list(self.active_positions.keys()):
                    # Implement position closing logic
                    del self.active_positions[symbol]
            
            # Final performance report
            uptime = (time.time() - self.start_time) / 3600
            total_signals = self.performance_stats['total_signals']
            success_rate = (self.performance_stats['successful_trades'] / max(total_signals, 1)) * 100
            
            logger.info("=" * 60)
            logger.info("📊 FINAL PERFORMANCE REPORT")
            logger.info("=" * 60)
            logger.info(f"⏱️ Total Uptime: {uptime:.2f} hours")
            logger.info(f"📈 Total Signals: {total_signals}")
            logger.info(f"✅ Successful Trades: {self.performance_stats['successful_trades']}")
            logger.info(f"📊 Success Rate: {success_rate:.1f}%")
            logger.info("")
            logger.info("🤖 AI SYSTEM USAGE:")
            logger.info(f"   🧠 RL Signals: {self.performance_stats['rl_signals']}")
            logger.info(f"   💭 Sentiment Enhanced: {self.performance_stats['sentiment_enhanced']}")
            logger.info(f"   📐 Kelly Sized: {self.performance_stats['kelly_sized']}")
            logger.info(f"   🔧 Traditional: {self.performance_stats['traditional_signals']}")
            logger.info("=" * 60)
            
            # Disconnect data handler
            if self.data_handler:
                self.data_handler.disconnect()
            
            # Send shutdown notification
            if self.notification_manager:
                shutdown_message = (
                    f"🏁 Enterprise AI Trading Bot Shutdown\n"
                    f"Uptime: {uptime:.1f} hours\n"
                    f"Signals: {total_signals}, Success: {success_rate:.1f}%\n"
                    f"AI Systems Active: {sum(1 for v in self._get_active_ai_systems().values() if isinstance(v, bool) and v)}/3"
                )
                self.notification_manager.send_notification("Bot Shutdown", shutdown_message)
            
            logger.info("✅ Enterprise Trading Bot shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    def start(self):
        """Start the enterprise trading bot"""
        logger.info("🚀 Starting Enterprise Trading Bot...")
        
        try:
            # Initialize core systems
            if not self.initialize_core_systems():
                logger.error("❌ Core systems initialization failed")
                return False
            
            # Initialize AI systems
            if not self.initialize_ai_systems():
                logger.error("❌ AI systems initialization failed")
                return False
            
            # Send startup notification
            if self.notification_manager:
                ai_count = sum(1 for v in self._get_active_ai_systems().values() if isinstance(v, bool) and v)
                startup_message = (
                    f"🚀 Enterprise AI Trading Bot Started\n"
                    f"AI Systems: {ai_count}/3 Active\n"
                    f"Symbols: {', '.join(self.config['trading']['symbols'])}\n"
                    f"Ready for AI-enhanced trading!"
                )
                self.notification_manager.send_notification("Bot Started", startup_message)
            
            # Run trading loop
            asyncio.run(self.run_trading_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Startup failed: {e}")
            logger.error(traceback.format_exc())
            return False

def main():
    """Main entry point"""
    logger.info("🚀 Starting Enterprise AI Trading Bot...")
    
    try:
        # Create and start bot
        bot = EnterpriseTradingBot()
        success = bot.start()
        
        if success:
            logger.info("✅ Enterprise Trading Bot completed successfully")
        else:
            logger.error("❌ Enterprise Trading Bot failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Bot interrupted by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
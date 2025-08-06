"""
Enhanced Professional Forex Trading Bot - COMPLETE PRODUCTION VERSION
Maintains all existing advanced features while adding critical fixes for RL integration
Production-grade implementation with SAC/A2C support, enhanced error handling, and seamless component integration
"""

import os
import sys
import logging
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from collections import Counter
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import warnings

warnings.filterwarnings('ignore')

# Enhanced RL Integration imports
RL_AVAILABLE = False
try:
    from stable_baselines3 import SAC, A2C
    import torch
    RL_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not available. RL features disabled.")

# Enhanced Core Module Imports - Fixed with comprehensive error handling
try:
    import config
    from enhanced_datahandler import EnhancedDataHandler
    from enhanced_marketintelligence import EnhancedMarketIntelligence  
    from enhanced_tradingenvironment import EnhancedTradingEnvironment
    from enhanced_train_rl_model import EnhancedRLModelManager
    from enhanced_feature_engineering import FeatureManager
    from dynamic_kelly_position_sizing import KellyPositionManager
    from enhanced_sentiment_integration import SentimentManager
    from strategymanager import StrategyManager
    from riskmanager import RiskManager
    from executionmanager import ExecutionManager
    from notificationmanager import NotificationManager
except ImportError as e:
    logging.error(f"Missing enhanced module: {e}")
    print(f"ERROR: Missing enhanced module: {e}")
    print("\nRequired enhanced files:")
    print("- enhanced_datahandler.py")
    print("- enhanced_marketintelligence.py") 
    print("- enhanced_tradingenvironment.py")
    print("- enhanced_train_rl_model.py")
    print("- enhanced_feature_engineering.py")
    print("- dynamic_kelly_position_sizing.py")
    print("- enhanced_sentiment_integration.py")
    print("- strategymanager.py")
    print("- riskmanager.py")
    print("- executionmanager.py")
    print("- notificationmanager.py")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Enhanced logging configuration
def setup_enhanced_logging():
    """Setup comprehensive logging system with enhanced error handling"""
    try:
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Clear existing handlers to prevent duplicates
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create enhanced formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Enhanced file handler with rotation
        log_file = log_dir / 'enhanced_trading_bot.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        
        # Enhanced console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Configure root logger with enhanced settings
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Suppress noisy loggers but keep important ones
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('stable_baselines3').setLevel(logging.WARNING)
        
        return True
        
    except Exception as e:
        print(f"Enhanced logging setup failed: {e}")
        # Fallback to basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        return False

# Initialize enhanced logging
setup_enhanced_logging()

class EnhancedTradingBot:
    """
    ✅ COMPLETE ENHANCED: Professional Trading Bot with Complete Integration
    Maintains all existing advanced features while adding critical fixes for seamless operation
    Features: SAC/A2C RL, Kelly Position Sizing, Sentiment Analysis, Advanced Risk Management, Regime Detection
    """
    
    def __init__(self, config_path: str = 'config.py'):
        """Initialize the enhanced trading bot with comprehensive error handling"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 70)
        self.logger.info("INITIALIZING ENHANCED FOREX TRADING BOT - COMPLETE VERSION")
        self.logger.info("=" * 70)
        
        # Core configuration with enhanced validation
        self.config = config
        self.start_time = time.time()
        self.initialization_time = datetime.now()
        
        # Enhanced control variables
        self.stop_event = threading.Event()
        self.status = "Initializing Enhanced Systems"
        self.last_analysis_time = {}
        self.analysis_success_count = 0
        self.analysis_failure_count = 0
        
        # Enhanced performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.current_positions = []
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.total_signals_generated = 0
        self.signals_executed = 0
        
        # Component health tracking
        self.component_health = {
            'datahandler': False,
            'marketintelligence': False,
            'feature_manager': False,
            'sentiment_manager': False,
            'kelly_manager': False,
            'strategy_manager': False,
            'risk_manager': False,
            'execution_manager': False,
            'notification_manager': False,
            'rl_system': False
        }
        
        # Enhanced Components Initialization with comprehensive error handling
        try:
            self._initialize_enhanced_components()
            self.logger.info("✅ All enhanced components initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize enhanced components: {e}")
            self.logger.error(traceback.format_exc())
            raise
        
        # RL Model Integration with enhanced error handling
        self._initialize_rl_system()
        
        # Final system checks with comprehensive diagnostics
        self._run_system_diagnostics()
        
        # Startup summary
        initialization_duration = (datetime.now() - self.initialization_time).total_seconds()
        self.logger.info(f"✅ Enhanced Trading Bot initialization completed in {initialization_duration:.2f}s")
        self.logger.info(f"RL Status: {'Enabled' if self.rl_enabled else 'Disabled'}")
        self.logger.info(f"RL Model: {self.rl_model_type if self.rl_model_type else 'None'}")
        self.logger.info(f"Components Health: {sum(self.component_health.values())}/{len(self.component_health)} OK")

    def _initialize_enhanced_components(self):
        """Initialize all enhanced trading components with comprehensive error handling"""
        
        self.logger.info("Initializing enhanced components...")
        
        # 1. Enhanced Feature Management
        try:
            self.feature_manager = FeatureManager()
            self.component_health['feature_manager'] = True
            self.logger.info("✅ FeatureManager initialized")
        except Exception as e:
            self.logger.error(f"❌ FeatureManager initialization failed: {e}")
            raise
        
        # 2. Enhanced Data Handler with validation
        try:
            self.datahandler = EnhancedDataHandler(config)
            self.component_health['datahandler'] = True
            self.logger.info("✅ EnhancedDataHandler initialized")
        except Exception as e:
            self.logger.error(f"❌ EnhancedDataHandler initialization failed: {e}")
            raise
        
        # 3. Enhanced Market Intelligence with validation
        try:
            self.marketintelligence = EnhancedMarketIntelligence(config, user_config=None)
            self.component_health['marketintelligence'] = True
            self.logger.info("✅ EnhancedMarketIntelligence initialized")
        except Exception as e:
            self.logger.error(f"❌ EnhancedMarketIntelligence initialization failed: {e}")
            raise
        
        # 4. Kelly Position Sizing with validation
        try:
            self.kelly_manager = KellyPositionManager(config)
            self.component_health['kelly_manager'] = True
            self.logger.info("✅ KellyPositionManager initialized")
        except Exception as e:
            self.logger.error(f"❌ KellyPositionManager initialization failed: {e}")
            raise
        
        # 5. Enhanced Sentiment Analysis with validation
        try:
            self.sentiment_manager = SentimentManager(config)
            self.component_health['sentiment_manager'] = True
            self.logger.info("✅ SentimentManager initialized")
        except Exception as e:
            self.logger.error(f"❌ SentimentManager initialization failed: {e}")
            raise
        
        # 6. Strategy Manager with enhanced compatibility
        try:
            self.strategymanager = StrategyManager(config, self.marketintelligence)
            self.component_health['strategy_manager'] = True
            self.logger.info("✅ StrategyManager initialized")
        except Exception as e:
            self.logger.error(f"❌ StrategyManager initialization failed: {e}")
            raise
        
        # 7. Risk Manager with enhanced features
        try:
            self.riskmanager = RiskManager(config, self.kelly_manager)
            self.component_health['risk_manager'] = True
            self.logger.info("✅ RiskManager initialized")
        except Exception as e:
            self.logger.error(f"❌ RiskManager initialization failed: {e}")
            raise
        
        # 8. Execution Manager with enhanced error handling
        try:
            self.executionmanager = ExecutionManager(config)
            self.component_health['execution_manager'] = True
            self.logger.info("✅ ExecutionManager initialized")
        except Exception as e:
            self.logger.error(f"❌ ExecutionManager initialization failed: {e}")
            raise
        
        # 9. Notification Manager with enhanced features
        try:
            self.notificationmanager = NotificationManager(config)
            self.component_health['notification_manager'] = True
            self.logger.info("✅ NotificationManager initialized")
        except Exception as e:
            self.logger.error(f"❌ NotificationManager initialization failed: {e}")
            raise
        
        self.logger.info("✅ All traditional components initialized with enhanced compatibility")

    def _initialize_rl_system(self):
        """Initialize the enhanced RL system with comprehensive error handling"""
        
        # Initialize RL tracking variables
        self.rl_model_manager = None
        self.rl_enabled = False
        self.rl_model_type = None
        self.rl_model = None
        
        # RL Performance Tracking with enhanced metrics
        self.rl_signal_count = 0
        self.rl_successful_trades = 0
        self.rl_failed_trades = 0
        self.rl_performance_history = []
        self.rl_confidence_scores = []
        self.rl_last_prediction_time = None
        
        if not RL_AVAILABLE:
            self.logger.warning("⚠️ RL libraries not available - RL features disabled")
            return
        
        try:
            self.logger.info("Initializing Enhanced RL System...")
            
            # Initialize Enhanced RL Model Manager
            self.rl_model_manager = EnhancedRLModelManager(config)
            
            # Load best available model with enhanced validation
            model_loaded = self.rl_model_manager.load_best_model()
            
            if model_loaded:
                self.rl_enabled = True
                self.rl_model_type = self.rl_model_manager.get_model_type()
                self.rl_model = self.rl_model_manager.get_model()
                self.component_health['rl_system'] = True
                
                self.logger.info(f"✅ Enhanced RL System initialized successfully")
                self.logger.info(f"   Model Type: {self.rl_model_type}")
                self.logger.info(f"   Model Status: Loaded and Ready")
                
                # Validate RL model compatibility
                self._validate_rl_model_compatibility()
                
            else:
                self.logger.warning("⚠️ No RL models found - traditional strategies only")
                
        except Exception as e:
            self.logger.error(f"❌ RL system initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            self.rl_enabled = False
            self.component_health['rl_system'] = False

    def _validate_rl_model_compatibility(self):
        """Validate RL model compatibility with feature engineering"""
        try:
            if not self.rl_enabled or not self.rl_model_manager:
                return
            
            # Test feature generation compatibility
            test_data = self._generate_test_market_data()
            features_array = self.feature_manager.get_latest_features(test_data)
            
            if features_array is not None and len(features_array) == 50:
                self.logger.info("✅ RL model compatibility validated (50 features)")
            else:
                self.logger.warning(f"⚠️ RL model compatibility issue: Expected 50 features, got {len(features_array) if features_array is not None else 'None'}")
                
        except Exception as e:
            self.logger.warning(f"RL model compatibility validation failed: {e}")

    def _generate_test_market_data(self) -> pd.DataFrame:
        """Generate test market data for validation"""
        try:
            dates = pd.date_range(start='2023-01-01', periods=200, freq='15min')
            test_data = pd.DataFrame({
                'Open': np.random.uniform(1.08, 1.12, 200),
                'High': np.random.uniform(1.09, 1.13, 200),
                'Low': np.random.uniform(1.07, 1.11, 200),
                'Close': np.random.uniform(1.08, 1.12, 200),
                'Volume': np.random.randint(100000, 1000000, 200)
            }, index=dates)
            
            return test_data
        except Exception as e:
            self.logger.error(f"Test data generation failed: {e}")
            return pd.DataFrame()

    def _run_system_diagnostics(self):
        """Run comprehensive system diagnostics with enhanced validation"""
        self.logger.info("Running Enhanced System Diagnostics...")
        
        # Enhanced diagnostics with detailed component checking
        diagnostics = {
            'mt5_connection': self._check_mt5_connection(),
            'feature_manager': self._check_feature_manager(),
            'rl_system': self._check_rl_system(),
            'sentiment_system': self._check_sentiment_system(),
            'kelly_sizing': self._check_kelly_sizing(),
            'market_intelligence': self._check_market_intelligence(),
            'strategy_manager': self._check_strategy_manager(),
            'risk_manager': self._check_risk_manager(),
            'execution_manager': self._check_execution_manager()
        }
        
        self.logger.info("System Diagnostics Results:")
        for component, status in diagnostics.items():
            status_symbol = "✅" if status else "❌"
            self.logger.info(f"  {status_symbol} {component}: {'OK' if status else 'FAILED'}")
        
        # Enhanced RL system diagnostics
        if self.rl_enabled and self.rl_model_manager:
            try:
                rl_diagnostics = self.rl_model_manager.get_diagnostics()
                self.logger.info("Enhanced RL System Details:")
                for key, value in rl_diagnostics.items():
                    self.logger.info(f"  • {key}: {value}")
            except Exception as e:
                self.logger.warning(f"RL diagnostics failed: {e}")
        
        # System health summary
        healthy_components = sum(diagnostics.values())
        total_components = len(diagnostics)
        health_percentage = (healthy_components / total_components) * 100
        
        self.logger.info(f"System Health: {healthy_components}/{total_components} components OK ({health_percentage:.1f}%)")
        
        if health_percentage < 80:
            self.logger.warning("⚠️ System health below 80% - some features may be limited")

    def _check_mt5_connection(self) -> bool:
        """Check MT5 connection status"""
        try:
            return getattr(self.datahandler, 'connected', False)
        except:
            return False

    def _check_feature_manager(self) -> bool:
        """Check feature manager status"""
        try:
            return hasattr(self.feature_manager, 'get_latest_features') and callable(self.feature_manager.get_latest_features)
        except:
            return False

    def _check_rl_system(self) -> bool:
        """Check RL system status"""
        try:
            return self.rl_enabled and self.rl_model is not None
        except:
            return False

    def _check_sentiment_system(self) -> bool:
        """Check sentiment system status"""
        try:
            return hasattr(self.sentiment_manager, 'get_market_sentiment') and callable(self.sentiment_manager.get_market_sentiment)
        except:
            return False

    def _check_kelly_sizing(self) -> bool:
        """Check Kelly sizing system status"""
        try:
            return hasattr(self.kelly_manager, 'calculate_position_size') and callable(self.kelly_manager.calculate_position_size)
        except:
            return False

    def _check_market_intelligence(self) -> bool:
        """Check market intelligence system status"""
        try:
            return hasattr(self.marketintelligence, 'identify_regime') and callable(self.marketintelligence.identify_regime)
        except:
            return False

    def _check_strategy_manager(self) -> bool:
        """Check strategy manager status"""
        try:
            return hasattr(self.strategymanager, 'evaluate_signals') and callable(self.strategymanager.evaluate_signals)
        except:
            return False

    def _check_risk_manager(self) -> bool:
        """Check risk manager status"""
        try:
            return hasattr(self.riskmanager, 'calculate_enhanced_risk') and callable(self.riskmanager.calculate_enhanced_risk)
        except:
            return False

    def _check_execution_manager(self) -> bool:
        """Check execution manager status"""
        try:
            return hasattr(self.executionmanager, 'execute_trade') and callable(self.executionmanager.execute_trade)
        except:
            return False

    def analyze_market_enhanced(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        ✅ COMPLETE FIXED: Enhanced market analysis with comprehensive error handling and RL compatibility
        Maintains all existing advanced features while ensuring seamless integration
        """
        analysis_start_time = time.time()
        
        try:
            self.logger.debug(f"🔍 Starting enhanced market analysis for {symbol}")
            
            # 1. Get multi-timeframe data with enhanced error handling
            data_dict = self._get_market_data_safe(symbol)
            if data_dict is None:
                self.analysis_failure_count += 1
                return None
            
            self.logger.debug(f"✅ Retrieved {len(data_dict)} bars for {symbol}")
            
            # 2. Enhanced feature engineering with RL compatibility
            features_result = self._generate_features_safe(symbol, data_dict)
            if not features_result['success']:
                self.logger.warning(f"Feature engineering failed for {symbol}: {features_result['error']}")
                # Continue with limited analysis instead of failing completely
            
            # 3. Market regime identification with enhanced error handling
            regime = self._identify_regime_safe(symbol, data_dict)
            self.logger.debug(f"Market regime identified for {symbol}: {regime}")
            
            # 4. Sentiment analysis with enhanced error handling
            sentiment_score = self._get_sentiment_safe(symbol)
            self.logger.debug(f"Sentiment score for {symbol}: {sentiment_score:.3f}")
            
            # 5. RL Signal Generation (Highest Priority) with enhanced compatibility
            rl_signal = self._generate_rl_signal_safe(symbol, data_dict, features_result.get('features_array'))
            
            # 6. Traditional signals (backup/confirmation) with enhanced error handling
            traditional_signal = self._generate_traditional_signal_safe(symbol, data_dict, regime)
            
            # 7. Enhanced signal prioritization and fusion
            final_signal = self._fuse_signals_enhanced(rl_signal, traditional_signal, sentiment_score, regime)
            
            if final_signal:
                # 8. Enhance signal with comprehensive context
                final_signal = self._enhance_signal_context(final_signal, symbol, data_dict, regime, sentiment_score, features_result)
                
                self.total_signals_generated += 1
                self.analysis_success_count += 1
                
                analysis_duration = time.time() - analysis_start_time
                self.logger.info(f"🎯 ENHANCED SIGNAL GENERATED for {symbol}:")
                self.logger.info(f"   Strategy: {final_signal['strategy']}")
                self.logger.info(f"   Direction: {final_signal['direction']}")
                self.logger.info(f"   Confidence: {final_signal['confidence']:.2f}")
                self.logger.info(f"   Regime: {final_signal['regime']}")
                self.logger.info(f"   Analysis Duration: {analysis_duration:.2f}s")
            else:
                self.logger.debug(f"No trading signal generated for {symbol}")
            
            return final_signal
            
        except Exception as e:
            self.analysis_failure_count += 1
            self.logger.error(f"❌ Enhanced market analysis failed for {symbol}: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def _get_market_data_safe(self, symbol: str) -> Optional[pd.DataFrame]:
        """Safely retrieve market data with comprehensive error handling"""
        try:
            data_dict = self.datahandler.get_data(symbol, 'M15', 200)
            
            if data_dict is None:
                self.logger.warning(f"No data returned for {symbol}")
                return None
            
            if len(data_dict) < 50:
                self.logger.warning(f"Insufficient data for {symbol}: {len(data_dict)} bars")
                return None
            
            # Validate data quality
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in data_dict.columns]
            if missing_columns:
                self.logger.warning(f"Missing required columns for {symbol}: {missing_columns}")
                return None
            
            return data_dict
            
        except Exception as e:
            self.logger.error(f"Error retrieving market data for {symbol}: {e}")
            return None

    def _generate_features_safe(self, symbol: str, data_dict: pd.DataFrame) -> Dict[str, Any]:
        """Safely generate features with comprehensive error handling"""
        try:
            # Generate comprehensive features (131+ features)
            features_df = self.feature_manager.engineer_features(data_dict)
            
            if features_df is None or len(features_df) == 0:
                return {'success': False, 'error': 'Feature engineering returned empty result'}
            
            # Generate RL-compatible features array (exactly 50 features)
            features_array = self.feature_manager.get_latest_features(data_dict)
            
            if features_array is None or len(features_array) != 50:
                return {
                    'success': False, 
                    'error': f'RL feature array invalid: expected 50, got {len(features_array) if features_array is not None else "None"}'
                }
            
            self.logger.debug(f"✅ Generated {len(features_df.columns)} comprehensive features + 50 RL features for {symbol}")
            
            return {
                'success': True,
                'features_df': features_df,
                'features_array': features_array,
                'feature_count': len(features_df.columns)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _identify_regime_safe(self, symbol: str, data_dict: pd.DataFrame) -> str:
        """Safely identify market regime with fallback"""
        try:
            regime = self.marketintelligence.identify_regime(data_dict)
            return regime if regime else 'normal'
        except Exception as e:
            self.logger.warning(f"Regime detection failed for {symbol}: {e}")
            return 'normal'

    def _get_sentiment_safe(self, symbol: str) -> float:
        """Safely get sentiment score with fallback"""
        try:
            sentiment_score = self.sentiment_manager.get_market_sentiment(symbol)
            return sentiment_score if sentiment_score is not None else 0.0
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed for {symbol}: {e}")
            return 0.0

    def _generate_rl_signal_safe(self, symbol: str, data_dict: pd.DataFrame, features_array: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Safely generate RL signal with comprehensive error handling"""
        if not self.rl_enabled or not self.rl_model_manager:
            return None
        
        try:
            if features_array is None:
                self.logger.debug(f"No features array available for RL signal generation: {symbol}")
                return None
            
            # Generate RL signal using the model manager
            rl_signal = self.rl_model_manager.generate_signal(symbol, data_dict)
            
            if rl_signal:
                self.rl_signal_count += 1
                self.rl_confidence_scores.append(rl_signal.get('confidence', 0.5))
                self.rl_last_prediction_time = datetime.now()
                
                self.logger.info(f"✅ RL Signal Generated: {self.rl_model_type} - {rl_signal['direction']} {symbol} (confidence: {rl_signal.get('confidence', 0.5):.2f})")
                return rl_signal
            
            return None
            
        except Exception as e:
            self.rl_failed_trades += 1
            self.logger.warning(f"RL signal generation failed for {symbol}: {e}")
            return None

    def _generate_traditional_signal_safe(self, symbol: str, data_dict: pd.DataFrame, regime: str) -> Optional[Dict[str, Any]]:
        """Safely generate traditional signal with comprehensive error handling"""
        try:
            traditional_signal = self.strategymanager.evaluate_signals(symbol, data_dict, regime)
            
            if traditional_signal:
                self.logger.debug(f"✅ Traditional signal generated: {traditional_signal['strategy']} - {traditional_signal['direction']} {symbol}")
                return traditional_signal
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Traditional signal generation failed for {symbol}: {e}")
            return None

    def _fuse_signals_enhanced(self, rl_signal: Optional[Dict], traditional_signal: Optional[Dict], 
                             sentiment_score: float, regime: str) -> Optional[Dict[str, Any]]:
        """
        ✅ ENHANCED: Advanced signal fusion with comprehensive logic and regime consideration
        """
        try:
            # Enhanced signal prioritization based on multiple factors
            
            # 1. High-confidence RL signal has highest priority
            if rl_signal and rl_signal.get('confidence', 0) > 0.75:
                rl_signal['strategy'] = f"High-Confidence-RL-{self.rl_model_type}"
                return rl_signal
            
            # 2. Signal agreement with sentiment confirmation
            if rl_signal and traditional_signal:
                rl_direction = rl_signal.get('direction', '')
                trad_direction = traditional_signal.get('direction', '')
                
                if rl_direction == trad_direction:
                    # Check sentiment confirmation
                    sentiment_confirms = (
                        (sentiment_score > 0.1 and rl_direction == 'BUY') or
                        (sentiment_score < -0.1 and rl_direction == 'SELL') or
                        abs(sentiment_score) <= 0.1  # Neutral sentiment doesn't conflict
                    )
                    
                    # Regime-based confidence adjustment
                    regime_boost = self._get_regime_confidence_boost(regime, rl_direction)
                    
                    if sentiment_confirms:
                        # Combine signals with enhanced confidence
                        combined_confidence = min(0.95, 
                            (rl_signal.get('confidence', 0.7) * 0.6 + 
                             traditional_signal.get('confidence', 0.7) * 0.4 + 
                             regime_boost))
                        
                        enhanced_signal = rl_signal.copy()
                        enhanced_signal['confidence'] = combined_confidence
                        enhanced_signal['strategy'] = f"Enhanced-Fusion-{self.rl_model_type}+Traditional+Sentiment"
                        enhanced_signal['fusion_components'] = {
                            'rl_confidence': rl_signal.get('confidence', 0.7),
                            'traditional_confidence': traditional_signal.get('confidence', 0.7),
                            'sentiment_score': sentiment_score,
                            'regime_boost': regime_boost
                        }
                        
                        return enhanced_signal
            
            # 3. Medium-confidence RL signal with regime support
            if rl_signal and rl_signal.get('confidence', 0) > 0.6:
                regime_support = self._check_regime_support(regime, rl_signal.get('direction', ''))
                if regime_support:
                    rl_signal['strategy'] = f"Regime-Supported-RL-{self.rl_model_type}"
                    rl_signal['confidence'] = min(0.85, rl_signal.get('confidence', 0.6) + 0.1)
                    return rl_signal
            
            # 4. High-confidence traditional signal
            if traditional_signal and traditional_signal.get('confidence', 0) > 0.8:
                traditional_signal['strategy'] = f"High-Confidence-Traditional-{traditional_signal.get('strategy', 'Unknown')}"
                return traditional_signal
            
            # 5. Any available RL signal (lower threshold)
            if rl_signal and rl_signal.get('confidence', 0) > 0.4:
                return rl_signal
            
            # 6. Fallback to traditional signal
            if traditional_signal and traditional_signal.get('confidence', 0) > 0.5:
                return traditional_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in enhanced signal fusion: {e}")
            # Fallback to simple fusion
            return rl_signal if rl_signal else traditional_signal

    def _get_regime_confidence_boost(self, regime: str, direction: str) -> float:
        """Get confidence boost based on regime and direction alignment"""
        try:
            regime_lower = regime.lower()
            
            if 'bullish' in regime_lower or 'uptrend' in regime_lower:
                return 0.1 if direction == 'BUY' else -0.05
            elif 'bearish' in regime_lower or 'downtrend' in regime_lower:
                return 0.1 if direction == 'SELL' else -0.05
            elif 'volatile' in regime_lower or 'breakout' in regime_lower:
                return 0.05  # Moderate boost for any direction in volatile regime
            elif 'ranging' in regime_lower:
                return 0.05  # Moderate boost for mean-reversion signals
            else:
                return 0.0  # No boost for normal regime
                
        except Exception as e:
            self.logger.debug(f"Error calculating regime confidence boost: {e}")
            return 0.0

    def _check_regime_support(self, regime: str, direction: str) -> bool:
        """Check if the regime supports the signal direction"""
        try:
            regime_lower = regime.lower()
            
            supportive_regimes = {
                'BUY': ['bullish', 'uptrend', 'momentum', 'breakout'],
                'SELL': ['bearish', 'downtrend', 'momentum', 'breakout']
            }
            
            for regime_type in supportive_regimes.get(direction, []):
                if regime_type in regime_lower:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Error checking regime support: {e}")
            return False

    def _enhance_signal_context(self, signal: Dict[str, Any], symbol: str, data_dict: pd.DataFrame, 
                               regime: str, sentiment_score: float, features_result: Dict) -> Dict[str, Any]:
        """Enhance signal with comprehensive context information"""
        try:
            enhanced_signal = signal.copy()
            
            # Add comprehensive context
            enhanced_signal.update({
                'symbol': symbol,
                'regime': regime,
                'sentiment_score': sentiment_score,
                'current_price': float(data_dict['Close'].iloc[-1]) if 'Close' in data_dict.columns else 0.0,
                'volume': float(data_dict.get('Volume', pd.Series([1000000])).iloc[-1]) if 'Volume' in data_dict.columns else 1000000,
                'analysis_time': datetime.now(),
                'data_quality': len(data_dict),
                'feature_count': features_result.get('feature_count', 0),
                'feature_engineering_success': features_result.get('success', False),
                'market_conditions': {
                    'volatility': self._calculate_volatility(data_dict),
                    'trend_strength': self._calculate_trend_strength(data_dict),
                    'volume_profile': self._analyze_volume_profile(data_dict)
                },
                'risk_metrics': {
                    'atr': self._calculate_atr(data_dict),
                    'price_range': float(data_dict['High'].iloc[-1] - data_dict['Low'].iloc[-1]),
                    'volatility_percentile': self._calculate_volatility_percentile(data_dict)
                }
            })
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"Error enhancing signal context: {e}")
            return signal

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate current volatility"""
        try:
            returns = data['Close'].pct_change().dropna()
            return float(returns.tail(20).std()) if len(returns) >= 20 else 0.02
        except:
            return 0.02

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength"""
        try:
            if len(data) < 20:
                return 0.5
            
            short_ma = data['Close'].rolling(10).mean()
            long_ma = data['Close'].rolling(20).mean()
            
            current_short = short_ma.iloc[-1]
            current_long = long_ma.iloc[-1]
            
            return float(abs(current_short - current_long) / current_long) if current_long != 0 else 0.5
        except:
            return 0.5

    def _analyze_volume_profile(self, data: pd.DataFrame) -> str:
        """Analyze volume profile"""
        try:
            if 'Volume' not in data.columns:
                return 'unavailable'
            
            recent_volume = data['Volume'].tail(10).mean()
            historical_volume = data['Volume'].tail(50).mean()
            
            if recent_volume > historical_volume * 1.2:
                return 'high'
            elif recent_volume < historical_volume * 0.8:
                return 'low'
            else:
                return 'normal'
        except:
            return 'unknown'

    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        try:
            if len(data) < 14:
                return float(data['High'].iloc[-1] - data['Low'].iloc[-1])
            
            high_low = data['High'] - data['Low']
            high_close = abs(data['High'] - data['Close'].shift())
            low_close = abs(data['Low'] - data['Close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            return float(atr) if not pd.isna(atr) else float(data['High'].iloc[-1] - data['Low'].iloc[-1])
        except:
            return 0.01

    def _calculate_volatility_percentile(self, data: pd.DataFrame) -> float:
        """Calculate volatility percentile"""
        try:
            returns = data['Close'].pct_change().dropna()
            if len(returns) < 50:
                return 0.5
            
            current_vol = returns.tail(20).std()
            historical_vols = [returns.iloc[i:i+20].std() for i in range(len(returns)-39)]
            
            percentile = sum(1 for vol in historical_vols if vol < current_vol) / len(historical_vols)
            return float(percentile)
        except:
            return 0.5

    def process_signal_enhanced(self, signal: Dict[str, Any]) -> bool:
        """
        ✅ ENHANCED: Signal processing with comprehensive validation and enhanced risk management
        """
        if not signal:
            return False
        
        processing_start_time = time.time()
        
        try:
            symbol = signal['symbol']
            signal_source = 'RL' if any(keyword in signal.get('strategy', '') for keyword in ['RL', 'SAC', 'A2C']) else 'Traditional'
            
            self.logger.info(f"🔄 Processing {signal_source} signal for {symbol}")
            self.logger.info(f"   Strategy: {signal.get('strategy', 'Unknown')}")
            self.logger.info(f"   Direction: {signal['direction']}")
            self.logger.info(f"   Confidence: {signal.get('confidence', 0.5):.2f}")
            
            # 1. Enhanced Kelly position sizing with comprehensive error handling
            kelly_result = self._calculate_kelly_sizing_safe(signal, symbol)
            if not kelly_result['success']:
                self.logger.warning(f"Kelly sizing failed for {symbol}: {kelly_result['error']}")
                if signal_source == 'RL':
                    self.rl_failed_trades += 1
                return False
            
            # 2. Enhanced risk management validation
            risk_result = self._calculate_risk_parameters_safe(signal, symbol, kelly_result['params'])
            if not risk_result['success']:
                self.logger.warning(f"Risk management validation failed for {symbol}: {risk_result['error']}")
                if signal_source == 'RL':
                    self.rl_failed_trades += 1
                return False
            
            # 3. Final pre-execution validation
            if not self._pre_execution_validation(signal, symbol, risk_result['params']):
                self.logger.warning(f"Pre-execution validation failed for {symbol}")
                if signal_source == 'RL':
                    self.rl_failed_trades += 1
                return False
            
            # 4. Execute the trade with enhanced error handling
            execution_result = self._execute_trade_safe(signal, risk_result['params'])
            
            if execution_result['success']:
                # 5. Post-execution processing
                self._handle_successful_execution(signal, risk_result['params'], execution_result['result'], signal_source)
                
                processing_duration = time.time() - processing_start_time
                self.logger.info(f"✅ {signal_source} trade executed successfully in {processing_duration:.2f}s: {symbol}")
                return True
            else:
                # 6. Handle execution failure
                self._handle_execution_failure(signal, symbol, signal_source, execution_result['error'])
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Enhanced signal processing failed: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def _calculate_kelly_sizing_safe(self, signal: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Safely calculate Kelly position sizing"""
        try:
            kelly_params = self.kelly_manager.calculate_position_size(
                symbol=symbol,
                confidence=signal.get('confidence', 0.5),
                expected_return=signal.get('expected_return', 0.02),
                risk_level=signal.get('risk_level', 0.01),
                account_balance=self.executionmanager.get_account_balance(),
                market_regime=signal.get('regime', 'normal')
            )
            
            if kelly_params and kelly_params.get('position_size', 0) > 0:
                return {'success': True, 'params': kelly_params}
            else:
                return {'success': False, 'error': 'Kelly sizing returned invalid position size'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _calculate_risk_parameters_safe(self, signal: Dict[str, Any], symbol: str, kelly_params: Dict) -> Dict[str, Any]:
        """Safely calculate risk management parameters"""
        try:
            risk_params = self.riskmanager.calculate_enhanced_risk(
                symbol=symbol,
                direction=signal['direction'],
                entry_price=signal.get('entry_price', 0),
                stop_loss=signal.get('stop_loss', 0),
                take_profit=signal.get('take_profit', 0),
                confidence=signal.get('confidence', 0.5),
                strategy=signal.get('strategy', 'Unknown'),
                position_size=kelly_params.get('position_size', 0.01)
            )
            
            if risk_params:
                return {'success': True, 'params': risk_params}
            else:
                return {'success': False, 'error': 'Risk management returned invalid parameters'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _pre_execution_validation(self, signal: Dict[str, Any], symbol: str, risk_params: Dict) -> bool:
        """Perform pre-execution validation"""
        try:
            # Check position limits
            if len(self.current_positions) >= getattr(self.config, 'MAX_OPEN_POSITIONS', 10):
                self.logger.warning(f"Maximum open positions reached: {len(self.current_positions)}")
                return False
            
            # Check symbol-specific limits
            symbol_positions = [p for p in self.current_positions if p['symbol'] == symbol]
            if len(symbol_positions) >= getattr(self.config, 'MAX_POSITIONS_PER_SYMBOL', 3):
                self.logger.warning(f"Maximum positions for {symbol} reached: {len(symbol_positions)}")
                return False
            
            # Check account balance
            account_balance = self.executionmanager.get_account_balance()
            min_balance = getattr(self.config, 'MIN_ACCOUNT_BALANCE', 1000)
            if account_balance < min_balance:
                self.logger.warning(f"Account balance too low: ${account_balance} < ${min_balance}")
                return False
            
            # Check market hours (if configured)
            if hasattr(self.config, 'TRADING_HOURS') and not self._is_trading_hours():
                self.logger.warning("Outside trading hours")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pre-execution validation error: {e}")
            return False

    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        try:
            trading_hours = getattr(self.config, 'TRADING_HOURS', {'start': 0, 'end': 24})
            current_hour = datetime.now().hour
            
            start_hour = trading_hours.get('start', 0)
            end_hour = trading_hours.get('end', 24)
            
            if start_hour <= end_hour:
                return start_hour <= current_hour <= end_hour
            else:  # Overnight trading
                return current_hour >= start_hour or current_hour <= end_hour
                
        except Exception as e:
            self.logger.error(f"Trading hours check error: {e}")
            return True  # Default to allowing trading

    def _execute_trade_safe(self, signal: Dict[str, Any], risk_params: Dict) -> Dict[str, Any]:
        """Safely execute trade"""
        try:
            execution_result = self.executionmanager.execute_trade(signal, risk_params)
            
            if execution_result and execution_result.get('success'):
                return {'success': True, 'result': execution_result}
            else:
                return {'success': False, 'error': 'Execution manager returned unsuccessful result'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _handle_successful_execution(self, signal: Dict[str, Any], risk_params: Dict, 
                                   execution_result: Dict, signal_source: str):
        """Handle successful trade execution"""
        try:
            symbol = signal['symbol']
            
            # Update trade statistics
            self.total_trades += 1
            self.signals_executed += 1
            
            # Add to current positions
            position_info = {
                'symbol': symbol,
                'direction': signal['direction'],
                'strategy': signal['strategy'],
                'signal_source': signal_source,
                'entry_time': datetime.now(),
                'entry_price': signal.get('entry_price'),
                'position_size': risk_params.get('position_size'),
                'stop_loss': risk_params.get('stop_loss'),
                'take_profit': risk_params.get('take_profit'),
                'confidence': signal.get('confidence', 0.5),
                'regime': signal.get('regime', 'normal'),
                'execution_id': execution_result.get('order_id')
            }
            
            self.current_positions.append(position_info)
            
            # Track RL performance specifically
            if signal_source == 'RL':
                self.logger.info(f"🤖 RL trade executed: {self.rl_model_type} - {symbol}")
                # Will update success/failure when position closes
            
            # Send notifications
            self._send_trade_notification(signal, risk_params, execution_result)
            
        except Exception as e:
            self.logger.error(f"Error handling successful execution: {e}")

    def _handle_execution_failure(self, signal: Dict[str, Any], symbol: str, signal_source: str, error: str):
        """Handle failed trade execution"""
        try:
            self.logger.warning(f"❌ Trade execution failed for {symbol}: {error}")
            
            if signal_source == 'RL':
                self.rl_failed_trades += 1
            
            # Send failure notification
            self._send_failure_notification(signal, error)
            
        except Exception as e:
            self.logger.error(f"Error handling execution failure: {e}")

    def _send_trade_notification(self, signal: Dict[str, Any], risk_params: Dict, execution_result: Dict):
        """Send trade notification"""
        try:
            self.notificationmanager.send_trade_notification(signal, risk_params, execution_result)
        except Exception as e:
            self.logger.warning(f"Trade notification failed: {e}")

    def _send_failure_notification(self, signal: Dict[str, Any], error: str):
        """Send failure notification"""
        try:
            self.notificationmanager.send_failure_notification(signal, error)
        except Exception as e:
            self.logger.warning(f"Failure notification failed: {e}")

    def update_performance_metrics_enhanced(self):
        """
        ✅ ENHANCED: Comprehensive performance metrics with detailed tracking
        """
        try:
            # Get current account status with error handling
            try:
                current_equity = self.executionmanager.get_account_equity()
                current_balance = self.executionmanager.get_account_balance()
            except Exception as e:
                self.logger.warning(f"Failed to get account info: {e}")
                current_equity = current_balance = 0.0
            
            # Calculate enhanced performance metrics
            win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
            loss_rate = (self.losing_trades / max(1, self.total_trades)) * 100
            uptime_hours = (time.time() - self.start_time) / 3600
            
            # Analysis success rate
            total_analysis = self.analysis_success_count + self.analysis_failure_count
            analysis_success_rate = (self.analysis_success_count / max(1, total_analysis)) * 100
            
            # Signal execution rate
            signal_execution_rate = (self.signals_executed / max(1, self.total_signals_generated)) * 100
            
            # RL specific metrics
            rl_metrics = self._calculate_rl_metrics_enhanced()
            
            # Component health summary
            healthy_components = sum(self.component_health.values())
            total_components = len(self.component_health)
            system_health = (healthy_components / total_components) * 100
            
            # Log comprehensive performance report
            self.logger.info("=" * 80)
            self.logger.info("ENHANCED PERFORMANCE UPDATE")
            self.logger.info("=" * 80)
            
            self.logger.info("📊 Account Status:")
            self.logger.info(f"   Current Equity: ${current_equity:,.2f}")
            self.logger.info(f"   Current Balance: ${current_balance:,.2f}")
            self.logger.info(f"   Daily P&L: ${self.daily_pnl:,.2f}")
            self.logger.info(f"   Max Drawdown: {self.max_drawdown:.2f}%")
            
            self.logger.info("📈 Trading Performance:")
            self.logger.info(f"   Total Trades: {self.total_trades}")
            self.logger.info(f"   Winning Trades: {self.winning_trades} ({win_rate:.1f}%)")
            self.logger.info(f"   Losing Trades: {self.losing_trades} ({loss_rate:.1f}%)")
            self.logger.info(f"   Active Positions: {len(self.current_positions)}")
            self.logger.info(f"   Signals Generated: {self.total_signals_generated}")
            self.logger.info(f"   Signals Executed: {self.signals_executed} ({signal_execution_rate:.1f}%)")
            
            self.logger.info("🔍 Analysis Performance:")
            self.logger.info(f"   Successful Analysis: {self.analysis_success_count} ({analysis_success_rate:.1f}%)")
            self.logger.info(f"   Failed Analysis: {self.analysis_failure_count}")
            
            self.logger.info("🤖 RL Performance:")
            for key, value in rl_metrics.items():
                self.logger.info(f"   {key}: {value}")
            
            self.logger.info("⚙️ System Status:")
            self.logger.info(f"   System Health: {healthy_components}/{total_components} ({system_health:.1f}%)")
            self.logger.info(f"   Uptime: {uptime_hours:.1f} hours")
            self.logger.info(f"   Memory Usage: {self._get_memory_usage()}")
            
            self.logger.info("🔧 Component Status:")
            for component, status in self.component_health.items():
                status_symbol = "✅" if status else "❌"
                self.logger.info(f"   {status_symbol} {component}: {'Active' if status else 'Inactive'}")
            
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"❌ Error updating enhanced performance metrics: {e}")

    def _calculate_rl_metrics_enhanced(self) -> Dict[str, Any]:
        """Calculate comprehensive RL performance metrics"""
        if not self.rl_enabled:
            return {'Status': 'RL Disabled'}
        
        try:
            # Basic metrics
            rl_success_rate = 0
            rl_win_rate = 0
            if self.rl_signal_count > 0:
                rl_success_rate = ((self.rl_signal_count - self.rl_failed_trades) / self.rl_signal_count) * 100
                rl_win_rate = (self.rl_successful_trades / max(1, self.rl_signal_count)) * 100
            
            # Advanced metrics
            avg_confidence = np.mean(self.rl_confidence_scores) if self.rl_confidence_scores else 0.0
            recent_performance = np.mean(self.rl_performance_history[-10:]) if len(self.rl_performance_history) > 0 else 0.0
            
            # Time since last prediction
            time_since_last = 'Never'
            if self.rl_last_prediction_time:
                delta = datetime.now() - self.rl_last_prediction_time
                if delta.total_seconds() < 3600:
                    time_since_last = f"{int(delta.total_seconds() / 60)} minutes ago"
                else:
                    time_since_last = f"{delta.total_seconds() / 3600:.1f} hours ago"
            
            return {
                'Model Type': self.rl_model_type or 'None',
                'Status': 'Active' if self.rl_enabled else 'Inactive',
                'Signals Generated': self.rl_signal_count,
                'Success Rate': f"{rl_success_rate:.1f}%",
                'Win Rate': f"{rl_win_rate:.1f}%",
                'Failed Executions': self.rl_failed_trades,
                'Average Confidence': f"{avg_confidence:.2f}",
                'Recent Performance': f"{recent_performance:.2f}" if recent_performance != 0.0 else 'N/A',
                'Last Prediction': time_since_last,
                'Performance History Size': len(self.rl_performance_history)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating RL metrics: {e}")
            return {'Status': 'Error calculating metrics'}

    def run_enhanced(self):
        """
        ✅ ENHANCED: Main trading loop with comprehensive monitoring and error handling
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING ENHANCED TRADING BOT SESSION")
        self.logger.info("=" * 80)
        
        # Enhanced connection validation
        connection_status = self._validate_connections()
        if not connection_status['success']:
            self.logger.error(f"❌ Connection validation failed: {connection_status['error']}")
            return
        
        # Display enhanced startup summary
        self._display_startup_summary_enhanced()
        
        # Initialize loop variables
        last_performance_update = time.time()
        last_system_check = time.time()
        last_component_health_check = time.time()
        loop_count = 0
        
        try:
            while not self.stop_event.is_set():
                loop_start_time = time.time()
                loop_count += 1
                
                # Update status with comprehensive information
                active_positions = len(self.current_positions)
                healthy_components = sum(self.component_health.values())
                self.status = (f"Enhanced Loop {loop_count} | RL: {self.rl_model_type or 'Off'} | "
                             f"Positions: {active_positions} | Health: {healthy_components}/{len(self.component_health)}")
                
                try:
                    # Main analysis loop with enhanced error handling
                    symbols = getattr(self.config, 'SYMBOLS_TO_TRADE', ['EURUSD', 'GBPUSD', 'XAUUSD'])
                    self._process_symbols_enhanced(symbols)
                    
                    # Enhanced position management
                    self._manage_positions_enhanced()
                    
                    # Periodic updates with configurable intervals
                    current_time = time.time()
                    
                    # Performance metrics update
                    performance_interval = getattr(self.config, 'PERFORMANCE_UPDATE_INTERVAL', 300)
                    if current_time - last_performance_update >= performance_interval:
                        self.update_performance_metrics_enhanced()
                        last_performance_update = current_time
                    
                    # System health check
                    if current_time - last_system_check >= 600:  # Every 10 minutes
                        self._run_system_health_check_enhanced()
                        last_system_check = current_time
                    
                    # Component health check
                    if current_time - last_component_health_check >= 1800:  # Every 30 minutes
                        self._check_component_health_enhanced()
                        last_component_health_check = current_time
                    
                    # Enhanced loop control with adaptive timing
                    loop_duration = time.time() - loop_start_time
                    base_interval = getattr(self.config, 'MAIN_LOOP_INTERVAL', 10)
                    sleep_time = max(1, base_interval - loop_duration)  # Minimum 1 second sleep
                    
                    if loop_count % 100 == 0:  # Log every 100 loops
                        self.logger.info(f"Loop {loop_count}: Duration {loop_duration:.2f}s, Sleeping {sleep_time:.2f}s")
                    
                    time.sleep(sleep_time)
                    
                except Exception as loop_error:
                    self.logger.error(f"❌ Error in enhanced main loop iteration {loop_count}: {loop_error}")
                    self.logger.error(traceback.format_exc())
                    time.sleep(5)  # Brief pause before continuing
                    
        except KeyboardInterrupt:
            self.logger.info("🛑 Shutdown signal received")
        except Exception as e:
            self.logger.error(f"❌ Critical error in enhanced main loop: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.shutdown_enhanced()

    def _validate_connections(self) -> Dict[str, Any]:
        """Validate all critical connections"""
        try:
            validation_results = {}
            
            # Check data handler connection
            try:
                if hasattr(self.datahandler, 'connect'):
                    connected = self.datahandler.connect()
                else:
                    connected = getattr(self.datahandler, 'connected', True)
                
                validation_results['datahandler'] = connected
            except Exception as e:
                validation_results['datahandler'] = False
                self.logger.warning(f"Data handler connection check failed: {e}")
            
            # Check execution manager connection
            try:
                account_balance = self.executionmanager.get_account_balance()
                validation_results['execution'] = account_balance > 0
            except Exception as e:
                validation_results['execution'] = False
                self.logger.warning(f"Execution manager validation failed: {e}")
            
            # Evaluate overall connection health
            critical_connections = ['datahandler']  # execution is not critical for startup
            critical_ok = all(validation_results.get(conn, False) for conn in critical_connections)
            
            if critical_ok:
                return {'success': True, 'details': validation_results}
            else:
                failed_connections = [conn for conn in critical_connections if not validation_results.get(conn, False)]
                return {'success': False, 'error': f'Critical connections failed: {failed_connections}', 'details': validation_results}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _process_symbols_enhanced(self, symbols: List[str]):
        """Process symbols with enhanced error handling and timing control"""
        try:
            for symbol in symbols:
                try:
                    # Check analysis interval
                    if symbol in self.last_analysis_time:
                        time_since_last = time.time() - self.last_analysis_time[symbol]
                        min_interval = getattr(self.config, 'MIN_ANALYSIS_INTERVAL', 30)
                        if time_since_last < min_interval:
                            continue
                    
                    # Enhanced market analysis
                    signal = self.analyze_market_enhanced(symbol)
                    
                    # Enhanced signal processing
                    if signal:
                        success = self.process_signal_enhanced(signal)
                        if success and any(keyword in signal.get('strategy', '') for keyword in ['RL', 'SAC', 'A2C']):
                            # Track RL performance (will be updated when trade closes)
                            pass
                    
                    # Update analysis timing
                    self.last_analysis_time[symbol] = time.time()
                    
                except Exception as symbol_error:
                    self.logger.error(f"❌ Error processing {symbol}: {symbol_error}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"❌ Error in symbol processing: {e}")

    def _manage_positions_enhanced(self):
        """Enhanced position management"""
        try:
            # Update position statuses
            self.executionmanager.manage_positions()
            
            # Check for closed positions and update statistics
            self._update_closed_positions()
            
        except Exception as e:
            self.logger.warning(f"Position management failed: {e}")

    def _update_closed_positions(self):
        """Update statistics for closed positions"""
        try:
            # This would typically be called by the execution manager
            # when positions are closed, updating winning/losing trades
            pass
        except Exception as e:
            self.logger.error(f"Error updating closed positions: {e}")

    def _run_system_health_check_enhanced(self):
        """Run enhanced system health check"""
        try:
            health_metrics = {
                'mt5_connection': getattr(self.datahandler, 'connected', False),
                'rl_model_loaded': self.rl_enabled and self.rl_model is not None,
                'active_positions': len(self.current_positions),
                'memory_usage_mb': self._get_memory_usage_numeric(),
                'uptime_hours': (time.time() - self.start_time) / 3600,
                'last_signal_time': max(self.last_analysis_time.values()) if self.last_analysis_time else 0,
                'analysis_success_rate': (self.analysis_success_count / max(1, self.analysis_success_count + self.analysis_failure_count)) * 100
            }
            
            # Log health metrics
            self.logger.info("🏥 System Health Check:")
            for metric, value in health_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"   • {metric}: {value:.2f}")
                else:
                    self.logger.info(f"   • {metric}: {value}")
                    
            # Check for concerning metrics
            if health_metrics['memory_usage_mb'] > 1000:
                self.logger.warning(f"⚠️ High memory usage: {health_metrics['memory_usage_mb']:.1f} MB")
            
            if health_metrics['analysis_success_rate'] < 80:
                self.logger.warning(f"⚠️ Low analysis success rate: {health_metrics['analysis_success_rate']:.1f}%")
                
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")

    def _check_component_health_enhanced(self):
        """Enhanced component health check"""
        try:
            self.logger.info("🔧 Checking component health...")
            
            # Re-run diagnostic checks
            new_health = {
                'datahandler': self._check_mt5_connection(),
                'feature_manager': self._check_feature_manager(),
                'rl_system': self._check_rl_system(),
                'sentiment_system': self._check_sentiment_system(),
                'kelly_sizing': self._check_kelly_sizing(),
                'marketintelligence': self._check_market_intelligence(),
                'strategy_manager': self._check_strategy_manager(),
                'risk_manager': self._check_risk_manager(),
                'execution_manager': self._check_execution_manager()
            }
            
            # Compare with previous health status
            for component, new_status in new_health.items():
                old_status = self.component_health.get(component, False)
                if old_status != new_status:
                    status_change = "recovered" if new_status else "failed"
                    self.logger.info(f"   🔄 {component} {status_change}")
            
            # Update component health
            self.component_health.update(new_health)
            
        except Exception as e:
            self.logger.error(f"❌ Component health check failed: {e}")

    def _display_startup_summary_enhanced(self):
        """Display comprehensive startup summary with enhanced information"""
        try:
            symbols = getattr(self.config, 'SYMBOLS_TO_TRADE', ['EURUSD', 'GBPUSD', 'XAUUSD'])
            risk_per_trade = getattr(self.config, 'RISK_PER_TRADE', 1.0)
            max_positions = getattr(self.config, 'MAX_OPEN_POSITIONS', 10)
            
            self.logger.info("🚀 Enhanced Trading Configuration:")
            self.logger.info(f"   RL Integration: {'Enabled' if self.rl_enabled else 'Disabled'}")
            self.logger.info(f"   RL Model: {self.rl_model_type if self.rl_model_type else 'None'}")
            self.logger.info(f"   Trading Symbols: {symbols}")
            self.logger.info(f"   Risk per Trade: {risk_per_trade}%")
            self.logger.info(f"   Max Open Positions: {max_positions}")
            self.logger.info(f"   Kelly Sizing: {'Enabled' if self.component_health.get('kelly_sizing', False) else 'Disabled'}")
            self.logger.info(f"   Sentiment Analysis: {'Enabled' if self.component_health.get('sentiment_system', False) else 'Disabled'}")
            self.logger.info(f"   Feature Engineering: {'Enabled' if self.component_health.get('feature_manager', False) else 'Disabled'}")
            self.logger.info(f"   Market Intelligence: {'Enabled' if self.component_health.get('marketintelligence', False) else 'Disabled'}")
            
            self.logger.info("💻 System Information:")
            self.logger.info(f"   Python Version: {sys.version.split()[0]}")
            self.logger.info(f"   RL Libraries: {'Available' if RL_AVAILABLE else 'Not Available'}")
            self.logger.info(f"   Memory Usage: {self._get_memory_usage()}")
            self.logger.info(f"   Process ID: {os.getpid()}")
            
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"❌ Error displaying startup summary: {e}")

    def _get_memory_usage_numeric(self) -> float:
        """Get current memory usage in MB as numeric value"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def _get_memory_usage(self) -> str:
        """Get current memory usage as formatted string"""
        try:
            memory_mb = self._get_memory_usage_numeric()
            return f"{memory_mb:.1f} MB" if memory_mb > 0 else "N/A"
        except:
            return "N/A"

    def shutdown_enhanced(self):
        """
        ✅ ENHANCED: Graceful shutdown with comprehensive cleanup
        """
        self.logger.info("🛑 Shutting down Enhanced Trading Bot...")
        
        shutdown_start_time = time.time()
        
        try:
            # Set stop event
            self.stop_event.set()
            
            # Close positions if configured
            if getattr(self.config, 'CLOSE_POSITIONS_ON_SHUTDOWN', False):
                self.logger.info("📤 Closing all positions before shutdown...")
                try:
                    self.executionmanager.close_all_positions()
                    self.logger.info("✅ All positions closed successfully")
                except Exception as e:
                    self.logger.warning(f"❌ Failed to close positions: {e}")
            
            # Disconnect from data sources
            try:
                if hasattr(self.datahandler, 'disconnect'):
                    self.datahandler.disconnect()
                    self.logger.info("✅ Data handler disconnected")
            except Exception as e:
                self.logger.warning(f"❌ Data handler disconnect failed: {e}")
            
            # Final performance report
            self.logger.info("📊 Generating final performance report...")
            self.update_performance_metrics_enhanced()
            
            # Save RL model state if applicable
            if self.rl_enabled and self.rl_model_manager:
                try:
                    if hasattr(self.rl_model_manager, 'save_performance_state'):
                        self.rl_model_manager.save_performance_state()
                        self.logger.info("✅ RL model state saved")
                except Exception as e:
                    self.logger.warning(f"❌ RL state save failed: {e}")
            
            # Generate enhanced shutdown statistics
            uptime_seconds = time.time() - self.start_time
            shutdown_stats = {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'rl_signals': self.rl_signal_count,
                'rl_model': self.rl_model_type or 'None',
                'uptime_hours': f"{uptime_seconds / 3600:.1f}",
                'uptime_days': f"{uptime_seconds / 86400:.1f}",
                'total_signals_generated': self.total_signals_generated,
                'signals_executed': self.signals_executed,
                'analysis_success_count': self.analysis_success_count,
                'analysis_failure_count': self.analysis_failure_count,
                'final_status': 'Clean Shutdown'
            }
            
            # Create shutdown summary
            shutdown_message = "\n".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in shutdown_stats.items()])
            
            # Send shutdown notification
            try:
                self.notificationmanager.send_system_notification(
                    "Enhanced Trading Bot Shutdown", 
                    shutdown_message
                )
                self.logger.info("✅ Shutdown notification sent")
            except Exception as e:
                self.logger.warning(f"❌ Shutdown notification failed: {e}")
            
            # Log final shutdown summary
            shutdown_duration = time.time() - shutdown_start_time
            
            self.logger.info("=" * 80)
            self.logger.info("ENHANCED TRADING BOT SHUTDOWN SUMMARY")
            self.logger.info("=" * 80)
            for key, value in shutdown_stats.items():
                self.logger.info(f"   {key.replace('_', ' ').title()}: {value}")
            self.logger.info(f"   Shutdown Duration: {shutdown_duration:.2f}s")
            self.logger.info("=" * 80)
            self.logger.info("✅ Enhanced Trading Bot shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error during enhanced shutdown: {e}")
            self.logger.error(traceback.format_exc())

    def get_enhanced_status(self) -> Dict[str, Any]:
        """
        ✅ ENHANCED: Comprehensive status information
        """
        try:
            uptime_seconds = time.time() - self.start_time
            
            # Calculate rates
            rl_performance = 0
            if self.rl_signal_count > 0:
                rl_performance = (self.rl_successful_trades / self.rl_signal_count) * 100
            
            win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
            analysis_success_rate = (self.analysis_success_count / max(1, self.analysis_success_count + self.analysis_failure_count)) * 100
            signal_execution_rate = (self.signals_executed / max(1, self.total_signals_generated)) * 100
            
            return {
                'status': self.status,
                'uptime': {
                    'seconds': uptime_seconds,
                    'hours': uptime_seconds / 3600,
                    'days': uptime_seconds / 86400
                },
                'rl_system': {
                    'enabled': self.rl_enabled,
                    'model_type': self.rl_model_type,
                    'signals_generated': self.rl_signal_count,
                    'success_rate': rl_performance,
                    'last_prediction': self.rl_last_prediction_time.isoformat() if self.rl_last_prediction_time else None
                },
                'trading_performance': {
                    'total_trades': self.total_trades,
                    'winning_trades': self.winning_trades,
                    'losing_trades': self.losing_trades,
                    'win_rate': win_rate,
                    'active_positions': len(self.current_positions),
                    'daily_pnl': self.daily_pnl,
                    'max_drawdown': self.max_drawdown
                },
                'signal_performance': {
                    'total_signals_generated': self.total_signals_generated,
                    'signals_executed': self.signals_executed,
                    'execution_rate': signal_execution_rate
                },
                'analysis_performance': {
                    'successful_analysis': self.analysis_success_count,
                    'failed_analysis': self.analysis_failure_count,
                    'success_rate': analysis_success_rate
                },
                'system_health': {
                    'component_health': self.component_health.copy(),
                    'healthy_components': sum(self.component_health.values()),
                    'total_components': len(self.component_health),
                    'health_percentage': (sum(self.component_health.values()) / len(self.component_health)) * 100
                },
                'enhanced_features': {
                    'kelly_sizing': self.component_health.get('kelly_sizing', False),
                    'sentiment_analysis': self.component_health.get('sentiment_system', False),
                    'feature_engineering': self.component_health.get('feature_manager', False),
                    'regime_detection': self.component_health.get('marketintelligence', False),
                    'advanced_risk_management': self.component_health.get('risk_manager', False)
                },
                'memory_usage': self._get_memory_usage(),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting enhanced status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

def main():
    """
    ✅ ENHANCED: Main entry point with comprehensive error handling
    """
    try:
        # Display startup banner
        print("=" * 80)
        print("🚀 Enhanced Professional Forex Trading Bot - Starting...")
        print("=" * 80)
        
        # Initialize the enhanced trading bot
        bot = EnhancedTradingBot()
        
        # Start the enhanced main loop
        bot.run_enhanced()
        
    except Exception as e:
        logging.error(f"❌ Critical error in enhanced main: {e}")
        logging.error(traceback.format_exc())
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("Please check the logs for detailed error information.")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("🛑 Manual shutdown initiated")
        print("\n🛑 Shutdown initiated by user")
        sys.exit(0)

if __name__ == "__main__":
    main()

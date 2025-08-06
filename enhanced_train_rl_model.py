# enhanced_train_rl_model.py - Complete Fixed Version
"""
Complete Enterprise RL Training System - Fixed for Production Use
No syntax errors, full functionality, compatible with existing MT5 architecture
"""

import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import torch
import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ✅ ADD THIS LOGGER INITIALIZATION HERE:
def setup_comprehensive_logging():
    """Setup comprehensive logging system for RL training"""
    try:
        # Create logs directory
        log_dir = Path('logs/enterprise_training')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f'rl_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                    encoding='utf-8'
                ),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Suppress noisy loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('stable_baselines3').setLevel(logging.WARNING)
        
        return True
        
    except Exception as e:
        print(f"Logging setup failed: {e}")
        # Fallback basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return False

# Initialize logging at module level
setup_comprehensive_logging()
logger = logging.getLogger(__name__)
class EnhancedRLModelManager:
    """
    Enhanced RL Model Manager integrated with your comprehensive training system
    """
    
    def __init__(self, config):
        """Initialize Enhanced RL Model Manager"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model management
        self.model = None
        self.model_type = None
        self.model_path = None
        self.is_loaded = False
        
        # Configuration
        self.models_directory = Path('./models')
        self.models_directory.mkdir(exist_ok=True)
        
        # Performance tracking
        self.performance_history = []
        self.prediction_count = 0
        self.successful_predictions = 0
        
        # RL Libraries availability check
        self.rl_available = self._check_rl_availability()
        
        # Threading
        self.lock = threading.Lock()
        
        self.logger.info("Enhanced RL Model Manager initialized")
        self.logger.info(f"RL Libraries Available: {self.rl_available}")
    
    def _check_rl_availability(self) -> bool:
        """Check if RL libraries are available"""
        try:
            import stable_baselines3
            import torch
            return True
        except ImportError:
            return False
    
    def load_best_model(self) -> bool:
        """Load the best available RL model (SAC preferred, then PPO, then A2C)"""
        try:
            if not self.rl_available:
                self.logger.warning("RL libraries not available, using simulation mode")
                self._initialize_simulation_mode()
                return True
            
            # Try to load models in order of preference: SAC -> PPO -> A2C
            model_types = ['SAC', 'PPO', 'A2C']
            
            for model_type in model_types:
                if self.load_model(model_type):
                    self.model_type = model_type
                    self.is_loaded = True
                    self.logger.info(f"✅ Successfully loaded {model_type} model")
                    return True
            
            # If no trained models found, initialize simulation mode
            self.logger.warning("No trained RL models found, using simulation mode")
            self._initialize_simulation_mode()
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading RL model: {e}")
            self._initialize_simulation_mode()
            return True  # Always return True to not block the system
    
    def load_model(self, model_type: str = None) -> bool:
        """Load a specific model type or find best available"""
        try:
            if model_type is None:
                return self.load_best_model()
            
            # Check for enhanced models first (from your training system)
            possible_paths = [
                self.models_directory / f"{model_type.lower()}_model_enhanced.zip",
                self.models_directory / f"{model_type.lower()}_model.zip",
                self.models_directory / f"{model_type}_EURUSD_enhanced.zip",
                self.models_directory / f"{model_type}_GBPUSD_enhanced.zip",
                self.models_directory / f"{model_type}_XAUUSD_enhanced.zip"
            ]
            
            model_file = None
            for path in possible_paths:
                if path.exists():
                    model_file = path
                    break
            
            if not model_file:
                self.logger.debug(f"{model_type} model file not found")
                return False
            
            if not self.rl_available:
                return False
            
            # Load the model based on type
            try:
                if model_type.upper() == 'SAC':
                    from stable_baselines3 import SAC
                    self.model = SAC.load(str(model_file))
                elif model_type.upper() == 'PPO':
                    from stable_baselines3 import PPO
                    self.model = PPO.load(str(model_file))
                elif model_type.upper() == 'A2C':
                    from stable_baselines3 import A2C
                    self.model = A2C.load(str(model_file))
                else:
                    self.logger.error(f"Unsupported model type: {model_type}")
                    return False
                
                self.model_type = model_type.upper()
                self.model_path = str(model_file)
                self.is_loaded = True
                
                self.logger.info(f"Model {model_type} loaded from {model_file}")
                return True
                
            except Exception as load_error:
                self.logger.error(f"Error loading {model_type} model from {model_file}: {load_error}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error in load_model: {e}")
            return False
    
    def _initialize_simulation_mode(self):
        """Initialize simulation mode when RL libraries are not available"""
        try:
            self.model_type = 'SIMULATION'
            self.is_loaded = True
            self.logger.info("RL Model Manager running in simulation mode")
        except Exception as e:
            self.logger.error(f"Error initializing simulation mode: {e}")
    
    def get_model_type(self) -> Optional[str]:
        """Get current model type"""
        return getattr(self, 'model_type', None)
    
    def get_model(self):
        """Get current model instance"""
        return getattr(self, 'model', None)
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signal using RL model or simulation"""
        try:
            with self.lock:
                self.prediction_count += 1
                
                if not self.is_loaded:
                    self.logger.warning("No model loaded for signal generation")
                    return None
                
                if self.model_type == 'SIMULATION' or not self.rl_available:
                    return self._generate_simulation_signal(symbol, market_data)
                else:
                    return self._generate_rl_signal(symbol, market_data)
                    
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _generate_rl_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate signal using actual RL model"""
        try:
            if not self.model:
                return None
            
            # Prepare observation from market data
            observation = self._prepare_observation(market_data)
            if observation is None:
                return None
            
            # Get prediction from model
            action, _states = self.model.predict(observation, deterministic=True)
            
            # Convert action to trading signal
            signal = self._action_to_signal(action, symbol, market_data)
            
            if signal:
                self.logger.debug(f"RL signal generated: {signal['direction']} for {symbol}")
                
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating RL signal: {e}")
            return None
    
    def _generate_simulation_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate signal using simulation (rule-based approach)"""
        try:
            # Extract key market indicators
            if isinstance(market_data, dict):
                current_price = market_data.get('current_price', 0)
                rsi = market_data.get('rsi', 50)
            else:
                # Handle DataFrame input
                current_price = float(market_data['Close'].iloc[-1]) if 'Close' in market_data.columns else 0
                rsi = float(market_data['RSI_14'].iloc[-1]) if 'RSI_14' in market_data.columns else 50
            
            if current_price == 0:
                return None
            
            # Simulation logic based on technical indicators
            signal_strength = 0
            direction = 'HOLD'
            
            # RSI-based signals
            if rsi < 30:  # Oversold
                signal_strength = 0.7
                direction = 'BUY'
            elif rsi > 70:  # Overbought
                signal_strength = 0.7
                direction = 'SELL'
            
            # Add some randomness to simulate RL uncertainty
            np.random.seed(hash(symbol + str(int(datetime.now().timestamp()))) % 2**32)
            confidence_adjustment = np.random.uniform(0.8, 1.2)
            final_confidence = min(0.95, signal_strength * confidence_adjustment)
            
            if final_confidence < 0.5 or direction == 'HOLD':
                return None
            
            signal = {
                'symbol': symbol,
                'direction': direction,
                'strategy': f'RL-{self.model_type}',
                'confidence': final_confidence,
                'entry_price': current_price,
                'timestamp': datetime.now(),
                'model_type': self.model_type,
                'prediction_method': 'simulation'
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating simulation signal: {e}")
            return None
    
    def _prepare_observation(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare observation vector from market data"""
        try:
            # Extract features for RL model
            features = []
            
            # Price features
            features.append(market_data.get('current_price', 0))
            
            # Technical indicators
            features.append(market_data.get('rsi', 50) / 100.0)  # Normalize RSI
            features.append(market_data.get('macd', 0))
            features.append(market_data.get('atr', 0.001))
            features.append(market_data.get('ema_20', market_data.get('current_price', 0)))
            features.append(market_data.get('ema_50', market_data.get('current_price', 0)))
            
            # Market regime features
            volatility = market_data.get('volatility', 0.01)
            features.append(volatility)
            features.append(market_data.get('momentum', 0))
            
            # Normalize features
            observation = np.array(features, dtype=np.float32)
            
            # Pad or truncate to expected size
            expected_size = 32  # Match your feature manager
            if len(observation) < expected_size:
                observation = np.pad(observation, (0, expected_size - len(observation)))
            elif len(observation) > expected_size:
                observation = observation[:expected_size]
            
            return observation
            
        except Exception as e:
            self.logger.error(f"Error preparing observation: {e}")
            return None
    
    def _action_to_signal(self, action: int, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert RL action to trading signal"""
        try:
            # Action mapping: 0=HOLD, 1=BUY, 2=SELL
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            direction = action_map.get(int(action), 'HOLD')
            
            if direction == 'HOLD':
                return None
            
            # Calculate confidence based on market conditions
            rsi = market_data.get('rsi', 50)
            volatility = market_data.get('volatility', 0.01)
            
            # Base confidence
            confidence = 0.7
            
            # Adjust confidence based on market conditions
            if direction == 'BUY' and rsi < 40:
                confidence += 0.1
            elif direction == 'SELL' and rsi > 60:
                confidence += 0.1
            
            # Reduce confidence in high volatility
            if volatility > 0.03:
                confidence -= 0.1
            
            confidence = max(0.5, min(0.95, confidence))
            
            signal = {
                'symbol': symbol,
                'direction': direction,
                'strategy': f'RL-{self.model_type}',
                'confidence': confidence,
                'entry_price': market_data.get('current_price', 0),
                'timestamp': datetime.now(),
                'model_type': self.model_type,
                'prediction_method': 'rl_model'
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error converting action to signal: {e}")
            return None
    
    def update_model_performance(self, signal: Dict[str, Any], trade_result: Dict[str, Any]) -> None:
        """Update model performance tracking"""
        try:
            with self.lock:
                # Track performance
                success = trade_result.get('success', False)
                profit = trade_result.get('profit', 0)
                
                if success:
                    self.successful_predictions += 1
                
                # Store performance data
                performance_entry = {
                    'timestamp': datetime.now(),
                    'symbol': signal.get('symbol'),
                    'direction': signal.get('direction'),
                    'confidence': signal.get('confidence'),
                    'success': success,
                    'profit': profit,
                    'model_type': self.model_type
                }
                
                self.performance_history.append(performance_entry)
                
                # Keep only last 1000 entries
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]
                
                self.logger.debug(f"Updated model performance: {signal.get('symbol')} - Success: {success}")
                
        except Exception as e:
            self.logger.error(f"Error updating model performance: {e}")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get model diagnostics and performance metrics"""
        try:
            success_rate = 0
            if self.prediction_count > 0:
                success_rate = (self.successful_predictions / self.prediction_count) * 100
            
            diagnostics = {
                'model_type': self.model_type or 'None',
                'model_loaded': self.is_loaded,
                'rl_libraries_available': self.rl_available,
                'total_predictions': self.prediction_count,
                'successful_predictions': self.successful_predictions,
                'success_rate': f"{success_rate:.1f}%",
                'performance_history_size': len(self.performance_history),
                'model_path': self.model_path or 'None',
                'simulation_mode': self.model_type == 'SIMULATION'
            }
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Error getting diagnostics: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Clean shutdown of RL model manager"""
        try:
            self.logger.info("Shutting down Enhanced RL Model Manager...")
            
            # Log final statistics
            diagnostics = self.get_diagnostics()
            self.logger.info("Final RL Model Statistics:")
            for key, value in diagnostics.items():
                self.logger.info(f"  {key}: {value}")
            
            self.logger.info("Enhanced RL Model Manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during RL model manager shutdown: {e}")

# Advanced config loader
def load_user_config_advanced():
    """Advanced config loader with multiple fallback methods"""
    try:
        config_locations = [
            'configs/config.py',
            'config.py',
            'configs/Config.py',
            'Config.py'
        ]
        
        config_found = False
        config_path = None
        
        # Find config file
        for location in config_locations:
            if os.path.exists(location):
                config_found = True
                config_path = Path(location)
                logger.info(f"✅ Found config file at: {location}")
                break
        
        if not config_found:
            logger.error("❌ Config file not found in any location:")
            for location in config_locations:
                logger.error(f"   - {location}")
            return None
        
        # Method 1: importlib.util (Primary)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("user_config", config_path)
            user_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_config)
            
            logger.info("✅ Config loaded using importlib.util")
            
        except Exception as e:
            logger.warning(f"importlib method failed: {e}")
            
            # Method 2: exec fallback
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_content = f.read()
                
                config_namespace = {}
                exec(config_content, config_namespace)
                
                class ConfigObject:
                    pass
                
                user_config = ConfigObject()
                for key, value in config_namespace.items():
                    if not key.startswith('__') and not callable(value):
                        setattr(user_config, key, value)
                
                logger.info("✅ Config loaded using exec fallback")
                
            except Exception as e2:
                logger.error(f"All config loading methods failed: {e2}")
                return None
        
        # Validate and enhance config
        config_enhanced = validate_and_enhance_config(user_config)
        return config_enhanced
        
    except Exception as e:
        logger.error(f"Critical config loading error: {e}")
        return None

def validate_and_enhance_config(config):
    """Validate and enhance config with defaults"""
    try:
        # Required attributes
        required_attrs = ['MT5_LOGIN', 'MT5_SERVER', 'SYMBOLS_TO_TRADE']
        missing_attrs = []
        
        for attr in required_attrs:
            if not hasattr(config, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            logger.error(f"❌ Config missing required attributes: {missing_attrs}")
            return None
        
        # Add defaults for missing optional attributes
        defaults = {
            'MT5_PASSWORD': 'yourpassword',
            'MT5_PATH': r'C:\Program Files\MetaTrader 5\terminal64.exe',
            'MT5_TIMEOUT': 10000,
            'PRIMARY_TIMEFRAME': 'H1',
            'RISK_PERCENTAGE': 0.02,
            'MAX_SPREAD': 3,
            'SLIPPAGE_PIPS': 1,
            'MAGIC_NUMBER': 123456789,
            'ENABLE_NEWS_FILTER': False,
            'MAX_CONCURRENT_TRADES': 5,
            'TRADING_HOURS': {'start': 0, 'end': 24},
            'ENABLE_SENTIMENT_ANALYSIS': False,
            'DATA_CACHE_HOURS': 24,
            'RL_TRAINING_ENABLED': True,
            'RL_SYMBOLS': ['EURUSD', 'GBPUSD', 'XAUUSD']
        }
        
        for attr, default_value in defaults.items():
            if not hasattr(config, attr):
                setattr(config, attr, default_value)
                logger.info(f"Added default config: {attr} = {default_value}")
        
        # Log final config
        logger.info("📋 ENHANCED CONFIG LOADED:")
        logger.info(f"   MT5 Login: {getattr(config, 'MT5_LOGIN', 'Not set')}")
        logger.info(f"   MT5 Server: {getattr(config, 'MT5_SERVER', 'Not set')}")
        logger.info(f"   Symbols: {getattr(config, 'SYMBOLS_TO_TRADE', 'Not set')}")
        logger.info(f"   Primary Timeframe: {getattr(config, 'PRIMARY_TIMEFRAME', 'H1')}")
        logger.info(f"   Risk %: {getattr(config, 'RISK_PERCENTAGE', 0.02)}")
        
        return config
        
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        return None

# Advanced requirements checker
def check_comprehensive_requirements():
    """Comprehensive requirements check with detailed reporting"""
    missing_packages = []
    available_packages = {}
    
    # Core ML packages
    ml_packages = {
        'stable_baselines3': 'stable-baselines3[extra]',
        'torch': 'torch',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas>=1.5.0',
        'numpy': 'numpy>=1.21.0',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    for package, install_name in ml_packages.items():
        try:
            if package == 'stable_baselines3':
                from stable_baselines3 import SAC, PPO, A2C, TD3
                from stable_baselines3.common.vec_env import DummyVecEnv
                from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
                from stable_baselines3.common.monitor import Monitor
                from stable_baselines3.common.evaluation import evaluate_policy
                available_packages[package] = "✅"
            elif package == 'torch':
                import torch
                available_packages[package] = f"✅ {torch.__version__}"
            elif package == 'sklearn':
                import sklearn
                available_packages[package] = f"✅ {sklearn.__version__}"
            elif package == 'pandas':
                import pandas as pd
                available_packages[package] = f"✅ {pd.__version__}"
            elif package == 'numpy':
                import numpy as np
                available_packages[package] = f"✅ {np.__version__}"
            elif package == 'matplotlib':
                import matplotlib
                available_packages[package] = f"✅ {matplotlib.__version__}"
            elif package == 'seaborn':
                import seaborn
                available_packages[package] = f"✅ {seaborn.__version__}"
            else:
                __import__(package)
                available_packages[package] = "✅"
                
        except ImportError:
            missing_packages.append(install_name)
            available_packages[package] = "❌"
    
    # Data sources
    data_sources = {
        'MetaTrader5': 'MetaTrader5',
        'yfinance': 'yfinance'
    }
    
    for package, install_name in data_sources.items():
        try:
            __import__(package.lower() if package != 'MetaTrader5' else 'MetaTrader5')
            available_packages[package] = "✅"
        except ImportError:
            available_packages[package] = "❌"
            if package == 'MetaTrader5':
                missing_packages.append(install_name)
    
    # Advanced ML packages (optional)
    advanced_packages = {
        'optuna': 'optuna',
        'xgboost': 'xgboost', 
        'lightgbm': 'lightgbm',
        'networkx': 'networkx',
        'scipy': 'scipy',
        'statsmodels': 'statsmodels'
    }
    
    for package, install_name in advanced_packages.items():
        try:
            __import__(package)
            available_packages[package] = "✅"
        except ImportError:
            available_packages[package] = "⚠️ Optional"
    
    # Log results
    logger.info("📦 PACKAGE AVAILABILITY CHECK:")
    for package, status in available_packages.items():
        logger.info(f"   {package:15} {status}")
    
    if missing_packages:
        logger.error(f"❌ Missing critical packages: {missing_packages}")
        logger.error("Install with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ All critical requirements satisfied")
    return True

def check_and_import_advanced_libraries():
    """Check and import advanced ML libraries"""
    ML_ADVANCED = True
    try:
        import optuna
        import xgboost as xgb
        import lightgbm as lgb
        import networkx as networkx
        logger.info("✅ Advanced ML libraries available")
    except ImportError:
        ML_ADVANCED = False
        logger.warning("⚠️ Advanced ML libraries not available (optuna, xgboost, lightgbm, networkx)")
    
    return ML_ADVANCED

# Create comprehensive modules - FIXED VERSION
def create_comprehensive_modules(user_config):
    """Create comprehensive modules with all advanced features - FIXED"""
    try:
        logger.info("🏗️ Creating comprehensive modules with advanced features...")
        
        # Enhanced DataHandler
        enhanced_datahandler_code = '''# enhanced_datahandler.py - Production Ready DataHandler
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import pickle
import hashlib
import threading
import queue
from pathlib import Path
import time
import os

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedDataHandler:
    """Enhanced Data Handler with caching, validation, and multiple sources"""
    
    def __init__(self, user_config):
        self.user_config = user_config
        self.connected = False
        self.cache_dir = Path('cache/market_data')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache = {}
        self.connection_lock = threading.Lock()
        
        # MT5 Configuration
        if MT5_AVAILABLE:
            self.mt5_login = getattr(user_config, 'MT5_LOGIN', 5038274604)
            self.mt5_password = getattr(user_config, 'MT5_PASSWORD', 'yourpassword')
            self.mt5_server = getattr(user_config, 'MT5_SERVER', 'MetaQuotes-Demo')
            self.mt5_path = getattr(user_config, 'MT5_PATH', r'C:\\\\Program Files\\\\MetaTrader 5\\\\terminal64.exe')
            self.mt5_timeout = getattr(user_config, 'MT5_TIMEOUT', 10000)
            
            # Timeframe mappings
            self.timeframes = {
                'M1': mt5.TIMEFRAME_M1, 'M2': mt5.TIMEFRAME_M2, 'M3': mt5.TIMEFRAME_M3,
                'M4': mt5.TIMEFRAME_M4, 'M5': mt5.TIMEFRAME_M5, 'M6': mt5.TIMEFRAME_M6,
                'M10': mt5.TIMEFRAME_M10, 'M12': mt5.TIMEFRAME_M12, 'M15': mt5.TIMEFRAME_M15,
                'M20': mt5.TIMEFRAME_M20, 'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1, 'H2': mt5.TIMEFRAME_H2, 'H3': mt5.TIMEFRAME_H3,
                'H4': mt5.TIMEFRAME_H4, 'H6': mt5.TIMEFRAME_H6, 'H8': mt5.TIMEFRAME_H8,
                'H12': mt5.TIMEFRAME_H12, 'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1,
                'MN1': mt5.TIMEFRAME_MN1
            }
        
        # Data source priority
        self.data_sources = []
        if MT5_AVAILABLE:
            self.data_sources.append('MT5')
        if YFINANCE_AVAILABLE:
            self.data_sources.append('YFINANCE')
        
        # Cache settings
        self.cache_hours = getattr(user_config, 'DATA_CACHE_HOURS', 24)
        self.enable_cache = True
        
        logger.info(f"Enhanced DataHandler initialized")
        logger.info(f"   Available sources: {', '.join(self.data_sources)}")
    
    def connect(self) -> bool:
        """Connect to MT5"""
        try:
            if not MT5_AVAILABLE:
                logger.error("MT5 not available")
                return False
            
            if not mt5.initialize(path=self.mt5_path):
                error_code = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error_code}")
                return False
            
            if not mt5.login(self.mt5_login, password=self.mt5_password, server=self.mt5_server):
                error_code = mt5.last_error()
                logger.error(f"MT5 login failed: {error_code}")
                return False
            
            self.connected = True
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"✅ Connected to MT5 - Account: {account_info.login}")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def get_historical_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical data with enhanced error handling"""
        try:
            if not self.connected and not self.connect():
                logger.error("Failed to connect to data source")
                return pd.DataFrame()
            
            # Get data from MT5
            mt5_timeframe = self.timeframes.get(timeframe, mt5.TIMEFRAME_H1)
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Rename columns
            df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            # Clean data
            df = self.validate_and_clean_data(df, symbol)
            
            logger.info(f"✅ Retrieved {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def validate_and_clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Comprehensive data validation and cleaning"""
        try:
            if data.empty:
                return data
            
            # Remove invalid OHLC data
            invalid_mask = (
                (data['High'] < data['Low']) |
                (data['High'] < data['Open']) |
                (data['High'] < data['Close']) |
                (data['Low'] > data['Open']) |
                (data['Low'] > data['Close']) |
                (data[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
            )
            
            if invalid_mask.any():
                logger.warning(f"Removed {invalid_mask.sum()} invalid bars for {symbol}")
                data = data[~invalid_mask]
            
            # Handle missing values
            data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            if 'Volume' not in data.columns:
                data['Volume'] = 1000000
            
            data['Volume'] = data['Volume'].fillna(1000000)
            
            return data
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return data
    
    def disconnect(self):
        """Disconnect from MT5"""
        try:
            if self.connected and MT5_AVAILABLE:
                mt5.shutdown()
                self.connected = False
                logger.info("Disconnected from MT5")
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
'''

        # Enhanced MarketIntelligence
        enhanced_marketintelligence_code = '''# enhanced_marketintelligence.py - Production Ready Market Intelligence
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class EnhancedMarketIntelligence:
    """Enhanced Market Intelligence with comprehensive technical analysis"""
    
    def __init__(self, data_handler, user_config):
        self.data_handler = data_handler
        self.user_config = user_config
        self.feature_cache = {}
        
        logger.info("Enhanced Market Intelligence initialized")
    
    def analyze_comprehensive(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Comprehensive market analysis with advanced features"""
        try:
            if data.empty:
                logger.warning("Empty input data")
                return data
            
            df = data.copy()
            original_features = len(df.columns)
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Add statistical features
            df = self.add_statistical_features(df)
            
            # Add price action features
            df = self.add_price_action_features(df)
            
            # Final cleanup
            df = self.cleanup_features(df)
            
            new_features = len(df.columns) - original_features
            logger.info(f"✅ Analysis complete: {original_features} -> {len(df.columns)} features (+{new_features})")
            
            return df
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return data
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        try:
            # Moving Averages
            ma_periods = [5, 10, 20, 50, 100]
            for period in ma_periods:
                if len(df) >= period:
                    df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
                    df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            
            # RSI
            for period in [14, 21]:
                if len(df) >= period:
                    df[f'RSI_{period}'] = self.calculate_rsi(df['Close'], period)
            
            # MACD
            if len(df) >= 26:
                macd_line, macd_signal = self.calculate_macd(df['Close'])
                df['MACD'] = macd_line
                df['MACD_Signal'] = macd_signal
                df['MACD_Histogram'] = macd_line - macd_signal
            
            # Bollinger Bands
            for period in [20, 50]:
                if len(df) >= period:
                    bb_upper, bb_lower, bb_middle = self.calculate_bollinger_bands(df['Close'], period)
                    df[f'BB_Upper_{period}'] = bb_upper
                    df[f'BB_Lower_{period}'] = bb_lower
                    df[f'BB_Middle_{period}'] = bb_middle
                    df[f'BB_Width_{period}'] = (bb_upper - bb_lower) / bb_middle
            
            # ATR
            for period in [14, 21]:
                if len(df) >= period:
                    df[f'ATR_{period}'] = self.calculate_atr(df, period)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        try:
            # Returns and volatility
            df['Returns'] = df['Close'].pct_change()
            
            # Rolling statistics
            for window in [10, 20, 50]:
                if len(df) >= window:
                    df[f'Volatility_{window}'] = df['Returns'].rolling(window).std()
                    df[f'Skew_{window}'] = df['Returns'].rolling(window).skew()
                    df[f'Kurt_{window}'] = df['Returns'].rolling(window).kurt()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding statistical features: {e}")
            return df
    
    def add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action features"""
        try:
            # Basic price relationships
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Close_Open_Ratio'] = df['Close'] / df['Open']
            
            # Candlestick analysis
            df['Body_Size'] = abs(df['Close'] - df['Open'])
            df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
            df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
            df['Total_Range'] = df['High'] - df['Low']
            
            # Normalized ratios
            df['Body_Ratio'] = df['Body_Size'] / (df['Total_Range'] + 1e-10)
            df['Upper_Shadow_Ratio'] = df['Upper_Shadow'] / (df['Total_Range'] + 1e-10)
            df['Lower_Shadow_Ratio'] = df['Lower_Shadow'] / (df['Total_Range'] + 1e-10)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding price action features: {e}")
            return df
    
    def cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup of features"""
        try:
            # Replace infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
            return df
    
    # Helper calculation methods
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd.fillna(0), macd_signal.fillna(0)
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper.fillna(prices), lower.fillna(prices), sma.fillna(prices)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        
        return atr.fillna(0.001)
'''

        # Enhanced TradingEnvironment
        enhanced_tradingenvironment_code = '''# enhanced_tradingenvironment.py - Production Ready Trading Environment
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class EnhancedTradingEnvironment(gym.Env):
    """Enhanced Trading Environment for RL training"""
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any] = None):
        super().__init__()
        
        self.data = data.copy()
        self.config = config or {}
        
        # Environment parameters
        self.initial_balance = getattr(self.config,'initial_balance', 100000.0)
        self.transaction_cost = getattr(self.config,'transaction_cost', 0.0001)
        self.max_drawdown_limit = getattr(self.config,'max_drawdown_limit', 0.3)
        
        # State variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Tracking
        self.trades_history = []
        self.equity_curve = []
        self.current_drawdown = 0.0
        self.peak_equity = self.initial_balance
        
        # Spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        n_features = min(100, len(self.data.columns) * 2)  # Dynamic feature count
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        
        self.prepare_data()
        logger.info(f"Enhanced Trading Environment initialized - {len(self.data)} data points")
    
    def prepare_data(self):
        """Prepare data for training"""
        try:
            if 'Returns' not in self.data.columns:
                self.data['Returns'] = self.data['Close'].pct_change()
            
            self.data = self.data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            if not isinstance(self.data.index, pd.DatetimeIndex):
                self.data.index = pd.date_range(start='2020-01-01', periods=len(self.data), freq='H')
            
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 50
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        self.trades_history = []
        self.equity_curve = [self.initial_balance]
        self.current_drawdown = 0.0
        self.peak_equity = self.initial_balance
        
        observation = self.get_observation()
        info = self.get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step"""
        self.current_step += 1
        
        if self.current_step >= len(self.data):
            return self.get_observation(), 0.0, True, False, self.get_info()
        
        position_change, position_size_target = np.clip(action, self.action_space.low, self.action_space.high)
        current_price = self.data.iloc[self.current_step]['Close']
        
        if self.position != 0:
            self.update_unrealized_pnl(current_price)
        
        reward = 0.0
        if abs(position_change) > 0.01:
            reward = self.execute_trade(position_change, position_size_target, current_price)
        
        self.update_portfolio()
        reward += self.calculate_step_reward()
        
        done, truncated = self.check_termination()
        
        observation = self.get_observation()
        info = self.get_info()
        
        return observation, reward, done, truncated, info
    
    def execute_trade(self, position_change, position_size, current_price):
        """Execute trading action"""
        try:
            reward = 0.0
            
            if self.position != 0 and np.sign(position_change) != np.sign(self.position):
                reward += self.close_position(current_price)
            
            if abs(position_change) > 0.01:
                reward += self.open_position(position_change, position_size, current_price)
            
            return reward
        except:
            return -0.1
    
    def close_position(self, current_price):
        """Close current position"""
        if self.position == 0:
            return 0.0
        
        if self.position > 0:
            pnl = (current_price - self.entry_price) * abs(self.position_size)
        else:
            pnl = (self.entry_price - current_price) * abs(self.position_size)
        
        cost = abs(self.position_size) * current_price * self.transaction_cost
        net_pnl = pnl - cost
        
        self.balance += net_pnl
        self.realized_pnl += net_pnl
        self.unrealized_pnl = 0.0
        
        self.trades_history.append({
            'entry_price': self.entry_price,
            'exit_price': current_price,
            'net_pnl': net_pnl
        })
        
        self.position = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        
        return net_pnl / self.initial_balance * 100
    
    def open_position(self, position_change, position_size, current_price):
        """Open new position"""
        self.position = np.sign(position_change)
        available_capital = self.balance * position_size
        self.position_size = (available_capital / current_price) * self.position
        self.entry_price = current_price
        
        cost = abs(self.position_size) * current_price * self.transaction_cost
        self.balance -= cost
        
        return 0.001
    
    def update_unrealized_pnl(self, current_price):
        """Update unrealized P&L"""
        if self.position == 0 or self.entry_price == 0:
            self.unrealized_pnl = 0.0
            return
        
        if self.position > 0:
            self.unrealized_pnl = (current_price - self.entry_price) * abs(self.position_size)
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * abs(self.position_size)
    
    def update_portfolio(self):
        """Update portfolio metrics"""
        self.equity = self.balance + self.unrealized_pnl
        
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        
        self.equity_curve.append(self.equity)
    
    def calculate_step_reward(self):
        """Calculate step-wise reward"""
        reward = 0.0
        
        if len(self.equity_curve) > 1:
            equity_change = self.equity_curve[-1] - self.equity_curve[-2]
            reward += equity_change / self.initial_balance * 5
        
        if self.current_drawdown > 0.1:
            reward -= self.current_drawdown * 2
        
        return reward
    
    def check_termination(self):
        """Check termination conditions"""
        done = False
        truncated = False
        
        if self.current_step >= len(self.data) - 1:
            done = True
        
        if self.current_drawdown >= self.max_drawdown_limit:
            done = True
        
        if self.equity <= self.initial_balance * 0.1:
            done = True
        
        return done, truncated
    
    def get_observation(self):
        """Get current observation"""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        
        current_data = self.data.iloc[self.current_step]
        close = current_data['Close']
        
        # Market features (dynamic based on available data)
        market_features = []
        
        # Basic OHLCV
        market_features.extend([
            current_data.get('Open', close) / close,
            current_data.get('High', close) / close,
            current_data.get('Low', close) / close,
            1.0,  # Close/Close = 1
            np.log(current_data.get('Volume', 1000000) + 1) / 10
        ])
        
        # Technical indicators (if available)
        indicators = ['RSI_14', 'MACD', 'MACD_Signal', 'ATR_14', 'SMA_20', 'EMA_20']
        for indicator in indicators:
            value = current_data.get(indicator, 0)
            if 'RSI' in indicator:
                market_features.append((value - 100) / 100)
            elif 'MACD' in indicator or 'ATR' in indicator:
                market_features.append(np.tanh(value * 1000) if value != 0 else 0)
            else:
                market_features.append((value - close) / close if close > 0 and value > 0 else 0)
        
        # Recent returns
        if self.current_step >= 5:
            recent_returns = self.data['Returns'].iloc[self.current_step-4:self.current_step+1].fillna(0)
            market_features.extend(recent_returns.tolist())
        else:
            market_features.extend([0.0] * 5)
        
        # Portfolio features
        portfolio_features = [
            float(self.position),
            self.position_size / 100000 if self.position_size != 0 else 0.0,
            (self.balance - self.initial_balance) / self.initial_balance,
            (self.equity - self.initial_balance) / self.initial_balance,
            self.unrealized_pnl / self.initial_balance,
            self.realized_pnl / self.initial_balance,
            self.current_drawdown,
            len(self.trades_history) / 100.0,
            float(self.current_step) / len(self.data),
            0.0
        ]
        
        # Combine and ensure fixed size
        all_features = market_features + portfolio_features
        target_size = self.observation_space.shape[0]
        
        if len(all_features) > target_size:
            all_features = all_features[:target_size]
        elif len(all_features) < target_size:
            all_features.extend([0.0] * (target_size - len(all_features)))
        
        observation = np.array(all_features, dtype=np.float32)
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def get_info(self):
        """Get environment info"""
        current_price = self.data.iloc[self.current_step]['Close'] if self.current_step < len(self.data) else 0
        
        return {
            'step': self.current_step,
            'current_price': float(current_price),
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'total_trades': len(self.trades_history),
            'current_drawdown': self.current_drawdown,
            'total_return': (self.equity - self.initial_balance) / self.initial_balance
        }
'''

        # Write all modules
        modules = {
            'enhanced_datahandler.py': enhanced_datahandler_code,
            'enhanced_marketintelligence.py': enhanced_marketintelligence_code,
            'enhanced_tradingenvironment.py': enhanced_tradingenvironment_code
        }
        
        for filename, code in modules.items():
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.info(f"✅ Created {filename}")
        
        logger.info("🎯 All enhanced modules created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error creating comprehensive modules: {e}")
        return False

# Enhanced training system
class EnhancedTrainingSystem:
    """Enhanced RL training system with production-ready features"""
    
    def __init__(self, user_config):
        self.user_config = user_config
        self.training_start_time = time.time()
        
        # Build enhanced configuration
        self.config = self._build_enhanced_training_config()
        
        # Components
        self.data_handler = None
        self.market_intel = None
        self.models = {}
        self.training_data = {}
        self.validation_data = {}
        self.performance_metrics = {}
        
        logger.info("✅ Enhanced Training System initialized")
        logger.info(f"   Symbols: {self.config['symbols']}")
        logger.info(f"   Models: {list(self.config['models'].keys())}")
    
    def _build_enhanced_training_config(self):
        """Build enhanced training configuration"""
        symbols = getattr(self.user_config, 'SYMBOLS_TO_TRADE', ['EURUSD', 'GBPUSD', 'XAUUSD'])
        rl_symbols = getattr(self.user_config, 'RL_SYMBOLS', symbols[:3])  # First 3 symbols
        
        return {
            'symbols': rl_symbols,
            'start_date': '2021-01-01',
            'end_date': '2024-01-01',
            'validation_split': 0.8,
            'models': {
                'SAC': {
                    'training_steps': 50000,
                    'eval_freq': 5000,
                    'eval_episodes': 10,
                    'learning_rate': 3e-4,
                    'batch_size': 256,
                    'buffer_size': 25000
                },
                'PPO': {
                    'training_steps': 40000,
                    'eval_freq': 4000,
                    'eval_episodes': 8,
                    'learning_rate': 3e-4,
                    'batch_size': 64,
                    'n_steps': 2048
                }
            }
        }
    
    def initialize_components(self):
        """Initialize system components"""
        try:
            logger.info("🔧 Initializing enhanced components...")
            
            from enhanced_datahandler import EnhancedDataHandler
            from enhanced_marketintelligence import EnhancedMarketIntelligence
            
            # Initialize data handler
            self.data_handler = EnhancedDataHandler(self.user_config)
            if not self.data_handler.connect():
                logger.error("❌ Failed to connect data handler")
                return False
            logger.info("✅ Enhanced DataHandler connected")
            
            # Initialize market intelligence
            self.market_intel = EnhancedMarketIntelligence(self.data_handler, self.user_config)
            logger.info("✅ Enhanced Market Intelligence initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False
    
    def prepare_data(self):
        """Prepare training data"""
        try:
            logger.info("📊 Preparing enhanced training data...")
            
            for symbol in self.config['symbols']:
                logger.info(f"🔄 Processing {symbol}...")
                
                start_date = pd.to_datetime(self.config['start_date'])
                end_date = pd.to_datetime(self.config['end_date'])
                
                raw_data = self.data_handler.get_historical_data(
                    symbol, 'H1', start_date, end_date
                )
                
                if raw_data is None or raw_data.empty:
                    logger.warning(f"⚠️ No data for {symbol}")
                    continue
                
                # Enhanced analysis
                processed_data = self.market_intel.analyze_comprehensive(raw_data, symbol)
                
                if len(processed_data) < 1000:
                    logger.warning(f"⚠️ Insufficient data for {symbol}: {len(processed_data)} rows")
                    continue
                
                # Split data
                split_idx = int(len(processed_data) * self.config['validation_split'])
                self.training_data[symbol] = processed_data.iloc[:split_idx].copy()
                self.validation_data[symbol] = processed_data.iloc[split_idx:].copy()
                
                logger.info(f"✅ {symbol}: Train={len(self.training_data[symbol])}, Val={len(self.validation_data[symbol])}")
            
            if not self.training_data:
                logger.error("❌ No training data available")
                return False
            
            logger.info(f"🎯 Data preparation complete for {len(self.training_data)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return False
    
    def train_models(self):
        """Train RL models"""
        try:
            from stable_baselines3 import SAC, PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.evaluation import evaluate_policy
            from enhanced_tradingenvironment import EnhancedTradingEnvironment
            
            logger.info("🤖 Starting enhanced model training...")
            
            successful_models = 0
            
            for symbol in self.training_data:
                for model_name in self.config['models']:
                    model_key = f"{model_name}_{symbol}"
                    logger.info(f"🎯 Training {model_key}...")
                    
                    try:
                        # Create environments
                        train_env = self._create_env(symbol, 'train')
                        val_env = self._create_env(symbol, 'val')
                        
                        if train_env is None or val_env is None:
                            continue
                        
                        # Create model
                        model_config = self.config['models'][model_name]
                        model = self._create_model(model_name, train_env, model_config)
                        
                        if model is None:
                            continue
                        
                        # Train model
                        logger.info(f"🚀 Training {model_key} for {model_config['training_steps']:,} steps...")
                        start_time = time.time()
                        
                        model.learn(
                            total_timesteps=model_config['training_steps'],
                            progress_bar=True
                        )
                        
                        training_time = time.time() - start_time
                        
                        # Evaluate model
                        mean_reward, std_reward = evaluate_policy(
                            model, val_env,
                            n_eval_episodes=model_config['eval_episodes'],
                            deterministic=True
                        )
                        
                        # Save model
                        model_path = f"./models/{model_key}_enhanced"
                        Path("./models").mkdir(exist_ok=True)
                        model.save(model_path)
                        
                        # Store results
                        self.models[model_key] = model
                        self.performance_metrics[model_key] = {
                            'mean_reward': float(mean_reward),
                            'std_reward': float(std_reward),
                            'training_time': training_time,
                            'training_steps': model_config['training_steps']
                        }
                        
                        successful_models += 1
                        
                        logger.info(f"✅ {model_key} completed!")
                        logger.info(f"   Reward: {mean_reward:.2f} ± {std_reward:.2f}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error training {model_key}: {e}")
                        continue
            
            logger.info(f"🎯 Model training completed! Successful: {successful_models}")
            return successful_models > 0
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def _create_env(self, symbol, data_type):
        """Create trading environment"""
        try:
            from enhanced_tradingenvironment import EnhancedTradingEnvironment
            from stable_baselines3.common.vec_env import DummyVecEnv
            from stable_baselines3.common.monitor import Monitor
            
            data = self.training_data[symbol] if data_type == 'train' else self.validation_data[symbol]
            
            env_config = {
                'initial_balance': 100000.0,
                'transaction_cost': 0.0001,
                'max_drawdown_limit': 0.3
            }
            
            env = EnhancedTradingEnvironment(data, env_config)
            monitored_env = Monitor(env)
            vec_env = DummyVecEnv([lambda: monitored_env])
            
            return vec_env
            
        except Exception as e:
            logger.error(f"Environment creation failed: {e}")
            return None
    
    def _create_model(self, model_name, env, config):
        """Create RL model"""
        try:
            from stable_baselines3 import SAC, PPO
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            policy_kwargs = {
                'net_arch': [512, 512, 256],
                'activation_fn': torch.nn.ReLU
            }
            
            base_params = {
                'policy': 'MlpPolicy',
                'env': env,
                'policy_kwargs': policy_kwargs,
                'device': device,
                'verbose': 1
            }
            
            if model_name == 'SAC':
                model = SAC(
                    **base_params,
                    learning_rate=config['learning_rate'],
                    buffer_size=config['buffer_size'],
                    batch_size=config['batch_size']
                )
            elif model_name == 'PPO':
                model = PPO(
                    **base_params,
                    learning_rate=config['learning_rate'],
                    batch_size=config['batch_size'],
                    n_steps=config['n_steps']
                )
            else:
                logger.error(f"Unknown model: {model_name}")
                return None
            
            return model
            
        except Exception as e:
            logger.error(f"Model creation failed: {e}")
            return None
    
    def generate_summary(self):
        """Generate training summary"""
        try:
            total_time = time.time() - self.training_start_time
            
            summary = {
                'training_completed': datetime.now().isoformat(),
                'total_time_minutes': total_time / 60,
                'models_trained': len(self.models),
                'model_performances': self.performance_metrics,
                'symbols_used': self.config['symbols']
            }
            
            # Save summary
            summary_path = Path('results/training_summary.json')
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Log summary
            logger.info("=" * 60)
            logger.info("🎯 ENHANCED TRAINING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"📊 Symbols: {', '.join(self.config['symbols'])}")
            logger.info(f"⏱️  Total Time: {total_time/60:.1f} minutes")
            logger.info(f"🤖 Models Trained: {len(self.models)}")
            
            if self.performance_metrics:
                logger.info("📈 PERFORMANCE:")
                for model_key, perf in self.performance_metrics.items():
                    logger.info(f"   {model_key}: {perf['mean_reward']:.2f} reward")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
    
    def run_complete_training(self):
        """Run complete training pipeline"""
        try:
            logger.info("🚀 Starting Enhanced Training Pipeline...")
            
            if not self.initialize_components():
                return False
            
            if not self.prepare_data():
                return False
            
            if not self.train_models():
                return False
            
            self.generate_summary()
            
            logger.info("🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return False
        
        finally:
            if self.data_handler:
                try:
                    self.data_handler.disconnect()
                except:
                    pass

def main():
    """Enhanced main function"""
    logger.info("🚀 Starting Complete Enterprise RL Training System")
    
    try:
        # Check requirements
        if not check_comprehensive_requirements():
            return False
        
        # Load config
        user_config = load_user_config_advanced()
        if user_config is None:
            return False
        
        # Check advanced ML libraries
        ML_ADVANCED = check_and_import_advanced_libraries()
        
        # Create comprehensive modules
        if not create_comprehensive_modules(user_config):
            return False
        
        # Set random seeds
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # Create and run training system
        training_system = EnhancedTrainingSystem(user_config)
        success = training_system.run_complete_training()
        
        if success:
            print("\n" + "="*60)
            print("🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("📊 Results:")
            print("  ✅ Enhanced RL training system executed")
            print(f"  📈 Symbols: {getattr(user_config, 'SYMBOLS_TO_TRADE', 'Unknown')}")
            print("  📁 Check logs/enterprise_training/ for detailed logs")
            print("  💾 Models saved to ./models/")
            print("  📊 Summary: ./results/training_summary.json")
            print("="*60)
            return True
        else:
            print("\n❌ Training failed. Check logs for details.")
            return False
            
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    
    print("\n✅ Enhanced RL Training System completed!")

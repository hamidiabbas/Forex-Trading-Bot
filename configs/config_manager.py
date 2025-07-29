"""
Enhanced YAML Configuration Manager with Environment Variable Support
Complete implementation with validation and error handling
"""
import yaml
import os
import re
import logging
from typing import Any, Dict, Optional, List
from utils.env_loader import get_env_var, validate_env_vars

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Complete configuration manager with environment variable substitution
    """
    
    def __init__(self, config_file: str = 'configs/bot_config.yaml'):
        self.config_file = config_file
        self.config = {}
        self._load_config()
        self._validate_critical_config()
    
    def _load_config(self) -> None:
        """Load and process YAML configuration with environment variable substitution"""
        try:
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"Configuration file {self.config_file} not found")
            
            with open(self.config_file, 'r', encoding='utf-8') as file:
                # Load raw YAML
                raw_config = yaml.safe_load(file)
                
                if not raw_config:
                    raise ValueError("Configuration file is empty or invalid")
                
                # Process environment variable substitutions
                self.config = self._substitute_env_vars(raw_config)
                
                logger.info(f"✅ Configuration loaded from {self.config_file}")
                
        except FileNotFoundError as e:
            logger.error(f"❌ Configuration file not found: {e}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"❌ Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
            raise
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables in configuration"""
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Look for ${VAR_NAME} pattern
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, obj)
            
            for match in matches:
                env_var = get_env_var(match)
                if env_var is not None:
                    obj = obj.replace(f'${{{match}}}', str(env_var))
                else:
                    logger.warning(f"Environment variable {match} not found, keeping placeholder")
            
            return obj
        else:
            return obj
    
    def _validate_critical_config(self) -> None:
        """Validate critical configuration parameters"""
        critical_checks = [
            ('trading.symbols', list, "Trading symbols must be configured"),
            ('mt5.magic_number', int, "MT5 magic number must be configured"),
            ('risk_management.max_risk_per_trade', (int, float), "Risk per trade must be configured")
        ]
        
        for key, expected_type, message in critical_checks:
            value = self.get(key)
            if value is None:
                raise ValueError(f"❌ {message}: {key} not found")
            if not isinstance(value, expected_type):
                raise ValueError(f"❌ {message}: {key} has invalid type")
        
        logger.info("✅ Critical configuration validation passed")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation with enhanced error handling"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except Exception as e:
            logger.debug(f"Error getting config key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config_section = self.config
        
        for k in keys[:-1]:
            if k not in config_section:
                config_section[k] = {}
            config_section = config_section[k]
        
        config_section[keys[-1]] = value
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def get_trading_symbols(self) -> List[str]:
        """Get trading symbols with validation and cleanup"""
        symbols = self.get('trading.symbols', ['EURUSD', 'GBPUSD', 'XAUUSD'])
        
        if not isinstance(symbols, list):
            logger.error("Trading symbols must be a list")
            return ['EURUSD', 'GBPUSD', 'XAUUSD']
        
        # Clean and validate symbols
        clean_symbols = []
        for symbol in symbols:
            if isinstance(symbol, str) and len(symbol) >= 6:
                clean_symbols.append(symbol.upper())
            else:
                logger.warning(f"Invalid symbol format: {symbol}")
        
        if not clean_symbols:
            logger.warning("No valid symbols found, using defaults")
            return ['EURUSD', 'GBPUSD', 'XAUUSD']
        
        # Warn about problematic symbols
        if 'USDJPY' in clean_symbols:
            logger.warning("⚠️ USDJPY found in symbols - consider replacing with XAUUSD")
        
        return clean_symbols
    
    def get_mt5_config(self) -> Dict[str, Any]:
        """Get MT5 configuration with security validation"""
        mt5_config = self.get('mt5', {})
        
        # Validate that credentials use environment variables
        sensitive_keys = ['login', 'password']
        for key in sensitive_keys:
            value = mt5_config.get(key, '')
            if isinstance(value, str):
                if value.isdigit() or not value.startswith('${'):
                    logger.warning(f"⚠️ {key} may be hardcoded - should use environment variable")
        
        return mt5_config
    
    def get_correlation_matrix(self) -> Dict[tuple, float]:
        """Get correlation matrix from config with validation"""
        correlations = self.get('risk_management.correlation_matrix', {})
        
        if not correlations:
            # Default correlation matrix
            logger.info("Using default correlation matrix")
            return {
                ('EURUSD', 'GBPUSD'): 0.85,
                ('EURUSD', 'XAUUSD'): -0.25,
                ('GBPUSD', 'XAUUSD'): -0.20
            }
        
        # Convert flat keys to tuple keys
        correlation_matrix = {}
        for key, value in correlations.items():
            if '_' in key:
                symbols = tuple(sorted(key.split('_')))
                try:
                    correlation_value = float(value)
                    if -1.0 <= correlation_value <= 1.0:
                        correlation_matrix[symbols] = correlation_value
                    else:
                        logger.warning(f"Invalid correlation value for {key}: {value}")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid correlation value format for {key}: {value}")
        
        return correlation_matrix
    
    def get_symbol_execution_params(self, symbol: str) -> Dict[str, Any]:
        """Get symbol-specific execution parameters"""
        symbol_params = self.get(f'symbol_params.{symbol}', {})
        
        # Default parameters if not configured
        defaults = {
            'slippage': 3,
            'max_retry_attempts': 3,
            'retry_delay': 1.0,
            'min_volume': 0.01,
            'max_volume': 10.0
        }
        
        # Special defaults for XAUUSD
        if symbol == 'XAUUSD':
            defaults.update({
                'slippage': 12,
                'max_retry_attempts': 8,
                'retry_delay': 0.3,
                'max_volume': 5.0
            })
        
        # Merge defaults with configured values
        for key, default_value in defaults.items():
            if key not in symbol_params:
                symbol_params[key] = default_value
        
        return symbol_params
    
    def is_symbol_tradeable(self, symbol: str) -> bool:
        """Check if symbol is configured for trading"""
        symbols = self.get_trading_symbols()
        return symbol in symbols
    
    def reload_config(self) -> bool:
        """Reload configuration from file"""
        try:
            self._load_config()
            self._validate_critical_config()
            logger.info("✅ Configuration reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to reload configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging"""
        return {
            'symbols': self.get_trading_symbols(),
            'mt5_configured': bool(self.get('mt5.login')),
            'rl_model_type': self.get('rl_model.model_type', 'None'),
            'risk_per_trade': self.get('risk_management.max_risk_per_trade', 0),
            'correlation_matrix_size': len(self.get_correlation_matrix()),
            'config_file': self.config_file
        }

# Create global configuration instance
try:
    config_manager = ConfigManager()
    logger.info("✅ Global configuration manager initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize configuration manager: {e}")
    raise

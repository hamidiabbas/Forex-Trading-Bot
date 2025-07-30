"""
Professional Configuration Management System
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = "configs/bot_config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    if self.config_path.suffix.lower() == '.yaml':
                        return yaml.safe_load(f)
                    else:
                        return json.load(f)
            else:
                self.logger.warning(f"Config file not found: {self.config_path}")
                return self._create_default_config()
                
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            'trading': {
                'symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
                'risk_per_trade': 1.0,
                'max_daily_risk': 5.0,
                'min_analysis_interval': 30,
                'loop_interval': 10,
                'close_positions_on_shutdown': False
            },
            'mt5': {
                'login': 5038274604,
                'server': 'MetaQuotes-Demo',
                'path': 'C:\\Program Files\\MetaTrader 5\\terminal64.exe'
            },
            'notifications': {
                'email_enabled': False,
                'slack_enabled': False,
                'console_enabled': True
            },
            'rl': {
                'enabled': True,
                'confidence_threshold': 0.6,
                'fallback_to_traditional': True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def save(self):
        """Save current configuration"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")

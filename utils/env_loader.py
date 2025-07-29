"""
Environment Variable Loader for Secure Configuration
Enhanced with validation and error handling
"""
import os
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

def load_env_file(env_file: str = '.env') -> bool:
    """Load environment variables from .env file with enhanced validation"""
    try:
        if not os.path.exists(env_file):
            logger.warning(f"Environment file {env_file} not found")
            return False
            
        loaded_vars = 0
        with open(env_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Validate format
                if '=' not in line:
                    logger.warning(f"Invalid format in {env_file} line {line_num}: {line}")
                    continue
                
                # Split and set environment variable
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")  # Remove quotes
                
                if key:
                    os.environ[key] = value
                    loaded_vars += 1
                    logger.debug(f"Loaded environment variable: {key}")
        
        logger.info(f"✅ Loaded {loaded_vars} environment variables from {env_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading environment file {env_file}: {e}")
        return False

def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get environment variable with enhanced validation"""
    value = os.getenv(key, default)
    
    if required and value is None:
        logger.error(f"❌ Required environment variable {key} not set")
        raise ValueError(f"Required environment variable {key} not set")
    
    if value is None and default is None:
        logger.warning(f"⚠️ Environment variable {key} not set, no default provided")
    
    return value

def validate_env_vars(required_vars: Dict[str, str]) -> bool:
    """Validate that all required environment variables are set"""
    missing_vars = []
    
    for var_name, description in required_vars.items():
        if not os.getenv(var_name):
            missing_vars.append(f"{var_name} ({description})")
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"   - {var}")
        return False
    
    logger.info("✅ All required environment variables are set")
    return True

# Load environment variables on import
load_env_file()

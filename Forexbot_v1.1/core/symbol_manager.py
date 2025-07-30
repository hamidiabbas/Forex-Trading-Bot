"""
Complete Symbol Manager for 53+ Forex Pairs
Handles all pair types with proper specifications and risk management
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, time
import numpy as np

logger = logging.getLogger(__name__)

class EnhancedSymbolManager:
    """
    Complete symbol manager for professional forex trading
    Supports 53+ forex pairs with proper categorization and risk management
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trading_config = config.get('trading', {})
        
        # All available symbols
        self.all_symbols = self.trading_config.get('all_symbols', [])
        
        # Active trading symbols (5 most liquid)
        self.active_symbols = self.trading_config.get('active_symbols', [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD'
        ])
        
        # Initialize symbol categories
        self.symbol_categories = self._categorize_symbols()
        self.symbol_specs = self._initialize_symbol_specifications()
        
        logger.info(f"✅ Symbol Manager initialized")
        logger.info(f"   Total symbols available: {len(self.all_symbols)}")
        logger.info(f"   Active trading symbols: {len(self.active_symbols)}")
        logger.info(f"   Major pairs: {len(self.symbol_categories['major'])}")
        logger.info(f"   Minor pairs: {len(self.symbol_categories['minor'])}")
        logger.info(f"   Exotic pairs: {len(self.symbol_categories['exotic'])}")
    
    def _categorize_symbols(self) -> Dict[str, List[str]]:
        """Categorize symbols by type"""
        major_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'
        ]
        
        minor_pairs = [
            'EURJPY', 'EURGBP', 'EURCHF', 'EURAUD', 'EURCAD', 'EURNZD',
            'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD',
            'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
            'NZDJPY', 'NZDCHF', 'NZDCAD',
            'CADJPY', 'CADCHF', 'CHFJPY'
        ]
        
        # Everything else is exotic
        exotic_pairs = [
            symbol for symbol in self.all_symbols 
            if symbol not in major_pairs and symbol not in minor_pairs
        ]
        
        return {
            'major': major_pairs,
            'minor': minor_pairs,
            'exotic': exotic_pairs
        }
    
    def _initialize_symbol_specifications(self) -> Dict[str, Dict[str, Any]]:
        """Initialize specifications for all symbols"""
        specs = {}
        
        for symbol in self.all_symbols:
            specs[symbol] = self._get_symbol_specification(symbol)
        
        return specs
    
    def _get_symbol_specification(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive specification for a symbol"""
        
        # Default specifications
        default_spec = {
            'pair_type': 'exotic',
            'pip_size': 0.0001,
            'pip_value': 1.0,
            'typical_spread': 0.0005,
            'slippage_tolerance': 0.0005,
            'min_stop_distance': 30,
            'max_spread_filter': 0.0010,
            'contract_size': 100000,
            'session_multiplier': 0.8
        }
        
        # Major pairs specifications
        if symbol in self.symbol_categories['major']:
            major_specs = {
                'EURUSD': {
                    'pair_type': 'major', 'pip_size': 0.0001, 'pip_value': 1.0,
                    'typical_spread': 0.0001, 'slippage_tolerance': 0.0001,
                    'min_stop_distance': 15, 'max_spread_filter': 0.0003,
                    'contract_size': 100000, 'session_multiplier': 1.0
                },
                'GBPUSD': {
                    'pair_type': 'major', 'pip_size': 0.0001, 'pip_value': 1.0,
                    'typical_spread': 0.0002, 'slippage_tolerance': 0.0002,
                    'min_stop_distance': 20, 'max_spread_filter': 0.0005,
                    'contract_size': 100000, 'session_multiplier': 1.0
                },
                'USDJPY': {
                    'pair_type': 'major', 'pip_size': 0.01, 'pip_value': 1.0,
                    'typical_spread': 0.02, 'slippage_tolerance': 0.02,
                    'min_stop_distance': 20, 'max_spread_filter': 0.05,
                    'contract_size': 100000, 'session_multiplier': 1.0
                },
                'USDCHF': {
                    'pair_type': 'major', 'pip_size': 0.0001, 'pip_value': 1.0,
                    'typical_spread': 0.0002, 'slippage_tolerance': 0.0002,
                    'min_stop_distance': 20, 'max_spread_filter': 0.0005,
                    'contract_size': 100000, 'session_multiplier': 1.0
                },
                'AUDUSD': {
                    'pair_type': 'major', 'pip_size': 0.0001, 'pip_value': 1.0,
                    'typical_spread': 0.0002, 'slippage_tolerance': 0.0002,
                    'min_stop_distance': 20, 'max_spread_filter': 0.0005,
                    'contract_size': 100000, 'session_multiplier': 1.0
                },
                'USDCAD': {
                    'pair_type': 'major', 'pip_size': 0.0001, 'pip_value': 1.0,
                    'typical_spread': 0.0002, 'slippage_tolerance': 0.0002,
                    'min_stop_distance': 20, 'max_spread_filter': 0.0005,
                    'contract_size': 100000, 'session_multiplier': 1.0
                },
                'NZDUSD': {
                    'pair_type': 'major', 'pip_size': 0.0001, 'pip_value': 1.0,
                    'typical_spread': 0.0003, 'slippage_tolerance': 0.0003,
                    'min_stop_distance': 25, 'max_spread_filter': 0.0006,
                    'contract_size': 100000, 'session_multiplier': 1.0
                }
            }
            return major_specs.get(symbol, default_spec)
        
        # Minor pairs specifications
        elif symbol in self.symbol_categories['minor']:
            minor_spec = default_spec.copy()
            minor_spec.update({
                'pair_type': 'minor',
                'typical_spread': 0.0003,
                'slippage_tolerance': 0.0003,
                'min_stop_distance': 25,
                'max_spread_filter': 0.0008,
                'session_multiplier': 0.9
            })
            
            # JPY pairs have different pip size
            if 'JPY' in symbol:
                minor_spec.update({
                    'pip_size': 0.01,
                    'typical_spread': 0.03,
                    'slippage_tolerance': 0.03,
                    'max_spread_filter': 0.08
                })
            
            return minor_spec
        
        # Exotic pairs specifications
        else:
            exotic_spec = default_spec.copy()
            exotic_spec.update({
                'pair_type': 'exotic',
                'typical_spread': 0.0008,
                'slippage_tolerance': 0.0008,
                'min_stop_distance': 40,
                'max_spread_filter': 0.0020,
                'session_multiplier': 0.7
            })
            
            # Special handling for specific exotic pairs
            if 'JPY' in symbol:
                exotic_spec.update({
                    'pip_size': 0.01,
                    'typical_spread': 0.08,
                    'slippage_tolerance': 0.08,
                    'max_spread_filter': 0.20
                })
            elif any(curr in symbol for curr in ['TRY', 'ZAR', 'MXN', 'BRL']):
                # High volatility emerging market currencies
                exotic_spec.update({
                    'typical_spread': 0.0015,
                    'slippage_tolerance': 0.0015,
                    'min_stop_distance': 60,
                    'max_spread_filter': 0.0040,
                    'session_multiplier': 0.5
                })
            
            return exotic_spec
    
    def get_active_symbols(self) -> List[str]:
        """Get list of active trading symbols"""
        return self.active_symbols.copy()
    
    def get_symbol_specification(self, symbol: str) -> Dict[str, Any]:
        """Get specification for specific symbol"""
        return self.symbol_specs.get(symbol, {})
    
    def get_pair_type(self, symbol: str) -> str:
        """Get pair type for symbol"""
        spec = self.get_symbol_specification(symbol)
        return spec.get('pair_type', 'unknown')
    
    def is_symbol_active(self, symbol: str) -> bool:
        """Check if symbol is active for trading"""
        return symbol in self.active_symbols
    
    def get_session_active_symbols(self, current_time: datetime = None) -> List[str]:
        """Get symbols active for current trading session"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        current_hour = current_time.hour
        
        # Define session times (UTC)
        if 22 <= current_hour or current_hour < 7:  # Asian session
            session_symbols = ['USDJPY', 'AUDUSD', 'NZDUSD']
        elif 7 <= current_hour < 16:  # London session
            session_symbols = ['EURUSD', 'GBPUSD', 'USDCHF']
        else:  # New York session (12-21 UTC)
            session_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
        
        # Return intersection of session symbols and active symbols
        return [symbol for symbol in session_symbols if symbol in self.active_symbols]
    
    def get_correlation_groups(self) -> Dict[str, List[str]]:
        """Get correlation groups for risk management"""
        return {
            'eur_group': ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF'],
            'gbp_group': ['GBPUSD', 'EURGBP', 'GBPJPY', 'GBPCHF'],
            'jpy_group': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY'],
            'usd_group': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD'],
            'aud_group': ['AUDUSD', 'AUDCHF', 'AUDJPY', 'AUDCAD'],
            'commodity_group': ['AUDUSD', 'NZDUSD', 'USDCAD']
        }
    
    def calculate_position_limits(self) -> Dict[str, int]:
        """Calculate position limits by pair type"""
        limits = self.config.get('trading', {}).get('position_limits', {})
        
        return {
            'major_pairs': limits.get('major_pairs', 3),
            'minor_pairs': limits.get('minor_pairs', 2),
            'exotic_pairs': limits.get('exotic_pairs', 1),
            'total_positions': limits.get('major_pairs', 3) + limits.get('minor_pairs', 2) + limits.get('exotic_pairs', 1)
        }
    
    def get_risk_multiplier(self, symbol: str) -> float:
        """Get risk multiplier for symbol based on type"""
        pair_type = self.get_pair_type(symbol)
        
        risk_limits = self.config.get('risk_management', {}).get('pair_risk_limits', {})
        
        multipliers = {
            'major': risk_limits.get('major_pairs', 0.015),
            'minor': risk_limits.get('minor_pairs', 0.010),
            'exotic': risk_limits.get('exotic_pairs', 0.005)
        }
        
        return multipliers.get(pair_type, 0.010)
    
    def validate_symbol_for_trading(self, symbol: str) -> Tuple[bool, str]:
        """Validate if symbol is suitable for trading"""
        
        # Check if symbol is in our list
        if symbol not in self.all_symbols:
            return False, f"Symbol {symbol} not in supported symbols list"
        
        # Check if symbol is active
        if not self.is_symbol_active(symbol):
            return False, f"Symbol {symbol} not in active trading list"
        
        # Check MT5 availability (if connected)
        try:
            import MetaTrader5 as mt5
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False, f"Symbol {symbol} not available on broker"
            
            if not symbol_info.visible:
                return False, f"Symbol {symbol} not visible on broker"
            
        except ImportError:
            # MT5 not available, skip broker check
            pass
        except Exception as e:
            return False, f"Error checking symbol {symbol}: {str(e)}"
        
        return True, "Symbol validated successfully"
    
    def get_symbols_summary(self) -> Dict[str, Any]:
        """Get comprehensive symbols summary"""
        return {
            'total_symbols': len(self.all_symbols),
            'active_symbols': len(self.active_symbols),
            'major_pairs': len(self.symbol_categories['major']),
            'minor_pairs': len(self.symbol_categories['minor']),
            'exotic_pairs': len(self.symbol_categories['exotic']),
            'active_by_type': {
                'major': [s for s in self.active_symbols if s in self.symbol_categories['major']],
                'minor': [s for s in self.active_symbols if s in self.symbol_categories['minor']],
                'exotic': [s for s in self.active_symbols if s in self.symbol_categories['exotic']]
            },
            'position_limits': self.calculate_position_limits()
        }

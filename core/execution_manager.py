"""
Enhanced Execution Manager with Advanced Order Management
Professional-grade trade execution for MetaTrader 5 - XAUUSD OPTIMIZED VERSION
"""

import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import MetaTrader5 as mt5
from dataclasses import dataclass
import threading
from enum import Enum

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIAL = "partial"

@dataclass
class EnhancedPosition:
    """Enhanced position data structure"""
    ticket: int
    symbol: str
    type: int
    volume: float
    open_price: float
    stop_loss: float
    take_profit: float
    open_time: datetime
    current_price: float
    profit: float
    strategy: str
    comment: str
    risk_amount: float
    expected_profit: float
    confidence: float

class EnhancedExecutionManager:
    """
    Enhanced execution manager with advanced order management and monitoring
    """
    
    def __init__(self, config, market_intelligence):
        self.config = config
        self.market_intelligence = market_intelligence
        self.logger = logging.getLogger(__name__)
        
        # MT5 connection parameters
        self.connected = False
        self.connection_attempts = 0
        self.max_connection_attempts = config.get('mt5.max_connection_attempts', 3)
        self.connection_lock = threading.Lock()
        
        # Execution parameters
        self.magic_number = config.get('mt5.magic_number', 123456789)
        self.max_slippage = config.get('mt5.max_slippage', 3)
        self.execution_timeout = config.get('mt5.execution_timeout', 30)
        
        # Advanced execution features
        self.enable_partial_fills = config.get('execution.enable_partial_fills', True)
        self.max_retry_attempts = config.get('execution.max_retry_attempts', 3)
        self.retry_delay = config.get('execution.retry_delay_seconds', 1)
        
        # Position tracking with thread safety
        self.position_lock = threading.Lock()
        self.open_positions = {}
        self.pending_orders = {}
        self.execution_history = []
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_profit = 0.0
        self.execution_times = []
        self.slippage_data = []
        
        # Risk monitoring
        self.last_account_update = None
        self.account_balance = 0.0
        self.account_equity = 0.0
        
        # MT5 Return Codes (using numeric values)
        self.MT5_RETURN_CODES = {
            'DONE': 10009,
            'REQUOTE': 10004,
            'PRICE_OFF': 10016,
            'INSUFFICIENT_MONEY': 10019,
            'INVALID_VOLUME': 10014,
            'MARKET_CLOSED': 10018,
            'NO_MONEY': 10019,
            'PRICE_CHANGED': 10020,
            'REJECT': 10006,
            'ERROR': 10013,
            'UNSUPPORTED_FILLING': 10030
        }
        
        # Symbol filling mode cache for performance
        self.symbol_filling_modes = {}
        
        # Initialize MT5 connection
        self._connect_mt5()
        
        self.logger.info("Enhanced ExecutionManager initialized successfully")
        self.logger.info(f"Connected to MT5: {'Yes' if self.connected else 'No'}")

    def _connect_mt5(self) -> bool:
        """Enhanced MT5 connection with comprehensive error handling"""
        with self.connection_lock:
            try:
                self.connection_attempts += 1
                self.logger.info(f"MT5 connection attempt {self.connection_attempts}...")
                
                if not mt5.initialize():
                    error_code = mt5.last_error()
                    self.logger.error(f"MT5 initialization failed: {error_code}")
                    return False
                
                # Validate terminal info
                terminal_info = mt5.terminal_info()
                if terminal_info is None:
                    self.logger.error("Failed to get terminal info")
                    return False
                
                # Validate account info
                account_info = mt5.account_info()
                if account_info is None:
                    self.logger.error("Failed to get account info")
                    return False
                
                # Update account information
                self.account_balance = float(account_info.balance)
                self.account_equity = float(account_info.equity)
                self.last_account_update = datetime.now()
                
                self.connected = True
                self.logger.info(f"✅ Connected to MT5 Terminal")
                self.logger.info(f"Account: {account_info.login}")
                self.logger.info(f"Balance: ${self.account_balance:.2f}")
                self.logger.info(f"Equity: ${self.account_equity:.2f}")
                self.logger.info(f"Server: {account_info.server}")
                
                # Test connection with a symbol info request
                test_symbol = self.config.get('trading.symbols', ['EURUSD'])[0]
                symbol_info = mt5.symbol_info(test_symbol)
                if symbol_info is None:
                    self.logger.warning(f"Test symbol {test_symbol} not available")
                else:
                    self.logger.info(f"Test symbol {test_symbol} accessible")
                    # Cache filling mode for test symbol
                    self._cache_symbol_filling_mode(test_symbol, symbol_info)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Enhanced MT5 connection error: {e}")
                return False

    def execute_trade(self, signal: Dict[str, Any], risk_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Enhanced trade execution with comprehensive error handling and monitoring
        """
        if not self.connected:
            self.logger.warning("MT5 not connected, attempting reconnection...")
            if not self._connect_mt5():
                self.logger.error("Failed to reconnect to MT5")
                return None
        
        execution_start_time = time.time()
        
        try:
            # Extract parameters
            symbol = signal.get('symbol', '')
            direction = signal.get('direction', '')
            strategy = signal.get('strategy', 'Unknown')
            confidence = signal.get('confidence', 0.7)
            
            position_size = risk_params.get('position_size', 0.01)
            entry_price = risk_params.get('entry_price', 0)
            stop_loss = risk_params.get('stop_loss', 0)
            take_profit = risk_params.get('take_profit', 0)
            
            # Validate execution parameters
            validation_result = self._validate_execution_parameters(
                symbol, direction, position_size, entry_price, stop_loss, take_profit
            )
            
            if not validation_result['valid']:
                self.logger.error(f"Execution validation failed: {validation_result['reason']}")
                self.failed_trades += 1
                return None
            
            # Prepare symbol for trading
            if not self._prepare_symbol(symbol):
                self.logger.error(f"Failed to prepare symbol {symbol}")
                self.failed_trades += 1
                return None
            
            # Get current market data with validation
            market_data = self._get_validated_market_data(symbol)
            if not market_data:
                self.logger.error(f"Failed to get market data for {symbol}")
                self.failed_trades += 1
                return None
            
            # Calculate optimal execution price
            execution_price = self._calculate_execution_price(
                direction, market_data, entry_price
            )
            
            # Adjust stop loss and take profit for market conditions
            adjusted_sl, adjusted_tp = self._adjust_levels_for_execution(
                symbol, direction, execution_price, stop_loss, take_profit, market_data
            )
            
            # Create order request
            order_request = self._create_order_request(
                symbol, direction, position_size, execution_price, 
                adjusted_sl, adjusted_tp, strategy, confidence
            )
            
            if not order_request:
                self.logger.error("Failed to create order request")
                self.failed_trades += 1
                return None
            
            # Execute order with retry logic
            execution_result = self._execute_order_with_retry(order_request)
            
            if execution_result and execution_result.get('success'):
                # Process successful execution
                trade_result = self._process_successful_execution(
                    execution_result, signal, risk_params, execution_start_time
                )
                
                self.successful_trades += 1
                self.total_trades += 1
                
                return trade_result
            else:
                # Process failed execution
                self._process_failed_execution(execution_result, symbol, direction)
                self.failed_trades += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Enhanced trade execution error: {e}")
            self.failed_trades += 1
            return None

    def _validate_execution_parameters(self, symbol: str, direction: str, position_size: float,
                                     entry_price: float, stop_loss: float, 
                                     take_profit: float) -> Dict[str, Any]:
        """Comprehensive parameter validation"""
        try:
            # Basic parameter validation
            if not all([symbol, direction, position_size > 0, entry_price > 0]):
                return {'valid': False, 'reason': 'Missing or invalid basic parameters'}
            
            # Direction validation
            if direction.upper() not in ['BUY', 'SELL']:
                return {'valid': False, 'reason': f'Invalid direction: {direction}'}
            
            # Symbol validation
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {'valid': False, 'reason': f'Symbol {symbol} not available'}
            
            # Position size validation
            if position_size < symbol_info.volume_min:
                return {'valid': False, 'reason': f'Position size {position_size} below minimum {symbol_info.volume_min}'}
            
            if position_size > symbol_info.volume_max:
                return {'valid': False, 'reason': f'Position size {position_size} above maximum {symbol_info.volume_max}'}
            
            # Price validation
            current_tick = mt5.symbol_info_tick(symbol)
            if current_tick is None:
                return {'valid': False, 'reason': f'No current price data for {symbol}'}
            
            # Stop loss validation
            if stop_loss > 0:
                min_stop_distance = symbol_info.trade_stops_level * symbol_info.point
                if direction.upper() == 'BUY':
                    if entry_price - stop_loss < min_stop_distance:
                        return {'valid': False, 'reason': f'Stop loss too close to entry price'}
                else:
                    if stop_loss - entry_price < min_stop_distance:
                        return {'valid': False, 'reason': f'Stop loss too close to entry price'}
            
            return {'valid': True, 'reason': 'All parameters valid'}
            
        except Exception as e:
            self.logger.error(f"Error validating execution parameters: {e}")
            return {'valid': False, 'reason': f'Validation error: {e}'}

    def _prepare_symbol(self, symbol: str) -> bool:
        """Prepare symbol for trading"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
            
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    self.logger.error(f"Failed to select symbol {symbol}")
                    return False
                self.logger.info(f"Symbol {symbol} selected for trading")
            
            # Cache filling mode for future use
            self._cache_symbol_filling_mode(symbol, symbol_info)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error preparing symbol {symbol}: {e}")
            return False

    def _get_validated_market_data(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get validated current market data"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            
            # Validate tick data freshness (within last 10 seconds)
            current_time = datetime.now().timestamp()
            if current_time - tick.time > 10:
                self.logger.warning(f"Stale tick data for {symbol}: {current_time - tick.time} seconds old")
            
            return {
                'bid': float(tick.bid),
                'ask': float(tick.ask),
                'spread': float(tick.ask - tick.bid),
                'time': datetime.fromtimestamp(tick.time),
                'point': float(symbol_info.point),
                'digits': symbol_info.digits
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    def _calculate_execution_price(self, direction: str, market_data: Dict[str, float], 
                                 entry_price: float) -> float:
        """Calculate optimal execution price"""
        try:
            if direction.upper() == 'BUY':
                # For buy orders, use ask price
                execution_price = market_data['ask']
            else:
                # For sell orders, use bid price
                execution_price = market_data['bid']
            
            # Apply slippage protection
            max_slippage_price = self.max_slippage * market_data['point']
            
            if direction.upper() == 'BUY':
                # Don't pay more than entry + slippage
                if execution_price > entry_price + max_slippage_price:
                    execution_price = entry_price + max_slippage_price
            else:
                # Don't sell for less than entry - slippage
                if execution_price < entry_price - max_slippage_price:
                    execution_price = entry_price - max_slippage_price
            
            return execution_price
            
        except Exception as e:
            self.logger.error(f"Error calculating execution price: {e}")
            return entry_price

    def _adjust_levels_for_execution(self, symbol: str, direction: str, execution_price: float,
                                   stop_loss: float, take_profit: float, 
                                   market_data: Dict[str, float]) -> Tuple[float, float]:
        """Adjust stop loss and take profit for current market conditions"""
        try:
            point = market_data['point']
            
            # Get minimum stop distance
            symbol_info = mt5.symbol_info(symbol)
            min_stop_distance = symbol_info.trade_stops_level * point
            
            adjusted_sl = stop_loss
            adjusted_tp = take_profit
            
            if direction.upper() == 'BUY':
                # Adjust stop loss for buy order
                if stop_loss > 0:
                    min_sl = execution_price - min_stop_distance
                    if stop_loss > min_sl:
                        adjusted_sl = min_sl
                        self.logger.warning(f"Adjusted SL from {stop_loss:.5f} to {adjusted_sl:.5f}")
                
                # Adjust take profit for buy order
                if take_profit > 0:
                    min_tp = execution_price + min_stop_distance
                    if take_profit < min_tp:
                        adjusted_tp = min_tp
                        self.logger.warning(f"Adjusted TP from {take_profit:.5f} to {adjusted_tp:.5f}")
            
            else:  # SELL
                # Adjust stop loss for sell order
                if stop_loss > 0:
                    max_sl = execution_price + min_stop_distance
                    if stop_loss < max_sl:
                        adjusted_sl = max_sl
                        self.logger.warning(f"Adjusted SL from {stop_loss:.5f} to {adjusted_sl:.5f}")
                
                # Adjust take profit for sell order
                if take_profit > 0:
                    max_tp = execution_price - min_stop_distance
                    if take_profit > max_tp:
                        adjusted_tp = max_tp
                        self.logger.warning(f"Adjusted TP from {take_profit:.5f} to {adjusted_tp:.5f}")
            
            return adjusted_sl, adjusted_tp
            
        except Exception as e:
            self.logger.error(f"Error adjusting levels: {e}")
            return stop_loss, take_profit

    def _create_order_request(self, symbol: str, direction: str, position_size: float,
                            execution_price: float, stop_loss: float, take_profit: float,
                            strategy: str, confidence: float) -> Optional[Dict[str, Any]]:
        """✅ UPDATED: Create order request with XAUUSD-optimized slippage settings"""
        try:
            order_type = mt5.ORDER_TYPE_BUY if direction.upper() == 'BUY' else mt5.ORDER_TYPE_SELL
            
            # Create unique comment
            timestamp = datetime.now().strftime('%H%M%S')
            comment = f"{strategy[:8]}_{timestamp}_C{int(confidence*100)}"
            
            # ✅ UPDATED: Dynamic slippage based on symbol (with XAUUSD optimization)
            if symbol == 'USDJPY':
                max_slippage = 5  # Higher slippage tolerance for USDJPY (legacy - not used)
                self.logger.debug(f"Using legacy USDJPY slippage ({max_slippage}) for {symbol}")
            elif symbol in ['GBPJPY', 'EURJPY', 'AUDJPY', 'CADJPY', 'CHFJPY']:
                max_slippage = 4  # Medium slippage for JPY pairs
                self.logger.debug(f"Using medium slippage ({max_slippage}) for JPY pair {symbol}")
            elif symbol == 'XAUUSD':  # ✅ NEW: Gold-specific optimized settings
                max_slippage = 4  # Medium slippage for Gold (excellent liquidity)
                self.logger.debug(f"Using Gold-optimized slippage ({max_slippage}) for {symbol}")
            elif symbol in ['GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD']:
                max_slippage = self.max_slippage  # Default slippage for major pairs
                self.logger.debug(f"Using standard slippage ({max_slippage}) for major pair {symbol}")
            elif symbol in ['EURUSD']:
                max_slippage = self.max_slippage  # Default slippage for EUR pairs
                self.logger.debug(f"Using standard slippage ({max_slippage}) for EUR pair {symbol}")
            elif symbol in ['XAGUSD', 'BTCUSD', 'ETHUSD']:  # Other precious metals and crypto
                max_slippage = 5  # Higher slippage for volatile instruments
                self.logger.debug(f"Using high slippage ({max_slippage}) for volatile instrument {symbol}")
            else:
                max_slippage = self.max_slippage  # Default slippage for other pairs
                self.logger.debug(f"Using default slippage ({max_slippage}) for {symbol}")
            
            # Determine best filling mode for the broker
            filling_mode = self._get_compatible_filling_mode(symbol)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position_size,
                "type": order_type,
                "price": execution_price,
                "sl": stop_loss if stop_loss > 0 else 0,
                "tp": take_profit if take_profit > 0 else 0,
                "deviation": max_slippage,  # Use symbol-specific slippage
                "magic": self.magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            
            self.logger.debug(f"Order request created: {symbol} with slippage={max_slippage}, filling={filling_mode}")
            return request
            
        except Exception as e:
            self.logger.error(f"Error creating order request: {e}")
            return None

    def _get_compatible_filling_mode(self, symbol: str):
        """Get broker-compatible filling mode for the symbol"""
        try:
            # Check cache first for performance
            if symbol in self.symbol_filling_modes:
                return self.symbol_filling_modes[symbol]
            
            # Get symbol info to check supported filling modes
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.warning(f"Cannot get symbol info for {symbol}, using default filling mode")
                return mt5.ORDER_FILLING_FOK
            
            # Check supported filling modes (bit flags)
            filling_modes = symbol_info.filling_mode
            compatible_mode = None
            
            # Priority order: FOK -> IOC -> Return (most compatible first)
            if filling_modes & 1:  # ORDER_FILLING_FOK (Fill or Kill)
                compatible_mode = mt5.ORDER_FILLING_FOK
                self.logger.debug(f"Using ORDER_FILLING_FOK for {symbol}")
            elif filling_modes & 2:  # ORDER_FILLING_IOC (Immediate or Cancel) 
                compatible_mode = mt5.ORDER_FILLING_IOC
                self.logger.debug(f"Using ORDER_FILLING_IOC for {symbol}")
            elif filling_modes & 4:  # ORDER_FILLING_RETURN (Return/Partial fills allowed)
                compatible_mode = mt5.ORDER_FILLING_RETURN
                self.logger.debug(f"Using ORDER_FILLING_RETURN for {symbol}")
            else:
                # Fallback to FOK if no specific mode detected
                compatible_mode = mt5.ORDER_FILLING_FOK
                self.logger.warning(f"No supported filling mode detected for {symbol}, using FOK")
            
            # Cache the result for future use
            self.symbol_filling_modes[symbol] = compatible_mode
            return compatible_mode
            
        except Exception as e:
            self.logger.error(f"Error determining filling mode for {symbol}: {e}")
            return mt5.ORDER_FILLING_FOK  # Safe fallback

    def _cache_symbol_filling_mode(self, symbol: str, symbol_info) -> None:
        """Cache symbol filling mode for performance"""
        try:
            if symbol_info and hasattr(symbol_info, 'filling_mode'):
                filling_modes = symbol_info.filling_mode
                
                if filling_modes & 1:  # ORDER_FILLING_FOK
                    self.symbol_filling_modes[symbol] = mt5.ORDER_FILLING_FOK
                elif filling_modes & 2:  # ORDER_FILLING_IOC
                    self.symbol_filling_modes[symbol] = mt5.ORDER_FILLING_IOC
                elif filling_modes & 4:  # ORDER_FILLING_RETURN
                    self.symbol_filling_modes[symbol] = mt5.ORDER_FILLING_RETURN
                else:
                    self.symbol_filling_modes[symbol] = mt5.ORDER_FILLING_FOK
                
                self.logger.debug(f"Cached filling mode for {symbol}: {self.symbol_filling_modes[symbol]}")
                
        except Exception as e:
            self.logger.error(f"Error caching filling mode for {symbol}: {e}")

    def _execute_order_with_retry(self, order_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute order with enhanced retry logic and requote handling"""
        last_error = None
        
        for attempt in range(self.max_retry_attempts):
            try:
                self.logger.info(f"🚀 Executing {order_request['type']} order for {order_request['symbol']} "
                               f"(Attempt {attempt + 1}/{self.max_retry_attempts})")
                self.logger.info(f"   Volume: {order_request['volume']} lots")
                self.logger.info(f"   Price: {order_request['price']:.5f}")
                self.logger.info(f"   Stop Loss: {order_request['sl']:.5f}")
                self.logger.info(f"   Take Profit: {order_request['tp']:.5f}")
                
                # Send order to MT5
                result = mt5.order_send(order_request)
                
                if result is None:
                    last_error = "No result returned from MT5"
                    self.logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                    if attempt < self.max_retry_attempts - 1:
                        time.sleep(self.retry_delay)
                    continue
                
                # Check result using numeric codes
                if result.retcode == self.MT5_RETURN_CODES['DONE']:  # 10009
                    # Success
                    self.logger.info(f"✅ Order executed successfully!")
                    self.logger.info(f"   Ticket: {result.order}")
                    self.logger.info(f"   Executed Volume: {result.volume} lots")
                    self.logger.info(f"   Executed Price: {result.price:.5f}")
                    
                    return {
                        'success': True,
                        'result': result,
                        'attempt': attempt + 1,
                        'symbol': order_request['symbol'],
                        'direction': 'BUY' if order_request['type'] == mt5.ORDER_TYPE_BUY else 'SELL'
                    }
                
                elif result.retcode in [self.MT5_RETURN_CODES['REQUOTE'], self.MT5_RETURN_CODES['PRICE_OFF']]:
                    # Enhanced requote handling
                    self.logger.warning(f"Requote received (code: {result.retcode}) for {order_request['symbol']}")
                    
                    # Get fresh market data
                    market_data = self._get_validated_market_data(order_request['symbol'])
                    if market_data:
                        if order_request['type'] == mt5.ORDER_TYPE_BUY:
                            order_request['price'] = market_data['ask']
                        else:
                            order_request['price'] = market_data['bid']
                        
                        # ✅ ENHANCED: Symbol-specific slippage increase on requotes
                        if order_request['symbol'] == 'XAUUSD':
                            order_request['deviation'] = min(8, order_request['deviation'] + 1)
                            self.logger.info(f"Increased XAUUSD slippage to {order_request['deviation']}")
                        elif order_request['symbol'] == 'USDJPY':
                            order_request['deviation'] = min(10, order_request['deviation'] + 2)
                            self.logger.info(f"Increased USDJPY slippage to {order_request['deviation']}")
                        elif order_request['symbol'] in ['GBPJPY', 'EURJPY', 'AUDJPY']:
                            order_request['deviation'] = min(8, order_request['deviation'] + 1)
                            self.logger.info(f"Increased JPY pair slippage to {order_request['deviation']}")
                        else:
                            order_request['deviation'] = min(6, order_request['deviation'] + 1)
                            self.logger.info(f"Increased slippage to {order_request['deviation']} for {order_request['symbol']}")
                        
                        last_error = f"Requote - retrying with price {order_request['price']:.5f}"
                        self.logger.info(last_error)
                    else:
                        last_error = "Requote but failed to get fresh price"
                        break
                
                elif result.retcode == self.MT5_RETURN_CODES['UNSUPPORTED_FILLING']:
                    # Handle unsupported filling mode by trying different modes
                    self.logger.warning(f"Unsupported filling mode (code: {result.retcode}), trying alternative...")
                    
                    # Try different filling modes
                    current_filling = order_request['type_filling']
                    alternative_modes = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]
                    
                    for alt_mode in alternative_modes:
                        if alt_mode != current_filling:
                            order_request['type_filling'] = alt_mode
                            self.symbol_filling_modes[order_request['symbol']] = alt_mode
                            self.logger.info(f"Trying alternative filling mode: {alt_mode}")
                            break
                    else:
                        last_error = f"All filling modes failed (Code: {result.retcode})"
                        break
                
                elif result.retcode == self.MT5_RETURN_CODES['INSUFFICIENT_MONEY']:
                    last_error = f"Insufficient money - cannot retry (Code: {result.retcode})"
                    self.logger.error(f"❌ {last_error}")
                    break
                
                elif result.retcode == self.MT5_RETURN_CODES['INVALID_VOLUME']:
                    last_error = f"Invalid volume - cannot retry (Code: {result.retcode})"
                    self.logger.error(f"❌ {last_error}")
                    break
                
                elif result.retcode == self.MT5_RETURN_CODES['MARKET_CLOSED']:
                    last_error = f"Market closed - cannot retry (Code: {result.retcode})"
                    self.logger.error(f"❌ {last_error}")
                    break
                
                else:
                    # Other errors - log and potentially retry
                    last_error = f"Order failed: Code {result.retcode} - {getattr(result, 'comment', 'No comment')}"
                    self.logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                
                # Wait before retry
                if attempt < self.max_retry_attempts - 1:
                    time.sleep(self.retry_delay)
                
            except Exception as e:
                last_error = f"Exception during execution: {e}"
                self.logger.error(f"Attempt {attempt + 1} exception: {last_error}")
                
                if attempt < self.max_retry_attempts - 1:
                    time.sleep(self.retry_delay)
        
        # All attempts failed
        return {
            'success': False,
            'error': last_error,
            'attempts': self.max_retry_attempts
        }

    def _process_successful_execution(self, execution_result: Dict[str, Any], 
                                    signal: Dict[str, Any], risk_params: Dict[str, Any],
                                    execution_start_time: float) -> Dict[str, Any]:
        """Process successful trade execution"""
        try:
            result = execution_result['result']
            execution_time = time.time() - execution_start_time
            
            # Calculate slippage
            requested_price = signal.get('entry_price', 0)
            executed_price = result.price
            slippage_pips = abs(executed_price - requested_price) / mt5.symbol_info(result.request.symbol).point
            
            # Create enhanced position object
            enhanced_position = EnhancedPosition(
                ticket=result.order,
                symbol=result.request.symbol,
                type=result.request.type,
                volume=result.volume,
                open_price=result.price,
                stop_loss=result.request.sl,
                take_profit=result.request.tp,
                open_time=datetime.now(),
                current_price=result.price,
                profit=0.0,
                strategy=signal.get('strategy', 'Unknown'),
                comment=result.request.comment,
                risk_amount=risk_params.get('risk_amount', 0),
                expected_profit=risk_params.get('max_gain_amount', 0),
                confidence=signal.get('confidence', 0.7)
            )
            
            # Add to position tracking
            with self.position_lock:
                self.open_positions[result.order] = enhanced_position
            
            # Record execution metrics
            self.execution_times.append(execution_time)
            self.slippage_data.append(slippage_pips)
            
            # Keep metrics manageable
            if len(self.execution_times) > 100:
                self.execution_times = self.execution_times[-50:]
            if len(self.slippage_data) > 100:
                self.slippage_data = self.slippage_data[-50:]
            
            # Create comprehensive execution result
            trade_result = {
                'success': True,
                'ticket': result.order,
                'symbol': result.request.symbol,
                'direction': execution_result['direction'],
                'volume': result.volume,
                'open_price': result.price,
                'stop_loss': result.request.sl,
                'take_profit': result.request.tp,
                'strategy': signal.get('strategy', 'Unknown'),
                'confidence': signal.get('confidence', 0.7),
                'execution_time_seconds': execution_time,
                'slippage_pips': slippage_pips,
                'timestamp': datetime.now(),
                'comment': result.request.comment,
                'magic_number': result.request.magic,
                'risk_amount': risk_params.get('risk_amount', 0),
                'expected_profit': risk_params.get('max_gain_amount', 0),
                'attempts_used': execution_result.get('attempt', 1),
                'filling_mode_used': result.request.type_filling
            }
            
            # Add to execution history
            self.execution_history.append(trade_result.copy())
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-500:]
            
            return trade_result
            
        except Exception as e:
            self.logger.error(f"Error processing successful execution: {e}")
            return {
                'success': True,
                'ticket': execution_result['result'].order,
                'error': f"Processing error: {e}"
            }

    def _process_failed_execution(self, execution_result: Dict[str, Any], 
                                symbol: str, direction: str) -> None:
        """Process failed trade execution"""
        try:
            error = execution_result.get('error', 'Unknown error')
            attempts = execution_result.get('attempts', 1)
            
            self.logger.error(f"❌ Order execution failed after {attempts} attempts")
            self.logger.error(f"   Symbol: {symbol}")
            self.logger.error(f"   Direction: {direction}")
            self.logger.error(f"   Error: {error}")
            
            # Record failed execution
            failed_execution = {
                'success': False,
                'symbol': symbol,
                'direction': direction,
                'error': error,
                'attempts_used': attempts,
                'timestamp': datetime.now()
            }
            
            self.execution_history.append(failed_execution)
            
        except Exception as e:
            self.logger.error(f"Error processing failed execution: {e}")

    def manage_positions(self) -> None:
        """Enhanced position management with comprehensive monitoring"""
        if not self.connected or not self.open_positions:
            return
        
        try:
            # Get current positions from MT5
            mt5_positions = mt5.positions_get()
            if mt5_positions is None:
                mt5_positions = []
            
            # Convert to dictionary for easier lookup
            mt5_positions_dict = {pos.ticket: pos for pos in mt5_positions}
            
            with self.position_lock:
                positions_to_remove = []
                
                for ticket, tracked_position in self.open_positions.items():
                    if ticket in mt5_positions_dict:
                        # Position still open - update current data
                        mt5_pos = mt5_positions_dict[ticket]
                        tracked_position.current_price = mt5_pos.price_current
                        tracked_position.profit = mt5_pos.profit
                        
                        # Advanced position monitoring
                        self._monitor_position_advanced(tracked_position, mt5_pos)
                        
                    else:
                        # Position closed - record closure
                        self.total_profit += tracked_position.profit
                        
                        closure_info = {
                            'ticket': ticket,
                            'symbol': tracked_position.symbol,
                            'profit': tracked_position.profit,
                            'close_time': datetime.now(),
                            'hold_duration': datetime.now() - tracked_position.open_time,
                            'strategy': tracked_position.strategy,
                            'close_reason': 'market_close'  # Could be SL, TP, or manual
                        }
                        
                        self.logger.info(f"🔒 Position closed: {ticket} ({tracked_position.symbol}) "
                                       f"P&L: ${tracked_position.profit:.2f}")
                        
                        positions_to_remove.append(ticket)
                
                # Remove closed positions
                for ticket in positions_to_remove:
                    del self.open_positions[ticket]
            
            # Update account info periodically
            if (self.last_account_update is None or 
                (datetime.now() - self.last_account_update).seconds > 60):
                self._update_account_info()
            
        except Exception as e:
            self.logger.error(f"Error in enhanced position management: {e}")

    def _monitor_position_advanced(self, tracked_position: EnhancedPosition, mt5_pos) -> None:
        """✅ FIXED: Advanced position monitoring with corrected profit percentage calculation"""
        try:
            # ✅ IMPROVED: Better profit percentage calculation
            if tracked_position.risk_amount > 10.0:  # Only calculate if risk_amount is reasonable
                profit_percent = (tracked_position.profit / tracked_position.risk_amount) * 100
            else:
                # Use position value as reference for percentage calculation
                if tracked_position.symbol == 'XAUUSD':
                    # For XAUUSD, position value calculation
                    position_value = tracked_position.volume * tracked_position.open_price * 100  # XAUUSD contract size
                else:
                    # For forex pairs, standard calculation
                    position_value = tracked_position.volume * 100000  # Standard lot size
                
                if position_value > 0:
                    profit_percent = (tracked_position.profit / position_value) * 100
                else:
                    profit_percent = 0.0
            
            # Cap percentage values to reasonable range
            profit_percent = max(-100.0, min(100.0, profit_percent))
            
            # Only log significant moves (>$1) with reasonable percentages
            if abs(tracked_position.profit) > 1.0:
                if tracked_position.profit > 0:
                    self.logger.info(f"📈 Profit on {tracked_position.symbol}: ${tracked_position.profit:.2f} ({profit_percent:.1f}%)")
                else:
                    self.logger.info(f"📉 Loss on {tracked_position.symbol}: ${tracked_position.profit:.2f} ({profit_percent:.1f}%)")
            
            # Check for approaching stop loss or take profit
            current_price = tracked_position.current_price
            
            if tracked_position.type == mt5.ORDER_TYPE_BUY:
                if tracked_position.stop_loss > 0:
                    distance_to_sl = current_price - tracked_position.stop_loss
                    price_range = tracked_position.open_price - tracked_position.stop_loss
                    if price_range > 0 and (distance_to_sl / price_range) < 0.1:  # Within 10% of SL
                        self.logger.warning(f"⚠️ {tracked_position.symbol} approaching stop loss")
                
                if tracked_position.take_profit > 0:
                    distance_to_tp = tracked_position.take_profit - current_price
                    price_range = tracked_position.take_profit - tracked_position.open_price
                    if price_range > 0 and (distance_to_tp / price_range) < 0.1:  # Within 10% of TP
                        self.logger.info(f"🎯 {tracked_position.symbol} approaching take profit")
            
            else:  # SELL position
                if tracked_position.stop_loss > 0:
                    distance_to_sl = tracked_position.stop_loss - current_price
                    price_range = tracked_position.stop_loss - tracked_position.open_price
                    if price_range > 0 and (distance_to_sl / price_range) < 0.1:
                        self.logger.warning(f"⚠️ {tracked_position.symbol} approaching stop loss")
                
                if tracked_position.take_profit > 0:
                    distance_to_tp = current_price - tracked_position.take_profit
                    price_range = tracked_position.open_price - tracked_position.take_profit
                    if price_range > 0 and (distance_to_tp / price_range) < 0.1:
                        self.logger.info(f"🎯 {tracked_position.symbol} approaching take profit")
            
        except Exception as e:
            self.logger.error(f"Error in advanced position monitoring: {e}")

    def _update_account_info(self) -> None:
        """Update account information"""
        try:
            account_info = mt5.account_info()
            if account_info:
                self.account_balance = float(account_info.balance)
                self.account_equity = float(account_info.equity)
                self.last_account_update = datetime.now()
                
                self.logger.debug(f"Account updated: Balance=${self.account_balance:.2f}, Equity=${self.account_equity:.2f}")
        except Exception as e:
            self.logger.error(f"Error updating account info: {e}")

    def close_position(self, ticket: int, reason: str = 'manual') -> bool:
        """Enhanced position closing with comprehensive logging"""
        try:
            if ticket not in self.open_positions:
                self.logger.warning(f"Position {ticket} not found in tracking")
                # Try to close anyway in case it exists in MT5
            
            # Get position from MT5
            position = mt5.positions_get(ticket=ticket)
            if not position:
                self.logger.warning(f"Position {ticket} not found in MT5")
                # Remove from tracking if it was there
                with self.position_lock:
                    if ticket in self.open_positions:
                        del self.open_positions[ticket]
                return False
            
            position = position[0]
            
            # Prepare close request
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            # Get current price for closing
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                self.logger.error(f"Cannot get current price for {position.symbol}")
                return False
            
            price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
            
            # Use compatible filling mode for closing orders
            filling_mode = self._get_compatible_filling_mode(position.symbol)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": self.max_slippage,
                "magic": self.magic_number,
                "comment": f"Close_{reason}_{datetime.now().strftime('%H%M%S')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            
            self.logger.info(f"🔒 Closing position {ticket} ({position.symbol}) - Reason: {reason}")
            
            result = mt5.order_send(request)
            
            if result and result.retcode == self.MT5_RETURN_CODES['DONE']:
                self.logger.info(f"✅ Position {ticket} closed successfully")
                self.logger.info(f"   Final P&L: ${position.profit:.2f}")
                self.logger.info(f"   Close Price: {result.price:.5f}")
                
                # Remove from tracking
                with self.position_lock:
                    if ticket in self.open_positions:
                        del self.open_positions[ticket]
                
                return True
            else:
                error_msg = result.comment if result else 'No result'
                self.logger.error(f"Failed to close position {ticket}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error closing position {ticket}: {e}")
            return False

    def close_all_positions(self, reason: str = 'emergency') -> bool:
        """Enhanced close all positions with detailed reporting"""
        try:
            self.logger.warning(f"🚨 Closing all positions - Reason: {reason}")
            
            # Get all positions from MT5
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                self.logger.info("No positions to close")
                return True
            
            success_count = 0
            total_positions = len(positions)
            
            for position in positions:
                # Only close positions with our magic number
                if position.magic == self.magic_number:
                    if self.close_position(position.ticket, reason):
                        success_count += 1
                    time.sleep(0.1)  # Small delay between closes
            
            self.logger.warning(f"Closed {success_count}/{total_positions} positions")
            
            # Clear tracking for any remaining positions
            with self.position_lock:
                self.open_positions.clear()
            
            return success_count == total_positions
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
            return False

    def get_account_equity(self) -> float:
        """Get current account equity with error handling"""
        try:
            if not self.connected:
                return 0.0
            
            account_info = mt5.account_info()
            if account_info:
                return float(account_info.equity)
            else:
                return self.account_equity  # Return cached value
                
        except Exception as e:
            self.logger.error(f"Error getting account equity: {e}")
            return self.account_equity

    def get_account_balance(self) -> float:
        """Get current account balance with error handling"""
        try:
            if not self.connected:
                return 0.0
            
            account_info = mt5.account_info()
            if account_info:
                return float(account_info.balance)
            else:
                return self.account_balance  # Return cached value
                
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return self.account_balance

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get comprehensive list of open positions"""
        try:
            positions = []
            with self.position_lock:
                for ticket, pos in self.open_positions.items():
                    position_dict = {
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                        'volume': pos.volume,
                        'open_price': pos.open_price,
                        'current_price': pos.current_price,
                        'stop_loss': pos.stop_loss,
                        'take_profit': pos.take_profit,
                        'profit': pos.profit,
                        'strategy': pos.strategy,
                        'confidence': pos.confidence,
                        'open_time': pos.open_time,
                        'comment': pos.comment,
                        'risk_amount': pos.risk_amount,
                        'expected_profit': pos.expected_profit,
                        'hold_duration_minutes': (datetime.now() - pos.open_time).seconds / 60
                    }
                    positions.append(position_dict)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}")
            return []

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        try:
            success_rate = (self.successful_trades / max(1, self.total_trades)) * 100
            
            # Calculate average execution time
            avg_execution_time = np.mean(self.execution_times) if self.execution_times else 0
            
            # Calculate average slippage
            avg_slippage = np.mean(self.slippage_data) if self.slippage_data else 0
            
            # Get recent performance
            recent_executions = self.execution_history[-20:] if len(self.execution_history) >= 20 else self.execution_history
            recent_success_rate = (sum(1 for ex in recent_executions if ex.get('success', False)) / max(1, len(recent_executions))) * 100
            
            return {
                'connected': self.connected,
                'total_trades': self.total_trades,
                'successful_trades': self.successful_trades,
                'failed_trades': self.failed_trades,
                'success_rate': success_rate,
                'recent_success_rate': recent_success_rate,
                'total_profit': self.total_profit,
                'open_positions_count': len(self.open_positions),
                'account_equity': self.get_account_equity(),
                'account_balance': self.get_account_balance(),
                'avg_execution_time_seconds': avg_execution_time,
                'avg_slippage_pips': avg_slippage,
                'connection_attempts': self.connection_attempts,
                'last_account_update': self.last_account_update,
                'execution_history_count': len(self.execution_history),
                'cached_filling_modes': len(self.symbol_filling_modes)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting execution statistics: {e}")
            return {'connected': self.connected, 'error': str(e)}

    def test_connection(self) -> bool:
        """Enhanced connection test"""
        try:
            if not self.connected:
                return self._connect_mt5()
            
            # Test with multiple queries
            account_info = mt5.account_info()
            if account_info is None:
                return False
            
            # Test symbol access
            test_symbol = self.config.get('trading.symbols', ['EURUSD'])[0]
            symbol_info = mt5.symbol_info(test_symbol)
            if symbol_info is None:
                return False
            
            # Test tick data
            tick = mt5.symbol_info_tick(test_symbol)
            if tick is None:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    def shutdown(self) -> None:
        """Enhanced shutdown with comprehensive cleanup"""
        try:
            self.logger.info("Shutting down Enhanced ExecutionManager...")
            
            # Log final statistics
            stats = self.get_execution_statistics()
            self.logger.info(f"Final Execution Stats:")
            self.logger.info(f"  Total Trades: {stats.get('total_trades', 0)}")
            self.logger.info(f"  Success Rate: {stats.get('success_rate', 0):.1f}%")
            self.logger.info(f"  Total Profit: ${stats.get('total_profit', 0):.2f}")
            self.logger.info(f"  Avg Execution Time: {stats.get('avg_execution_time_seconds', 0):.2f}s")
            self.logger.info(f"  Avg Slippage: {stats.get('avg_slippage_pips', 0):.1f} pips")
            self.logger.info(f"  Cached Filling Modes: {stats.get('cached_filling_modes', 0)}")
            
            # Clear caches
            self.symbol_filling_modes.clear()
            
            # Disconnect from MT5
            if self.connected:
                mt5.shutdown()
                self.connected = False
                self.logger.info("MT5 connection closed")
            
            self.logger.info("✅ Enhanced ExecutionManager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during enhanced shutdown: {e}")

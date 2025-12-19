"""
Broker abstraction interface for multi-broker support.

This module defines a common interface that all broker clients must implement,
allowing the ExecutionEngine to work with any broker (Alpaca, DHAN, etc.)
without changing core trading logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BrokerClient(ABC):
    """
    Abstract base class for broker API clients.
    
    All broker implementations (Alpaca, DHAN, etc.) must implement this interface
    to ensure compatibility with the ExecutionEngine.
    """
    
    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        """
        Return account details (buying power, equity, etc.).
        
        Returns:
            Dict with account information including:
            - equity: float
            - buying_power: float
            - cash: float
            - etc. (broker-specific fields)
        """
        pass
    
    @abstractmethod
    def list_positions(self) -> list[Dict[str, Any]]:
        """
        Return a list of all open positions.
        
        Returns:
            List of position dicts, each containing:
            - symbol: str
            - qty: float (positive for long, negative for short)
            - market_value: float
            - avg_entry_price: float
            - etc. (broker-specific fields)
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Return open position for a symbol, or None if no position.
        
        Args:
            symbol: Trading symbol (broker-specific format)
            
        Returns:
            Position dict or None if no position exists
        """
        pass
    
    @abstractmethod
    def submit_order(
        self,
        *,
        symbol: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        side: str,
        order_type: str = "market",
        time_in_force: str = "gtc",
        take_profit_limit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit an order (simple or bracket).
        
        Args:
            symbol: Trading symbol
            qty: Quantity (mutually exclusive with notional)
            notional: Dollar amount (mutually exclusive with qty)
            side: "buy" or "sell"
            order_type: "market", "limit", etc.
            time_in_force: "gtc", "day", etc.
            take_profit_limit_price: Optional take-profit price
            stop_loss_price: Optional stop-loss price
            client_order_id: Optional client order ID for tracking
            
        Returns:
            Order response dict with order_id, status, etc.
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order by ID.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancellation response
        """
        pass
    
    @abstractmethod
    def get_last_trade(
        self,
        symbol: str,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        force_retry: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the last trade price for a symbol.
        
        Args:
            symbol: Trading symbol
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            force_retry: If True, retry even on first failure
            
        Returns:
            Dict with price information or None if unavailable
        """
        pass
    
    @property
    @abstractmethod
    def broker_name(self) -> str:
        """Return the broker name (e.g., 'alpaca', 'dhan')."""
        pass

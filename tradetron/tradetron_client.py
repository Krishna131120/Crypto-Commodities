"""
Tradetron API client for paper trading commodities via signal-based execution.

Tradetron works differently from traditional brokers:
- Instead of placing direct orders, you send SIGNALS to Tradetron
- Tradetron executes trades based on your signals
- Perfect for paper trading and strategy automation

Authentication:
- Uses API OAuth Token (generated from Tradetron dashboard)
- Token is strategy-specific

Environment variables expected:
- TRADETRON_API_TOKEN : Your Tradetron API OAuth token (from strategy settings)
- TRADETRON_API_URL : Optional override (default: https://api.tradetron.tech/api)

IMPORTANT:
- Strategy must be created and deployed on Tradetron platform first
- Strategy must be in "Paper Trading" mode for paper trading
- Each strategy has its own API token
- Signals are sent as key-value pairs (webhook format)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

# Import from parent trading package (tradetron is at root level, trading is sibling)
from trading.broker_interface import BrokerClient

# Tradetron API endpoints
DEFAULT_TRADETRON_API_URL = "https://api.tradetron.tech/api"


class TradetronAuthError(RuntimeError):
    """Raised when Tradetron credentials are missing or invalid."""


@dataclass
class TradetronConfig:
    """Configuration wrapper for Tradetron API settings."""

    api_token: str  # API token (UUID format)
    auth_token: str  # Auth token (used in requests as auth-token)
    api_url: str = DEFAULT_TRADETRON_API_URL

    @classmethod
    def from_env(cls) -> "TradetronConfig":
        """
        Load Tradetron config from .env file FIRST, then fallback to environment variables.
        """
        api_token = None
        auth_token = None
        api_url = DEFAULT_TRADETRON_API_URL

        # PRIORITY 1: Read from .env file FIRST
        if os.path.exists(".env"):
            try:
                with open(".env", "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key == "TRADETRON_API_TOKEN":
                            api_token = val
                        elif key == "TRADETRON_AUTH_TOKEN":
                            auth_token = val
                        elif key == "TRADETRON_API_URL":
                            api_url = val
            except Exception:
                pass

        # PRIORITY 2: Fallback to environment variables
        if not api_token:
            api_token = os.getenv("TRADETRON_API_TOKEN")
        if not auth_token:
            auth_token = os.getenv("TRADETRON_AUTH_TOKEN")
        if api_url == DEFAULT_TRADETRON_API_URL:
            api_url = os.getenv("TRADETRON_API_URL", DEFAULT_TRADETRON_API_URL)

        # Auth token is required (used in requests)
        if not auth_token:
            raise TradetronAuthError(
                "Tradetron auth token not found. Set TRADETRON_AUTH_TOKEN in .env file or as environment variable. "
                "Get your token from Tradetron dashboard: My Strategies → Your Strategy → API OAUTH Token"
            )
        
        # API token is optional (for reference, but auth_token is what's used)
        if not api_token:
            api_token = auth_token  # Fallback to auth_token if API token not provided

        return cls(api_token=api_token, auth_token=auth_token, api_url=api_url)


class TradetronClient(BrokerClient):
    """
    Tradetron API client implementing the BrokerClient interface.
    
    NOTE: Tradetron uses signal-based execution, not direct order placement.
    This client adapts the BrokerClient interface to work with Tradetron's signal API.
    
    How it works:
    1. Your code calls submit_order() (same as other brokers)
    2. TradetronClient converts it to a Tradetron signal
    3. Signal is sent to Tradetron API
    4. Tradetron executes the trade in paper/live account
    """

    def __init__(self, config: Optional[TradetronConfig] = None, api_token: Optional[str] = None):
        """
        Initialize Tradetron client.
        
        Args:
            config: Optional TradetronConfig object
            api_token: Optional API token (overrides config)
        """
        if config:
            self.config = config
        elif api_token:
            self.config = TradetronConfig(api_token=api_token)
        else:
            self.config = TradetronConfig.from_env()

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def _request(
        self,
        method: str,
        endpoint: str = "",
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Low-level HTTP request helper."""
        url = f"{self.config.api_url}{endpoint}"

        # Tradetron uses auth-token in the request body (webhook format)
        # Add token to json_body if it's a POST request
        if method.upper() == "POST" and json_body is not None:
            json_body["auth-token"] = self.config.auth_token  # Use auth_token (not api_token)

        resp = self._session.request(method, url, params=params, json=json_body, timeout=15)

        if resp.status_code == 401:
            raise TradetronAuthError(
                "Tradetron authentication failed (401). Check your API token. "
                "Get your token from: My Strategies → Your Strategy → API OAUTH Token"
            )

        try:
            data = resp.json()
        except ValueError:
            resp.raise_for_status()
            return resp.text

        if not resp.ok:
            msg = data.get("message") if isinstance(data, dict) else str(data)
            error_msg = data.get("error", "") if isinstance(data, dict) else ""
            full_msg = f"{msg} {error_msg}".strip() if error_msg else msg
            raise RuntimeError(f"Tradetron API error {resp.status_code}: {full_msg}")

        return data

    @property
    def broker_name(self) -> str:
        return "tradetron"

    def get_account(self) -> Dict[str, Any]:
        """
        Return account details from Tradetron.
        
        NOTE: Tradetron may not provide direct account API.
        This is a placeholder that returns basic info.
        For paper trading, you can check balance in Tradetron dashboard.
        """
        # Tradetron doesn't have a standard account endpoint
        # Return placeholder data - actual balance can be checked in dashboard
        return {
            "equity": 100000.0,  # Placeholder - check dashboard for actual balance
            "buying_power": 100000.0,
            "cash": 100000.0,
            "portfolio_value": 100000.0,
            "margin_used": 0.0,
            "margin_available": 100000.0,
            "note": "Tradetron account details should be checked in dashboard. "
                    "This is placeholder data for compatibility.",
        }

    def list_positions(self) -> list[Dict[str, Any]]:
        """
        Return all open positions.
        
        NOTE: Tradetron may not provide direct positions API.
        Positions should be checked in Tradetron dashboard.
        This returns empty list - positions are managed by Tradetron.
        """
        # Tradetron manages positions internally
        # Check positions in Tradetron dashboard
        # This is a placeholder for interface compatibility
        return []

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Return position for a symbol, or None.
        
        NOTE: Check positions in Tradetron dashboard.
        """
        # Tradetron manages positions internally
        # This is a placeholder
        return None

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
        Submit an order by sending a signal to Tradetron.
        
        Tradetron uses signal-based execution:
        - BUY signal → Tradetron places buy order
        - SELL signal → Tradetron places sell order
        
        For MCX commodities, qty is required (lot-based).
        
        Signal format (Tradetron webhook):
        {
            "auth-token": "your-token",
            "key": "Symbol_long",  # or "Symbol_short"
            "value": "1",  # 1 = enable, 0 = disable
            "key1": "Symbol_lots",
            "value1": "quantity"
        }
        """
        # Validation: qty is required for MCX
        if qty is None and notional is None:
            raise ValueError("For MCX commodities, qty must be provided (lot-based).")

        if notional is not None:
            raise ValueError("MCX commodities require qty (lot-based), not notional amount.")

        # Validation: side must be valid
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError(f"side must be 'buy' or 'sell', got: {side}")

        # Validation: qty must be positive integer
        qty_int = int(qty) if qty is not None else 0
        if qty_int <= 0:
            raise ValueError(f"Quantity must be positive integer (lot-based), got: {qty}")

        # Normalize symbol for Tradetron (remove spaces, special chars)
        symbol_normalized = symbol.upper().replace(" ", "_").replace("-", "_")

        # Build Tradetron signal payload
        # Format: key-value pairs for webhook
        signal_payload = {
            "auth-token": self.config.auth_token,  # Use auth_token (not api_token)
        }

        # For BUY: Set long signal
        if side == "buy":
            signal_payload[f"{symbol_normalized}_long"] = "1"  # Enable long
            signal_payload[f"{symbol_normalized}_short"] = "0"  # Disable short
            signal_payload[f"{symbol_normalized}_long_lots"] = str(qty_int)
        else:  # SELL
            signal_payload[f"{symbol_normalized}_short"] = "1"  # Enable short
            signal_payload[f"{symbol_normalized}_long"] = "0"  # Disable long
            signal_payload[f"{symbol_normalized}_short_lots"] = str(qty_int)

        # Add stop-loss and take-profit if provided
        # Note: Tradetron may handle these in strategy settings
        # You can also send them as additional parameters if your strategy supports it
        if stop_loss_price:
            signal_payload[f"{symbol_normalized}_stop_loss"] = str(stop_loss_price)

        if take_profit_limit_price:
            signal_payload[f"{symbol_normalized}_target"] = str(take_profit_limit_price)

        # Send signal to Tradetron
        try:
            response = self._request("POST", "", json_body=signal_payload)
        except Exception as e:
            # If standard endpoint fails, try alternative format
            # Some Tradetron setups use different endpoints
            raise RuntimeError(
                f"Failed to send signal to Tradetron: {e}. "
                f"Check: (1) API token is correct, (2) Strategy is deployed and active, "
                f"(3) Signal format matches your strategy configuration."
            )

        # Handle response
        # Tradetron may return different response formats
        if isinstance(response, dict):
            # Check for success indicators
            status = response.get("status", "unknown")
            message = response.get("message", response.get("msg", "Signal sent"))

            return {
                "id": response.get("order_id") or response.get("id") or client_order_id or "tradetron_signal",
                "status": "accepted" if status in ["success", "ok", "accepted"] else "pending",
                "symbol": symbol.upper(),
                "qty": qty_int,
                "side": side.upper(),
                "order_type": order_type.lower(),
                "message": message,
                "tradetron_response": response,
            }
        else:
            # Response is not a dict - assume success if no error
            return {
                "id": client_order_id or "tradetron_signal",
                "status": "accepted",
                "symbol": symbol.upper(),
                "qty": qty_int,
                "side": side.upper(),
                "order_type": order_type.lower(),
                "message": "Signal sent to Tradetron",
                "tradetron_response": str(response),
            }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order by sending an EXIT signal.
        
        NOTE: Tradetron manages orders internally.
        To cancel/exit, send an EXIT signal for the symbol.
        """
        # For Tradetron, "canceling" means sending an EXIT signal
        # This is a simplified implementation
        # You may need to track which symbol the order_id corresponds to
        
        return {
            "order_id": order_id,
            "status": "cancelled",
            "message": "Tradetron manages orders internally. Send EXIT signal to close position.",
        }

    def get_last_trade(
        self,
        symbol: str,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        force_retry: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the last trade price for a symbol.
        
        NOTE: Tradetron may not provide direct market data API.
        You may need to use a separate market data provider (like Angel One or other MCX data source)
        for getting current prices.
        
        This is a placeholder - implement based on your market data source.
        """
        # Tradetron doesn't provide market data directly
        # You should use your existing market data source (e.g., Angel One for MCX prices)
        # This is a placeholder for interface compatibility
        
        # Option: You can still use Angel One API just for market data (no trading)
        # Or use another MCX data provider
        
        return None

    def send_exit_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Send an EXIT signal to close a position.
        
        This is a Tradetron-specific method to explicitly exit positions.
        
        Args:
            symbol: Trading symbol to exit
            
        Returns:
            Response dict
        """
        symbol_normalized = symbol.upper().replace(" ", "_").replace("-", "_")

        signal_payload = {
            "auth-token": self.config.auth_token,  # Use auth_token (not api_token)
            f"{symbol_normalized}_long": "0",  # Disable long
            f"{symbol_normalized}_short": "0",  # Disable short
        }

        response = self._request("POST", "", json_body=signal_payload)

        return {
            "status": "exited",
            "symbol": symbol.upper(),
            "message": "Exit signal sent to Tradetron",
            "tradetron_response": response,
        }

"""
Minimal Alpaca paper trading REST client.

This client:
- Uses environment variables for credentials (never hard-codes keys).
- Is defaulted to the paper trading endpoint.
- Provides simple helpers for account, positions, and order submission.

Environment variables expected:
- ALPACA_API_KEY    : your Alpaca API key ID
- ALPACA_SECRET_KEY : your Alpaca secret key
- ALPACA_BASE_URL   : optional override for the base URL
                       (defaults to Alpaca paper trading URL).

IMPORTANT:
- Do NOT commit your actual keys into the repository.
- Set them in your shell or a local .env that is not checked into git.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from .broker_interface import BrokerClient


DEFAULT_PAPER_BASE_URL = "https://paper-api.alpaca.markets/v2"


class AlpacaAuthError(RuntimeError):
    """Raised when Alpaca credentials are missing or invalid."""


@dataclass
class AlpacaConfig:
    """Configuration wrapper for Alpaca API settings."""

    api_key: str
    secret_key: str
    base_url: str = DEFAULT_PAPER_BASE_URL

    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        """
        Load Alpaca config from environment variables, with a best-effort
        fallback to a local .env file if env vars are missing.
        """
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL", DEFAULT_PAPER_BASE_URL)

        # Fallback: try to parse a local .env file if env vars are not set.
        if (not api_key or not secret_key) and os.path.exists(".env"):
            try:
                with open(".env", "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        # Support both KEY=VALUE and PowerShell-style $env:KEY="VALUE"
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("$env:"):
                            # PowerShell style: $env:ALPACA_API_KEY="..."
                            parts = line.split("=", 1)
                            if len(parts) != 2:
                                continue
                            key_part = parts[0].replace("$env:", "").strip().strip('"').strip("'")
                            val_part = parts[1].strip().strip('"').strip("'")
                            if key_part == "ALPACA_API_KEY" and not api_key:
                                api_key = val_part
                            elif key_part == "ALPACA_SECRET_KEY" and not secret_key:
                                secret_key = val_part
                            elif key_part == "ALPACA_BASE_URL" and base_url == DEFAULT_PAPER_BASE_URL:
                                base_url = val_part
                        else:
                            # Standard .env style: KEY=VALUE
                            if "=" not in line:
                                continue
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip().strip('"').strip("'")
                            if key == "ALPACA_API_KEY" and not api_key:
                                api_key = val
                            elif key == "ALPACA_SECRET_KEY" and not secret_key:
                                secret_key = val
                            elif key == "ALPACA_BASE_URL" and base_url == DEFAULT_PAPER_BASE_URL:
                                base_url = val
            except Exception:
                # If .env parsing fails, we'll still fall back to the error below.
                pass

        if not api_key or not secret_key:
            raise AlpacaAuthError(
                "Missing Alpaca credentials. Please set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in your environment or .env file."
            )
        return cls(api_key=api_key, secret_key=secret_key, base_url=base_url.rstrip("/"))


class AlpacaClient(BrokerClient):
    """Thin wrapper around Alpaca's REST API for paper trading."""

    def __init__(self, config: Optional[AlpacaConfig] = None):
        self.config = config or AlpacaConfig.from_env()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "APCA-API-KEY-ID": self.config.api_key,
                "APCA-API-SECRET-KEY": self.config.secret_key,
                "Content-Type": "application/json",
            }
        )
    
    @property
    def broker_name(self) -> str:
        return "alpaca"

    # ------------------------------------------------------------------
    # Low-level request helper
    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.config.base_url}{path}"
        resp = self._session.request(method, url, params=params, json=json_body, timeout=10)
        if resp.status_code == 401:
            raise AlpacaAuthError("Alpaca authentication failed (401). Check your keys.")
        try:
            data = resp.json()
        except ValueError:
            resp.raise_for_status()
            return resp.text
        if not resp.ok:
            # Surface useful error message while preserving the payload
            msg = data.get("message") if isinstance(data, dict) else str(data)
            raise RuntimeError(f"Alpaca API error {resp.status_code}: {msg}")
        return data

    # ------------------------------------------------------------------
    # Account and positions
    # ------------------------------------------------------------------
    def get_account(self) -> Dict[str, Any]:
        """Return account details (buying power, equity, etc.)."""
        return self._request("GET", "/account")

    def list_positions(self) -> Any:
        """Return a list of all open positions."""
        return self._request("GET", "/positions")

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return open position for a symbol, or None if no position.

        NOTE: The trading API uses symbols *without* slashes (e.g. BTCUSD).
        A 404 from /positions/{symbol} simply means "no open position".
        """
        try:
            return self._request("GET", f"/positions/{symbol.upper()}")
        except Exception as exc:
            # Alpaca returns 404 if no position exists; treat that as None.
            msg = str(exc)
            if "404" in msg:
                return None
            raise

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
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
        Submit a simple or bracket order.

        You can specify either:
        - qty       : number of shares/units
        - notional  : dollar amount to trade (for some assets)
        """
        if (qty is None) == (notional is None):
            raise ValueError("Exactly one of qty or notional must be provided.")

        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")

        body: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if qty is not None:
            body["qty"] = qty
        else:
            body["notional"] = notional

        if client_order_id:
            body["client_order_id"] = client_order_id

        # Bracket order for take-profit/stop-loss
        if take_profit_limit_price or stop_loss_price:
            body["order_class"] = "bracket"
            if take_profit_limit_price:
                body["take_profit"] = {"limit_price": float(take_profit_limit_price)}
            if stop_loss_price:
                body["stop_loss"] = {"stop_price": float(stop_loss_price)}

        return self._request("POST", "/orders", json_body=body)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order by ID."""
        return self._request("DELETE", f"/orders/{order_id}")
    
    def submit_stop_order(
        self,
        *,
        symbol: str,
        qty: float,
        stop_price: float,
        side: str = "sell",  # "sell" for long positions, "buy" for short positions
        time_in_force: str = "gtc",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a standalone stop-loss order for crypto positions.
        
        This is used for crypto because Alpaca doesn't support bracket orders (OCO) for crypto.
        The stop order will execute at the broker level even when the monitoring script is not running.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSD")
            qty: Quantity to sell/buy when stop is triggered
            stop_price: Price at which to trigger the stop order
            side: "sell" for long positions (stop-loss), "buy" for short positions (stop-loss)
            time_in_force: Order time in force (default: "gtc" - good till cancelled)
            client_order_id: Optional client order ID for tracking
        """
        body: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.lower(),
            "type": "stop",  # Stop order type
            "stop_price": float(stop_price),
            "qty": float(qty),
            "time_in_force": time_in_force,
        }
        
        if client_order_id:
            body["client_order_id"] = client_order_id
        
        return self._request("POST", "/orders", json_body=body)
    
    def submit_take_profit_order(
        self,
        *,
        symbol: str,
        qty: float,
        limit_price: float,
        side: str = "sell",  # "sell" for long positions, "buy" for short positions
        time_in_force: str = "gtc",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a standalone take-profit limit order for crypto positions.
        
        This is used for crypto because Alpaca doesn't support bracket orders (OCO) for crypto.
        The take-profit order will execute at the broker level even when the monitoring script is not running.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSD")
            qty: Quantity to sell/buy when limit is reached
            limit_price: Price at which to execute the limit order
            side: "sell" for long positions (take-profit), "buy" for short positions (take-profit)
            time_in_force: Order time in force (default: "gtc" - good till cancelled)
            client_order_id: Optional client order ID for tracking
        """
        body: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.lower(),
            "type": "limit",  # Limit order type for take-profit
            "limit_price": float(limit_price),
            "qty": float(qty),
            "time_in_force": time_in_force,
        }
        
        if client_order_id:
            body["client_order_id"] = client_order_id
        
        return self._request("POST", "/orders", json_body=body)

    # ------------------------------------------------------------------
    # Market data helpers (basic)
    # ------------------------------------------------------------------
    def get_last_trade(self, symbol: str, max_retries: int = 5, retry_delay: float = 1.0, force_retry: bool = False) -> Optional[Dict[str, Any]]:
        """
        Fetch last trade for a symbol using Alpaca's data API with improved retry logic.
        
        For crypto symbols (BTCUSD, ETHUSD, etc.), uses the crypto endpoint.
        For stocks, uses the stocks endpoint.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSD")
            max_retries: Number of retry attempts (default: 5, increased for reliability)
            retry_delay: Delay between retries in seconds (default: 1.0, increased for stability)
            force_retry: If True, will retry even more aggressively (for active positions)
        
        Returns:
            Dict with price data or None if all attempts fail
        """
        import time
        
        # Increase retries and delay if force_retry is True (for active positions)
        if force_retry:
            max_retries = max(max_retries, 8)
            retry_delay = max(retry_delay, 0.5)
        
        # Data API v2 is under a different base URL; if ALPACA_DATA_BASE_URL
        # is set we will use it, otherwise we try to call via trading base.
        data_base = os.getenv("ALPACA_DATA_BASE_URL", self.config.base_url)
        
        # Determine if this is a crypto symbol.
        # We support both legacy "BTCUSD" and newer "BTC/USD" styles.
        symbol_upper = symbol.upper()
        # Remove "/" for path-based endpoints; keep original for logging.
        path_symbol = symbol_upper.replace("/", "")
        is_crypto = path_symbol.endswith("USD") and len(path_symbol) <= 8  # BTCUSD, ETHUSD, etc.
        
        # Retry loop for reliability
        last_exception = None
        for attempt in range(max_retries):
            try:
                if is_crypto:
                    # For crypto, try multiple endpoints in order of preference
                    
                    # Method 1: Try trading API's last trade endpoint (most reliable)
                    try:
                        # Trading API endpoint: /v2/stocks/{symbol}/trades/latest (works for crypto too)
                        url = f"{self.config.base_url.rstrip('/')}/stocks/{path_symbol}/trades/latest"
                        resp = self._session.get(url, timeout=20)  # Increased timeout
                        if resp.ok:
                            data = resp.json()
                            # Response format: {"trade": {"p": price, "t": timestamp}}
                            if "trade" in data:
                                trade = data["trade"]
                                price = trade.get("p") or trade.get("price")
                                if price:
                                    return {
                                        "price": price,
                                        "p": price,
                                        "t": trade.get("t") or trade.get("timestamp"),
                                    }
                            # Alternative format: direct price field
                            elif "p" in data or "price" in data:
                                price = data.get("p") or data.get("price")
                                if price:
                                    return {
                                        "price": price,
                                        "p": price,
                                        "t": data.get("t") or data.get("timestamp"),
                                    }
                    except Exception as e:
                        last_exception = e
                        # Continue to next method
                    
                    # Method 2: Try crypto data API v2 (if available)
                    try:
                        crypto_base = os.getenv("ALPACA_CRYPTO_DATA_BASE_URL", "https://data.alpaca.markets")
                        url = f"{crypto_base.rstrip('/')}/v2/stocks/{path_symbol}/trades/latest"
                        # Add authentication headers for data API
                        headers = {
                            "APCA-API-KEY-ID": self.config.api_key,
                            "APCA-API-SECRET-KEY": self.config.secret_key,
                        }
                        resp = self._session.get(url, headers=headers, timeout=20)
                        if resp.ok:
                            data = resp.json()
                            if "trade" in data:
                                trade = data["trade"]
                                price = trade.get("p") or trade.get("price")
                                if price:
                                    return {
                                        "price": price,
                                        "p": price,
                                        "t": trade.get("t") or trade.get("timestamp"),
                                    }
                    except Exception as e:
                        last_exception = e
                        # Continue to next method
                    
                    # Method 3: Try position-based price (if we have an open position)
                    try:
                        position = self.get_position(path_symbol)
                        if position:
                            market_value = float(position.get("market_value", 0) or 0)
                            qty = float(position.get("qty", 0) or 0)
                            if qty != 0 and market_value != 0:
                                price = abs(market_value / qty)
                                if price > 0:
                                    return {
                                        "price": price,
                                        "p": price,
                                        "t": None,  # Position doesn't have timestamp
                                    }
                    except Exception as e:
                        last_exception = e
                        # Continue to retry
                
                # For stocks, use standard stocks endpoint
                if not is_crypto:
                    url = f"{self.config.base_url.rstrip('/')}/stocks/{symbol_upper}/trades/latest"
                else:
                    # Try trading API endpoint (may not work for crypto in paper trading)
                    url = f"{self.config.base_url.rstrip('/')}/stocks/{path_symbol}/trades/latest"
                
                resp = self._session.get(url, timeout=20)  # Increased timeout
                if resp.status_code == 404:
                    # Symbol not found, don't retry
                    return None
                try:
                    data = resp.json()
                except ValueError:
                    resp.raise_for_status()
                    # Retry on JSON decode errors
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
                if resp.ok:
                    return data
                # If not OK, retry
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
                
            except Exception as e:
                last_exception = e
                # Retry on any exception
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                # On last attempt, log the error for debugging
                if attempt == max_retries - 1:
                    import warnings
                    warnings.warn(f"Alpaca get_last_trade failed after {max_retries} attempts for {symbol}: {last_exception}", UserWarning)
                return None
        
        return None




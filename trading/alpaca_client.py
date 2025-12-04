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


class AlpacaClient:
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

    # ------------------------------------------------------------------
    # Market data helpers (basic)
    # ------------------------------------------------------------------
    def get_last_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch last trade for a symbol using Alpaca's data API.
        
        For crypto symbols (BTCUSD, ETHUSD, etc.), uses the crypto endpoint.
        For stocks, uses the stocks endpoint.
        
        NOTE: Depending on your account and plan, you may or may not have
        access to all market data endpoints. This is a best-effort helper.
        """
        # Data API v2 is under a different base URL; if ALPACA_DATA_BASE_URL
        # is set we will use it, otherwise we try to call via trading base.
        data_base = os.getenv("ALPACA_DATA_BASE_URL", self.config.base_url)
        
        # Determine if this is a crypto symbol.
        # We support both legacy "BTCUSD" and newer "BTC/USD" styles.
        symbol_upper = symbol.upper()
        # Remove "/" for path-based endpoints; keep original for logging.
        path_symbol = symbol_upper.replace("/", "")
        is_crypto = path_symbol.endswith("USD") and len(path_symbol) <= 8  # BTCUSD, ETHUSD, etc.
        
        if is_crypto:
            # Use crypto endpoint: /v1beta1/crypto/{symbol}/trades/latest
            # Note: Alpaca crypto data API might be at a different base URL
            crypto_base = os.getenv("ALPACA_CRYPTO_DATA_BASE_URL", "https://data.alpaca.markets")
            url = f"{crypto_base.rstrip('/')}/v1beta1/crypto/{path_symbol}/trades/latest"
        else:
            # Use stocks endpoint
            url = f"{data_base.rstrip('/')}/stocks/{symbol_upper}/trades/latest"
        
        try:
            # For crypto, try without auth headers first (crypto data API is free)
            if is_crypto:
                # Try crypto data API without authentication (free for crypto)
                try:
                    import requests as req_lib
                    crypto_base = os.getenv("ALPACA_CRYPTO_DATA_BASE_URL", "https://data.alpaca.markets")
                    url = f"{crypto_base.rstrip('/')}/v1beta1/crypto/{path_symbol}/trades/latest"
                    # Crypto data API doesn't require auth
                    resp = req_lib.get(url, timeout=10)
                    if resp.ok:
                        data = resp.json()
                        if "trade" in data:
                            trade = data["trade"]
                            return {
                                "price": trade.get("p"),  # price
                                "p": trade.get("p"),
                                "t": trade.get("t"),  # timestamp
                            }
                except Exception:
                    pass
                
                # Fallback: try latest bar
                try:
                    import requests as req_lib
                    crypto_base = os.getenv("ALPACA_CRYPTO_DATA_BASE_URL", "https://data.alpaca.markets")
                    url = f"{crypto_base.rstrip('/')}/v1beta1/crypto/{path_symbol}/bars/latest"
                    resp = req_lib.get(url, timeout=10)
                    if resp.ok:
                        data = resp.json()
                        if "bar" in data:
                            bar = data["bar"]
                            return {
                                "price": bar.get("c"),  # close price
                                "p": bar.get("c"),
                                "t": bar.get("t"),  # timestamp
                            }
                except Exception:
                    pass
            
            # For stocks or if crypto failed, use authenticated session
            resp = self._session.get(url, timeout=10)
            if resp.status_code == 404:
                return None
            try:
                data = resp.json()
            except ValueError:
                resp.raise_for_status()
                return None
            if not resp.ok:
                return None
            return data
        except Exception:
            return None




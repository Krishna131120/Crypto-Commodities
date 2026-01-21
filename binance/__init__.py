"""
Binance trading integration package.

This package contains all Binance-specific trading components:
- BinanceClient: REST API client for Binance spot trading
- Live trader: Binance-specific trading loop
- Position management: Binance positions stored separately
- Logs: Binance trading logs stored in binance/logs/

All components use the same concepts as Alpaca but are organized separately
for better interactivity and clarity.
"""

from .binance_client import BinanceClient, BinanceConfig, BinanceAuthError

__all__ = [
    "BinanceClient",
    "BinanceConfig",
    "BinanceAuthError",
]

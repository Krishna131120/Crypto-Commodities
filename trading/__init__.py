"""
Trading integration package.

This module contains:
- Symbol universe and data_symbol <-> trading_symbol mappings.
- Broker abstraction interface for multi-broker support.
- Alpaca paper-trading REST client wrapper.
- Angel One API client for MCX commodities trading.
- Execution engine and live trading loop that connect
  model predictions to broker orders.

Note: Tradetron is a separate package. Import directly:
    from tradetron.tradetron_client import TradetronClient
"""

from .alpaca_client import AlpacaClient, AlpacaConfig, AlpacaAuthError
from .binance_client import BinanceClient, BinanceConfig, BinanceAuthError
from .broker_interface import BrokerClient
from .angelone_client import AngelOneClient, AngelOneConfig, AngelOneAuthError
from .execution_engine import ExecutionEngine, TradingRiskConfig
from .position_manager import PositionManager
from .symbol_universe import AssetMapping, all_enabled, find_by_data_symbol, find_by_trading_symbol

__all__ = [
    "AlpacaClient",
    "AlpacaConfig",
    "AlpacaAuthError",
    "BinanceClient",
    "BinanceConfig",
    "BinanceAuthError",
    "BrokerClient",
    "AngelOneClient",
    "AngelOneConfig",
    "AngelOneAuthError",
    "ExecutionEngine",
    "TradingRiskConfig",
    "PositionManager",
    "AssetMapping",
    "all_enabled",
    "find_by_data_symbol",
    "find_by_trading_symbol",
]


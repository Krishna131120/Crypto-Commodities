"""
Trading integration package.

This module contains:
- Symbol universe and data_symbol <-> trading_symbol mappings.
- Broker abstraction interface for multi-broker support.
- Alpaca paper-trading REST client wrapper.
- DHAN API client (template - needs API documentation).
- Execution engine and live trading loop that connect
  model predictions to broker orders.
"""

from .alpaca_client import AlpacaClient, AlpacaConfig, AlpacaAuthError
from .broker_interface import BrokerClient
from .dhan_client import DhanClient, DhanConfig, DhanAuthError
from .execution_engine import ExecutionEngine, TradingRiskConfig
from .position_manager import PositionManager
from .symbol_universe import AssetMapping, all_enabled, find_by_data_symbol, find_by_trading_symbol

__all__ = [
    "AlpacaClient",
    "AlpacaConfig",
    "AlpacaAuthError",
    "BrokerClient",
    "DhanClient",
    "DhanConfig",
    "DhanAuthError",
    "ExecutionEngine",
    "TradingRiskConfig",
    "PositionManager",
    "AssetMapping",
    "all_enabled",
    "find_by_data_symbol",
    "find_by_trading_symbol",
]


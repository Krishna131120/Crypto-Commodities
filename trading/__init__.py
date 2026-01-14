"""
Trading integration package.

This module contains:
- Symbol universe and data_symbol <-> trading_symbol mappings.
- Broker abstraction interface for multi-broker support.
- Alpaca paper-trading REST client wrapper.
- Angel One API client for MCX commodities trading.
- Tradetron API client for paper trading via signals.
- Execution engine and live trading loop that connect
  model predictions to broker orders.
"""

from .alpaca_client import AlpacaClient, AlpacaConfig, AlpacaAuthError
from .broker_interface import BrokerClient
from .angelone_client import AngelOneClient, AngelOneConfig, AngelOneAuthError
from tradetron import TradetronClient, TradetronConfig, TradetronAuthError
from .execution_engine import ExecutionEngine, TradingRiskConfig
from .position_manager import PositionManager
from .symbol_universe import AssetMapping, all_enabled, find_by_data_symbol, find_by_trading_symbol

__all__ = [
    "AlpacaClient",
    "AlpacaConfig",
    "AlpacaAuthError",
    "BrokerClient",
    "AngelOneClient",
    "AngelOneConfig",
    "AngelOneAuthError",
    "TradetronClient",
    "TradetronConfig",
    "TradetronAuthError",
    "ExecutionEngine",
    "TradingRiskConfig",
    "PositionManager",
    "AssetMapping",
    "all_enabled",
    "find_by_data_symbol",
    "find_by_trading_symbol",
]


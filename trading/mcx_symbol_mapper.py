"""
MCX Symbol Mapping System

Maps Yahoo Finance commodity symbols (GC=F, CL=F, etc.) to MCX contract symbols.
Handles contract expiry and auto-selects current month contract.

MCX Contract Format:
- Gold: GOLDFEB24, GOLDMAR24, etc. (GOLD + Month + Year)
- Crude Oil: CRUDEOILFEB24, CRUDEOILMAR24, etc.
- Silver: SILVERFEB24, SILVERMAR24, etc.

Contract Expiry:
- MCX contracts typically expire on the last business day of the month
- We auto-select the current month contract
- Handle rollover when contract expires
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

# MCX Symbol Mapping: Yahoo Symbol -> MCX Base Symbol
# For commodities without Yahoo equivalents, we use MCX-specific symbols (MCX_ prefix)
MCX_BASE_SYMBOLS: Dict[str, str] = {
    # Bullion
    "GC=F": "GOLD",           # Gold futures
    "SI=F": "SILVER",          # Silver futures
    "PL=F": "PLATINUM",        # Platinum futures
    "MCX_GOLDM": "GOLDM",      # Gold Mini
    "MCX_GOLDGUINEA": "GOLDGUINEA",  # Gold Guinea
    "MCX_GOLDPETAL": "GOLDPETAL",    # Gold Petal
    "MCX_SILVERM": "SILVERM",   # Silver Mini
    "MCX_SILVERMIC": "SILVERMIC",    # Silver Micro
    "MCX_SILVER1000": "SILVER1000",  # Silver 1000
    
    # Energy
    "CL=F": "CRUDEOIL",        # Crude oil futures
    "BZ=F": "BRENTCRUDE",      # Brent crude oil
    "NG=F": "NATURALGAS",      # Natural gas futures
    "MCX_CRUDEOILM": "CRUDEOILM",  # Crude Oil Mini
    
    # Base Metals
    "MCX_ALUMINIUM": "ALUMINIUM",  # Aluminium
    "MCX_ALUMINI": "ALUMINI",      # Aluminium Mini
    "HG=F": "COPPER",          # Copper (HG=F is COMEX copper, closest Yahoo equivalent)
    "MCX_LEAD": "LEAD",        # Lead
    "MCX_LEADMINI": "LEADMINI",    # Lead Mini
    "MCX_NICKEL": "NICKEL",    # Nickel
    "MCX_ZINC": "ZINC",        # Zinc
    "MCX_ZINCMINI": "ZINCMINI",    # Zinc Mini
    
    # Agricultural
    "ZC=F": "CORN",            # Corn futures
    "ZS=F": "SOYBEAN",         # Soybean futures
    "ZW=F": "WHEAT",           # Wheat futures
    "MCX_CARDAMOM": "CARDAMOM",    # Cardamom
    "CT=F": "COTTON",          # Cotton (CT=F is ICE cotton, closest Yahoo equivalent)
    "MCX_CPO": "CPO",          # Crude Palm Oil
    "MCX_MENTHAOIL": "MENTHAOIL",  # Mentha Oil
}

# MCX Month Codes
MCX_MONTH_CODES = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR",
    5: "MAY", 6: "JUN", 7: "JUL", 8: "AUG",
    9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


def get_mcx_contract_symbol(yahoo_symbol: str, month: Optional[int] = None, year: Optional[int] = None) -> str:
    """
    Convert Yahoo Finance symbol or MCX-specific symbol to MCX contract symbol.
    
    Args:
        yahoo_symbol: Yahoo Finance symbol (e.g., "GC=F") or MCX-specific symbol (e.g., "MCX_GOLDM")
        month: Optional month (1-12). If None, uses current month.
        year: Optional year (2-digit, e.g., 24). If None, uses current year.
        
    Returns:
        MCX contract symbol (e.g., "GOLDFEB24")
    """
    base_symbol = MCX_BASE_SYMBOLS.get(yahoo_symbol.upper())
    if not base_symbol:
        # If symbol is already an MCX base symbol (no mapping needed), use it directly
        # This handles cases where data_symbol is already "GOLD", "SILVER", etc.
        if yahoo_symbol.upper() in MCX_BASE_SYMBOLS.values():
            base_symbol = yahoo_symbol.upper()
        else:
            raise ValueError(f"No MCX mapping found for symbol: {yahoo_symbol}. Available symbols: {list(MCX_BASE_SYMBOLS.keys())}")
    
    now = datetime.now()
    contract_month = month if month is not None else now.month
    contract_year = year if year is not None else int(str(now.year)[-2:])
    
    month_code = MCX_MONTH_CODES[contract_month]
    
    return f"{base_symbol}{month_code}{contract_year:02d}"


def get_current_mcx_contract(yahoo_symbol: str) -> str:
    """
    Get the current month MCX contract for a Yahoo symbol.
    
    Args:
        yahoo_symbol: Yahoo Finance symbol (e.g., "GC=F")
        
    Returns:
        Current month MCX contract symbol
    """
    return get_mcx_contract_symbol(yahoo_symbol)


def get_next_mcx_contract(yahoo_symbol: str) -> str:
    """
    Get the next month MCX contract for a Yahoo symbol.
    
    Args:
        yahoo_symbol: Yahoo Finance symbol (e.g., "GC=F")
        
    Returns:
        Next month MCX contract symbol
    """
    now = datetime.now()
    next_month = (now.month % 12) + 1
    next_year = now.year if next_month > now.month else now.year + 1
    
    return get_mcx_contract_symbol(
        yahoo_symbol,
        month=next_month,
        year=int(str(next_year)[-2:])
    )


def get_mcx_contract_for_horizon(yahoo_symbol: str, horizon: str) -> str:
    """
    Get MCX contract based on trading horizon.
    
    Args:
        yahoo_symbol: Yahoo Finance symbol (e.g., "GC=F")
        horizon: Trading horizon ("intraday", "short", "long")
        
    Returns:
        MCX contract symbol for the horizon
    """
    horizon = horizon.lower()
    
    if horizon == "intraday":
        # Intraday: Use current month contract
        return get_current_mcx_contract(yahoo_symbol)
    elif horizon == "short":
        # Short-term: Use current or next month contract
        return get_current_mcx_contract(yahoo_symbol)
    elif horizon == "long":
        # Long-term: Use next month contract
        return get_next_mcx_contract(yahoo_symbol)
    else:
        # Default: Current month
        return get_current_mcx_contract(yahoo_symbol)


def get_mcx_lot_size(yahoo_symbol: str) -> int:
    """
    Get MCX lot size for a commodity.
    
    MCX has fixed lot sizes for each commodity:
    - Gold: 1 kg (1 lot)
    - Silver: 30 kg (1 lot)
    - Crude Oil: 100 barrels (1 lot)
    - etc.
    
    Args:
        yahoo_symbol: Yahoo Finance symbol
        
    Returns:
        Lot size (minimum tradable quantity)
    """
    lot_sizes: Dict[str, int] = {
        # Bullion
        "GC=F": 1,              # Gold: 1 kg per lot
        "SI=F": 30,             # Silver: 30 kg per lot
        "PL=F": 1,              # Platinum: 1 kg per lot
        "MCX_GOLDM": 100,       # Gold Mini: 100 grams per lot
        "MCX_GOLDGUINEA": 8,    # Gold Guinea: 8 grams per lot
        "MCX_GOLDPETAL": 1,     # Gold Petal: 1 gram per lot
        "MCX_SILVERM": 5,       # Silver Mini: 5 kg per lot
        "MCX_SILVERMIC": 1,     # Silver Micro: 1 kg per lot
        "MCX_SILVER1000": 30,   # Silver 1000: 30 kg per lot
        
        # Energy
        "CL=F": 100,            # Crude Oil: 100 barrels per lot
        "BZ=F": 100,            # Brent Crude: 100 barrels per lot
        "NG=F": 1250,           # Natural Gas: 1250 MMBtu per lot
        "MCX_CRUDEOILM": 10,    # Crude Oil Mini: 10 barrels per lot
        
        # Base Metals
        "MCX_ALUMINIUM": 5,     # Aluminium: 5 MT per lot
        "MCX_ALUMINI": 1,       # Aluminium Mini: 1 MT per lot
        "HG=F": 25000,          # Copper: 25,000 lbs per lot (COMEX standard, MCX may differ)
        "MCX_LEAD": 5,          # Lead: 5 MT per lot
        "MCX_LEADMINI": 1,      # Lead Mini: 1 MT per lot
        "MCX_NICKEL": 1,         # Nickel: 1 MT per lot
        "MCX_ZINC": 5,          # Zinc: 5 MT per lot
        "MCX_ZINCMINI": 1,      # Zinc Mini: 1 MT per lot
        
        # Agricultural
        "ZC=F": 100,            # Corn: 100 bushels per lot
        "ZS=F": 100,            # Soybean: 100 bushels per lot
        "ZW=F": 100,            # Wheat: 100 bushels per lot
        "MCX_CARDAMOM": 100,    # Cardamom: 100 kg per lot
        "CT=F": 50000,          # Cotton: 50,000 lbs per lot (ICE standard, MCX may differ)
        "MCX_CPO": 10,          # Crude Palm Oil: 10 MT per lot
        "MCX_MENTHAOIL": 360,   # Mentha Oil: 360 kg per lot
    }
    
    return lot_sizes.get(yahoo_symbol.upper(), 1)  # Default to 1 if not found


def round_to_lot_size(quantity: float, yahoo_symbol: str) -> int:
    """
    Round quantity to nearest MCX lot size.
    
    Args:
        quantity: Desired quantity
        yahoo_symbol: Yahoo Finance symbol
        
    Returns:
        Rounded quantity in lots (integer)
    """
    lot_size = get_mcx_lot_size(yahoo_symbol)
    lots = int(round(quantity / lot_size))
    return max(1, lots) * lot_size  # Minimum 1 lot

"""
MCX Symbol Mapping System

Maps Yahoo Finance commodity symbols (GC=F, CL=F, etc.) to MCX contract symbols.
Handles contract expiry and auto-selects current month contract.

MCX Contract Format (AngelOne):
- Format: {BASE_SYMBOL}{DAY:02d}{MONTH}{YEAR:02d}FUT
- Example: GOLD05FEB26FUT = GOLD + 05 (day) + FEB (month) + 26 (year) + FUT
- Example: GOLDGUINEA30JAN26FUT = GOLDGUINEA + 30 (day) + JAN (month) + 26 (year) + FUT

Contract Expiry:
- MCX contracts expire on specific days of the month (varies by commodity)
- Common expiry days: 05 (GOLD), 27/30/31 (GOLDGUINEA, GOLDPETAL, SILVERM), 18/19/20 (CRUDEOIL)
- We auto-select the current month contract with appropriate expiry day
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
    "MCX_COPPERMI": "COPPERMI",    # Copper Mini
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

# MCX Contract Expiry Days (typical expiry day for each base symbol)
# Format: {BASE_SYMBOL: [list of common expiry days to try]}
# If first day doesn't work, symbol lookup will try others
MCX_EXPIRY_DAYS: Dict[str, list[int]] = {
    # Bullion
    "GOLD": [5, 4, 2],           # Usually 05, sometimes 04, 02
    "SILVER": [5, 4, 3],         # Usually 05, sometimes 04, 03
    "PLATINUM": [5],             # Usually 05
    "GOLDM": [5, 3],             # Usually 05, sometimes 03
    "GOLDGUINEA": [30, 27, 31, 29],  # Usually 30, sometimes 27, 31, 29
    "GOLDPETAL": [30, 27, 31, 29],   # Usually 30, sometimes 27, 31, 29
    "SILVERM": [30, 27, 31],     # Usually 30, sometimes 27, 31
    "SILVERMIC": [30, 27, 31],   # Usually 30, sometimes 27, 31
    "SILVER1000": [30, 27, 31],  # Usually 30, sometimes 27, 31
    
    # Energy
    "CRUDEOIL": [20, 19, 18],    # Usually 20, sometimes 19, 18
    "BRENTCRUDE": [20, 19, 18],  # Usually 20, sometimes 19, 18
    "NATURALGAS": [20, 19, 18],  # Usually 20, sometimes 19, 18
    "CRUDEOILM": [20, 19, 18],   # Usually 20, sometimes 19, 18
    
    # Base Metals (default to 30 if not specified)
    "ALUMINIUM": [30, 27, 31],
    "ALUMINI": [30, 27, 31],
    "COPPER": [30, 27, 31],
    "COPPERMI": [30, 27, 31],
    "LEAD": [30, 27, 31],
    "LEADMINI": [30, 27, 31],
    "NICKEL": [30, 27, 31],
    "ZINC": [30, 27, 31],
    "ZINCMINI": [30, 27, 31],
    
    # Agricultural (default to 30 if not specified)
    "CORN": [30, 27, 31],
    "SOYBEAN": [30, 27, 31],
    "WHEAT": [30, 27, 31],
    "CARDAMOM": [30, 27, 31],
    "COTTON": [30, 27, 31],
    "CPO": [30, 27, 31],
    "MENTHAOIL": [30, 27, 31],
}


def get_mcx_contract_symbol(yahoo_symbol: str, month: Optional[int] = None, year: Optional[int] = None, expiry_day: Optional[int] = None) -> str:
    """
    Convert Yahoo Finance symbol or MCX-specific symbol to MCX contract symbol.
    
    AngelOne MCX format: {BASE_SYMBOL}{DAY:02d}{MONTH}{YEAR:02d}FUT
    Example: GOLD05FEB26FUT = GOLD + 05 (day) + FEB (month) + 26 (year) + FUT
    
    Args:
        yahoo_symbol: Yahoo Finance symbol (e.g., "GC=F") or MCX-specific symbol (e.g., "MCX_GOLDM")
        month: Optional month (1-12). If None, uses current month.
        year: Optional year (2-digit, e.g., 26). If None, uses current year.
        expiry_day: Optional expiry day (1-31). If None, uses default for the commodity.
        
    Returns:
        MCX contract symbol (e.g., "GOLD05FEB26FUT")
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
    
    # Get expiry day (default based on commodity)
    if expiry_day is None:
        expiry_days = MCX_EXPIRY_DAYS.get(base_symbol, [30])  # Default to 30 if not specified
        expiry_day = expiry_days[0]  # Use first (most common) expiry day
    
    month_code = MCX_MONTH_CODES[contract_month]
    
    # AngelOne format: {BASE}{DAY:02d}{MONTH}{YEAR:02d}FUT
    return f"{base_symbol}{expiry_day:02d}{month_code}{contract_year:02d}FUT"


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
        "MCX_COPPERMI": 1,      # Copper Mini: 1 MT per lot
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

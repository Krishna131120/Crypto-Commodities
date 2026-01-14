"""
Comprehensive trade logging with profit/loss tracking.

Logs all trades, positions, and outcomes with detailed information.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dataclasses import dataclass, asdict


@dataclass
class TradeLog:
    """Comprehensive trade log entry."""
    
    timestamp: str  # ISO timestamp
    symbol: str  # Trading symbol
    data_symbol: str  # Data symbol
    asset_type: str  # crypto or commodities
    action: str  # buy, sell, hold, exit
    side: str  # long, short, flat
    quantity: float
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    current_price: Optional[float] = None
    
    # P/L information
    realized_pl: Optional[float] = None
    realized_pl_pct: Optional[float] = None
    unrealized_pl: Optional[float] = None
    unrealized_pl_pct: Optional[float] = None
    
    # Position information
    position_status: Optional[str] = None  # open, closed, profit_target_hit, stop_loss_hit
    cost_basis: Optional[float] = None
    current_value: Optional[float] = None
    
    # Trading decision information
    decision: Optional[str] = None  # enter, exit, hold, pyramid, etc.
    reason: Optional[str] = None
    exit_reason: Optional[str] = None
    
    # Model/prediction information
    consensus_action: Optional[str] = None
    consensus_confidence: Optional[float] = None
    consensus_return: Optional[float] = None
    predicted_price: Optional[float] = None
    
    # Risk management
    profit_target_pct: Optional[float] = None
    profit_target_price: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    stop_loss_price: Optional[float] = None
    
    # Order information
    order_id: Optional[str] = None
    dry_run: bool = False
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None


class TradeLogger:
    """Logs all trades with comprehensive profit/loss tracking."""
    
    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize trade logger.
        
        Args:
            log_file: Path to log file. Defaults to data/logs/trades.jsonl
        """
        if log_file is None:
            log_file = Path("data/logs/trades.jsonl")
        
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_trade(self, trade: TradeLog) -> None:
        """Log a trade entry."""
        try:
            # Convert to dict
            entry = asdict(trade)
            
            # Remove None values for cleaner logs (optional - keep if you want to see all fields)
            # entry = {k: v for k, v in entry.items() if v is not None}
            
            # Write as JSONL (one JSON object per line)
            with open(self.log_file, "a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            print(f"[WARN] Failed to log trade: {e}")
    
    def log_position_update(
        self,
        symbol: str,
        data_symbol: str,
        asset_type: str,
        side: str,
        quantity: float,
        entry_price: float,
        current_price: float,
        position_status: str,
        cost_basis: float,
        profit_target_pct: float,
        profit_target_price: float,
        stop_loss_pct: float,
        stop_loss_price: float,
        consensus_action: Optional[str] = None,
        consensus_confidence: Optional[float] = None,
        dry_run: bool = False,
    ) -> None:
        """Log a position update (for monitoring active positions)."""
        current_value = quantity * current_price if side == "long" else -(quantity * current_price)
        
        # Calculate unrealized P/L
        if side == "long":
            unrealized_pl = (current_price - entry_price) * quantity
            unrealized_pl_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # short
            unrealized_pl = (entry_price - current_price) * quantity
            unrealized_pl_pct = ((entry_price - current_price) / entry_price) * 100
        
        trade = TradeLog(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            symbol=symbol,
            data_symbol=data_symbol,
            asset_type=asset_type,
            action="monitor",
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pl=unrealized_pl,
            unrealized_pl_pct=unrealized_pl_pct,
            position_status=position_status,
            cost_basis=cost_basis,
            current_value=current_value,
            decision="monitor",
            reason="position_monitoring",
            consensus_action=consensus_action,
            consensus_confidence=consensus_confidence,
            profit_target_pct=profit_target_pct,
            profit_target_price=profit_target_price,
            stop_loss_pct=stop_loss_pct,
            stop_loss_price=stop_loss_price,
            dry_run=dry_run,
        )
        
        self.log_trade(trade)
    
    def log_position_closed(
        self,
        symbol: str,
        data_symbol: str,
        asset_type: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        exit_reason: str,
        realized_pl: float,
        realized_pl_pct: float,
        cost_basis: float,
        profit_target_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        order_id: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        """Log when a position is closed."""
        trade = TradeLog(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            symbol=symbol,
            data_symbol=data_symbol,
            asset_type=asset_type,
            action="exit",
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            realized_pl=realized_pl,
            realized_pl_pct=realized_pl_pct,
            position_status="closed",
            cost_basis=cost_basis,
            current_value=0.0,  # Position closed, value is 0
            decision="exit",
            exit_reason=exit_reason,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            order_id=order_id,
            dry_run=dry_run,
        )
        
        self.log_trade(trade)
    
    def log_order_execution(
        self,
        symbol: str,
        data_symbol: str,
        asset_type: str,
        action: str,
        side: str,
        quantity: float,
        price: float,
        decision: str,
        reason: str,
        consensus_action: Optional[str] = None,
        consensus_confidence: Optional[float] = None,
        consensus_return: Optional[float] = None,
        predicted_price: Optional[float] = None,
        order_id: Optional[str] = None,
        dry_run: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log when an order is executed."""
        trade = TradeLog(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            symbol=symbol,
            data_symbol=data_symbol,
            asset_type=asset_type,
            action=action,
            side=side,
            quantity=quantity,
            entry_price=price if action in ("buy", "enter") else None,
            exit_price=price if action in ("sell", "exit") else None,
            current_price=price,
            decision=decision,
            reason=reason,
            consensus_action=consensus_action,
            consensus_confidence=consensus_confidence,
            consensus_return=consensus_return,
            predicted_price=predicted_price,
            order_id=order_id,
            dry_run=dry_run,
            metadata=metadata,
        )
        
        self.log_trade(trade)

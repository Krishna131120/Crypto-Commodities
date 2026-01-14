"""
Symbol-level loss tracking to prevent over-trading losing symbols.

This module tracks:
- Consecutive losses per symbol
- Daily loss limits per symbol
- Win rate per symbol
- Cooldown periods after losses
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional


@dataclass
class SymbolStats:
    """Statistics for a single trading symbol."""
    symbol: str
    consecutive_losses: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    daily_loss: float = 0.0  # Total loss for current day
    last_loss_time: Optional[str] = None
    cooldown_until: Optional[str] = None  # ISO timestamp when cooldown expires
    is_blocked: bool = False  # True if symbol should not be traded
    block_reason: Optional[str] = None


class SymbolLossTracker:
    """Tracks losses per symbol and enforces trading limits."""
    
    def __init__(self, stats_file: Optional[Path] = None):
        """
        Initialize symbol loss tracker.
        
        Args:
            stats_file: Path to JSON file storing symbol statistics. Defaults to data/positions/symbol_stats.json
        """
        if stats_file is None:
            stats_file = Path("data/positions/symbol_stats.json")
        self.stats_file = Path(stats_file)
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        self._stats: Dict[str, SymbolStats] = {}
        self._load_stats()
    
    def _load_stats(self) -> None:
        """Load symbol statistics from JSON file."""
        if not self.stats_file.exists():
            self._stats = {}
            return
        
        try:
            with open(self.stats_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._stats = {}
            for symbol, stat_data in data.items():
                self._stats[symbol] = SymbolStats(**stat_data)
        except Exception as exc:
            print(f"[WARNING] Failed to load symbol stats from {self.stats_file}: {exc}")
            self._stats = {}
    
    def _save_stats(self) -> None:
        """Save symbol statistics to JSON file."""
        try:
            # Convert SymbolStats dataclasses to dicts
            data = {}
            for symbol, stats in self._stats.items():
                data[symbol] = asdict(stats)
            
            # Write to file atomically
            temp_file = self.stats_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            temp_file.replace(self.stats_file)
        except Exception as exc:
            print(f"[WARNING] Failed to save symbol stats to {self.stats_file}: {exc}")
    
    def record_trade(self, symbol: str, realized_pl: float, realized_pl_pct: float) -> None:
        """
        Record a completed trade for a symbol.
        
        Args:
            symbol: Trading symbol
            realized_pl: Realized profit/loss in currency
            realized_pl_pct: Realized profit/loss percentage
        """
        if symbol not in self._stats:
            self._stats[symbol] = SymbolStats(symbol=symbol)
        
        stats = self._stats[symbol]
        stats.total_trades += 1
        
        if realized_pl < 0:
            # Loss
            stats.losing_trades += 1
            stats.consecutive_losses += 1
            stats.daily_loss += abs(realized_pl)
            stats.last_loss_time = datetime.now(timezone.utc).isoformat()
            
            # Check if we should block this symbol
            if stats.consecutive_losses >= 3:
                # Block after 3 consecutive losses
                stats.is_blocked = True
                stats.block_reason = f"{stats.consecutive_losses} consecutive losses"
                # Set 24-hour cooldown
                cooldown_end = datetime.now(timezone.utc) + timedelta(hours=24)
                stats.cooldown_until = cooldown_end.isoformat()
                print(f"  [BLOCK] {symbol}: Blocked due to {stats.consecutive_losses} consecutive losses (cooldown until {cooldown_end.strftime('%Y-%m-%d %H:%M:%S')} UTC)")
        else:
            # Win
            stats.winning_trades += 1
            stats.consecutive_losses = 0  # Reset consecutive losses on win
            if stats.is_blocked and stats.consecutive_losses == 0:
                # Unblock if we get a win
                stats.is_blocked = False
                stats.block_reason = None
                stats.cooldown_until = None
        
        self._save_stats()
    
    def can_trade(self, symbol: str, max_daily_loss: float = 500.0, min_win_rate: float = 0.30) -> tuple[bool, Optional[str]]:
        """
        Check if a symbol can be traded.
        
        Args:
            symbol: Trading symbol to check
            max_daily_loss: Maximum daily loss per symbol (default: $500)
            min_win_rate: Minimum win rate required after 10 trades (default: 30%)
        
        Returns:
            Tuple of (can_trade: bool, reason: Optional[str])
        """
        if symbol not in self._stats:
            return True, None
        
        stats = self._stats[symbol]
        now = datetime.now(timezone.utc)
        
        # Check cooldown period
        if stats.cooldown_until:
            try:
                cooldown_end = datetime.fromisoformat(stats.cooldown_until.replace('Z', '+00:00'))
                if now < cooldown_end:
                    remaining = (cooldown_end - now).total_seconds() / 3600  # hours
                    return False, f"Cooldown period active ({remaining:.1f} hours remaining)"
            except Exception:
                pass
        
        # Check if blocked
        if stats.is_blocked:
            return False, stats.block_reason or "Symbol is blocked"
        
        # Check consecutive losses
        if stats.consecutive_losses >= 3:
            return False, f"{stats.consecutive_losses} consecutive losses"
        
        # Check daily loss limit
        if stats.daily_loss >= max_daily_loss:
            return False, f"Daily loss limit reached (${stats.daily_loss:.2f} >= ${max_daily_loss:.2f})"
        
        # Check win rate after 10 trades
        if stats.total_trades >= 10:
            win_rate = stats.winning_trades / stats.total_trades
            if win_rate < min_win_rate:
                return False, f"Win rate too low ({win_rate*100:.1f}% < {min_win_rate*100:.0f}% after {stats.total_trades} trades)"
        
        return True, None
    
    def reset_daily_losses(self) -> None:
        """Reset daily loss counters (call this at start of each trading day)."""
        for stats in self._stats.values():
            stats.daily_loss = 0.0
    
    def get_stats(self, symbol: str) -> Optional[SymbolStats]:
        """Get statistics for a symbol."""
        return self._stats.get(symbol)
    
    def unblock_symbol(self, symbol: str) -> None:
        """Manually unblock a symbol (for testing or manual override)."""
        if symbol in self._stats:
            stats = self._stats[symbol]
            stats.is_blocked = False
            stats.block_reason = None
            stats.cooldown_until = None
            stats.consecutive_losses = 0
            self._save_stats()

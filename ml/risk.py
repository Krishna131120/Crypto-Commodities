"""
Risk management utilities for inference-time guardrails.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RiskManagerConfig:
    max_position: float = 1.0
    min_position: float = 0.05
    max_drawdown_pct: float = 0.15
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_daily_loss_pct: float = 0.03
    min_confidence: float = 0.55
    volatility_scaler: float = 0.5
    paper_trade: bool = False
    paper_trade_size: float = 0.25


class RiskManager:
    """
    Simple risk guardrail module used by inference pipeline.
    """

    def __init__(self, config: Optional[RiskManagerConfig] = None):
        self.config = config or RiskManagerConfig()
        self._equity_peak = 0.0
        self._equity = 0.0

    def update_equity(self, pnl: float):
        self._equity += pnl
        self._equity_peak = max(self._equity_peak, self._equity)

    def max_position_allowed(self, confidence: float, volatility: float) -> float:
        if self.config.paper_trade:
            return self.config.paper_trade_size
        if confidence < self.config.min_confidence:
            return 0.0
        vol_adj = max(0.1, 1.0 - self.config.volatility_scaler * volatility)
        size = self.config.max_position * confidence * vol_adj
        return float(max(self.config.min_position, min(self.config.max_position, size)))

    def check_drawdown(self) -> bool:
        if self._equity_peak <= 0:
            return True
        drawdown = (self._equity - self._equity_peak) / max(abs(self._equity_peak), 1e-9)
        return drawdown >= -self.config.max_drawdown_pct

    def apply_stop_take(self, entry_price: float, current_price: float, direction: str) -> Dict[str, bool]:
        if entry_price <= 0 or current_price <= 0:
            return {"stop_triggered": False, "take_profit_triggered": False}
        change = (current_price / entry_price) - 1.0
        stop_triggered = False
        take_triggered = False
        if direction == "long":
            stop_triggered = change <= -self.config.stop_loss_pct
            take_triggered = change >= self.config.take_profit_pct
        elif direction == "short":
            stop_triggered = change >= self.config.stop_loss_pct
            take_triggered = change <= -self.config.take_profit_pct
        return {
            "stop_triggered": stop_triggered,
            "take_profit_triggered": take_triggered,
        }

    def should_trade(self, confidence: float, volatility: float) -> bool:
        drawdown_ok = self.check_drawdown()
        pos_allowed = self.max_position_allowed(confidence, volatility)
        return drawdown_ok and pos_allowed > 0



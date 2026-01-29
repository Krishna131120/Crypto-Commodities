"""
Simple Dashboard Server for Paper Trading
Displays positions, P/L, account balance, and trade history in real-time.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional


def get_live_price(symbol: str) -> Optional[float]:
    """
    Fetch live price from yfinance for a commodity symbol.
    Returns None if fetch fails.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        # Try to get current price
        info = ticker.info
        price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("lastPrice")
        if price and price > 0:
            return float(price)
        # Fallback: get last close from history
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        print(f"[DASHBOARD] Failed to fetch live price for {symbol}: {e}")
    return None


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for dashboard."""
    
    def _send_response(self, status_code: int, data: Any, content_type: str = "application/json"):
        """Send response."""
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        if content_type == "application/json":
            response = json.dumps(data, indent=2, ensure_ascii=False)
            self.wfile.write(response.encode("utf-8"))
        else:
            self.wfile.write(data.encode("utf-8"))
    
    def _load_positions(self) -> List[Dict[str, Any]]:
        """Load active positions from active_positions.json."""
        positions_file = Path("data/positions/active_positions.json")
        if positions_file.exists():
            try:
                data = json.loads(positions_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    # Format 1: {"positions": [...]}
                    if "positions" in data and isinstance(data["positions"], list):
                        return data["positions"]
                    # Format 2: {symbol: {...}} (PositionManager format)
                    # We convert dict to list but keep symbol info
                    positions = []
                    for symbol, pos in data.items():
                        if isinstance(pos, dict):
                            if "symbol" not in pos:
                                pos["symbol"] = symbol
                            positions.append(pos)
                    return positions
                elif isinstance(data, list):
                    return data
                return []
            except Exception as e:
                print(f"[DASHBOARD ERROR] Failed to load positions: {e}")
                return []
        return []
    
    def _load_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Load recent trade history."""
        trades_file = Path("data/logs/trades.jsonl")
        trades = []
        if trades_file.exists():
            try:
                with open(trades_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                trades.append(json.loads(line))
                            except Exception:
                                pass
            except Exception:
                pass
        # Return most recent trades first
        return list(reversed(trades[-limit:]))
    
    def _load_account(self) -> Dict[str, Any]:
        """
        Dynamically calculate account state from trades and positions.
        This ensures the dashboard always matches reality (Demat style).
        """
        INITIAL_CAPITAL = 1000000.0  # 10 Lakhs starting capital
        
        # 1. Calculate Total Realized P/L from trade history
        trades = self._load_trades(limit=10000) # Load all trades
        total_realized_pl = 0.0
        for trade in trades:
            if "realized_pl" in trade and trade["realized_pl"] is not None:
                total_realized_pl += float(trade["realized_pl"])
        
        # 2. Calculate Position Metrics
        positions = self._load_positions()
        open_positions = [p for p in positions if p.get("status") == "open"]
        
        total_invested = 0.0
        current_market_value = 0.0
        total_unrealized_pl = 0.0
        
        for pos in open_positions:
            qty = float(pos.get("quantity") or pos.get("qty") or 0)
            entry_price = float(pos.get("entry_price") or 0)
            symbol = pos.get("symbol") or pos.get("data_symbol") or ""
            
            # LIVE PRICE: Try to fetch from yfinance, fallback to stored prices
            current_price = None
            if symbol:
                current_price = get_live_price(symbol)
            if current_price is None:
                # Fallback to stored price
                current_price = float(pos.get("current_price") or pos.get("last_price") or entry_price)
            
            # Cost Basis (Invested Amount)
            # For LONG: Cost = Entry * Qty
            position_cost = entry_price * abs(qty)
            total_invested += position_cost
            
            # Market Value & Unrealized P/L (LONG only since we disabled shorting)
            side = str(pos.get("side") or "long").lower()
            market_val = current_price * abs(qty)
            if side == "long":
                unrealized = (current_price - entry_price) * abs(qty)
            else: # short (legacy - shouldn't happen anymore)
                unrealized = (entry_price - current_price) * abs(qty)
            
            current_market_value += market_val
            total_unrealized_pl += unrealized
            
        # 3. Calculate Demat-style Account Balances
        # Cash = Initial + Realized P/L - Invested Amount (Cost Basis)
        # Note: In a real margin account, it's more complex, but this works for Demat view
        cash_balance = INITIAL_CAPITAL + total_realized_pl - total_invested
        
        # Equity (Net Worth) = Cash + Market Value (Longs) + (Short Credit - Short Liability)
        # Simpler: Equity = Initial + Realized P/L + Unrealized P/L
        equity = INITIAL_CAPITAL + total_realized_pl + total_unrealized_pl
        
        return {
            "equity": round(equity, 2),
            "cash": round(cash_balance, 2), # Available Margin / Cash
            "buying_power": round(cash_balance * 1.0, 2), # Assuming 1x leverage for simplicity
            "invested_amount": round(total_invested, 2),
            "current_value": round(current_market_value, 2),
            "total_realized_pl": round(total_realized_pl, 2),
            "total_unrealized_pl": round(total_unrealized_pl, 2),
            "initial_capital": INITIAL_CAPITAL
        }
    
    def _calculate_summary(self, positions: List[Dict], trades: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        # Count positions
        open_positions = [p for p in positions if p.get("status") == "open"]
        
        # Calculate total unrealized P/L
        total_unrealized_pl = 0.0
        for pos in open_positions:
            entry_price = float(pos.get("entry_price") or 0)
            current_price = float(pos.get("current_price") or pos.get("last_price") or 0)
            qty = float(pos.get("quantity") or pos.get("qty") or 0)
            side = str(pos.get("side") or "long").lower()
            
            if side == "long":
                unrealized_pl = (current_price - entry_price) * qty
            else:
                unrealized_pl = (entry_price - current_price) * qty
            
            total_unrealized_pl += unrealized_pl
        
        # Calculate realized P/L from trades
        total_realized_pl = 0.0
        winning_trades = 0
        losing_trades = 0
        
        for trade in trades:
            if "realized_pl" in trade:
                realized_pl = trade["realized_pl"]
                total_realized_pl += realized_pl
                if realized_pl > 0:
                    winning_trades += 1
                elif realized_pl < 0:
                    losing_trades += 1
        
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        return {
            "total_positions": len(open_positions),
            "total_unrealized_pl": round(total_unrealized_pl, 2),
            "total_realized_pl": round(total_realized_pl, 2),
            "total_pl": round(total_unrealized_pl + total_realized_pl, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2)
        }
    
    def do_GET(self):
        """Handle GET requests."""
        path = self.path.split("?")[0]
        
        # DEBUG: Print exact path for troubleshooting
        print(f"[DEBUG] Received path: '{path}' (length={len(path)})")
        
        try:
            if path == "/" or path == "/dashboard":
                # Serve HTML dashboard with no-cache headers
                html = self._generate_dashboard_html()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))
            
            elif path == "/api/positions":
                # Return positions as JSON
                positions = self._load_positions()
                self._send_response(200, {"positions": positions})
            
            elif path == "/api/trades":
                # Return trade history as JSON
                trades = self._load_trades(limit=50)
                self._send_response(200, {"trades": trades})
            
            elif path == "/api/account":
                # Return account balance as JSON
                account = self._load_account()
                self._send_response(200, account)
            
            elif path == "/api/summary":
                # Return summary statistics
                positions = self._load_positions()
                trades = self._load_trades(limit=1000)  # All trades for stats
                account = self._load_account()
                summary = self._calculate_summary(positions, trades)
                summary["account"] = account
                self._send_response(200, summary)
            
            elif path == "/api/liquidate":
                # Liquidate all positions and update account
                if self.command == "POST":
                    self._liquidate_all_positions()
                    self._send_response(200, {"status": "success", "message": "All positions liquidated"})
                else:
                     self._send_response(405, {"error": "Method not allowed"})

            elif path == "/api/reset_account":
                # Reset account to initial state
                if self.command == "POST":
                    self._reset_account()
                    self._send_response(200, {"status": "success", "message": "Account reset successfully"})
                else:
                     self._send_response(405, {"error": "Method not allowed"})
            
            else:
                self._send_response(404, {"error": "Not found"})
        
        except Exception as e:
            self._send_response(500, {"error": str(e)})

    def do_POST(self):
        """Handle POST requests."""
        self.do_GET() # Reuse dispatch logic since we check method inside

    def _liquidate_all_positions(self):
        """
        Liquidate all open positions:
        1. Calculate realized P/L for each position
        2. Log liquidation trades to trades.jsonl
        3. Clear active_positions.json
        """
        positions_file = Path("data/positions/active_positions.json")
        trades_file = Path("data/logs/trades.jsonl")
        
        # Load current positions
        positions = self._load_positions()
        open_positions = [p for p in positions if p.get("status") == "open"]
        
        if not open_positions:
            print("[DASHBOARD] No open positions to liquidate.")
            return
            
        liquidated_trades = []
        timestamp = datetime.now().isoformat()
        
        for pos in open_positions:
            symbol = pos.get("symbol") or pos.get("trading_symbol") or pos.get("data_symbol") or "Unknown"
            qty = float(pos.get("quantity") or pos.get("qty") or 0)
            entry_price = float(pos.get("entry_price") or 0)
            
            # LIVE PRICE: Fetch from yfinance for accurate liquidation P/L
            current_price = None
            if symbol != "Unknown":
                current_price = get_live_price(symbol)
            if current_price is None:
                # Fallback to stored price
                current_price = float(pos.get("current_price") or pos.get("last_price") or entry_price)
            
            side = str(pos.get("side") or "long").lower()
            
            # Calculate P/L (LONG only since we disabled shorting)
            realized_pl = (current_price - entry_price) * qty
            realized_pl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                
            # Create exit trade record
            exit_trade = {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "exit_price": current_price,
                "quantity": qty,
                "realized_pl": realized_pl,
                "realized_pl_pct": realized_pl_pct,
                "exit_reason": "MANUAL_LIQUIDATION_ALL"
            }
            liquidated_trades.append(exit_trade)
            
        # Append to trades.jsonl
        trades_file.parent.mkdir(parents=True, exist_ok=True)
        with open(trades_file, "a", encoding="utf-8") as f:
            for trade in liquidated_trades:
                f.write(json.dumps(trade) + "\n")
                
        # Clear positions (save empty dict {} for PositionManager compatibility)
        positions_file.write_text("{}", encoding="utf-8")
        
        print(f"[DASHBOARD] Liquidated {len(open_positions)} positions.")

    def _reset_account(self):
        """
        Reset account:
        1. Liquidate any open positions first
        2. Clear trades.jsonl
        3. Ensure active_positions.json is empty
        """
        # 1. Liquidate first (to log the closing trades if possible, though they will be deleted)
        # Actually, user wants equity reset to 1,000,000, which means history MUST be cleared.
        self._liquidate_all_positions()
        
        positions_file = Path("data/positions/active_positions.json")
        trades_file = Path("data/logs/trades.jsonl")
        
        # 2. Clear trades log (effectively resets realized P/L to 0)
        trades_file.write_text("", encoding="utf-8")
        
        # 3. Clear positions
        positions_file.write_text("{}", encoding="utf-8")
        
        print(f"[DASHBOARD] Account reset to initial state (Rs. 1,000,000).")
    
    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paper Trading Dashboard - Commodities</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0a0e27;
            color: #e0e6ed;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 32px;
            font-weight: 700;
            color: #fff;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #8b93a7;
            font-size: 14px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: #1a1f3a;
            border: 1px solid #2d3561;
            border-radius: 12px;
            padding: 20px;
        }
        
        .card-title {
            font-size: 14px;
            color: #8b93a7;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .card-value {
            font-size: 28px;
            font-weight: 700;
            color: #fff;
        }
        
        .card-value.positive {
            color: #10b981;
        }
        
        .card-value.negative {
            color: #ef4444;
        }
        
        .section {
            margin-bottom: 30px;
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .section-actions {
            display: flex;
            gap: 10px;
        }
        
        .section-title {
            font-size: 20px;
            font-weight: 600;
            color: #fff;
        }
        
        .btn {
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
            font-weight: 500;
        }
        
        .refresh-btn {
            background: #3b82f6;
            color: #fff;
        }
        
        .refresh-btn:hover {
            background: #2563eb;
        }
        
        .danger-btn {
            background: #ef4444;
            color: #fff;
        }
        
        .danger-btn:hover {
            background: #dc2626;
        }
        
        .warning-btn {
            background: #f59e0b;
            color: #fff;
        }
        
        .warning-btn:hover {
            background: #d97706;
        }
        
        table {
            width: 100%;
            background: #1a1f3a;
            border: 1px solid #2d3561;
            border-radius: 12px;
            border-collapse: collapse;
            overflow: hidden;
        }
        
        thead {
            background: #252b4a;
        }
        
        th {
            padding: 12px;
            text-align: left;
            font-size: 12px;
            font-weight: 600;
            color: #8b93a7;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        td {
            padding: 12px;
            border-top: 1px solid #2d3561;
            font-size: 14px;
        }
        
        tr:hover {
            background: #252b4a;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .badge.long {
            background: #10b98120;
            color: #10b981;
        }
        
        .badge.short {
            background: #ef444420;
            color: #ef4444;
        }
        
        .badge.open {
            background: #3b82f620;
            color: #3b82f6;
        }
        
        .pl-positive {
            color: #10b981;
        }
        
        .pl-negative {
            color: #ef4444;
        }
        
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #8b93a7;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #8b93a7;
        }
        
        .auto-refresh {
            font-size: 12px;
            color: #8b93a7;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div style="display: flex; justify-content: space-between; align-items: start;">
                 <div>
                    <h1>📊 Paper Trading Dashboard</h1>
                    <p class="subtitle">Real-time commodities trading monitor | Auto-refresh: <span id="countdown">30s</span></p>
                 </div>
                 <button class="btn warning-btn" onclick="resetAccount()">🔄 Reset Account</button>
            </div>
        </header>
        
        <!-- Summary Cards -->
        <div class="grid">
            <div class="card">
                <div class="card-title">Account Equity</div>
                <div class="card-value" id="equity">Loading...</div>
            </div>
             <div class="card">
                <div class="card-title">Invested Amount</div>
                <div class="card-value" id="invested">Loading...</div>
            </div>
            <div class="card">
                <div class="card-title">Cash Available</div>
                <div class="card-value" id="cash">Loading...</div>
            </div>
            <div class="card">
                <div class="card-title">Total P/L</div>
                <div class="card-value" id="total-pl">Loading...</div>
            </div>
            <div class="card">
                <div class="card-title">Win Rate</div>
                <div class="card-value" id="win-rate">0%</div>
            </div>
        </div>
        
        <!-- Active Positions -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Active Positions (<span id="position-count">0</span>)</h2>
                <div class="section-actions">
                    <button class="btn danger-btn" onclick="liquidateAll()">⚠️ Liquidate All</button>
                    <button class="btn refresh-btn" onclick="loadData()">Refresh</button>
                </div>
            </div>
            <div id="positions-table">
                <div class="loading">Loading positions...</div>
            </div>
        </div>
        
        <!-- Trade History -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Recent Trades (Closed)</h2>
            </div>
            <div id="trades-table">
                <div class="loading">Loading trades...</div>
            </div>
        </div>
    </div>
    
    <script>
        let refreshInterval;
        let countdownInterval;
        let countdown = 30;
        
        function formatCurrency(value) {
            return '₹' + Math.abs(value).toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }
        
        function formatPL(value) {
            const sign = value >= 0 ? '+' : '-';
            return sign + formatCurrency(value);
        }
        
        async function loadData() {
            console.log('[DASHBOARD] Starting loadData...');
            
            try {
                // Load summary with timeout
                console.log('[DASHBOARD] Fetching /api/summary...');
                const summaryRes = await Promise.race([
                    fetch('/api/summary'),
                    new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), 5000))
                ]);
                
                if (!summaryRes.ok) {
                    throw new Error(`HTTP ${summaryRes.status}`);
                }
                
                const summary = await summaryRes.json();
                console.log('[DASHBOARD] Got summary:', summary);
                
                // Update summary cards - IMMEDIATE UPDATE
                const equity = summary.account?.equity ?? 1000000;
                const cash = summary.account?.cash ?? 1000000;
                const invested = summary.account?.invested_amount ?? 0;
                const totalPL = summary.total_pl ?? 0;
                
                console.log('[DASHBOARD] Updating DOM with equity:', equity);
                
                document.getElementById('equity').textContent = formatCurrency(equity);
                document.getElementById('cash').textContent = formatCurrency(cash);
                document.getElementById('invested').textContent = formatCurrency(invested);
                document.getElementById('total-pl').textContent = formatPL(totalPL);
                document.getElementById('win-rate').textContent = (summary.win_rate || 0).toFixed(1) + '%';
                document.getElementById('position-count').textContent = summary.total_positions || 0;
                
                console.log('[DASHBOARD] DOM updated successfully');
                
                // Load positions
                try {
                    const positionsRes = await fetch('/api/positions');
                    const positionsData = await positionsRes.json();
                    renderPositions(positionsData.positions || [], equity);
                } catch (posErr) {
                    console.error('[DASHBOARD] Error loading positions:', posErr);
                }
                
                // Load trades
                try {
                    const tradesRes = await fetch('/api/trades');
                    const tradesData = await tradesRes.json();
                    renderTrades(tradesData.trades || []);
                } catch (tradeErr) {
                    console.error('[DASHBOARD] Error loading trades:', tradeErr);
                }
                
                // Reset countdown
                countdown = 30;
                
            } catch (error) {
                console.error('[DASHBOARD] FATAL ERROR in loadData:', error);
                alert('Dashboard Error: ' + error.message + '. Check console for details.');
                document.getElementById('equity').textContent = 'Error: ' + error.message;
            }
        }
        
        function renderPositions(positions, totalEquity) {
            const container = document.getElementById('positions-table');
            
            if (positions.length === 0) {
                container.innerHTML = '<div class="empty-state">No active positions</div>';
                return;
            }
            
            const openPositions = positions.filter(p => p.status === 'open');
            
            let html = '<table><thead><tr>';
            html += '<th>Symbol</th>';
            html += '<th>Side</th>';
            html += '<th>Quantity</th>';
            html += '<th>Avg Entry</th>';
            html += '<th>LTP</th>';
            html += '<th>Invested</th>';
            html += '<th>P/L</th>';
            html += '<th>P/L %</th>';
            html += '<th>Port %</th>'; // New Column
            html += '<th>Target</th>';
            html += '<th>Stop</th>';
            html += '</tr></thead><tbody>';
            
            openPositions.forEach(pos => {
                const symbol = pos.symbol || pos.trading_symbol || pos.data_symbol || 'Unknown';
                const entryPrice = pos.entry_price || 0;
                const currentPrice = pos.current_price || pos.last_price || 0;
                const qty = pos.quantity || pos.qty || 0;
                const side = (pos.side || 'long').toLowerCase();
                
                let unrealizedPL = 0;
                let unrealizedPLPct = 0;
                
                if (side === 'long') {
                    unrealizedPL = (currentPrice - entryPrice) * qty;
                    unrealizedPLPct = entryPrice > 0 ? ((currentPrice - entryPrice) / entryPrice * 100) : 0;
                } else {
                    unrealizedPL = (entryPrice - currentPrice) * qty;
                    unrealizedPLPct = entryPrice > 0 ? ((entryPrice - currentPrice) / entryPrice * 100) : 0;
                }
                
                const investedAmt = entryPrice * qty;
                const marketVal = currentPrice * qty;
                
                // Calculate Portfolio %
                // For Short, we use the excessive margin/exposure as proxy for "Portfolio Value" impact, 
                // or just Market Value (Liability) / Equity. Demat usually shows Asset Value / Net Worth.
                // We'll use Abs(Market Value) / Equity
                const portPct = totalEquity > 0 ? (Math.abs(marketVal) / totalEquity * 100) : 0;
                
                const plClass = unrealizedPL >= 0 ? 'pl-positive' : 'pl-negative';
                
                html += '<tr>';
                html += `<td><strong>${symbol}</strong></td>`;
                html += `<td><span class="badge ${side}">${side.toUpperCase()}</span></td>`;
                html += `<td>${qty.toFixed(4)}</td>`;
                html += `<td>${formatCurrency(entryPrice)}</td>`;
                html += `<td>${formatCurrency(currentPrice)}</td>`;
                html += `<td>${formatCurrency(investedAmt)}</td>`;
                html += `<td class="${plClass}">${formatPL(unrealizedPL)}</td>`;
                html += `<td class="${plClass}">${unrealizedPLPct >= 0 ? '+' : ''}${unrealizedPLPct.toFixed(2)}%</td>`;
                html += `<td>${portPct.toFixed(2)}%</td>`;
                html += `<td>${formatCurrency(pos.profit_target_price || 0)}</td>`;
                html += `<td>${formatCurrency(pos.stop_loss_price || 0)}</td>`;
                html += '</tr>';
            });
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }
        
        function renderTrades(trades) {
            const container = document.getElementById('trades-table');
            
            // Show ALL trades (entries and exits)
            // If it's an entry, realized_pl might be undefined/null
            if (trades.length === 0) {
                container.innerHTML = '<div class="empty-state">No trades yet</div>';
                return;
            }
            
            let html = '<table><thead><tr>';
            html += '<th>Time</th>';
            html += '<th>Symbol</th>';
            html += '<th>Side</th>';
            html += '<th>Entry Price</th>';
            html += '<th>Exit Price</th>';
            html += '<th>Quantity</th>';
            html += '<th>P/L</th>';
            html += '<th>P/L %</th>';
            html += '<th>Type/Reason</th>';
            html += '</tr></thead><tbody>';
            
            trades.slice().reverse().forEach(trade => {
                const symbol = trade.symbol || trade.trading_symbol || trade.data_symbol || 'Unknown';
                const realizedPL = trade.realized_pl;
                const realizedPLPct = trade.realized_pl_pct;
                
                // Determine if it's a closed trade or just an entry
                const isClosed = realizedPL !== undefined && realizedPL !== null;
                const plClass = isClosed ? (realizedPL >= 0 ? 'pl-positive' : 'pl-negative') : '';
                
                const timestamp = trade.timestamp || trade.exit_time || trade.entry_time || '';
                const time = timestamp ? new Date(timestamp).toLocaleString('en-IN') : '-';
                
                const side = trade.side || trade.final_side || 'long';
                
                // For entries, we might not have exit price. For exits, we do.
                const exitPriceDisplay = trade.exit_price ? formatCurrency(trade.exit_price) : '-';
                const plDisplay = isClosed ? formatPL(realizedPL) : '-';
                const plPctDisplay = isClosed ? (realizedPLPct >= 0 ? '+' : '') + realizedPLPct.toFixed(2) + '%' : '-';
                
                // Display decision or exit reason
                let typeReason = trade.exit_reason || trade.reason || trade.decision || '-';
                if (typeReason.includes('enter')) typeReason = 'ENTRY';
                
                html += '<tr>';
                html += `<td>${time}</td>`;
                html += `<td><strong>${symbol}</strong></td>`;
                html += `<td><span class="badge ${side}">${side.toUpperCase()}</span></td>`;
                html += `<td>${formatCurrency(trade.entry_price || trade.price || 0)}</td>`;
                html += `<td>${exitPriceDisplay}</td>`;
                html += `<td>${(trade.quantity || trade.exit_qty || trade.entry_qty || 0).toFixed(4)}</td>`;
                html += `<td class="${plClass}">${plDisplay}</td>`;
                html += `<td class="${plClass}">${plPctDisplay}</td>`;
                html += `<td>${typeReason}</td>`;
                html += '</tr>';
            });
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }
        
        function updateCountdown() {
            countdown--;
            document.getElementById('countdown').textContent = countdown + 's';
            
            if (countdown <= 0) {
                loadData();
            }
        }
        
        async function liquidateAll() {
            if (!confirm(`⚠️ WARNING: This will immediately close ALL open positions at current market prices.\n\nAre you sure you want to LIQUIDATE ALL?`)) {
                return;
            }
            
            try {
                const res = await fetch('/api/liquidate', { method: 'POST' });
                const data = await res.json();
                
                if (res.ok) {
                    alert('Success: ' + data.message);
                    loadData(); // Reload immediately
                } else {
                    alert('Error: ' + (data.error || 'Failed to liquidate'));
                }
            } catch (error) {
                console.error('Error liquidating:', error);
                alert('Connection error while liquidating');
            }
        }
        
        async function resetAccount() {
            if (!confirm(`⚠️ DANGER: This will DELETE all trade history, CLEAR all positions, and RESET your account balance to ₹10,00,000.\n\nThis action cannot be undone.\n\nAre you sure you want to RESET EVERYTHING?`)) {
                return;
            }
            
            try {
                const res = await fetch('/api/reset_account', { method: 'POST' });
                const data = await res.json();
                 
                if (res.ok) {
                    alert('Success: ' + data.message);
                    loadData(); // Reload immediately
                } else {
                     alert('Error: ' + (data.error || 'Failed to reset account'));
                }
            } catch (error) {
                console.error('Error resetting account:', error);
                alert('Connection error while resetting account');
            }
        }
        
        // Initial load
        loadData();
        
        // Auto-refresh every 30 seconds
        refreshInterval = setInterval(loadData, 30000);
        countdownInterval = setInterval(updateCountdown, 1000);
    </script>
</body>
</html>"""
    
    def log_message(self, format, *args):
        """Override to customize logging."""
        print(f"[DASHBOARD] {self.address_string()} - {format % args}")


def start_dashboard(port: int = 8080):
    """Start the dashboard server."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, DashboardHandler)
    
    print("=" * 80)
    print(f"PAPER TRADING DASHBOARD STARTED")
    print("=" * 80)
    print(f"Dashboard URL: http://localhost:{port}/dashboard")
    print(f"Auto-refresh: Every 30 seconds")
    print()
    print("API Endpoints:")
    print(f"  - Positions: http://localhost:{port}/api/positions")
    print(f"  - Trades:    http://localhost:{port}/api/trades")
    print(f"  - Account:   http://localhost:{port}/api/account")
    print(f"  - Summary:   http://localhost:{port}/api/summary")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[DASHBOARD] Shutting down...")
        httpd.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper Trading Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    
    args = parser.parse_args()
    start_dashboard(port=args.port)

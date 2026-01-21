"""
Binance Trading Dashboard.

Live dashboard showing:
- Active positions
- Recent trades
- Trading cycles
- Account status
- Real-time predictions

Access at: http://localhost:8081
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from binance.binance_client import BinanceClient
from trading.position_manager import PositionManager

# Binance-specific paths
BINANCE_ROOT = Path(__file__).parent
BINANCE_LOGS_DIR = BINANCE_ROOT / "logs"
BINANCE_POSITIONS_FILE = BINANCE_ROOT / "data" / "positions" / "active_positions.json"
BINANCE_TRADES_LOG = BINANCE_LOGS_DIR / "trades.jsonl"
BINANCE_CYCLES_LOG = BINANCE_LOGS_DIR / "cycles.jsonl"


class BinanceDashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Binance dashboard."""
    
    def _send_response(self, status_code: int, data: Dict[str, Any], content_type: str = "application/json"):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        
        response = json.dumps(data, indent=2, ensure_ascii=False)
        self.wfile.write(response.encode("utf-8"))
    
    def _send_error(self, status_code: int, message: str):
        """Send error response."""
        self._send_response(status_code, {"error": True, "message": message})
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        path = self.path.split("?")[0]
        
        try:
            if path == "/" or path == "/dashboard":
                # Serve HTML dashboard
                self._serve_dashboard()
            elif path == "/api/account":
                # Get Binance account status
                self._get_account()
            elif path == "/api/positions":
                # Get active positions
                self._get_positions()
            elif path == "/api/trades":
                # Get recent trades
                self._get_trades()
            elif path == "/api/cycles":
                # Get trading cycles
                self._get_cycles()
            elif path == "/api/status":
                # Get overall status
                self._get_status()
            else:
                self._send_error(404, f"Endpoint not found: {path}")
        except Exception as e:
            self._send_error(500, f"Server error: {str(e)}")
    
    def _serve_dashboard(self):
        """Serve HTML dashboard."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Binance Trading Dashboard</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { background: #1e1e1e; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .section { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .section h2 { margin-top: 0; color: #1e1e1e; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f8f8; font-weight: bold; }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .status-badge { padding: 4px 8px; border-radius: 4px; font-size: 12px; }
        .status-open { background: #d4edda; color: #155724; }
        .status-closed { background: #f8d7da; color: #721c24; }
        .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .refresh-btn:hover { background: #0056b3; }
        .stats { display: flex; gap: 20px; }
        .stat-box { flex: 1; background: #f8f9fa; padding: 15px; border-radius: 4px; }
        .stat-value { font-size: 24px; font-weight: bold; }
        .stat-label { color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Binance Trading Dashboard</h1>
            <p>Live trading monitor for Binance paper trading</p>
        </div>
        
        <div class="section">
            <h2>Account Status</h2>
            <div id="account-status">Loading...</div>
        </div>
        
        <div class="section">
            <h2>Active Positions</h2>
            <div id="positions">Loading...</div>
        </div>
        
        <div class="section">
            <h2>Recent Trades</h2>
            <button class="refresh-btn" onclick="loadData()">Refresh</button>
            <div id="trades">Loading...</div>
        </div>
        
        <div class="section">
            <h2>Trading Cycles</h2>
            <div id="cycles">Loading...</div>
        </div>
    </div>
    
    <script>
        async function loadData() {
            try {
                // Load account
                const accountRes = await fetch('/api/account');
                const account = await accountRes.json();
                document.getElementById('account-status').innerHTML = `
                    <div class="stats">
                        <div class="stat-box">
                            <div class="stat-label">Equity</div>
                            <div class="stat-value">$${account.equity?.toFixed(2) || '0.00'}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Buying Power</div>
                            <div class="stat-value">$${account.buying_power?.toFixed(2) || '0.00'}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Cash</div>
                            <div class="stat-value">$${account.cash?.toFixed(2) || '0.00'}</div>
                        </div>
                    </div>
                `;
                
                // Load positions
                const positionsRes = await fetch('/api/positions');
                const positions = await positionsRes.json();
                if (positions.positions && positions.positions.length > 0) {
                    let html = '<table><tr><th>Symbol</th><th>Side</th><th>Quantity</th><th>Entry Price</th><th>Current Price</th><th>P/L</th><th>P/L %</th><th>Status</th></tr>';
                    positions.positions.forEach(pos => {
                        const pl = pos.realized_pl || pos.unrealized_pl || 0;
                        const plPct = pos.realized_pl_pct || pos.unrealized_pl_pct || 0;
                        const plClass = pl >= 0 ? 'positive' : 'negative';
                        html += `<tr>
                            <td>${pos.symbol}</td>
                            <td>${pos.side?.toUpperCase() || 'N/A'}</td>
                            <td>${pos.quantity?.toFixed(8) || '0'}</td>
                            <td>$${pos.entry_price?.toFixed(2) || '0.00'}</td>
                            <td>$${pos.current_price?.toFixed(2) || '0.00'}</td>
                            <td class="${plClass}">$${pl.toFixed(2)}</td>
                            <td class="${plClass}">${plPct.toFixed(2)}%</td>
                            <td><span class="status-badge status-${pos.status || 'open'}">${pos.status || 'open'}</span></td>
                        </tr>`;
                    });
                    html += '</table>';
                    document.getElementById('positions').innerHTML = html;
                } else {
                    document.getElementById('positions').innerHTML = '<p>No active positions</p>';
                }
                
                // Load trades
                const tradesRes = await fetch('/api/trades');
                const trades = await tradesRes.json();
                if (trades.trades && trades.trades.length > 0) {
                    let html = '<table><tr><th>Time</th><th>Symbol</th><th>Action</th><th>Quantity</th><th>Price</th><th>P/L</th></tr>';
                    trades.trades.slice(0, 50).forEach(trade => {
                        const pl = trade.realized_pl || 0;
                        const plClass = pl >= 0 ? 'positive' : 'negative';
                        html += `<tr>
                            <td>${new Date(trade.timestamp).toLocaleString()}</td>
                            <td>${trade.symbol || trade.data_symbol || 'N/A'}</td>
                            <td>${trade.decision || trade.action || 'N/A'}</td>
                            <td>${trade.trade_qty?.toFixed(8) || '0'}</td>
                            <td>$${trade.trade_price?.toFixed(2) || '0.00'}</td>
                            <td class="${plClass}">$${pl.toFixed(2)}</td>
                        </tr>`;
                    });
                    html += '</table>';
                    document.getElementById('trades').innerHTML = html;
                } else {
                    document.getElementById('trades').innerHTML = '<p>No trades yet</p>';
                }
                
                // Load cycles
                const cyclesRes = await fetch('/api/cycles');
                const cycles = await cyclesRes.json();
                if (cycles.cycles && cycles.cycles.length > 0) {
                    let html = '<table><tr><th>Time</th><th>Monitored</th><th>Traded</th><th>Skipped</th><th>Duration</th></tr>';
                    cycles.cycles.slice(0, 20).forEach(cycle => {
                        html += `<tr>
                            <td>${new Date(cycle.cycle_start).toLocaleString()}</td>
                            <td>${cycle.symbols_processed || 0}</td>
                            <td>${cycle.symbols_traded || 0}</td>
                            <td>${cycle.symbols_skipped || 0}</td>
                            <td>${(cycle.cycle_duration_seconds || 0).toFixed(1)}s</td>
                        </tr>`;
                    });
                    html += '</table>';
                    document.getElementById('cycles').innerHTML = html;
                } else {
                    document.getElementById('cycles').innerHTML = '<p>No cycles yet</p>';
                }
            } catch (error) {
                console.error('Error loading data:', error);
            }
        }
        
        // Load data on page load and refresh every 5 seconds
        loadData();
        setInterval(loadData, 5000);
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))
    
    def _get_account(self):
        """Get Binance account status."""
        try:
            client = BinanceClient()
            account = client.get_account()
            self._send_response(200, account)
        except Exception as e:
            self._send_error(500, f"Failed to get account: {str(e)}")
    
    def _get_positions(self):
        """Get active positions."""
        try:
            position_manager = PositionManager(positions_file=BINANCE_POSITIONS_FILE)
            positions = position_manager.list_positions()
            
            # Get current prices for positions
            client = BinanceClient()
            for pos in positions:
                try:
                    broker_pos = client.get_position(pos.symbol)
                    if broker_pos:
                        pos["current_price"] = broker_pos.get("current_price", pos.entry_price)
                        # Calculate unrealized P/L
                        if pos.side == "long":
                            pos["unrealized_pl"] = (pos["current_price"] - pos.entry_price) * pos.quantity
                            pos["unrealized_pl_pct"] = ((pos["current_price"] - pos.entry_price) / pos.entry_price) * 100
                        else:
                            pos["unrealized_pl"] = (pos.entry_price - pos["current_price"]) * pos.quantity
                            pos["unrealized_pl_pct"] = ((pos.entry_price - pos["current_price"]) / pos.entry_price) * 100
                except:
                    pass
            
            self._send_response(200, {"positions": positions})
        except Exception as e:
            self._send_error(500, f"Failed to get positions: {str(e)}")
    
    def _get_trades(self):
        """Get recent trades from log."""
        trades = []
        if BINANCE_TRADES_LOG.exists():
            try:
                with open(BINANCE_TRADES_LOG, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                trade = json.loads(line)
                                if trade.get("decision") in ["entered_long", "entered_short", "exited_position", "closed_position"]:
                                    trades.append(trade)
                            except:
                                pass
                # Sort by timestamp (newest first)
                trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            except Exception as e:
                pass
        
        self._send_response(200, {"trades": trades[:100]})  # Last 100 trades
    
    def _get_cycles(self):
        """Get trading cycles."""
        cycles = []
        if BINANCE_CYCLES_LOG.exists():
            try:
                with open(BINANCE_CYCLES_LOG, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                cycle = json.loads(line)
                                cycles.append(cycle)
                            except:
                                pass
                # Sort by timestamp (newest first)
                cycles.sort(key=lambda x: x.get("cycle_start", ""), reverse=True)
            except Exception as e:
                pass
        
        self._send_response(200, {"cycles": cycles[:50]})  # Last 50 cycles
    
    def _get_status(self):
        """Get overall status."""
        try:
            client = BinanceClient()
            account = client.get_account()
            position_manager = PositionManager(positions_file=BINANCE_POSITIONS_FILE)
            positions = position_manager.list_positions()
            
            # Count trades
            trade_count = 0
            if BINANCE_TRADES_LOG.exists():
                try:
                    with open(BINANCE_TRADES_LOG, "r", encoding="utf-8") as f:
                        trade_count = sum(1 for line in f if line.strip())
                except:
                    pass
            
            status = {
                "broker": "binance",
                "account_equity": account.get("equity", 0),
                "active_positions": len(positions),
                "total_trades": trade_count,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            
            self._send_response(200, status)
        except Exception as e:
            self._send_error(500, f"Failed to get status: {str(e)}")
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    port = 8081
    server_address = ("", port)
    httpd = HTTPServer(server_address, BinanceDashboardHandler)
    
    print("=" * 80)
    print("BINANCE TRADING DASHBOARD")
    print("=" * 80)
    print(f"Dashboard: http://localhost:{port}")
    print(f"API: http://localhost:{port}/api/")
    print()
    print("Endpoints:")
    print(f"  GET /                    - Dashboard (HTML)")
    print(f"  GET /api/account         - Account status")
    print(f"  GET /api/positions       - Active positions")
    print(f"  GET /api/trades          - Recent trades")
    print(f"  GET /api/cycles          - Trading cycles")
    print(f"  GET /api/status          - Overall status")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[STOPPED] Dashboard server stopped")
        httpd.shutdown()


if __name__ == "__main__":
    main()

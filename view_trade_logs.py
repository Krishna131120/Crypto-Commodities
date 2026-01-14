"""
View and analyze trade logs from both old and new log files.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    if not file_path.exists():
        return []
    
    entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def analyze_trades(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze trade entries for profit/loss summary."""
    analysis = {
        "total_entries": len(entries),
        "entries_entered": 0,
        "entries_exited": 0,
        "entries_monitored": 0,
        "total_realized_pl": 0.0,
        "total_unrealized_pl": 0.0,
        "wins": 0,
        "losses": 0,
        "breakeven": 0,
        "by_symbol": {},
    }
    
    for entry in entries:
        decision = entry.get("decision", "").lower()
        symbol = entry.get("trading_symbol") or entry.get("symbol", "UNKNOWN")
        data_symbol = entry.get("data_symbol", "")
        
        if "enter" in decision:
            analysis["entries_entered"] += 1
        elif "exit" in decision or "close" in decision:
            analysis["entries_exited"] += 1
            realized_pl = entry.get("realized_pl")
            if realized_pl is not None:
                pl = float(realized_pl)
                analysis["total_realized_pl"] += pl
                if pl > 0:
                    analysis["wins"] += 1
                elif pl < 0:
                    analysis["losses"] += 1
                else:
                    analysis["breakeven"] += 1
        elif "hold" in decision or "monitor" in decision:
            analysis["entries_monitored"] += 1
            # Check for unrealized P/L
            position_status = entry.get("position_status", {})
            unrealized_pl = position_status.get("unrealized_pl") or entry.get("unrealized_pl")
            if unrealized_pl is not None:
                analysis["total_unrealized_pl"] += float(unrealized_pl)
        
        # Track by symbol
        if symbol not in analysis["by_symbol"]:
            analysis["by_symbol"][symbol] = {
                "trades": 0,
                "realized_pl": 0.0,
                "wins": 0,
                "losses": 0,
            }
        
        if "enter" in decision or "exit" in decision:
            analysis["by_symbol"][symbol]["trades"] += 1
            realized_pl = entry.get("realized_pl")
            if realized_pl is not None:
                pl = float(realized_pl)
                analysis["by_symbol"][symbol]["realized_pl"] += pl
                if pl > 0:
                    analysis["by_symbol"][symbol]["wins"] += 1
                elif pl < 0:
                    analysis["by_symbol"][symbol]["losses"] += 1
    
    return analysis


def main():
    """Main function to view trade logs."""
    print("=" * 80)
    print("TRADE LOG VIEWER")
    print("=" * 80)
    print()
    
    # Check old log file
    old_log = Path("logs/trading/crypto_trades.jsonl")
    print(f"[1] Checking old log file: {old_log}")
    print("-" * 80)
    
    old_entries = load_jsonl(old_log)
    if old_entries:
        print(f"  Found {len(old_entries)} entries in old log file")
        old_analysis = analyze_trades(old_entries)
        print(f"\n  Summary:")
        print(f"    Total Entries: {old_analysis['total_entries']}")
        print(f"    Entries Entered: {old_analysis['entries_entered']}")
        print(f"    Entries Exited: {old_analysis['entries_exited']}")
        print(f"    Entries Monitored: {old_analysis['entries_monitored']}")
        print(f"    Total Realized P/L: ${old_analysis['total_realized_pl']:.2f}")
        print(f"    Total Unrealized P/L: ${old_analysis['total_unrealized_pl']:.2f}")
        print(f"    Wins: {old_analysis['wins']}")
        print(f"    Losses: {old_analysis['losses']}")
        print(f"    Breakeven: {old_analysis['breakeven']}")
        
        if old_analysis['by_symbol']:
            print(f"\n  By Symbol:")
            for symbol, stats in sorted(old_analysis['by_symbol'].items(), key=lambda x: x[1]['realized_pl'], reverse=True):
                print(f"    {symbol}: {stats['trades']} trades, P/L: ${stats['realized_pl']:.2f} (W: {stats['wins']}, L: {stats['losses']})")
    else:
        print("  No entries found in old log file")
    
    print()
    
    # Check new log file
    new_log = Path("data/logs/trades.jsonl")
    print(f"[2] Checking new log file: {new_log}")
    print("-" * 80)
    
    new_entries = load_jsonl(new_log)
    if new_entries:
        print(f"  Found {len(new_entries)} entries in new log file")
        new_analysis = analyze_trades(new_entries)
        print(f"\n  Summary:")
        print(f"    Total Entries: {new_analysis['total_entries']}")
        print(f"    Total Realized P/L: ${new_analysis['total_realized_pl']:.2f}")
        print(f"    Total Unrealized P/L: ${new_analysis['total_unrealized_pl']:.2f}")
        print(f"    Wins: {new_analysis['wins']}")
        print(f"    Losses: {new_analysis['losses']}")
    else:
        print("  No entries yet - new file will be populated when trades execute")
    
    print()
    print("=" * 80)
    print("NOTE: All future trades will be logged to the new file:")
    print(f"  {new_log}")
    print("=" * 80)


if __name__ == "__main__":
    main()

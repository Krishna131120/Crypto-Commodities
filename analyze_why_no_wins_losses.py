"""
Analyze why some symbols show W:0, L:0 (no wins, no losses).
"""

import json
from pathlib import Path
from collections import Counter


def analyze_symbol(symbol: str, entries: list) -> dict:
    """Analyze entries for a specific symbol."""
    symbol_entries = [e for e in entries if e.get("trading_symbol") == symbol]
    
    decisions = Counter(e.get("decision", "unknown") for e in symbol_entries)
    
    # Count different types
    rejected = [e for e in symbol_entries if "rejected" in e.get("decision", "").lower()]
    entered = [e for e in symbol_entries if "enter" in e.get("decision", "").lower() and "rejected" not in e.get("decision", "").lower()]
    exited = [e for e in symbol_entries if "exit" in e.get("decision", "").lower()]
    with_pl = [e for e in exited if e.get("realized_pl") is not None]
    
    return {
        "symbol": symbol,
        "total_entries": len(symbol_entries),
        "decisions": dict(decisions),
        "rejected_count": len(rejected),
        "entered_count": len(entered),
        "exited_count": len(exited),
        "with_realized_pl": len(with_pl),
        "sample_rejected": rejected[0] if rejected else None,
    }


def main():
    """Main analysis."""
    log_file = Path("logs/trading/crypto_trades.jsonl")
    
    print("=" * 80)
    print("WHY W:0, L:0 ANALYSIS")
    print("=" * 80)
    print()
    
    # Load entries
    entries = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entries.append(json.loads(line))
            except:
                continue
    
    print(f"Total entries loaded: {len(entries)}")
    print()
    
    # Analyze symbols with W:0, L:0
    symbols_to_check = ["BTCUSD", "SHIBUSD", "XRPUSD", "LTCUSD"]
    
    for symbol in symbols_to_check:
        analysis = analyze_symbol(symbol, entries)
        
        print(f"Symbol: {symbol}")
        print("-" * 80)
        print(f"  Total log entries: {analysis['total_entries']}")
        print(f"  Decision breakdown:")
        for decision, count in sorted(analysis['decisions'].items(), key=lambda x: -x[1]):
            print(f"    - {decision}: {count}")
        print(f"  Rejected orders: {analysis['rejected_count']}")
        print(f"  Successfully entered: {analysis['entered_count']}")
        print(f"  Successfully exited: {analysis['exited_count']}")
        print(f"  Exits with P/L data: {analysis['with_realized_pl']}")
        
        if analysis['sample_rejected']:
            sample = analysis['sample_rejected']
            error = sample.get("alpaca_error", "N/A")
            print(f"\n  Sample rejected reason:")
            print(f"    {error[:150]}...")
        
        print()
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()

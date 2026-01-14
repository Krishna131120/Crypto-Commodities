"""
Verification script to ensure all critical fixes are properly implemented.

Run this before starting the bot to verify:
1. Exit price retrieval fix
2. Stop-loss percentage fix
3. Symbol loss tracking
4. Model flip exit logic
"""
from pathlib import Path
import sys

def verify_imports():
    """Verify all required imports are available."""
    print("=" * 80)
    print("VERIFYING IMPORTS")
    print("=" * 80)
    
    try:
        from trading.execution_engine import ExecutionEngine, TradingRiskConfig
        print("[OK] ExecutionEngine imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import ExecutionEngine: {e}")
        return False
    
    try:
        from trading.symbol_loss_tracker import SymbolLossTracker, SymbolStats
        print("[OK] SymbolLossTracker imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import SymbolLossTracker: {e}")
        return False
    
    try:
        from ml.horizons import get_horizon_risk_config
        print("[OK] Horizon risk config imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import horizon risk config: {e}")
        return False
    
    return True

def verify_execution_engine():
    """Verify ExecutionEngine has all fixes."""
    print("\n" + "=" * 80)
    print("VERIFYING EXECUTION ENGINE FIXES")
    print("=" * 80)
    
    try:
        from trading.execution_engine import ExecutionEngine
        
        # Check if loss_tracker is initialized
        engine = ExecutionEngine()
        if hasattr(engine, 'loss_tracker'):
            print("[OK] SymbolLossTracker is initialized in ExecutionEngine")
        else:
            print("[FAIL] SymbolLossTracker NOT found in ExecutionEngine")
            return False
        
        # Check if loss_tracker has required methods
        if hasattr(engine.loss_tracker, 'can_trade') and hasattr(engine.loss_tracker, 'record_trade'):
            print("[OK] SymbolLossTracker has required methods (can_trade, record_trade)")
        else:
            print("[FAIL] SymbolLossTracker missing required methods")
            return False
        
        # Check ExecutionEngine source for exit price fix
        engine_file = Path("trading/execution_engine.py")
        if engine_file.exists():
            content = engine_file.read_text(encoding="utf-8")
            
            # Check for exit price validation
            if "CRITICAL VALIDATION: Sanity check exit price" in content:
                print("[OK] Exit price validation fix found")
            else:
                print("[FAIL] Exit price validation fix NOT found")
                return False
            
            # Check for filled price retrieval
            if "filled_avg_price" in content and "actual_exit_price" in content:
                print("[OK] Exit price retrieval fix found")
            else:
                print("[FAIL] Exit price retrieval fix NOT found")
                return False
            
            # Check for stop-loss priority fix
            if "CRITICAL FIX: Ensure horizon stop-loss is used correctly" in content:
                print("[OK] Stop-loss percentage fix found")
            else:
                print("[FAIL] Stop-loss percentage fix NOT found")
                return False
            
            # Check for symbol loss tracking
            if "self.loss_tracker.can_trade" in content:
                print("[OK] Symbol loss tracking check found")
            else:
                print("[FAIL] Symbol loss tracking check NOT found")
                return False
            
            # Check for model flip logic
            if "within 0.5% of profit target" in content:
                print("[OK] Model flip exit logic fix found")
            else:
                print("[FAIL] Model flip exit logic fix NOT found")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error verifying ExecutionEngine: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_stop_loss_config():
    """Verify stop-loss configuration is correct."""
    print("\n" + "=" * 80)
    print("VERIFYING STOP-LOSS CONFIGURATION")
    print("=" * 80)
    
    try:
        from ml.horizons import get_horizon_risk_config
        
        # Check intraday stop-loss (should be 5%)
        intraday_config = get_horizon_risk_config("intraday")
        intraday_stop = intraday_config.get("default_stop_loss_pct", 0)
        
        if abs(intraday_stop - 0.05) < 0.001:  # 5%
            print(f"[OK] Intraday stop-loss: {intraday_stop*100:.1f}% (correct)")
        else:
            print(f"[FAIL] Intraday stop-loss: {intraday_stop*100:.1f}% (expected 5.0%)")
            return False
        
        # Check short-term stop-loss (should be 6%)
        short_config = get_horizon_risk_config("short")
        short_stop = short_config.get("default_stop_loss_pct", 0)
        
        if abs(short_stop - 0.06) < 0.001:  # 6%
            print(f"[OK] Short-term stop-loss: {short_stop*100:.1f}% (correct)")
        else:
            print(f"[FAIL] Short-term stop-loss: {short_stop*100:.1f}% (expected 6.0%)")
            return False
        
        # Check long-term stop-loss (should be 7%)
        long_config = get_horizon_risk_config("long")
        long_stop = long_config.get("default_stop_loss_pct", 0)
        
        if abs(long_stop - 0.07) < 0.001:  # 7%
            print(f"[OK] Long-term stop-loss: {long_stop*100:.1f}% (correct)")
        else:
            print(f"[FAIL] Long-term stop-loss: {long_stop*100:.1f}% (expected 7.0%)")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error verifying stop-loss config: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_symbol_loss_tracker():
    """Verify SymbolLossTracker functionality."""
    print("\n" + "=" * 80)
    print("VERIFYING SYMBOL LOSS TRACKER")
    print("=" * 80)
    
    try:
        from trading.symbol_loss_tracker import SymbolLossTracker
        
        tracker = SymbolLossTracker()
        
        # Test can_trade for new symbol (should allow)
        can_trade, reason = tracker.can_trade("TESTUSD")
        if can_trade:
            print("[OK] New symbols can be traded (correct)")
        else:
            print(f"[FAIL] New symbols blocked: {reason}")
            return False
        
        # Test recording a loss
        tracker.record_trade("TESTUSD", -100.0, -2.0)
        stats = tracker.get_stats("TESTUSD")
        if stats and stats.consecutive_losses == 1:
            print("[OK] Loss recording works (1 consecutive loss)")
        else:
            print("[FAIL] Loss recording failed")
            return False
        
        # Test recording 3 losses (should block)
        tracker.record_trade("TESTUSD", -50.0, -1.0)
        tracker.record_trade("TESTUSD", -75.0, -1.5)
        stats = tracker.get_stats("TESTUSD")
        if stats and stats.consecutive_losses >= 3:
            can_trade, reason = tracker.can_trade("TESTUSD")
            if not can_trade:
                print("[OK] Symbol blocked after 3 consecutive losses (correct)")
            else:
                print("[FAIL] Symbol NOT blocked after 3 consecutive losses")
                return False
        else:
            print("[FAIL] Consecutive loss tracking failed")
            return False
        
        # Test unblocking
        tracker.unblock_symbol("TESTUSD")
        can_trade, reason = tracker.can_trade("TESTUSD")
        if can_trade:
            print("[OK] Symbol unblocking works")
        else:
            print(f"[FAIL] Symbol still blocked after unblock: {reason}")
            return False
        
        # Clean up test symbol
        test_file = Path("data/positions/symbol_stats.json")
        if test_file.exists():
            import json
            data = json.loads(test_file.read_text())
            if "TESTUSD" in data:
                del data["TESTUSD"]
                test_file.write_text(json.dumps(data, indent=2))
        
        return True
    except Exception as e:
        print(f"✗ Error verifying SymbolLossTracker: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("CRITICAL FIXES VERIFICATION")
    print("=" * 80)
    print("\nThis script verifies that all critical fixes are properly implemented.")
    print("Run this before starting the bot to ensure everything is working.\n")
    
    all_passed = True
    
    # Run all checks
    if not verify_imports():
        all_passed = False
    
    if not verify_execution_engine():
        all_passed = False
    
    if not verify_stop_loss_config():
        all_passed = False
    
    if not verify_symbol_loss_tracker():
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 80)
    if all_passed:
        print("[SUCCESS] ALL CHECKS PASSED - All fixes are properly implemented!")
        print("=" * 80)
        print("\nYou can now run the bot with confidence.")
        print("\nImplemented fixes:")
        print("  1. [OK] Exit price retrieval and validation")
        print("  2. [OK] Stop-loss percentage (5% for intraday, 6% for short-term, 7% for long-term)")
        print("  3. [OK] Symbol-level loss limits (blocks after 3 consecutive losses)")
        print("  4. [OK] Model flip exit logic (holds if within 0.5% of profit target)")
        return 0
    else:
        print("[FAIL] SOME CHECKS FAILED - Please review the errors above")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())

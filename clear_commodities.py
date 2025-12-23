"""
Utility script to clear ONLY commodities-related data, features, models, and logs.

This script will:
- Clear commodities raw data (data/json/raw/commodities)
- Clear commodities features (data/features/commodities)
- Clear commodities models (models/commodities)
- Clear commodities training logs (logs/training/commodities)
- Clear commodities trading logs (logs/trading/commodities_trades.jsonl)
- Clear commodities DQN summaries (models/dqn/commodities_*)
- Clear commodities tensorboard logs (logs/tensorboard/DQN_* for commodities)

PRESERVES:
- All crypto data, features, models, and logs
- General trading logs (crypto_trades.jsonl)
- Active positions (unless --include-positions flag is used)
"""

from pathlib import Path
import shutil
import json


def _delete_directory(path: Path, label: str):
    """Delete a directory if it exists."""
    if path.exists():
        try:
            shutil.rmtree(path)
            print(f"[DELETED] {label}: {path}")
            return True
        except Exception as exc:
            print(f"[ERROR] {label}: {exc}")
            return False
    else:
        print(f"[SKIP] {label}: {path} (not found)")
        return False


def _delete_file(path: Path, label: str):
    """Delete a file if it exists."""
    if path.exists():
        try:
            path.unlink()
            print(f"[DELETED] {label}: {path}")
            return True
        except Exception as exc:
            print(f"[ERROR] {label}: {exc}")
            return False
    else:
        print(f"[SKIP] {label}: {path} (not found)")
        return False


def _clear_commodities_positions(active_positions_file: Path):
    """Clear only commodities positions from active_positions.json."""
    if not active_positions_file.exists():
        print(f"[SKIP] Active positions file not found: {active_positions_file}")
        return
    
    try:
        with open(active_positions_file, "r", encoding="utf-8") as f:
            positions = json.load(f)
        
        # Filter out commodities positions
        commodities_symbols = ["GC=F", "CL=F", "SI=F", "PL=F", "HG=F", "NG=F"]  # Add more as needed
        original_count = len(positions)
        filtered_positions = {}
        
        for symbol, pos_data in positions.items():
            # Check if it's a commodities position
            asset_type = pos_data.get("asset_type", "")
            is_commodity = asset_type == "commodities"
            
            # Also check if symbol matches commodities pattern
            if not is_commodity:
                for comm_symbol in commodities_symbols:
                    if comm_symbol in symbol or symbol in comm_symbol:
                        is_commodity = True
                        break
            
            if not is_commodity:
                filtered_positions[symbol] = pos_data
        
        removed_count = original_count - len(filtered_positions)
        
        if removed_count > 0:
            with open(active_positions_file, "w", encoding="utf-8") as f:
                json.dump(filtered_positions, f, indent=2)
            print(f"[CLEARED] Removed {removed_count} commodities position(s) from active_positions.json")
        else:
            print(f"[SKIP] No commodities positions found in active_positions.json")
            
    except Exception as exc:
        print(f"[ERROR] Failed to clear commodities positions: {exc}")


def clear_commodities_data(include_positions: bool = False):
    """
    Clear all commodities-related data, features, models, and logs.
    
    Args:
        include_positions: If True, also clear commodities positions from active_positions.json
    """
    project_root = Path(__file__).parent
    
    print("=" * 80)
    print("CLEARING COMMODITIES DATA, FEATURES, MODELS, AND LOGS")
    print("=" * 80)
    print()
    print("This will delete:")
    print("  - Commodities raw data (data/json/raw/commodities)")
    print("  - Commodities features (data/features/commodities)")
    print("  - Commodities models (models/commodities)")
    print("  - Commodities training logs (logs/training/commodities)")
    print("  - Commodities trading logs (logs/trading/commodities_trades.jsonl)")
    print("  - Commodities DQN summaries (models/dqn/commodities_*)")
    print()
    print("PRESERVES:")
    print("  - All crypto data, features, models, and logs")
    print("  - General trading logs (crypto_trades.jsonl)")
    if not include_positions:
        print("  - Active positions (use --include-positions to clear commodities positions)")
    print("=" * 80)
    print()
    
    deleted_count = 0
    
    # 1. Clear commodities raw data
    commodities_data_path = project_root / "data" / "json" / "raw" / "commodities"
    if _delete_directory(commodities_data_path, "Commodities raw data"):
        deleted_count += 1
    
    # 2. Clear commodities features
    commodities_features_path = project_root / "data" / "features" / "commodities"
    if _delete_directory(commodities_features_path, "Commodities features"):
        deleted_count += 1
    
    # 3. Clear commodities models
    commodities_models_path = project_root / "models" / "commodities"
    if _delete_directory(commodities_models_path, "Commodities models"):
        deleted_count += 1
    
    # 4. Clear commodities training logs
    commodities_training_logs_path = project_root / "logs" / "training" / "commodities"
    if _delete_directory(commodities_training_logs_path, "Commodities training logs"):
        deleted_count += 1
    
    # 5. Clear commodities trading logs
    commodities_trades_file = project_root / "logs" / "trading" / "commodities_trades.jsonl"
    if _delete_file(commodities_trades_file, "Commodities trading logs"):
        deleted_count += 1
    
    # 6. Clear commodities DQN summaries
    dqn_path = project_root / "models" / "dqn"
    if dqn_path.exists():
        dqn_files = list(dqn_path.glob("commodities_*.json"))
        for dqn_file in dqn_files:
            if _delete_file(dqn_file, f"Commodities DQN summary ({dqn_file.name})"):
                deleted_count += 1
        if not dqn_files:
            print(f"[SKIP] No commodities DQN summaries found in {dqn_path}")
    
    # 7. Clear commodities tensorboard logs (if any)
    tensorboard_path = project_root / "logs" / "tensorboard"
    if tensorboard_path.exists():
        # Check DQN directories for commodities-related logs
        # Note: TensorBoard logs are typically named by trial number, so we check metadata
        # For now, we'll clear all DQN tensorboard logs (they're typically small)
        dqn_dirs = list(tensorboard_path.glob("DQN_*"))
        for dqn_dir in dqn_dirs:
            # Check if it contains commodities-related events
            event_files = list(dqn_dir.glob("events.out.tfevents.*"))
            if event_files:
                # For safety, we'll note these but not delete automatically
                # User can manually delete if needed
                print(f"[INFO] Found TensorBoard logs in {dqn_dir.name} (not auto-deleted, delete manually if needed)")
    
    # 8. Clear commodities positions from active_positions.json (if requested)
    if include_positions:
        active_positions_file = project_root / "data" / "positions" / "active_positions.json"
        _clear_commodities_positions(active_positions_file)
    
    print()
    print("=" * 80)
    if deleted_count > 0:
        print(f"Commodities data cleared successfully ({deleted_count} item(s) deleted).")
    else:
        print("No commodities data found to clear.")
    print("=" * 80)
    print()
    print("NOTE: Crypto data, features, models, and logs have been preserved.")


def main():
    """Main entry point."""
    import sys
    
    include_positions = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print(__doc__)
            print()
            print("Usage:")
            print("  python clear_commodities.py              # Clear commodities data (preserve positions)")
            print("  python clear_commodities.py --include-positions  # Also clear commodities positions")
            sys.exit(0)
        elif sys.argv[1] == "--include-positions":
            include_positions = True
    
    clear_commodities_data(include_positions=include_positions)


if __name__ == "__main__":
    main()


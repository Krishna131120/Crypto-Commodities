"""
Utility script to wipe all generated datasets, features, models, and logs so
you can restart the ingestion + training pipeline from scratch.
"""
from pathlib import Path
import shutil


def _reset_directory(path: Path, label: str):
    if path.exists():
        try:
            shutil.rmtree(path)
            print(f"[DELETED] {label}: {path}")
        except Exception as exc:
            print(f"[ERROR] {label}: {exc}")
            return
    path.mkdir(parents=True, exist_ok=True)
    print(f"[CREATED] {label}: {path}")


def _clean_pycache(project_root: Path):
    """Remove all __pycache__ directories recursively."""
    pycache_count = 0
    for pycache_dir in project_root.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            pycache_count += 1
        except Exception as exc:
            print(f"[ERROR] Failed to delete {pycache_dir}: {exc}")
    
    if pycache_count > 0:
        print(f"[DELETED] {pycache_count} __pycache__ directory/ies")
    else:
        print("[SKIP] No __pycache__ directories found")


def clear_trading_logs():
    """Clear only trading logs (crypto_trades.jsonl) and active positions."""
    project_root = Path(__file__).parent
    trading_log_file = project_root / "logs" / "trading" / "crypto_trades.jsonl"
    active_positions_file = project_root / "data" / "positions" / "active_positions.json"
    
    print("=" * 60)
    print("CLEARING TRADING LOGS AND POSITIONS")
    print("=" * 60)
    
    # Clear trading logs
    if trading_log_file.exists():
        try:
            trading_log_file.unlink()
            print(f"[DELETED] Trading logs: {trading_log_file}")
        except Exception as exc:
            print(f"[ERROR] Failed to delete trading logs: {exc}")
    else:
        print("[SKIP] Trading logs file not found")
    
    # Clear active positions
    if active_positions_file.exists():
        try:
            # Clear the file by writing an empty JSON object
            active_positions_file.write_text("{}")
            print(f"[CLEARED] Active positions: {active_positions_file}")
        except Exception as exc:
            print(f"[ERROR] Failed to clear active positions: {exc}")
    else:
        # Create the directory if it doesn't exist and create empty file
        active_positions_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            active_positions_file.write_text("{}")
            print(f"[CREATED] Active positions file: {active_positions_file}")
        except Exception as exc:
            print(f"[ERROR] Failed to create active positions file: {exc}")
    
    print("=" * 60)
    print("Trading logs and positions cleared successfully.")
    print("=" * 60)


def clear_logs(include_trading_logs=True):
    project_root = Path(__file__).parent
    crypto_path = project_root / "data" / "json" / "raw" / "crypto"
    commodities_path = project_root / "data" / "json" / "raw" / "commodities"
    features_path = project_root / "data" / "features"
    models_path = project_root / "models"
    logs_path = project_root / "logs"

    print("=" * 60)
    if include_trading_logs:
        print("RESETTING DATA / FEATURES / MODELS / LOGS (INCLUDING TRADING LOGS)")
    else:
        print("RESETTING DATA / FEATURES / MODELS / LOGS (PRESERVING TRADING LOGS)")
    print("=" * 60)

    _reset_directory(crypto_path, "Crypto data")
    _reset_directory(commodities_path, "Commodities data")
    _reset_directory(features_path, "Feature outputs")
    _reset_directory(models_path, "Model artifacts")
    _reset_directory(logs_path, "Training logs")
    
    # Only clear trading logs if explicitly requested
    if include_trading_logs:
        trading_log_file = project_root / "logs" / "trading" / "crypto_trades.jsonl"
        if trading_log_file.exists():
            try:
                trading_log_file.unlink()
                print(f"[DELETED] Trading logs: {trading_log_file}")
            except Exception as exc:
                print(f"[ERROR] Failed to delete trading logs: {exc}")
        
        # Clear active positions
        active_positions_file = project_root / "data" / "positions" / "active_positions.json"
        if active_positions_file.exists():
            try:
                # Clear the file by writing an empty JSON object
                active_positions_file.write_text("{}")
                print(f"[CLEARED] Active positions: {active_positions_file}")
            except Exception as exc:
                print(f"[ERROR] Failed to clear active positions: {exc}")
        else:
            # Create the directory if it doesn't exist and create empty file
            active_positions_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                active_positions_file.write_text("{}")
                print(f"[CREATED] Active positions file: {active_positions_file}")
            except Exception as exc:
                print(f"[ERROR] Failed to create active positions file: {exc}")
    
    print("\n" + "-" * 60)
    print("CLEANING PYTHON CACHE")
    print("-" * 60)
    _clean_pycache(project_root)

    print("=" * 60)
    print("Environment cleared successfully.")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--trading-only":
        clear_trading_logs()
    elif len(sys.argv) > 1 and sys.argv[1] == "--include-trading":
        # Include trading logs if explicitly requested
        clear_logs(include_trading_logs=True)
    else:
        # Default: clear everything EXCEPT trading logs (preserve trade history)
        clear_logs(include_trading_logs=False)


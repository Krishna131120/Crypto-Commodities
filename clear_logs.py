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


def clear_logs():
    project_root = Path(__file__).parent
    crypto_path = project_root / "data" / "json" / "raw" / "crypto"
    commodities_path = project_root / "data" / "json" / "raw" / "commodities"
    features_path = project_root / "data" / "features"
    models_path = project_root / "models"
    logs_path = project_root / "logs"

    print("=" * 60)
    print("RESETTING DATA / FEATURES / MODELS / LOGS")
    print("=" * 60)

    _reset_directory(crypto_path, "Crypto data")
    _reset_directory(commodities_path, "Commodities data")
    _reset_directory(features_path, "Feature outputs")
    _reset_directory(models_path, "Model artifacts")
    _reset_directory(logs_path, "Training logs")
    
    print("\n" + "-" * 60)
    print("CLEANING PYTHON CACHE")
    print("-" * 60)
    _clean_pycache(project_root)

    print("=" * 60)
    print("Environment cleared successfully.")
    print("=" * 60)


if __name__ == "__main__":
    clear_logs()


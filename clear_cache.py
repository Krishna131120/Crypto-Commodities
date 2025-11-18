"""
Clear all cache files and directories from the project.
Removes Python bytecode cache (__pycache__), .pyc files, and all fetched data.
"""
import os
import shutil
from pathlib import Path

def clear_cache():
    """Delete all cache files, directories, and fetched data."""
    project_root = Path(__file__).parent
    cache_removed = []
    files_removed = 0
    dirs_removed = 0
    
    print("=" * 60)
    print("CLEARING CACHE AND FETCHED DATA")
    print("=" * 60)
    
    # Remove all fetched data (crypto and commodities)
    data_dir = project_root / "data" / "json" / "raw"
    if data_dir.exists():
        try:
            shutil.rmtree(data_dir)
            dirs_removed += 1
            print(f"  [DELETED] {data_dir.relative_to(project_root)}/ (all crypto and commodities data)")
        except Exception as e:
            print(f"  [ERROR] Failed to delete {data_dir}: {e}")
    else:
        print(f"  [INFO] No data directory found at {data_dir.relative_to(project_root)}")
    
    # Find and remove all __pycache__ directories
    for pycache_dir in project_root.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            cache_removed.append(str(pycache_dir.relative_to(project_root)))
            dirs_removed += 1
            print(f"  [DELETED] {pycache_dir.relative_to(project_root)}")
        except Exception as e:
            print(f"  [ERROR] Failed to delete {pycache_dir}: {e}")
    
    # Find and remove all .pyc files (standalone bytecode files)
    for pyc_file in project_root.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            files_removed += 1
            print(f"  [DELETED] {pyc_file.relative_to(project_root)}")
        except Exception as e:
            print(f"  [ERROR] Failed to delete {pyc_file}: {e}")
    
    # Find and remove all .pyo files (optimized bytecode)
    for pyo_file in project_root.rglob("*.pyo"):
        try:
            pyo_file.unlink()
            files_removed += 1
            print(f"  [DELETED] {pyo_file.relative_to(project_root)}")
        except Exception as e:
            print(f"  [ERROR] Failed to delete {pyo_file}: {e}")
    
    print("=" * 60)
    print(f"Cleanup complete!")
    print(f"  Directories removed: {dirs_removed}")
    print(f"  Files removed: {files_removed}")
    print(f"  Total items: {dirs_removed + files_removed}")
    print("=" * 60)
    
    if dirs_removed == 0 and files_removed == 0:
        print("  No cache or data files found. Project is already clean!")


if __name__ == "__main__":
    clear_cache()


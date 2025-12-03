#!/usr/bin/env python3
"""Clear Numba cache to force recompilation with latest code."""
import os
import shutil
from pathlib import Path

def clear_numba_cache():
    """Find and clear all __pycache__ and Numba cache directories."""
    project_root = Path(__file__).parent
    cleared = []
    
    # Find all __pycache__ directories
    for pycache_dir in project_root.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            cleared.append(str(pycache_dir))
            print(f"✓ Cleared: {pycache_dir}")
        except Exception as e:
            print(f"✗ Failed to clear {pycache_dir}: {e}")
    
    # Also clear .pyc files
    for pyc_file in project_root.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            cleared.append(str(pyc_file))
            print(f"✓ Deleted: {pyc_file}")
        except Exception as e:
            print(f"✗ Failed to delete {pyc_file}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Cleared {len(cleared)} cached files/directories")
    print(f"{'='*60}")
    print("\n✅ Numba cache cleared. Next run will recompile with latest code.")
    print("\nNow run your training again:")
    print("  python -m training.parallel_self_play")

if __name__ == "__main__":
    clear_numba_cache()

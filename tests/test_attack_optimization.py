#!/usr/bin/env python3
"""
Test script to verify attack detection optimizations.
Run with: python -m pytest tests/test_attack_optimization.py -v
Or directly: python tests/test_attack_optimization.py
"""
import sys
import os

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time


def test_special_attacker_in_range_kernel():
    """Test the _check_special_attacker_in_range_kernel function."""
    from game3d.attacks.check import _check_special_attacker_in_range_kernel
    from game3d.common.shared_types import COORD_DTYPE
    
    # Test BOMB (type 26) at (3, 3, 3) with radius 2
    positions = np.array([[3, 3, 3]], dtype=COORD_DTYPE)
    types = np.array([26], dtype=np.int8)
    
    # Target in range (distance <= 2)
    assert _check_special_attacker_in_range_kernel(positions, types, 3, 3, 5) == True  # dist 2
    assert _check_special_attacker_in_range_kernel(positions, types, 4, 4, 4) == True  # dist sqrt(3)
    assert _check_special_attacker_in_range_kernel(positions, types, 5, 5, 5) == True  # Chebyshev 2
    
    # Target out of range (distance > 2)
    assert _check_special_attacker_in_range_kernel(positions, types, 6, 6, 6) == False  # Chebyshev 3
    assert _check_special_attacker_in_range_kernel(positions, types, 0, 0, 0) == False  # dist 3
    
    print("✅ BOMB range test passed")
    
    # Test ARCHER (type 25) - attacks at exactly squared distance 4
    positions = np.array([[3, 3, 3]], dtype=COORD_DTYPE)
    types = np.array([25], dtype=np.int8)
    
    # Exactly distance 2 (squared = 4)
    assert _check_special_attacker_in_range_kernel(positions, types, 5, 3, 3) == True  # dx=2
    assert _check_special_attacker_in_range_kernel(positions, types, 3, 5, 3) == True  # dy=2
    assert _check_special_attacker_in_range_kernel(positions, types, 3, 3, 5) == True  # dz=2
    
    # Not exactly distance 2
    assert _check_special_attacker_in_range_kernel(positions, types, 4, 3, 3) == False  # dx=1
    assert _check_special_attacker_in_range_kernel(positions, types, 6, 3, 3) == False  # dx=3
    
    print("✅ ARCHER range test passed")


def test_special_attacker_in_range_wrapper():
    """Test the _special_attacker_in_range wrapper function with mock cache."""
    from game3d.attacks.check import _special_attacker_in_range
    from game3d.common.shared_types import COORD_DTYPE
    
    class MockOccCache:
        def __init__(self, positions, types):
            self._positions = positions
            self._types = types
        
        def get_positions(self, color):
            return self._positions
        
        def batch_get_types_only(self, positions):
            return self._types
    
    # Case 1: Bomb in range
    positions = np.array([[3, 3, 3]], dtype=COORD_DTYPE)
    types = np.array([26], dtype=np.int8)  # BOMB
    mock_cache = MockOccCache(positions, types)
    target = np.array([4, 4, 4], dtype=COORD_DTYPE)  # In range
    
    assert _special_attacker_in_range(mock_cache, 1, target) == True
    
    # Case 2: Bomb out of range
    target = np.array([7, 7, 7], dtype=COORD_DTYPE)  # Out of range
    assert _special_attacker_in_range(mock_cache, 1, target) == False
    
    # Case 3: No special attackers
    types = np.array([2], dtype=np.int8)  # KNIGHT
    mock_cache = MockOccCache(positions, types)
    assert _special_attacker_in_range(mock_cache, 1, target) == False
    
    print("✅ _special_attacker_in_range wrapper test passed")


def test_check_module_imports():
    """Verify check.py can be imported without errors."""
    try:
        from game3d.attacks.check import (
            square_attacked_by,
            square_attacked_by_incremental,
            move_would_leave_king_in_check,
            _special_attacker_in_range,
            _check_special_attacker_in_range_kernel
        )
        print("✅ All check.py imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def run_quick_benchmark():
    """Quick benchmark to ensure Numba JIT is working."""
    from game3d.attacks.check import _check_special_attacker_in_range_kernel
    from game3d.common.shared_types import COORD_DTYPE
    
    # First call triggers JIT compilation
    positions = np.array([[3, 3, 3]], dtype=COORD_DTYPE)
    types = np.array([26], dtype=np.int8)
    _ = _check_special_attacker_in_range_kernel(positions, types, 4, 4, 4)
    
    # Now benchmark
    n_iterations = 1000
    start = time.perf_counter()
    for _ in range(n_iterations):
        _check_special_attacker_in_range_kernel(positions, types, 4, 4, 4)
    elapsed = time.perf_counter() - start
    
    per_call_us = (elapsed / n_iterations) * 1_000_000
    print(f"✅ Kernel benchmark: {per_call_us:.2f} µs/call ({n_iterations} iterations)")


if __name__ == "__main__":
    print("=" * 60)
    print("Attack Detection Optimization Tests")
    print("=" * 60)
    
    if not test_check_module_imports():
        sys.exit(1)
    
    test_special_attacker_in_range_kernel()
    test_special_attacker_in_range_wrapper()
    run_quick_benchmark()
    
    print()
    print("=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)

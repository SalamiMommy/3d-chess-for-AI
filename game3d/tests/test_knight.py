#!/usr/bin/env python3
"""Test to verify Knight movement patterns."""
import numpy as np
from game3d.pieces.pieces.knight import KNIGHT_MOVEMENT_VECTORS, BUFFED_KNIGHT_MOVEMENT_VECTORS

print("Testing Knight movement patterns:")

print(f"\nUnbuffed (2,1,0) pattern: {len(KNIGHT_MOVEMENT_VECTORS)} directions")
print("Sample vectors:", KNIGHT_MOVEMENT_VECTORS[:4])

print(f"\nBuffed (2,1,1) pattern: {len(BUFFED_KNIGHT_MOVEMENT_VECTORS)} directions")
print("Sample vectors:", BUFFED_KNIGHT_MOVEMENT_VECTORS[:4])

# Verify the patterns
def check_knight_pattern(vectors, name, expected_z_values):
    """Check if vectors match expected (2,1,z) pattern."""
    all_valid = True
    for v in vectors:
        x, y, z = sorted([abs(v[0]), abs(v[1]), abs(v[2])])
        # Should be sorted as (0, 1, 2) or (1, 1, 2) depending on z value
        if z not in expected_z_values:
            print(f"  ❌ Unexpected vector {v} (sorted abs: {x},{y},{z})")
            all_valid = False
            
    if all_valid:
        print(f"✅ {name} pattern verified!")
    return all_valid

print("\nVerifying patterns:")
unbuffed_ok = check_knight_pattern(KNIGHT_MOVEMENT_VECTORS, "Unbuffed (2,1,0)", [0])
buffed_ok = check_knight_pattern(BUFFED_KNIGHT_MOVEMENT_VECTORS, "Buffed (2,1,1)", [1])

if unbuffed_ok and buffed_ok:
    print("\n✅ All Knight patterns verified!")
else:
    print("\n❌ Some patterns need review")

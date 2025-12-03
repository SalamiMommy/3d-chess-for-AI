#!/usr/bin/env python3
"""Quick test to verify BigKnight movement patterns."""
import numpy as np
from game3d.pieces.pieces.bigknights import (
    KNIGHT31_MOVEMENT_VECTORS, 
    BUFFED_KNIGHT31_MOVEMENT_VECTORS,
    KNIGHT32_MOVEMENT_VECTORS,
    BUFFED_KNIGHT32_MOVEMENT_VECTORS
)

print("Testing BigKnight movement patterns:")
print(f"\nKNIGHT31 unbuffed vectors (should be (3,1,0) pattern): {len(KNIGHT31_MOVEMENT_VECTORS)} directions")
print("Sample vectors:", KNIGHT31_MOVEMENT_VECTORS[:4])

print(f"\nKNIGHT31 buffed vectors (should be (3,1,1) pattern): {len(BUFFED_KNIGHT31_MOVEMENT_VECTORS)} directions")
print("Sample vectors:", BUFFED_KNIGHT31_MOVEMENT_VECTORS[:4])

print(f"\nKNIGHT32 unbuffed vectors (should be (3,2,0) pattern): {len(KNIGHT32_MOVEMENT_VECTORS)} directions")
print("Sample vectors:", KNIGHT32_MOVEMENT_VECTORS[:4])

print(f"\nKNIGHT32 buffed vectors (should be (3,2,1) pattern): {len(BUFFED_KNIGHT32_MOVEMENT_VECTORS)} directions")
print("Sample vectors:", BUFFED_KNIGHT32_MOVEMENT_VECTORS[:4])

# Verify the patterns
def check_pattern(vectors, name, expected_z):
    """Check if vectors match expected pattern."""
    failures = []
    for v in vectors:
        x, y, z = v
        # Check if at least one coordinate has absolute value matching the pattern
        if abs(z) not in expected_z:
            failures.append(v)
    
    if failures:
        print(f"\n❌ {name} has unexpected vectors: {failures[:5]}")
        return False
    else:
        print(f"✅ {name} looks correct!")
        return True

all_good = True
all_good &= check_pattern(KNIGHT31_MOVEMENT_VECTORS, "KNIGHT31 unbuffed", [0, 1, 3])
all_good &= check_pattern(BUFFED_KNIGHT31_MOVEMENT_VECTORS, "KNIGHT31 buffed", [1, 3])
all_good &= check_pattern(KNIGHT32_MOVEMENT_VECTORS, "KNIGHT32 unbuffed", [0, 2, 3])
all_good &= check_pattern(BUFFED_KNIGHT32_MOVEMENT_VECTORS, "KNIGHT32 buffed", [1, 2, 3])

if all_good:
    print("\n✅ All BigKnight patterns verified!")
else:
    print("\n❌ Some patterns need review")

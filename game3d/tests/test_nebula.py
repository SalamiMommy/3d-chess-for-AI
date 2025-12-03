#!/usr/bin/env python3
"""Test to verify Nebula sphere changes."""
import numpy as np
from game3d.pieces.pieces.nebula import _NEBULA_DIRECTIONS, _BUFFED_NEBULA_DIRECTIONS

print("Testing Nebula movement patterns:")

# Check unbuffed (radius 2)
print(f"\nUnbuffed (radius 2): {len(_NEBULA_DIRECTIONS)} positions")
# Calculate Chebyshev distances (max of abs coordinates)
cheb_dists_unbuffed = np.max(np.abs(_NEBULA_DIRECTIONS), axis=1)
print(f"Chebyshev distances: min={cheb_dists_unbuffed.min()}, max={cheb_dists_unbuffed.max()}")
print("Sample positions:", _NEBULA_DIRECTIONS[:5])

# Check buffed (radius 3)
print(f"\nBuffed (radius 3): {len(_BUFFED_NEBULA_DIRECTIONS)} positions")
cheb_dists_buffed = np.max(np.abs(_BUFFED_NEBULA_DIRECTIONS), axis=1)
print(f"Chebyshev distances: min={cheb_dists_buffed.min()}, max={cheb_dists_buffed.max()}")
print("Sample positions:", _BUFFED_NEBULA_DIRECTIONS[:5])

# Verify ranges
if cheb_dists_unbuffed.max() == 2:
    print("\n✅ Unbuffed Nebula uses radius 2")
else:
    print(f"\n❌ Unbuffed Nebula range incorrect: max={cheb_dists_unbuffed.max()}")

if cheb_dists_buffed.max() == 3:
    print("✅ Buffed Nebula uses radius 3")
else:
    print(f"❌ Buffed Nebula range incorrect: max={cheb_dists_buffed.max()}")

# Check that origin is excluded
has_origin_unbuffed = any((d == [0, 0, 0]).all() for d in _NEBULA_DIRECTIONS)
has_origin_buffed = any((d == [0, 0, 0]).all() for d in _BUFFED_NEBULA_DIRECTIONS)

if not has_origin_unbuffed and not has_origin_buffed:
    print("✅ Origin (0,0,0) correctly excluded from both")
else:
    print("❌ Origin should be excluded!")

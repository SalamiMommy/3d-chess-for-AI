#!/usr/bin/env python3
"""Test to verify Geomancer radius changes."""
import numpy as np
from game3d.pieces.pieces.geomancer import GEOMANCY_OFFSETS, BUFFED_GEOMANCY_OFFSETS

print("Testing Geomancer geomancy offsets:")

# Check unbuffed (radius 3)
print(f"\nUnbuffed (radius 3, Cheb dist >= 2): {len(GEOMANCY_OFFSETS)} positions")
cheb_dists_unbuffed = np.max(np.abs(GEOMANCY_OFFSETS), axis=1)
print(f"Chebyshev distances: min={cheb_dists_unbuffed.min()}, max={cheb_dists_unbuffed.max()}")
print("Sample positions:", GEOMANCY_OFFSETS[:5])

# Check buffed (radius 4)
print(f"\nBuffed (radius 4, Cheb dist >= 2): {len(BUFFED_GEOMANCY_OFFSETS)} positions")
cheb_dists_buffed = np.max(np.abs(BUFFED_GEOMANCY_OFFSETS), axis=1)
print(f"Chebyshev distances: min={cheb_dists_buffed.min()}, max={cheb_dists_buffed.max()}")
print("Sample positions:", BUFFED_GEOMANCY_OFFSETS[:5])

# Verify ranges
if cheb_dists_unbuffed.min() >= 2 and cheb_dists_unbuffed.max() == 3:
    print("\n✅ Unbuffed geomancy uses radius 3 (Cheb dist 2-3)")
else:
    print(f"\n❌ Unbuffed geomancy range incorrect: {cheb_dists_unbuffed.min()} to {cheb_dists_unbuffed.max()}")

if cheb_dists_buffed.min() >= 2 and cheb_dists_buffed.max() == 4:
    print("✅ Buffed geomancy uses radius 4 (Cheb dist 2-4)")
else:
    print(f"❌ Buffed geomancy range incorrect: {cheb_dists_buffed.min()} to {cheb_dists_buffed.max()}")

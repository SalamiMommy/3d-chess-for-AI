#!/usr/bin/env python3
"""Test to verify Orbiter sphere surface movement."""
import numpy as np
from game3d.pieces.pieces.orbiter import _ORBITAL_DIRS, _BUFFED_ORBITAL_DIRS

print("Testing Orbiter movement patterns:")

# Calculate distances for unbuffed
unbuffed_sqr_dists = np.array([dx*dx + dy*dy + dz*dz for dx, dy, dz in _ORBITAL_DIRS])
print(f"\nUnbuffed (radius 3 sphere surface): {len(_ORBITAL_DIRS)} directions")
print(f"Squared distances range: {unbuffed_sqr_dists.min()} to {unbuffed_sqr_dists.max()}")
print(f"Expected: around 9 (r^2 for radius 3)")
print(f"Sample vectors:", _ORBITAL_DIRS[:5])

# Calculate distances for buffed
buffed_sqr_dists = np.array([dx*dx + dy*dy + dz*dz for dx, dy, dz in _BUFFED_ORBITAL_DIRS])
print(f"\nBuffed (radius 4 sphere surface): {len(_BUFFED_ORBITAL_DIRS)} directions")
print(f"Squared distances range: {buffed_sqr_dists.min()} to {buffed_sqr_dists.max()}")
print(f"Expected: around 16 (r^2 for radius 4)")
print(f"Sample vectors:", _BUFFED_ORBITAL_DIRS[:5])

# Check distribution
print(f"\nUnbuffed squared distance distribution:")
unique_unbuffed, counts_unbuffed = np.unique(unbuffed_sqr_dists, return_counts=True)
for dist, count in zip(unique_unbuffed, counts_unbuffed):
    print(f"  d^2={dist}: {count} vectors (radius={np.sqrt(dist):.2f})")

print(f"\nBuffed squared distance distribution:")
unique_buffed, counts_buffed = np.unique(buffed_sqr_dists, return_counts=True)
for dist, count in zip(unique_buffed, counts_buffed):
    print(f"  d^2={dist}: {count} vectors (radius={np.sqrt(dist):.2f})")

# Verify they're on the sphere surface (not inside)
if 8 <= unbuffed_sqr_dists.min() and unbuffed_sqr_dists.max() <= 11:
    print("\n✅ Unbuffed vectors approximate radius 3 sphere surface")
else:
    print("\n❌ Unbuffed vectors don't match radius 3 sphere")

if 14 <= buffed_sqr_dists.min() and buffed_sqr_dists.max() <= 18:
    print("✅ Buffed vectors approximate radius 4 sphere surface")
else:
    print("❌ Buffed vectors don't match radius 4 sphere")

#!/usr/bin/env python3
"""Quick test to verify Echo buffed anchors."""
import numpy as np
from game3d.pieces.pieces.echo import _ANCHORS, _BUFFED_ANCHORS, _ECHO_DIRECTIONS, _BUFFED_ECHO_DIRECTIONS

print("Testing Echo movement patterns:")
print(f"\nUnbuffed anchors (offset 2): {len(_ANCHORS)} anchors")
print("Anchors:", _ANCHORS)

print(f"\nBuffed anchors (offset 3): {len(_BUFFED_ANCHORS)} anchors")
print("Anchors:", _BUFFED_ANCHORS)

print(f"\nUnbuffed directions (6 anchors × 26 bubble): {len(_ECHO_DIRECTIONS)} directions")
print("Sample:", _ECHO_DIRECTIONS[:3])

print(f"\nBuffed directions (6 anchors × 26 bubble): {len(_BUFFED_ECHO_DIRECTIONS)} directions")
print("Sample:", _BUFFED_ECHO_DIRECTIONS[:3])

# Verify anchors are at correct distance
unbuffed_distances = np.abs(_ANCHORS).max(axis=1)
buffed_distances = np.abs(_BUFFED_ANCHORS).max(axis=1)

print(f"\n✅ Unbuffed anchor max coordinates: {unbuffed_distances} (should all be 2)")
print(f"✅ Buffed anchor max coordinates: {buffed_distances} (should all be 3)")

if np.all(unbuffed_distances == 2) and np.all(buffed_distances == 3):
    print("\n✅ Echo buffed anchors verified - 1 space further away!")
else:
    print("\n❌ Something is wrong with anchor distances")

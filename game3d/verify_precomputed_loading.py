#!/usr/bin/env python3
"""
Quick verification script to check that precomputed moves are loading correctly
with both buffed and unbuffed variants.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.movement.jump_engine import _load_precomputed_moves, _PRECOMPUTED_MOVES_FLAT, _PRECOMPUTED_OFFSETS, _PRECOMPUTED_MOVES
from game3d.common.shared_types import PieceType

# Force load
_load_precomputed_moves()

print("=" * 60)
print("Precomputed Moves Loading Verification")
print("=" * 60)

# Check a few piece types
test_pieces = [PieceType.KING, PieceType.KNIGHT, PieceType.NEBULA, PieceType.ORBITER, PieceType.WALL]

for piece_type in test_pieces:
    pt_val = piece_type.value
    print(f"\n{piece_type.name}:")
    
    # Check flat moves
    if pt_val in _PRECOMPUTED_MOVES_FLAT:
        variants_flat = list(_PRECOMPUTED_MOVES_FLAT[pt_val].keys())
        print(f"  Flat moves variants: {variants_flat}")
        
        for variant in variants_flat:
            flat_moves = _PRECOMPUTED_MOVES_FLAT[pt_val][variant]
            offsets = _PRECOMPUTED_OFFSETS[pt_val][variant]
            print(f"    {variant}: {len(flat_moves)} total moves, {len(offsets)} positions")
    else:
        print(f"  ❌ No flat moves loaded")
    
    # Check object moves
    if pt_val in _PRECOMPUTED_MOVES:
        variants_obj = list(_PRECOMPUTED_MOVES[pt_val].keys())
        print(f"  Object moves variants: {variants_obj}")
    else:
        print(f"  ⚠️  No object moves loaded (optional)")

print("\n" + "=" * 60)
print("✅ Verification complete!")
print("=" * 60)

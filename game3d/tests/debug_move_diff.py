"""
Debug script to identify which piece types have different move counts.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from collections import defaultdict

def main():
    print("=== Move Count Comparison by Piece Type ===\n")
    
    # Setup
    from game3d.game.gamestate import GameState
    from game3d.core.buffer import state_to_buffer
    from game3d.movement.generator import generate_legal_moves as legacy_gen
    from game3d.core.api import generate_legal_moves as new_gen, invalidate_cache
    from game3d.common.shared_types import PieceType, SIZE
    
    state = GameState.from_startpos()
    buffer = state_to_buffer(state)
    
    # Generate moves
    legacy_moves = legacy_gen(state)
    invalidate_cache()
    new_moves = new_gen(buffer)
    
    print(f"Total legacy moves: {len(legacy_moves)}")
    print(f"Total new moves: {len(new_moves)}")
    print(f"Difference: {len(new_moves) - len(legacy_moves):+d}\n")
    
    # Check move format
    print("Legacy move format (first 3):")
    for i, m in enumerate(legacy_moves[:3]):
        print(f"  {i}: {m}")
    print("New move format (first 3):")
    for i, m in enumerate(new_moves[:3]):
        print(f"  {i}: {m}")
    
    # Normalize moves to (fx, fy, fz, tx, ty, tz) format
    def normalize_moves(moves, label):
        """Convert moves to 6-tuple format."""
        normalized = []
        for m in moves:
            if hasattr(m, '__len__'):
                if len(m) == 6:
                    # Already in (fx, fy, fz, tx, ty, tz) format
                    fx, fy, fz, tx, ty, tz = int(m[0]), int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5])
                elif len(m) == 2:
                    # (from_coord, to_coord) format
                    fx, fy, fz = int(m[0][0]), int(m[0][1]), int(m[0][2])
                    tx, ty, tz = int(m[1][0]), int(m[1][1]), int(m[1][2])
                else:
                    print(f"Unknown format in {label}: {m}")
                    continue
            else:
                print(f"Unknown type in {label}: {type(m)}")
                continue
            
            # Validate bounds
            if not (0 <= fx < SIZE and 0 <= fy < SIZE and 0 <= fz < SIZE and
                    0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE):
                # Skip out-of-bounds (shouldn't happen)
                continue
            
            normalized.append((fx, fy, fz, tx, ty, tz))
        return set(normalized)
    
    legacy_set = normalize_moves(legacy_moves, "legacy")
    new_set = normalize_moves(new_moves, "new")
    
    print(f"\nNormalized legacy moves: {len(legacy_set)}")
    print(f"Normalized new moves: {len(new_set)}")
    
    # Find differences
    extra_in_new = new_set - legacy_set
    missing_in_new = legacy_set - new_set
    
    print(f"\nExtra in new: {len(extra_in_new)}")
    print(f"Missing in new: {len(missing_in_new)}\n")
    
    # Get piece type at each source square
    board_type = buffer.board_type
    
    # Count by piece type
    def count_by_type(moves, label):
        counts = defaultdict(int)
        examples = defaultdict(list)
        for m in moves:
            fx, fy, fz = m[0], m[1], m[2]
            pt = int(board_type[fx, fy, fz])
            counts[pt] += 1
            if len(examples[pt]) < 3:
                examples[pt].append(m)
        
        print(f"\n{label}:")
        print("-" * 50)
        for pt in sorted(counts.keys()):
            try:
                name = PieceType(pt).name
            except:
                name = f"Type_{pt}"
            print(f"  {name:20s}: {counts[pt]:4d} moves")
            for ex in examples[pt][:2]:
                print(f"    Example: ({ex[0]},{ex[1]},{ex[2]}) -> ({ex[3]},{ex[4]},{ex[5]})")
    
    if extra_in_new:
        count_by_type(extra_in_new, "EXTRA moves in new system (not in legacy)")
    
    if missing_in_new:
        count_by_type(missing_in_new, "MISSING moves in new system (in legacy only)")

if __name__ == "__main__":
    main()


"""
Simple test of filter_legal_moves with manual moves.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from game3d.core.buffer import GameBuffer, create_empty_buffer
from game3d.core.attacks import filter_legal_moves, is_check
from game3d.common.shared_types import COORD_DTYPE

def create_pinned_pawn_buffer():
    """Create a position where the pawn is pinned."""
    buffer = create_empty_buffer(max_pieces=100)
    
    buffer.meta[0] = 1  # White to move
    
    idx = 0
    
    # White King at (4, 4, 0)
    buffer.occupied_coords[idx] = np.array([4, 4, 0], dtype=COORD_DTYPE)
    buffer.occupied_types[idx] = 6
    buffer.occupied_colors[idx] = 1
    buffer.board_type[4, 4, 0] = 6
    buffer.board_color[4, 4, 0] = 1
    buffer.meta[4] = idx
    idx += 1
    
    # White Pawn at (4, 4, 1) - pinned
    buffer.occupied_coords[idx] = np.array([4, 4, 1], dtype=COORD_DTYPE)
    buffer.occupied_types[idx] = 1
    buffer.occupied_colors[idx] = 1
    buffer.board_type[4, 4, 1] = 1
    buffer.board_color[4, 4, 1] = 1
    idx += 1
    
    # Black King at (0, 0, 8)
    buffer.occupied_coords[idx] = np.array([0, 0, 8], dtype=COORD_DTYPE)
    buffer.occupied_types[idx] = 6
    buffer.occupied_colors[idx] = 2
    buffer.board_type[0, 0, 8] = 6
    buffer.board_color[0, 0, 8] = 2
    buffer.meta[5] = idx
    idx += 1
    
    # Black Rook at (4, 4, 7) - pinning the pawn
    buffer.occupied_coords[idx] = np.array([4, 4, 7], dtype=COORD_DTYPE)
    buffer.occupied_types[idx] = 4
    buffer.occupied_colors[idx] = 2
    buffer.board_type[4, 4, 7] = 4
    buffer.board_color[4, 4, 7] = 2
    idx += 1
    
    return GameBuffer(
        buffer.occupied_coords,
        buffer.occupied_types,
        buffer.occupied_colors,
        idx,
        buffer.board_type,
        buffer.board_color,
        buffer.board_color_flat,
        buffer.is_buffed,
        buffer.is_debuffed,
        buffer.is_frozen,
        buffer.meta,
        0,
        buffer.history,
        0
    )

def test_filter():
    print("=== Test filter_legal_moves ===\n")
    
    buffer = create_pinned_pawn_buffer()
    
    print(f"Is White in check initially? {is_check(buffer, 1)}")
    
    # Create some test moves manually
    # Pawn moves:
    moves = np.array([
        [4, 4, 1, 4, 4, 2],  # Pawn forward (legal - stays on pin line)
        [4, 4, 1, 5, 5, 2],  # Pawn diagonal (illegal - breaks pin)
        [4, 4, 1, 3, 3, 2],  # Pawn diagonal (illegal - breaks pin)
        [4, 4, 0, 3, 4, 0],  # King lateral (legal if not attacked)
        [4, 4, 0, 5, 4, 0],  # King lateral (legal if not attacked)
    ], dtype=COORD_DTYPE)
    
    print(f"\nTest moves: {len(moves)}")
    for m in moves:
        print(f"  {m[:3]} -> {m[3:]}")
    
    # Filter
    legal = filter_legal_moves(buffer, moves)
    
    print(f"\nLegal moves: {len(legal)}")
    for m in legal:
        print(f"  {m[:3]} -> {m[3:]}")
    
    # Expected:
    # - (4,4,1) -> (4,4,2): LEGAL (pawn forward, stays on pin line)
    # - (4,4,1) -> (5,5,2): ILLEGAL (pawn diagonal, breaks pin)
    # - (4,4,1) -> (3,3,2): ILLEGAL (pawn diagonal, breaks pin)
    # - King moves: LEGAL if not attacked

if __name__ == "__main__":
    test_filter()


import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from game3d.common.shared_types import Color, PieceType
from game3d.game.gamestate import GameState
from game3d.pieces.pieces.infiltrator import generate_infiltrator_moves
from game3d.pieces.pieces.pawn import generate_pawn_moves

def test_pawn_z_axis_movement():
    """Verify that pawns move along the Z-axis (index 2)."""
    game = GameState.from_startpos()
    cache_manager = game.cache_manager
    cache_manager.occupancy_cache.clear()
    
    # Place White Pawn at (4, 4, 2)
    pawn_pos = np.array([4, 4, 2])
    cache_manager.occupancy_cache.set_position_fast(pawn_pos, PieceType.PAWN.value, Color.WHITE.value)
    
    moves = generate_pawn_moves(cache_manager, Color.WHITE.value, pawn_pos)
    
    # Expect move to (4, 4, 3)
    found_z_push = False
    for move in moves:
        tx, ty, tz = move[3], move[4], move[5]
        if tx == 4 and ty == 4 and tz == 3:
            found_z_push = True
            
    assert found_z_push, "Pawn should move along Z-axis to (4, 4, 3)"

def test_infiltrator_pawn_front_targeting():
    """Verify that Infiltrator targets the square IN FRONT of the pawn."""
    game = GameState.from_startpos()
    cache_manager = game.cache_manager
    cache_manager.occupancy_cache.clear()
    
    # Place White Infiltrator at (0, 0, 0)
    inf_pos = np.array([0, 0, 0])
    cache_manager.occupancy_cache.set_position_fast(inf_pos, PieceType.INFILTRATOR.value, Color.WHITE.value)
    
    # Place Black Pawn at (4, 4, 6)
    # Black pawns move -Z (to 5), so front is (4, 4, 5)
    black_pawn_pos = np.array([4, 4, 6])
    cache_manager.occupancy_cache.set_position_fast(black_pawn_pos, PieceType.PAWN.value, Color.BLACK.value)
    
    moves = generate_infiltrator_moves(cache_manager, Color.WHITE.value, inf_pos)
    
    found_front = False
    for move in moves:
        tx, ty, tz = move[3], move[4], move[5]
        if tx == 4 and ty == 4 and tz == 5:
            found_front = True
            
    assert found_front, "Infiltrator should target square in front of Black pawn (4, 4, 5)"

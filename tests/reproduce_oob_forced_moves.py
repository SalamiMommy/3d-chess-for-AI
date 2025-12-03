"""Reproduce out-of-bounds coordinates from forced moves."""
import numpy as np
from game3d.board.board import Board
from game3d.cache.manager import OptimizedCacheManager
from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.pieces.pieces.whitehole import push_candidates_vectorized
from game3d.pieces.pieces.blackhole import suck_candidates_vectorized

def test_wall_immunity():
    """Test that Walls are immune to physics."""
    print("Testing Wall immunity to blackhole/whitehole physics...")
    
    board = Board()
    cache = OptimizedCacheManager(board)
    state = GameState(board=board, cache_manager=cache, color=Color.WHITE)
    
    # Place a whitehole at [7, 2, 7]
    whitehole_pos = np.array([7, 2, 7], dtype=COORD_DTYPE)
    cache.occupancy_cache.set_position(whitehole_pos, np.array([PieceType.WHITEHOLE, Color.WHITE]))
    
    # Place a Wall anchor at [8, 2, 7] (would push to [9, 2, 7] which is OOB)
    # Wall occupies [8,2,7], [9,2,7], [8,3,7], [9,3,7]
    # Wait, [9,2,7] is already OOB! So this is an invalid wall position.
    # Let's use [7, 2, 7] which occupies [7,2,7], [8,2,7], [7,3,7], [8,3,7]
    
    # Actually, let's place the wall at [6, 1, 7]
    wall_anchor = np.array([6, 1, 7], dtype=COORD_DTYPE)
    wall_offsets = np.array([
        [0, 0, 0],
        [1, 0, 0],  # [7, 1, 7]
        [0, 1, 0],  # [6, 2, 7]
        [1, 1, 0]   # [7, 2, 7]
    ], dtype=COORD_DTYPE)
    
    for offset in wall_offsets:
        pos = wall_anchor + offset
        cache.occupancy_cache.set_position(pos, np.array([PieceType.WALL, Color.BLACK]))
    
    print(f"Whitehole at: {whitehole_pos}")
    print(f"Wall anchor at: {wall_anchor}")
    print(f"Wall squares: {wall_anchor + wall_offsets}")
    
    # Calculate forced moves
    forced_moves = push_candidates_vectorized(cache, Color.WHITE)
    
    print(f"\nForced push moves: {len(forced_moves)}")
    for move in forced_moves:
        from_pos = move[:3]
        to_pos = move[3:]
        print(f"  {from_pos} -> {to_pos}")
        
        # Check if destination is out of bounds
        if np.any(to_pos < 0) or np.any(to_pos >= 9):
            print(f"    ❌ OUT OF BOUNDS!")
            piece_info = cache.occupancy_cache.get(from_pos)
            print(f"    Piece type: {PieceType(piece_info['piece_type']).name if piece_info else 'None'}")
    
    if len(forced_moves) > 0:
        print("\n❌ FAILURE: Walls should be immune to physics!")
    else:
        print("\n✅ SUCCESS: No forced moves (Walls are immune)")

def test_normal_piece_near_edge():
    """Test that normal pieces near the edge are handled correctly."""
    print("\n" + "="*60)
    print("Testing normal piece near board edge...")
    
    board = Board()
    cache = OptimizedCacheManager(board)
    state = GameState(board=board, cache_manager=cache, color=Color.WHITE)
    
    # Place a whitehole at [7, 2, 7]
    whitehole_pos = np.array([7, 2, 7], dtype=COORD_DTYPE)
    cache.occupancy_cache.set_position(whitehole_pos, np.array([PieceType.WHITEHOLE, Color.WHITE]))
    
    # Place a pawn at [8, 2, 7] (should try to push to [9, 2, 7] which is OOB)
    pawn_pos = np.array([8, 2, 7], dtype=COORD_DTYPE)
    cache.occupancy_cache.set_position(pawn_pos, np.array([PieceType.PAWN, Color.BLACK]))
    
    print(f"Whitehole at: {whitehole_pos}")
    print(f"Pawn at: {pawn_pos}")
    
    # Calculate forced moves
    forced_moves = push_candidates_vectorized(cache, Color.WHITE)
    
    print(f"\nForced push moves: {len(forced_moves)}")
    for move in forced_moves:
        from_pos = move[:3]
        to_pos = move[3:]
        print(f"  {from_pos} -> {to_pos}")
        
        # Check if destination is out of bounds
        if np.any(to_pos < 0) or np.any(to_pos >= 9):
            print(f"    ❌ OUT OF BOUNDS!")
    
    if len(forced_moves) == 0:
        print("\n✅ SUCCESS: No forced moves (pawn at edge cannot be pushed OOB)")
    else:
        print("\n❌ FAILURE: Pawn can be pushed OOB!")

if __name__ == "__main__":
    test_wall_immunity()
    test_normal_piece_near_edge()

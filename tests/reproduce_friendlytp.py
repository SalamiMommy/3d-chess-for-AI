
import numpy as np
from game3d.game.gamestate import GameState
from game3d.common.shared_types import PieceType, Color
from game3d.pieces.pieces.friendlytp import generate_friendlytp_moves

def test_friendlytp():
    print("Initializing GameState...")
    game = GameState.from_startpos()
    
    # Clear board
    game.cache_manager.occupancy_cache.clear()
    
    # Setup:
    # Friendly Teleporter at [0, 0, 0] (White)
    # Friendly Pawn at [5, 5, 5] (White)
    # Enemy Pawn at [0, 1, 0] (Black) - for capture test
    
    tp_pos = np.array([0, 0, 0])
    friend_pos = np.array([5, 5, 5])
    enemy_pos = np.array([0, 1, 0])
    
    game.cache_manager.occupancy_cache.set_position(tp_pos, np.array([PieceType.FRIENDLYTELEPORTER, Color.WHITE]))
    game.cache_manager.occupancy_cache.set_position(friend_pos, np.array([PieceType.PAWN, Color.WHITE]))
    game.cache_manager.occupancy_cache.set_position(enemy_pos, np.array([PieceType.PAWN, Color.BLACK]))
    
    print("Generating moves for Friendly Teleporter at [0, 0, 0]...")
    moves = generate_friendlytp_moves(game.cache_manager, Color.WHITE, tp_pos)
    
    print(f"Total moves generated: {len(moves)}")
    
    # Check King Moves
    # Should be able to capture enemy at [0, 1, 0]
    capture_moves = [m for m in moves if np.array_equal(m[3:6], enemy_pos)]
    print(f"Capture moves to [0, 1, 0]: {len(capture_moves)}")
    if len(capture_moves) > 0:
        print("PASS: King capture move generated.")
    else:
        print("FAIL: King capture move NOT generated.")
        
    # Should be able to move to empty adjacent squares, e.g., [1, 0, 0]
    adj_pos = np.array([1, 0, 0])
    adj_moves = [m for m in moves if np.array_equal(m[3:6], adj_pos)]
    print(f"Moves to adjacent empty [1, 0, 0]: {len(adj_moves)}")
    if len(adj_moves) > 0:
        print("PASS: King move to empty square generated.")
    else:
        print("FAIL: King move to empty square NOT generated.")

    # Check Teleport Moves
    # Should be able to teleport to squares adjacent to friendly pawn at [5, 5, 5]
    # e.g., [5, 5, 4], [4, 5, 5], etc.
    teleport_target = np.array([5, 5, 4])
    tele_moves = [m for m in moves if np.array_equal(m[3:6], teleport_target)]
    print(f"Teleport moves to [5, 5, 4] (adjacent to friend): {len(tele_moves)}")
    if len(tele_moves) > 0:
        print("PASS: Teleport move generated.")
    else:
        print("FAIL: Teleport move NOT generated.")
        
    # Check Invalid Teleport
    # Should NOT be able to teleport to random square [3, 3, 3] (not adjacent to any friend)
    invalid_target = np.array([3, 3, 3])
    invalid_moves = [m for m in moves if np.array_equal(m[3:6], invalid_target)]
    print(f"Moves to invalid target [3, 3, 3]: {len(invalid_moves)}")
    if len(invalid_moves) == 0:
        print("PASS: Invalid teleport move NOT generated.")
    else:
        print("FAIL: Invalid teleport move generated!")

if __name__ == "__main__":
    test_friendlytp()

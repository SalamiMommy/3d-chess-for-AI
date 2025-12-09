
import time
import numpy as np
from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.attacks.check import square_attacked_by_incremental
from game3d.common.shared_types import Color, PieceType

def benchmark_incremental_check():
    gs = GameState(Board())
    # Setup a complex board
    # Add many pieces to make export_buffer_data slow
    # Place pieces to create "affected" relationships
    
    # White King
    gs.cache_manager.occupancy_cache.set_position(np.array([4, 4, 0]), np.array([PieceType.KING, Color.WHITE]))
    
    # Add noise pieces
    for x in range(9):
        for y in range(9):
            if (x+y) % 3 == 0:
                pos = np.array([x, y, 8])
                gs.cache_manager.occupancy_cache.set_position(pos, np.array([PieceType.PAWN, Color.BLACK]))

    # Setup specific move scenario
    # Move a white pawn
    from_pos = np.array([4, 4, 1])
    to_pos = np.array([4, 4, 2])
    gs.cache_manager.occupancy_cache.set_position(from_pos, np.array([PieceType.PAWN, Color.WHITE]))
    
    # Add attacker that would be affected (e.g. rook blocked by pawn)
    attacker_pos = np.array([4, 4, 5])
    gs.cache_manager.occupancy_cache.set_position(attacker_pos, np.array([PieceType.ROOK, Color.BLACK]))
    
    # Warmup caches
    gs.gen_moves()
    
    # Run Benchmark
    start = time.time()
    iterations = 1000
    
    # We need to simulate the call context
    # square_attacked_by_incremental requires 'cache' and specific args
    
    cache = gs.cache_manager
    target_square = np.array([4, 4, 0]) # Check if King square is attacked
    
    # Pre-populate cache so we hit the "incremental" path
    # But we want to measure the function itself, not the setup.
    # The function calls get_pieces_affected_by_move inside.
    
    # We need to fake the "old moves" being in cache.
    # gen_moves() did that.
    
    # Simulate: Move has happened on board?
    # move_would_leave_king_in_check updates the board via set_position_fast BEFORE calling incremental.
    # So we should update the board.
    
    gs.cache_manager.occupancy_cache.set_position_fast(from_pos, 0, 0)
    gs.cache_manager.occupancy_cache.set_position_fast(to_pos, PieceType.PAWN, Color.WHITE)
    
    for _ in range(iterations):
        # We need to ensure we don't hit the "cached mask result" immediately if we want to exercise the logic.
        # But real usage HITS the logic when pieces are affected.
        # Here we have affected pieces (the rook).
        
        # We also need to clear attack mask cache? 
        # Actually square_attacked_by_incremental takes a cached mask and patches it.
        # If we invoke it repeatedly, it should do the same work (computing the patch).
        
        is_attacked = square_attacked_by_incremental(
            gs.board,
            target_square,
            Color.BLACK,
            cache,
            from_pos,
            to_pos
        )
    
    end = time.time()
    print(f"Time for {iterations} iterations: {end - start:.4f}s")
    print(f"Per iteration: {(end - start)/iterations*1000:.4f}ms")

if __name__ == "__main__":
    benchmark_incremental_check()

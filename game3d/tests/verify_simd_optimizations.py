
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.common.shared_types import SIZE, SIZE_SQUARED, COORD_DTYPE, Color, PieceType, RADIUS_2_OFFSETS
from game3d.cache.manager import OptimizedCacheManager
from game3d.movement.jump_engine import get_jump_movement_generator, _generate_and_filter_jump_moves
from game3d.pieces.pieces.slower import get_debuffed_squares, _get_slower_debuff_squares_fast

class MockBoard:
    def __init__(self):
        self.generation = 0
        self._cache_manager = None
        
    def get_initial_setup(self):
        # Return empty board setup
        return (np.empty((0, 3), dtype=COORD_DTYPE), 
                np.empty(0, dtype=np.int8), 
                np.empty(0, dtype=np.int8))

def test_jump_engine_optimization():
    print("Testing Jump Engine Optimization...")
    
    # Setup
    board = MockBoard()
    cache_manager = OptimizedCacheManager(board)
    jump_engine = get_jump_movement_generator()
    
    # Create a board with some pieces
    # Center piece (White Knight)
    pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
    
    # Add some obstacles and enemies
    # Friendly at [5, 6, 5] (Knight jump)
    cache_manager.occupancy_cache.set_position(np.array([5, 6, 5], dtype=COORD_DTYPE), np.array([PieceType.PAWN, Color.WHITE]))
    # Enemy at [3, 2, 3] (Knight jump)
    cache_manager.occupancy_cache.set_position(np.array([3, 2, 3], dtype=COORD_DTYPE), np.array([PieceType.PAWN, Color.BLACK]))
    
    # Directions (Knight moves)
    # Just a few sample directions
    directions = np.array([
        [1, 2, 1],
        [-1, -2, -1],
        [1, 2, -1], # Blocked by friendly
        [-1, -2, 1] # Capture enemy
    ], dtype=COORD_DTYPE)
    
    # 1. Test Correctness
    flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
    
    # Expected moves:
    # [5, 6, 5] is friendly -> Blocked
    # [3, 2, 3] is enemy -> Capture
    # [5, 6, 3] (4+1, 4+2, 4-1) -> Empty -> Valid
    # [3, 2, 5] (4-1, 4-2, 4+1) -> Empty -> Valid
    
    moves = _generate_and_filter_jump_moves(pos, directions, flattened_occ, True, Color.WHITE)
    
    print(f"Generated {len(moves)} moves")
    for m in moves:
        print(f"Move: {m}")
        
    # Verify count
    # Should be 3 moves (1 capture, 2 empty)
    assert len(moves) == 3
    
    # Verify capture
    capture_move = None
    for m in moves:
        if m[3] == 3 and m[4] == 2 and m[5] == 3:
            capture_move = m
            break
    assert capture_move is not None
    
    # Verify blocked
    blocked_move = None
    for m in moves:
        if m[3] == 5 and m[4] == 6 and m[5] == 5:
            blocked_move = m
            break
    assert blocked_move is None

    print("Jump Engine Correctness: PASS")
    
    # 2. Benchmark
    # Run many times
    start_time = time.time()
    for _ in range(10000):
        _generate_and_filter_jump_moves(pos, directions, flattened_occ, True, Color.WHITE)
    end_time = time.time()
    print(f"Jump Engine Benchmark (10k iters): {end_time - start_time:.4f}s")


def test_slower_optimization():
    print("\nTesting Slower Piece Optimization...")
    
    board = MockBoard()
    cache_manager = OptimizedCacheManager(board)
    
    # Place Slower pieces
    slower_pos1 = np.array([4, 4, 4], dtype=COORD_DTYPE)
    slower_pos2 = np.array([2, 2, 2], dtype=COORD_DTYPE)
    
    cache_manager.occupancy_cache.set_position(slower_pos1, np.array([PieceType.SLOWER, Color.WHITE]))
    cache_manager.occupancy_cache.set_position(slower_pos2, np.array([PieceType.SLOWER, Color.WHITE]))
    
    # Place enemies within range
    # [4, 4, 5] (dist 1) -> Affected
    enemy1 = np.array([4, 4, 5], dtype=COORD_DTYPE)
    cache_manager.occupancy_cache.set_position(enemy1, np.array([PieceType.PAWN, Color.BLACK]))
    
    # [6, 4, 4] (dist 2) -> Affected
    enemy2 = np.array([6, 4, 4], dtype=COORD_DTYPE)
    cache_manager.occupancy_cache.set_position(enemy2, np.array([PieceType.PAWN, Color.BLACK]))
    
    # [7, 4, 4] (dist 3) -> Not Affected
    enemy3 = np.array([7, 4, 4], dtype=COORD_DTYPE)
    cache_manager.occupancy_cache.set_position(enemy3, np.array([PieceType.PAWN, Color.BLACK]))
    
    # Place friendly -> Not Affected
    friendly = np.array([4, 5, 4], dtype=COORD_DTYPE)
    cache_manager.occupancy_cache.set_position(friendly, np.array([PieceType.PAWN, Color.WHITE]))
    
    # 1. Test Correctness
    debuffed = get_debuffed_squares(cache_manager, Color.WHITE)
    
    print(f"Found {len(debuffed)} debuffed squares")
    for sq in debuffed:
        print(f"Square: {sq}")
        
    # Should find enemy1 and enemy2
    assert len(debuffed) == 2
    
    # Check coordinates
    found_enemy1 = False
    found_enemy2 = False
    for sq in debuffed:
        if np.array_equal(sq, enemy1): found_enemy1 = True
        if np.array_equal(sq, enemy2): found_enemy2 = True
        
    assert found_enemy1
    assert found_enemy2
    
    print("Slower Optimization Correctness: PASS")
    
    # 2. Benchmark
    start_time = time.time()
    for _ in range(1000):
        get_debuffed_squares(cache_manager, Color.WHITE)
    end_time = time.time()
    print(f"Slower Benchmark (1k iters): {end_time - start_time:.4f}s")

if __name__ == "__main__":
    test_jump_engine_optimization()
    test_slower_optimization()

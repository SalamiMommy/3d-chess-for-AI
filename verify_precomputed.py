
import numpy as np
import time
from game3d.movement.jump_engine import JumpMovementEngine
from game3d.common.shared_types import PieceType, SIZE, COORD_DTYPE, COLOR_WHITE

class MockOccupancyCache:
    def get_flattened_occupancy(self):
        return np.zeros(SIZE**3, dtype=np.int8)

class MockCacheManager:
    def __init__(self):
        self.occupancy_cache = MockOccupancyCache()

def verify():
    cm = MockCacheManager()
    engine = JumpMovementEngine(cm)
    
    pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
    
    # Test Knight (should use precomputed)
    print("Testing Knight move generation...")
    start_time = time.time()
    # We need to pass directions even if precomputed is used (as fallback/argument requirement)
    # Just pass dummy directions to prove precomputed is used? 
    # No, if I pass dummy directions and it works, it proves precomputed is used.
    # But jump_engine requires valid directions array shape.
    
    dummy_directions = np.zeros((1, 3), dtype=COORD_DTYPE) 
    # If I pass zeros, and allow_zero_direction=False, it returns empty if not using precomputed.
    # But wait, jump_engine filters zero directions.
    
    # Let's pass a single valid direction that is NOT a knight move.
    # If it returns knight moves, then it's using precomputed.
    dummy_directions = np.array([[1, 0, 0]], dtype=COORD_DTYPE)
    
    moves = engine.generate_jump_moves(
        color=COLOR_WHITE,
        pos=pos,
        directions=dummy_directions,
        piece_type=PieceType.KNIGHT
    )
    
    print(f"Generated {len(moves)} moves.")
    if len(moves) > 1:
        print("SUCCESS: Generated multiple moves despite dummy directions. Precomputed moves are being used!")
        print(f"Sample move: {moves[0]}")
    else:
        print("FAILURE: Generated moves match dummy directions (or empty). Precomputed moves NOT used.")
        print(moves)

if __name__ == "__main__":
    verify()


import numpy as np

from game3d.common.shared_types import PieceType, COORD_DTYPE, FLOAT_DTYPE, Color
from training.opponents import OpponentBase, _compute_king_proximity_rewards_vectorized, AdaptiveOpponent
from game3d.game.gamestate import GameState
from game3d.cache.manager import OptimizedCacheManager
from game3d.movement.movepiece import Move

class MockOccupancyCache:
    def __init__(self, enemy_king_pos):
        self.enemy_king_pos = enemy_king_pos
        self._occ = np.zeros((9, 9, 9), dtype=np.int32) # Dummy

    def has_priest(self, color):
        return False

    def find_king(self, color):
        return self.enemy_king_pos
    
    def batch_get_attributes_unsafe(self, coords):
        # Dummy implementation
        n = len(coords)
        return np.zeros(n, dtype=int), np.zeros(n, dtype=int)

class MockCacheManager:
    def __init__(self, enemy_king_pos):
        self.occupancy_cache = MockOccupancyCache(enemy_king_pos)
        self.move_cache = None # Not needed for this specific test usually

# We can test the _apply_check_rewards method directly by subclassing or monkeypatching, 
# or just test the logic that we plan to inject.
# Actually, let's create a minimal test that calls _apply_check_rewards using a partial mock.

def test_king_proximity_exclusion():
    # Setup
    # Enemy King at (4, 4, 4)
    enemy_king_pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
    
    opponent = OpponentBase(Color.WHITE)
    
    # We want to test _apply_check_rewards
    # It requires: rewards, cache_manager, to_coords, from_types
    
    n_moves = 2
    # Move 0: Queen moves to (3, 4, 4) -> dist 1 to King -> Should get reward
    # Move 1: King moves to (5, 4, 4) -> dist 1 to King -> Should NOT get reward (after fix)
    
    to_coords = np.array([
        [3, 4, 4],
        [5, 4, 4]
    ], dtype=COORD_DTYPE)
    
    from_types = np.array([
        PieceType.QUEEN.value,
        PieceType.KING.value
    ], dtype=np.int32)
    
    rewards = np.zeros(n_moves, dtype=FLOAT_DTYPE)
    
    cache_manager = MockCacheManager(enemy_king_pos)
    
    # Call the method
    opponent._apply_check_rewards(
        rewards, 
        cache_manager, 
        to_coords, 
        from_types, 
        check_reward=0.0, # Ignore check reward for this test
        proximity_reward=1.0
    )
    
    print(f"Queen Reward: {rewards[0]}")
    print(f"King Reward: {rewards[1]}")
    
    # Assertions
    assert rewards[0] == 1.0, "Queen should get proximity reward"
    
    # This assertion is expected to FAIL before the fix
    if rewards[1] != 0.0:
        print("FAIL: King received proximity reward!")
        exit(1)
    else:
        print("PASS: King did not receive proximity reward.")

if __name__ == "__main__":
    try:
        test_king_proximity_exclusion()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

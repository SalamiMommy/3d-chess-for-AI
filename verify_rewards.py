
import numpy as np
from unittest.mock import MagicMock
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, FLOAT_DTYPE
from training.opponents import create_opponent, AVAILABLE_OPPONENTS

def test_opponents():
    print("Testing opponents...")
    
    # Mock CacheManager and OccupancyCache
    mock_occupancy = MagicMock()
    # batch_get_attributes returns (colors, types)
    # Let's say we have 1 move
    mock_occupancy.batch_get_attributes.return_value = (
        np.array([2], dtype=np.int8), # captured_colors (enemy)
        np.array([PieceType.PAWN.value], dtype=np.int8) # captured_types
    )
    mock_occupancy.has_priest.return_value = False # No enemy priests -> trigger check reward logic
    mock_occupancy.find_king.return_value = np.array([4, 4, 4], dtype=COORD_DTYPE) # Enemy king at center
    mock_occupancy.get_positions.return_value = np.empty((0, 3), dtype=COORD_DTYPE) # No priests for PriestHunter
    mock_occupancy.get.return_value = None # For neighbor checks

    mock_move_cache = MagicMock()
    mock_move_cache.get_cached_moves.return_value = np.zeros((0, 6), dtype=COORD_DTYPE)

    mock_cache_manager = MagicMock()
    mock_cache_manager.occupancy_cache = mock_occupancy
    mock_cache_manager.move_cache = mock_move_cache

    mock_state = MagicMock()
    mock_state.cache_manager = mock_cache_manager
    mock_state.halfmove_clock = 0
    mock_state.board.byte_hash.return_value = b'123'

    # Create dummy moves: 1 move from (0,0,0) to (1,1,1)
    moves = np.array([[0, 0, 0, 1, 1, 1]], dtype=COORD_DTYPE)

    for opp_name in AVAILABLE_OPPONENTS:
        print(f"Testing {opp_name}...")
        try:
            opponent = create_opponent(opp_name, Color.WHITE)
            rewards = opponent.batch_reward(mock_state, moves)
            print(f"  Reward: {rewards[0]}")
            assert rewards.shape == (1,)
            assert not np.isnan(rewards[0])
        except Exception as e:
            print(f"  FAILED: {e}")
            raise e

    print("All opponents passed!")

if __name__ == "__main__":
    test_opponents()

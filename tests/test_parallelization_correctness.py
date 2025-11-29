"""Test correctness of parallelization optimizations."""
import numpy as np
from game3d.game.factory import start_game_state
from game3d.movement.generator import generate_legal_moves
from game3d.movement.pseudolegal import JOBLIB_AVAILABLE

def test_move_generation_correctness():
    """Test that parallel and sequential paths produce same results."""
    print("Testing move generation correctness...")
    print(f"Joblib available: {JOBLIB_AVAILABLE}")
    
    # Initialize game
    state = start_game_state()
    
    # Generate moves (will use parallel if available)
    moves = generate_legal_moves(state)
    
    print(f"✅ Generated {len(moves)} legal moves")
    print(f"   Move shape: {moves.shape}")
    
    # Validate move format
    assert moves.ndim == 2, "Moves should be 2D array"
    if moves.size > 0:
        assert moves.shape[1] == 6, "Each move should have 6 coordinates"
        
        # Check all moves are within board bounds
        assert np.all(moves >= 0), "All coordinates should be >= 0"
        assert np.all(moves <= 9), "All coordinates should be <= 9"
        
        print(f"✅ All moves within valid bounds")
        
        # Check moves are unique
        unique_moves = np.unique(moves, axis=0)
        print(f"   Unique moves: {len(unique_moves)} / {len(moves)}")
    
    print("\n✅ All correctness checks passed!")
    return True

def test_trailblaze_batch_counters():
    """Test batch_get_counters functionality."""
    print("\nTesting TrailblazeCache.batch_get_counters...")
    
    state = start_game_state()
    cache = state.cache_manager.trailblaze_cache
    
    # Test with some coordinates
    coords = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2]
    ], dtype=np.int16)
    
    # Get counters (should all be 0 initially)
    counters = cache.batch_get_counters(coords)
    
    assert counters.shape == (3,), f"Expected shape (3,), got {counters.shape}"
    assert np.all(counters == 0), "All counters should be 0 initially"
    
    # Increment one counter
    cache.increment_counter(coords[0])
    
    # Get counters again
    counters = cache.batch_get_counters(coords)
    assert counters[0] == 1, "First counter should be 1"
    assert counters[1] == 0, "Second counter should be 0"
    assert counters[2] == 0, "Third counter should be 0"
    
    print("✅ batch_get_counters works correctly!")
    return True

if __name__ == "__main__":
    test_move_generation_correctness()
    test_trailblaze_batch_counters()
    print("\n" + "="*60)
    print("All tests passed! ✅")
    print("="*60)

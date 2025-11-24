import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from game3d.cache.effectscache.trailblazecache import TrailblazeCache
from game3d.common.shared_types import Color, COORD_DTYPE

class TestTrailblazerColor:
    def test_friendly_fire_off(self):
        """Test that trails do not affect friendly pieces."""
        cache = TrailblazeCache()
        
        # Create a trail for White
        trailblazer_pos = np.array([[0, 0, 0]], dtype=COORD_DTYPE)
        trail_squares = np.array([[1, 0, 0], [2, 0, 0]], dtype=COORD_DTYPE)
        cache.add_trail(trailblazer_pos, trail_squares, color=Color.WHITE)
        
        # Check intersection for White piece (Friendly)
        path = np.array([[1, 0, 0]], dtype=COORD_DTYPE)
        assert not cache.check_trail_intersection(path, avoider_color=Color.WHITE)
        assert cache.get_intersecting_squares(path, avoider_color=Color.WHITE).size == 0
        
        # Check intersection for Black piece (Enemy)
        assert cache.check_trail_intersection(path, avoider_color=Color.BLACK)
        assert cache.get_intersecting_squares(path, avoider_color=Color.BLACK).size > 0

    def test_enemy_trails_damage(self):
        """Test that trails damage enemy pieces."""
        cache = TrailblazeCache()
        
        # Create a trail for Black
        trailblazer_pos = np.array([[3, 3, 3]], dtype=COORD_DTYPE)
        trail_squares = np.array([[3, 3, 2], [3, 3, 1]], dtype=COORD_DTYPE)
        cache.add_trail(trailblazer_pos, trail_squares, color=Color.BLACK)
        
        # Check intersection for White piece (Enemy)
        path = np.array([[3, 3, 2]], dtype=COORD_DTYPE)
        assert cache.check_trail_intersection(path, avoider_color=Color.WHITE)
        
        # Check intersection for Black piece (Friendly)
        assert not cache.check_trail_intersection(path, avoider_color=Color.BLACK)

    def test_multiple_trails_mixed_colors(self):
        """Test mixed color trails."""
        cache = TrailblazeCache()
        
        # White Trail
        cache.add_trail(
            np.array([[0,0,0]], dtype=COORD_DTYPE), 
            np.array([[1,0,0]], dtype=COORD_DTYPE), 
            color=Color.WHITE
        )
        
        # Black Trail
        cache.add_trail(
            np.array([[5,5,5]], dtype=COORD_DTYPE), 
            np.array([[1,0,0]], dtype=COORD_DTYPE), # Overlapping square!
            color=Color.BLACK
        )
        
        path = np.array([[1,0,0]], dtype=COORD_DTYPE)
        
        # Debug prints
        print(f"Active trails count: {np.sum(cache._trail_data['active'])}")
        active_indices = np.where(cache._trail_data['active'])[0]
        for idx in active_indices:
            print(f"Entry {idx}: Color={cache._trail_data[idx]['color']}, FlatIdx={cache._trail_data[idx]['flat_idx']}")
            print(f"  Coords: {cache._trail_data[idx]['trail_coords'][0]}")

        # White piece moving: Should hit Black trail (even if White trail is there too)
        print("Checking intersection for WHITE piece (avoider=WHITE)...")
        res_white = cache.check_trail_intersection(path, avoider_color=Color.WHITE)
        print(f"Result: {res_white}")
        assert res_white
        
        print("Checking intersection for BLACK piece (avoider=BLACK)...")
        res_black = cache.check_trail_intersection(path, avoider_color=Color.BLACK)
        print(f"Result: {res_black}")
        assert res_black

if __name__ == "__main__":
    # Manually run tests if executed directly
    t = TestTrailblazerColor()
    try:
        t.test_friendly_fire_off()
        print("test_friendly_fire_off passed")
        t.test_enemy_trails_damage()
        print("test_enemy_trails_damage passed")
        t.test_multiple_trails_mixed_colors()
        print("test_multiple_trails_mixed_colors passed")
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

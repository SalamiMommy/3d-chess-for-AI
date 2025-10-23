# game3d/cache/cache_consistency.py
"""Tools to validate cache consistency across modules."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

def validate_cache_consistency(game_state: 'GameState') -> bool:
    """Validate that all modules are using the same cache manager instance."""
    cache_manager = game_state.cache_manager

    # Check board references
    if game_state.board.cache_manager is not cache_manager:
        print(f"ERROR: Board cache manager mismatch")
        return False

    # Check occupancy cache
    if game_state.cache_manager.occupancy._manager is not cache_manager:
        print(f"ERROR: Occupancy cache manager mismatch")
        return False

    # Check effect caches
    effect_caches = [
        cache_manager.aura_cache,
        cache_manager.trailblaze_cache,
        cache_manager.geomancy_cache,
        cache_manager.attacks_cache
    ]

    for cache in effect_caches:
        cache_manager_attr = getattr(cache, '_cache_manager', None)
        if cache_manager_attr is not None and cache_manager_attr is not cache_manager:
            print(f"ERROR: Effect cache manager mismatch: {type(cache).__name__}")
            return False

    # Check move cache
    if cache_manager._move_cache and hasattr(cache_manager._move_cache, '_manager'):
        if cache_manager._move_cache._manager is not cache_manager:
            print(f"ERROR: Move cache manager mismatch")
            return False

    print("SUCCESS: All cache manager references are consistent")
    return True

def validate_incremental_updates(game_state: 'GameState', move) -> bool:
    """Validate that incremental updates work correctly."""
    original_hash = game_state.zkey
    original_aura = game_state.cache_manager.aura_cache._state.copy() if hasattr(game_state.cache_manager.aura_cache, '_state') else None

    # Make move
    new_state = game_state.make_move(move)

    # Check that caches were updated incrementally
    if new_state.zkey == original_hash:
        print("ERROR: Zobrist hash not updated incrementally")
        return False

    if original_aura is not None:
        new_aura = new_state.cache_manager.aura_cache._state
        if new_aura is original_aura:  # Should be a new/different object
            print("ERROR: Aura cache not updated incrementally")
            return False

    print("SUCCESS: Incremental updates working correctly")
    return True

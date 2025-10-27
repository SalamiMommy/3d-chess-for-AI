# game3d/cache/cache_consistency.py
"""Tools to validate cache consistency across modules."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

def validate_cache_consistency(game_state: 'GameState') -> bool:
    """Enhanced validation that all modules use the same cache manager instance."""
    cache_manager = game_state.cache_manager

    if cache_manager is None:
        print("ERROR: GameState has no cache manager")
        return False

    # Check board references
    if game_state.board.cache_manager is not cache_manager:
        print("ERROR: Board cache manager mismatch")
        return False

    # Check all caches use the same manager instance
    caches_to_check = [
        ("Occupancy", cache_manager.occupancy),
        ("Aura", cache_manager.aura_cache),
        ("Trailblaze", cache_manager.trailblaze_cache),
        ("Geomancy", cache_manager.geomancy_cache),
        ("Attacks", cache_manager.attacks_cache),
    ]

    for cache_name, cache in caches_to_check:
        cache_manager_attr = getattr(cache, '_cache_manager', None)
        if cache_manager_attr is not None and cache_manager_attr is not cache_manager:
            print(f"ERROR: {cache_name} cache manager mismatch")
            return False

    # Check move cache
    if cache_manager._move_cache:
        move_cache_manager = getattr(cache_manager._move_cache, '_cache_manager', None)
        if move_cache_manager is not None and move_cache_manager is not cache_manager:
            print("ERROR: Move cache manager mismatch")
            return False

    # Verify Zobrist hash consistency
    current_hash = cache_manager._current_zobrist_hash
    computed_hash = cache_manager._zobrist.compute_from_scratch(game_state.board, game_state.color)

    if current_hash != computed_hash:
        print(f"ERROR: Zobrist hash desync - current: {current_hash}, computed: {computed_hash}")
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

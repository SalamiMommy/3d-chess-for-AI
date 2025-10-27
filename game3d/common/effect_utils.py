# effect_utils.py
def apply_standard_effects(
    cache_manager: "OptimizedCacheManager",
    move: "Move",
    mover: Color,
    current_ply: int,
    board: "Board"
) -> Set[str]:
    """Apply standard effects in consistent order."""
    affected_caches = set()

    # Determine affected caches
    from_piece = cache_manager.occupancy_cache.get(move.from_coord)
    if from_piece:
        effect_type = get_piece_effect_type(from_piece.ptype)
        if effect_type:
            affected_caches.add(effect_type)

    # Apply effects in consistent order using pre-defined tuple
    effect_order = ('aura', 'trailblaze', 'geomancy', 'attacks')
    for effect_name in effect_order:
        if effect_name in affected_caches:
            cache = cache_manager._get_cache_by_name(effect_name)
            if cache is not None and hasattr(cache, 'apply_move'):
                cache.apply_move(move, mover, current_ply, board)

    return affected_caches

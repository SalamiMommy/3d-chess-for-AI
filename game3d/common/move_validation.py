# game3d/common/move_validation.py
def validate_move_basic(game_state, move, expected_color=None):
    """Basic move validation used in turnmove.py and moveeffects.py."""
    cache = get_cache_manager(game_state)

    # Check piece exists
    from_piece = cache.occupancy.get(move.from_coord)
    if not from_piece:
        return False

    # Check color
    if expected_color and from_piece.color != expected_color:
        return False

    # Check frozen status
    if cache.is_frozen(move.from_coord, from_piece.color):
        return False

    return True

def validate_move_destination(game_state, move, piece_color):
    """Destination validation."""
    cache = get_cache_manager(game_state)
    to_piece = cache.occupancy.get(move.to_coord)
    return not (to_piece and to_piece.color == piece_color)

def validate_move_comprehensive(
    cache_manager: "OptimizedCacheManager",
    move: "Move",
    expected_color: Color,
    check_effects: bool = True
) -> bool:
    """Comprehensive move validation with all effect checks."""
    # Basic validation
    if not validate_move_basic(cache_manager, move, expected_color):
        return False

    # Destination validation
    if not validate_move_destination(cache_manager, move, expected_color):
        return False

    # Effect-based validations
    if check_effects:
        if cache_manager.is_frozen(move.from_coord, expected_color):
            return False

        if cache_manager.is_movement_debuffed(move.from_coord, expected_color):
            # Handle debuff logic
            pass

        if cache_manager.is_geomancy_blocked(move.to_coord, cache_manager.halfmove_clock):
            return False

    return True

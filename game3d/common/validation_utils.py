# game3d/common/validation_utils.py
def validate_game_state(state):
    """Comprehensive state validation."""
    issues = []

    # Board-cache consistency
    if not validate_cache_integrity(state):
        issues.append("Cache-board desync")

    # Piece count sanity
    white_pieces = list(state.cache_manager.get_pieces_of_color(Color.WHITE))
    black_pieces = list(state.cache_manager.get_pieces_of_color(Color.BLACK))

    if len(white_pieces) == 0 or len(black_pieces) == 0:
        issues.append("No pieces for one color")

    return issues

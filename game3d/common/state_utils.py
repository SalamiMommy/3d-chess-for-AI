# game3d/common/state_utils.py
def create_new_state(original_state, new_board, new_color, move=None,
                    increment_turn=True, reuse_cache=True):
    """Unified state creation logic."""
    if reuse_cache:
        cache_manager = original_state.cache_manager
        cache_manager.board = new_board
    else:
        from game3d.cache.manager import get_cache_manager
        cache_manager = get_cache_manager(new_board, new_color)

    new_history = original_state.history
    if move:
        new_history = new_history + (move,)

    turn_number = original_state.turn_number
    halfmove_clock = original_state.halfmove_clock

    if increment_turn:
        turn_number += 1
        halfmove_clock += 1

    from game3d.game.gamestate import GameState
    return GameState(
        board=new_board,
        color=new_color,
        cache_manager=cache_manager,
        history=new_history,
        halfmove_clock=halfmove_clock,
        game_mode=original_state.game_mode,
        turn_number=turn_number,
    )

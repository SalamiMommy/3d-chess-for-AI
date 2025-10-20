from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import torch

from game3d.board.board import Board
from game3d.common.enums import Color
from game3d.cache.manager import get_cache_manager  # already imported

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

from .gamestate import GameState, GameMode


# ---------- public helpers --------------------------------------------------

def start_game_state(cache: 'OptimizedCacheManager' | None = None) -> GameState:
    if cache is None:
        raise RuntimeError("start_game_state() requires an external OptimizedCacheManager")
    return GameState(
        board=cache.board,
        color=Color.WHITE,
        cache=cache,
        history=(),
        halfmove_clock=0,
        game_mode=GameMode.STANDARD,
        turn_number=1,
    )


def create_game_state_from_tensor(
    tensor: torch.Tensor,
    color: Color,
    cache: Optional['OptimizedCacheManager'] = None,
) -> GameState:
    board = Board(tensor)
    if cache is None:
        cache = get_cache_manager(board, color)
    else:
        cache.board = board
        cache._current = color
        cache.refresh_all()          # full incremental rebuild
    return GameState(
        board=board,
        color=color,
        cache=cache,
        history=(),
        halfmove_clock=0,
        game_mode=GameMode.STANDARD,
        turn_number=1,
    )


def clone_game_state_for_search(original: GameState) -> GameState:
    return original.clone_with_new_cache()


def new_board_with_manager(color: Color = Color.WHITE) -> Board:
    """
    Create a start-position board and attach a **fresh** cache manager
    that is guaranteed to be rebuilt *after* the pieces are on the board.
    """
    # 1. Build the position first – no cache attached yet
    board = Board.empty()
    board.init_startpos()            # fills the tensor

    # 2. Destroy any half-initialised manager that might already be lurking
    board.cache_manager = None

    # 3. Now create the real manager – rebuild() will see the full tensor
    cm = get_cache_manager(board, color)
    return board

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import torch

from game3d.board.board import Board
from game3d.pieces.enums import Color

if TYPE_CHECKING:                         # ← type-only
    from game3d.cache.manager import OptimizedCacheManager

# DELETE this line
# from game3d.cache.manager import OptimizedCacheManager, get_cache_manager

from .gamestate import GameState, GameMode

def start_game_state(cache: Optional[OptimizedCacheManager] = None) -> GameState:
    """Create starting position with optimized cache."""
    from game3d.cache.manager import get_cache_manager   # ← local import

    board = Board.empty()
    board.init_startpos()
    cache = cache or get_cache_manager(board, Color.WHITE)
    return GameState(
        board=board,
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
    cache: Optional[OptimizedCacheManager] = None,
) -> GameState:
    from game3d.cache.manager import get_cache_manager   # ← local import

    board = Board(tensor)
    cache = cache or get_cache_manager(board, color)
    return GameState(
        board=board,
        color=color,
        cache=cache,
        history=(),
        halfmove_clock=0,
        game_mode=GameMode.STANDARD,
        turn_number=1,
    )

def clone_game_state_for_search(state: GameState) -> GameState:
    """Create a clone optimized for search operations."""
    return state.clone_with_new_cache()

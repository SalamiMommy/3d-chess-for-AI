from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import torch

from game3d.board.board import Board
from game3d.common.enums import Color
from game3d.cache.manager import get_cache_manager  # Added missing import

from .gamestate import GameState, GameMode

def start_game_state(cache: 'OptimizedCacheManager' | None = None) -> GameState:
    """
    Build the initial position.
    The caller *must* supply an external cache; we only wire it in.
    """
    if cache is None:
        raise RuntimeError("start_game_state() requires an external OptimizedCacheManager")
    # Board is already owned by the cache – do **not** create another one.
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
    """
    Build a state out of an arbitrary tensor.
    If the caller already has a cache, reuse it; otherwise create *one*
    and return it so that the caller can keep it for later.
    """

    board = Board(tensor)
    if cache is None:                       # caller did not have one
        cache = get_cache_manager(board, color)
    else:                                   # caller gave us one – adopt the board
        cache.board = board
        cache._current = color
        cache.refresh_all()                 # incremental update after tensor swap
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
    """Create a deep clone for search algorithms."""
    return original.clone_with_new_cache()

def new_board_with_manager(color: Color = Color.WHITE) -> Board:
    """Create a start-position board and attach a cache-manager."""
    board = Board.startpos()                     # low-level, no manager
    _ = get_cache_manager(board, color)          # attaches itself to board
    return board                                 # ready to use

# factory.py - FIXED
from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import torch

from game3d.board.board import Board
from game3d.common.enums import Color
from game3d.cache.manager import get_cache_manager

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

from .gamestate import GameState, GameMode


def start_game_state(cache_manager: 'OptimizedCacheManager' | None = None) -> GameState:
    """
    Create a new game state from start position.
    """
    if cache_manager is None:
        raise RuntimeError("start_game_state() requires an external OptimizedCacheManager")
    return GameState(
        board=cache_manager.board,
        color=Color.WHITE,
        cache_manager=cache_manager,
        history=(),
        halfmove_clock=0,
        game_mode=GameMode.STANDARD,
        turn_number=1,
    )


def create_game_state_from_tensor(
    tensor: torch.Tensor,
    color: Color,
    cache_manager: Optional['OptimizedCacheManager'] = None,
) -> GameState:
    """
    Create game state from tensor representation.
    Reuse existing cache manager if provided.
    """
    board = Board(tensor)
    if cache_manager is None:
        cache_manager = get_cache_manager(board, color)
    else:
        # Reuse existing cache manager with updated board
        cache_manager.board = board
        cache_manager._current = color
        cache_manager.rebuild(board, color)  # Incremental rebuild
    return GameState(
        board=board,
        color=color,
        cache_manager=cache_manager,
        history=(),
        halfmove_clock=0,
        game_mode=GameMode.STANDARD,
        turn_number=1,
    )


def clone_game_state_for_search(original: GameState) -> GameState:
    """Clone game state for search - only create new cache when needed for threading."""
    return original.clone(deep_cache=True)  # Use deep cache for thread safety


def new_board_with_manager(color: Color = Color.WHITE) -> Board:
    """
    Create a start-position board and attach a fresh cache manager.
    """
    board = Board.empty()
    board.init_startpos()
    board.cache_manager = None

    # Create the manager (Zobrist computed once in get_cache_manager)
    cm = get_cache_manager(board, color)
    return board

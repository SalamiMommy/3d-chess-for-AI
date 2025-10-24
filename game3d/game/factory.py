# game3d/game/factory.py - FIXED
from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import torch

from game3d.board.board import Board
from game3d.common.enums import Color

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from .gamestate import GameState

from .gamestate import GameState, GameMode

def start_game_state(cache_manager: 'OptimizedCacheManager') -> GameState:
    """
    Create a new game state from start position - REQUIRES external cache manager.
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

    if cache_manager is not None:
        # Reuse existing cache manager with incremental update
        cache_manager.rebuild(board, color)
        new_cache_manager = cache_manager
    else:
        # Only create new cache manager when absolutely necessary
        from game3d.cache.manager import get_cache_manager
        new_cache_manager = get_cache_manager(board, color)

    return GameState(
        board=board,
        color=color,
        cache_manager=new_cache_manager,
        history=(),
        halfmove_clock=0,
        game_mode=GameMode.STANDARD,
        turn_number=1,
    )

def clone_game_state_for_search(original: GameState) -> GameState:
    """Clone game state for search - reuse cache manager when possible."""
    return original.clone(deep_cache=False)  # Prefer shallow clone

def new_board_with_manager(color: Color = Color.WHITE) -> tuple[Board, 'OptimizedCacheManager']:
    """
    Create a start-position board and cache manager together.
    """
    from game3d.cache.manager import get_cache_manager

    board = Board.empty()
    board.init_startpos()

    # Create manager once
    cm = get_cache_manager(board, color)
    return board, cm

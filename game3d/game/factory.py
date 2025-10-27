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

def start_game_state(cache_manager: Optional['OptimizedCacheManager'] = None) -> GameState:
    """Create a new game state from start position."""
    from game3d.cache.manager import get_cache_manager

    board = Board.empty()
    board.init_startpos()

    if cache_manager is not None:
        # Reuse and FULLY reset provided manager
        cache_manager.rebuild(board, Color.WHITE)
        board.cache_manager = cache_manager
    else:
        # Create new manager
        cache_manager = get_cache_manager(board, Color.WHITE)

    return GameState(
        board=board,
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
    Create game state from tensor - reuse cache manager if provided.
    """
    from game3d.cache.manager import get_cache_manager

    board = Board(tensor)

    if cache_manager is not None:
        # Reuse and update existing manager
        cache_manager.rebuild(board, color)
        board.cache_manager = cache_manager
    else:
        # Get singleton manager for this board
        cache_manager = get_cache_manager(board, color)

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
    """Clone game state for search - ALWAYS reuse cache manager."""
    return original.clone(deep_cache=False)

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

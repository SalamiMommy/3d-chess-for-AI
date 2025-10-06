# factory.py - Fixed version
from __future__ import annotations
from typing import Optional

import torch

from game3d.board.board import Board
from game3d.pieces.enums import Color
from game3d.cache.manager import OptimizedCacheManager, get_cache_manager

from .gamestate import GameState, GameMode  # Add GameMode import

def start_game_state(cache: Optional[OptimizedCacheManager] = None) -> GameState:
    """Create starting position with optimized cache."""
    board = Board.empty()
    board.init_startpos()
    cache = cache or get_cache_manager(board, Color.WHITE)
    return GameState(
        board=board,
        color=Color.WHITE,
        cache=cache,
        history=(),
        halfmove_clock=0,
        game_mode=GameMode.STANDARD,  # Add this
        turn_number=1,  # Add this
    )

def create_game_state_from_tensor(tensor: torch.Tensor, color: Color,
                                  cache: Optional[OptimizedCacheManager] = None) -> GameState:
    """Create GameState from tensor representation."""
    board = Board(tensor)
    cache = cache or get_cache_manager(board, color)
    return GameState(
        board=board,
        color=color,
        cache=cache,
        history=(),
        halfmove_clock=0,
        game_mode=GameMode.STANDARD,  # Add this
        turn_number=1,  # Add this
    )

def clone_game_state_for_search(state: GameState) -> GameState:
    """Create a clone optimized for search operations."""
    return state.clone_with_new_cache()

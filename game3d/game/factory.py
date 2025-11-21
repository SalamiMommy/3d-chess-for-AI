# game3d/game/factory.py
"""Factory functions for 9x9x9 chess engine - refactored to use common modules."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Union
import numpy as np

from game3d.common.shared_types import (
    FLOAT_DTYPE, BOARD_SHAPE_4D, Color, MOVE_DTYPE
)
from game3d.common.validation import validate_array
from game3d.common.state_utils import create_new_state
from game3d.cache.manager import get_cache_manager, OptimizedCacheManager

if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.game.gamestate import GameState
else:
    # Runtime import
    from game3d.game.gamestate import GameState

def start_game_state(
    cache_manager: Optional[OptimizedCacheManager] = None,
    ensure_start_pos: bool = True
) -> GameState:
    """Create game state from start position - main factory entry point."""
    from game3d.board.board import Board

    board = Board.empty()
    if ensure_start_pos:
        board.init_startpos()

    # Use centralized cache manager
    cm = cache_manager or get_cache_manager(board, Color.WHITE)
    board._cache_manager = cm

    # ✅ CRITICAL FIX: Rebuild occupancy cache from board data
    cm._rebuild_occupancy_from_board()

    return GameState(
        board=board,
        color=Color.WHITE,
        cache_manager=cm,
        history=np.empty(0, dtype=MOVE_DTYPE),
        halfmove_clock=0,
        turn_number=1,
    )


def create_game_state_from_tensor(
    tensor: np.ndarray,
    color: Union[Color, int],
    cache_manager: Optional[OptimizedCacheManager] = None,
    validate_input: bool = False
) -> GameState:
    """Create game state from tensor using centralized utilities."""
    from game3d.board import Board
    from game3d.game.gamestate import GameState

    # Validate using common validation module
    if validate_input:
        tensor = validate_array(
            tensor, name="tensor", dtype=FLOAT_DTYPE,
            shape=BOARD_SHAPE_4D, strict_bounds=False
        )

    # Normalize color
    if isinstance(color, int):
        color = Color(color) if 0 <= color <= 2 else Color.EMPTY

    board = Board(tensor)
    cm = cache_manager or get_cache_manager(board, color)
    board._cache_manager = cm

    return GameState(
        board=board,
        color=color,
        cache_manager=cm,
        history=np.empty(0, dtype=MOVE_DTYPE),
        halfmove_clock=0,
        turn_number=1,
    )


def clone_game_state_for_search(
    original: GameState,
    deep_cache: bool = False,
    optimize_memory: bool = True
) -> GameState:
    """Clone game state for search using state_utils."""
    cloned_state = original.clone(deep_cache=deep_cache)

    # Optimize arrays using validation module
    if optimize_memory and hasattr(cloned_state.board, '_array'):
        cloned_state.board._array = validate_array(
            cloned_state.board._array, name="board_array",
            dtype=FLOAT_DTYPE, contiguous=True
        )

    return cloned_state


def create_tensor_batch(states: list, dtype: np.dtype = FLOAT_DTYPE) -> np.ndarray:
    """Create batch tensor from game states using state_utils."""
    if not states:
        raise ValueError("Empty state list provided")

    # Validate batch using common validation
    from game3d.common.validation import validate_not_none

    validate_not_none(states, "states")

    # Extract arrays
    arrays = [state.board.array() for state in states]

    # Stack using numpy
    batch = np.stack(arrays, axis=0).astype(dtype, copy=False)

    return validate_array(
        batch, name="batch_tensor", dtype=dtype,
        ndim=5, shape=(len(states),) + arrays[0].shape
    )


def validate_batch_states(states: list, min_size: int = 1) -> list:
    """Validate batch states using centralized validation."""
    from game3d.common.validation import validate_not_none

    validate_not_none(states, "states")

    if not isinstance(states, list):
        raise TypeError(f"Expected list, got {type(states)}")

    if len(states) < min_size:
        raise ValueError(f"Need {min_size} states, got {len(states)}")

    # Validate each state has required methods
    for i, state in enumerate(states):
        try:
            validate_not_none(state, f"state[{i}]")
            if not (hasattr(state, 'to_array') or
                    (hasattr(state, 'board') and hasattr(state.board, 'array'))):
                raise AttributeError(f"State {i} missing array methods")
        except Exception as e:
            raise ValueError(f"Invalid state at index {i}: {e}")

    return states


def optimize_game_state_arrays(state: GameState, force_contiguous: bool = True) -> GameState:
    """Optimize game state arrays using validation module."""
    if hasattr(state.board, '_array'):
        state.board._array = validate_array(
            state.board._array, name="board_array", dtype=FLOAT_DTYPE,
            contiguous=force_contiguous
        )
    return state


def new_board_with_manager(
    color: Union[Color, int] = Color.WHITE,
    return_start_tensor: bool = False,
    cache_manager: Optional[OptimizedCacheManager] = None
) -> Union[tuple[Board, np.ndarray], Board]:
    """Create board with cache manager using centralized utilities."""
    from game3d.board import Board

    # Normalize color
    if isinstance(color, int):
        color = Color(color) if 0 <= color <= 2 else Color.WHITE

    board = Board.empty()
    board.init_startpos()

    # Setup cache manager
    if cache_manager is None:
        cm = get_cache_manager(board, color)
    else:
        cm = cache_manager

    board._cache_manager = cm

    # Get start tensor from cache manager utilities
    if return_start_tensor:
        from game3d.cache.manager import get_start_tensor
        return board, get_start_tensor(cm)

    return board


# Cache-related utilities moved to cache.manager module
def clear_start_tensor_cache(cache_manager: Optional[OptimizedCacheManager] = None) -> None:
    """Clear start tensor cache - delegated to cache manager."""
    if cache_manager is not None:
        cache_manager.clear_start_tensor_cache()


def get_start_tensor_cache_size(cache_manager: Optional[OptimizedCacheManager] = None) -> int:
    """Get cached tensor size - delegated to cache manager."""
    if cache_manager is not None:
        return cache_manager.get_start_tensor_cache_size()
    return 0


def optimize_cache_performance(cache_manager: OptimizedCacheManager) -> bool:
    """Optimize cache performance - delegated to cache manager."""
    return cache_manager.optimize_performance() if cache_manager else False


# Simplified start position tensor creation
def create_start_tensor(cache_manager: Optional[OptimizedCacheManager] = None) -> np.ndarray:
    """Create start position tensor using board utilities."""
    from game3d.board import Board

    board = Board.empty()
    board.init_startpos()

    # Use board's native array export
    tensor = board.get_board_array()

    # Cache in manager if available
    if cache_manager is not None:
        cache_manager._start_tensor_cache = tensor

    return tensor


__all__ = [
    # Core factory functions
    'start_game_state',
    'create_game_state_from_tensor',
    'clone_game_state_for_search',
    'new_board_with_manager',

    # Batch utilities
    'create_tensor_batch',
    'validate_batch_states',

    # Optimization utilities
    'optimize_game_state_arrays',
    'optimize_cache_performance',

    # Cache utilities (delegated)
    'clear_start_tensor_cache',
    'get_start_tensor_cache_size',
    'create_start_tensor',
]

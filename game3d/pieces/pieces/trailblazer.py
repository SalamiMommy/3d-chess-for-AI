# trailblazer.py - FULLY NUMPY-NATIVE
"""
Trailblazer piece implementation.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Union
import numpy as np

from game3d.common.shared_types import (
    COLOR_DTYPE, PIECE_TYPE_DTYPE, COORD_DTYPE,
    PieceType, Color, Result, TRAILBLAZER
)
from game3d.common.registry import register
from game3d.movement.slider_engine import get_slider_movement_generator
from game3d.movement.movepiece import Move
from game3d.movement.movementmodifiers import get_range_modifier

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Trailblazer movement vectors - rook-like orthogonal directions
ROOK_DIRECTIONS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=COORD_DTYPE)

# Maximum movement distance
MAX_TRAILBLAZER_DISTANCE = 3

def generate_trailblazer_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray,
    max_distance: Union[int, np.ndarray] = MAX_TRAILBLAZER_DISTANCE,
    ignore_occupancy: bool = False
) -> np.ndarray:
    """Generate trailblazer moves from numpy-native position array."""
    pos_arr = pos.astype(COORD_DTYPE)
    
    # Validate position
    if pos_arr.ndim == 1:
        # Lazy import to avoid circular dependency
        from game3d.common.coord_utils import in_bounds_vectorized
        if not in_bounds_vectorized(pos_arr.reshape(1, 3))[0]:
            return np.empty((0, 6), dtype=COORD_DTYPE)

    slider_engine = get_slider_movement_generator()
    moves = slider_engine.generate_slider_moves_array(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=ROOK_DIRECTIONS,
        max_distance=max_distance,
        ignore_occupancy=ignore_occupancy
    )

    # Note: Path metadata removed as it was already being lost when
    # generator.py converted Move objects to arrays. The trailblaze effect
    # should be calculated when moves are executed, not during generation.
    return moves

class TrailblazeRecorder:
    """Records trail history."""
    __slots__ = ("_history",)

    def __init__(self) -> None:
        self._history: list[np.ndarray] = []

    def add_trail(self, squares: np.ndarray) -> None:
        """Add trail squares."""
        squares_arr = np.asarray(squares, dtype=COORD_DTYPE)
        if squares_arr.size == 0:
            return
        self._history.append(squares_arr)
        if len(self._history) > 3:
            self._history.pop(0)

    def current_trail_numpy(self) -> np.ndarray:
        """Return union of last 3 trails."""
        if not self._history:
            return np.empty((0, 3), dtype=COORD_DTYPE)
        return np.unique(np.vstack(self._history), axis=0)

def apply_trailblaze_step_numpy(
    enemy_sq: np.ndarray,
    enemy_color: int,
    cache_manager: 'OptimizedCacheManager',
    board,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply trailblaze effect to enemy piece."""
    enemy_sq_arr = np.asarray(enemy_sq, dtype=COORD_DTYPE).reshape(3)

    trail = cache_manager.trailblaze_cache.current_trail_squares(
        Color(enemy_color).opposite(), board
    )

    if isinstance(trail, set):
        trail_array = (
            np.empty((0, 3), dtype=COORD_DTYPE) if not trail
            else np.array([*trail], dtype=COORD_DTYPE)
        )
    elif isinstance(trail, np.ndarray):
        trail_array = trail if trail.ndim == 2 else trail.reshape(-1, 3)
    else:
        trail_array = np.empty((0, 3), dtype=COORD_DTYPE)

    if trail_array.size == 0:
        return (
            np.empty((0, 3), dtype=COORD_DTYPE),
            np.empty(0, dtype=object)
        )

    if not np.any(np.all(trail_array == enemy_sq_arr, axis=1)):
        return (
            np.empty((0, 3), dtype=COORD_DTYPE),
            np.empty(0, dtype=object)
        )

    if cache_manager.trailblaze_cache.increment_counter(enemy_sq_arr, enemy_color, board):
        victim = cache_manager.occupancy_cache.get(enemy_sq_arr.reshape(1, 3))
        if victim is not None:
            piece_type = victim.get("piece_type")
            victim_color = victim.get("color")
            if piece_type != PieceType.KING or not _any_priest_alive(board, victim_color):
                cache_manager.occupancy.set_piece(enemy_sq_arr, None)
                return enemy_sq_arr.reshape(1, 3), np.array([victim], dtype=object)

    return (
        np.empty((0, 3), dtype=COORD_DTYPE),
        np.empty(0, dtype=object)
    )

def _any_priest_alive(board, color: Color) -> bool:
    """Check if any priest of the given color is alive."""
    # Implementation depends on board structure
    # This is a placeholder - actual implementation should check board state
    return False

@register(PieceType.TRAILBLAZER)
def trailblazer_move_dispatcher(state: 'GameState', pos: np.ndarray, ignore_occupancy: bool = False) -> np.ndarray:
    """Registered dispatcher for Trailblazer moves (max 3 steps unbuffed, 4 buffed)."""
    # Check if buffed
    pos_arr = pos.astype(COORD_DTYPE)
    if pos_arr.ndim == 1:
        pos_arr = pos_arr.reshape(1, 3)
    
    x, y, z = pos_arr[0]
    buffed_squares = state.cache_manager.consolidated_aura_cache._buffed_squares
    is_buffed = buffed_squares[x, y, z]
    
    # Use max distance 4 if buffed, 3 otherwise
    max_dist = 4 if is_buffed else MAX_TRAILBLAZER_DISTANCE
    
    return generate_trailblazer_moves(state.cache_manager, state.color, pos, max_dist, ignore_occupancy)

__all__ = [
    "generate_trailblazer_moves",
    "TrailblazeRecorder",
    "apply_trailblaze_step_numpy",
    "ROOK_DIRECTIONS",
    "MAX_TRAILBLAZER_DISTANCE",
    "trailblazer_move_dispatcher"
]

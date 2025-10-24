# game3d/movement/trailblazer.py - FIXED
"""
Unified Trailblazer – 3-step rook slider + trailblaze side-effects.
"""

from __future__ import annotations

from typing import List, Tuple, Set, Dict, Optional, TYPE_CHECKING
from collections import deque
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.slidermovement import generate_moves
from game3d.movement.movepiece import Move, MOVE_FLAGS, convert_legacy_move_args
from game3d.common.coord_utils import reconstruct_path
from game3d.attacks.check import _any_priest_alive

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ----------------------------------------------------------
# 6 orthogonal directions (±X, ±Y, ±Z)
# ----------------------------------------------------------
_ROOK_DIRS = np.array([
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
], dtype=np.int8)

# ----------------------------------------------------------
# 1. Movement generator – FIXED parameter name
# ----------------------------------------------------------
def generate_trailblazer_moves(
    cache_manager,  # FIXED: Changed from 'cache' to 'cache_manager' for consistency
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Orthogonal slider, max 3 squares, with trailblaze metadata."""
    moves = generate_moves(
        piece_type='rook',
        pos=(x, y, z),
        color=color,
        max_distance=3,
        directions=_ROOK_DIRS,
        cache_manager=cache_manager,  # FIXED: Use parameter name
    )

    # Annotate every move with the **slid path** (intermediates only)
    for mv in moves:
        mv.metadata["slid_path"] = list(reconstruct_path(
            mv.from_coord, mv.to_coord,
            include_start=False, include_end=False
        ))

    return moves

# ----------------------------------------------------------
# 2. Trailblaze recorder (unchanged)
# ----------------------------------------------------------
class TrailblazeRecorder:
    __slots__ = ("_history",)

    def __init__(self) -> None:
        self._history: deque[Set[Tuple[int, int, int]]] = deque(maxlen=3)

    def add_trail(self, squares: Set[Tuple[int, int, int]]) -> None:
        self._history.append(squares)

    def current_trail(self) -> Set[Tuple[int, int, int]]:
        """Union of last 3 trails."""
        out: Set[Tuple[int, int, int]] = set()
        for s in self._history:
            out.update(s)
        return out

# ----------------------------------------------------------
# 3. Side-effect helpers (unchanged)
# ----------------------------------------------------------
def apply_trailblaze_step(
    enemy_sq: Tuple[int, int, int],
    enemy_color: Color,
    cache_manager,  # FIXED: Consistent parameter name
    board,
) -> List[Tuple[Tuple[int, int, int], "Piece"]]:
    """
    Enemy *enemy_sq* just moved; if it lands on a trail-marked square,
    increment its counter and remove if counter == 3 (king spared if priests).
    """
    removed: List[Tuple[Tuple[int, int, int], "Piece"]] = []

    # 1. Get current trail squares
    trail = cache_manager.trailblaze_cache.current_trail_squares(enemy_color.opposite(), board)
    if enemy_sq not in trail:
        return removed

    # 2. Increment counter
    if cache_manager.trailblaze_cache.increment_counter(enemy_sq, enemy_color, board):
        # 3. Remove if limit reached (king spared if priests alive)
        victim = cache_manager.occupancy.get(enemy_sq)
        if victim is not None:
            if victim.ptype is PieceType.KING and _any_priest_alive(board, victim.color):
                return removed  # king spared
            removed.append((enemy_sq, victim))
            cache_manager.occupancy.set_piece(enemy_sq, None)

    return removed

# ----------------------------------------------------------
# 4. Dispatcher – FIXED: Use cache_manager property
# ----------------------------------------------------------
@register(PieceType.TRAILBLAZER)
def trailblazer_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_trailblazer_moves(state.cache_manager, state.color, x, y, z)

__all__ = ["generate_trailblazer_moves", "TrailblazeRecorder", "apply_trailblaze_step"]

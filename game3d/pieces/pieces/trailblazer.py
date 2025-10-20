# game3d/movement/trailblazer.py
"""
Unified Trailblazer – 3-step rook slider + trailblaze side-effects.

Movement:
  - orthogonal slides up to 3 squares (like Rook)

Side-effect:
  - every square the piece **slides through** is **recorded** (last 3 moves)
  - when an **enemy piece moves into** any recorded square (Trailblazer owner's recorded) → **+1 trail counter**
  - counter reaches 3 → piece is **removed** (king spared if priests alive)
"""

from __future__ import annotations

from typing import List, Tuple, Set, Dict, Optional, TYPE_CHECKING
from collections import deque
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.slidermovement import generate_moves
from game3d.movement.movepiece import Move, MOVE_FLAGS, convert_legacy_move_args
from game3d.common.common import reconstruct_path
from game3d.attacks.check import _any_priest_alive  # king spared if priests

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
# 1.  Movement generator – 3-step rook + trail metadata
# ----------------------------------------------------------
def generate_trailblazer_moves(
    cache,          # OptimizedCacheManager
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Orthogonal slider, max 3 squares, with trailblaze metadata."""
    moves = generate_moves(
        piece_type='rook',
        pos=(x, y, z),
        color=color.value,
        max_distance=3,
        directions=_ROOK_DIRS,
        occupancy=cache.occupancy._occ,
    )

    # Annotate every move with the **slid path** (intermediates only)
    for mv in moves:
        mv.metadata["slid_path"] = list(reconstruct_path(
            mv.from_coord, mv.to_coord,
            include_start=False, include_end=False
        ))

    return moves

# ----------------------------------------------------------
# 2.  Trailblaze recorder – per-trailblazer FIFO (last 3 slid sets)
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
# 3.  Side-effect helpers – called by cache manager when enemy moves
# ----------------------------------------------------------
def apply_trailblaze_step(
    enemy_sq: Tuple[int, int, int],
    enemy_color: Color,
    cache,
    board,
) -> List[Tuple[Tuple[int, int, int], "Piece"]]:
    """
    Enemy *enemy_sq* just moved; if it lands on a trail-marked square,
    increment its counter and remove if counter == 3 (king spared if priests).
    Returns list of (sq, piece) that were actually removed.
    """
    removed: List[Tuple[Tuple[int, int, int], "Piece"]] = []

    # 1.  Get current trail squares (union of last 3 Trailblazer slides)
    # FIXED: current_trail_squares takes controller (owner), but for enemy check, use owner's opposite
    trail = cache.trailblaze_cache.current_trail_squares(enemy_color.opposite(), board)
    if enemy_sq not in trail:
        return removed

    # 2.  Increment counter
    if cache.trailblaze_cache.increment_counter(enemy_sq, enemy_color, board):
        # 3.  Remove if limit reached (king spared if priests alive)
        victim = cache.occupancy.get(enemy_sq)
        if victim is not None:
            if victim.ptype is PieceType.KING and _any_priest_alive(board, victim.color):
                return removed  # king spared
            removed.append((enemy_sq, victim))
            cache.occupancy.set_piece(enemy_sq, None)

    return removed

# ----------------------------------------------------------
# 4.  Dispatcher – ONLY Trailblazer
# ----------------------------------------------------------
@register(PieceType.TRAILBLAZER)
def trailblazer_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_trailblazer_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_trailblazer_moves", "TrailblazeRecorder", "apply_trailblaze_step"]

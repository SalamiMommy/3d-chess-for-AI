# Updated bomb.py (full file content with fixes: changed 'DETONATE' to 'SELF_DETONATE')

# game3d/movement/pieces/bomb.py
"""
Unified Bomb generator – king steps + self-detonation.
Bomb pieces:
  - move exactly like a King (one step in any of the 26 directions);
  - may **effect-move** to their own square to detonate immediately;
  - automatically detonate when **captured** (handled elsewhere);
  - kill every enemy piece inside a 2-radius sphere **except** kings
    whose side still has at least one priest alive.
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from game3d.common.enums import Color, PieceType
from game3d.common.common import (
    get_aura_squares,      # 2-radius aura from common
    in_bounds,             # fast bounds check
    subtract_coords,
    add_coords,
)
from game3d.movement.movepiece import Move, MOVE_FLAGS, convert_legacy_move_args
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.attacks.check import _any_priest_alive

if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.common.enums import Color

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ------------------------------------------------------------------
# Public generator – used by dispatcher and AI
# ------------------------------------------------------------------
def generate_bomb_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """All legal Bomb moves from (x,y,z) including self-detonation."""
    pos = (x, y, z)

    # 1.  Re-use the king engine for normal 1-step moves
    moves = generate_king_moves(state.cache, state.color, x, y, z)

    # 2.  Add the self-detonation pseudo-move
    self_det = convert_legacy_move_args(
        pos, pos,
        flags=MOVE_FLAGS['SELF_DETONATE']
    )
    self_det.metadata['detonate'] = True

    # 3.  Prune pointless detonations
    if detonate(state.board, pos, state.color):   # ← use the real function
        moves.append(self_det)

    return moves

# ------------------------------------------------------------------
# Helper – compute enemy pieces that would die in a 2-sphere detonation
# ------------------------------------------------------------------
def detonate(board: "Board", center: Tuple[int, int, int], current_color: "Color") -> List[Tuple[int, int, int]]:
    """
    Return every square that would be cleared by a bomb exploding at *center*.
    Kings are spared if their side still has a priest alive.
    """
    from game3d.attacks.check import _any_priest_alive   # keep the priest rule

    victims: List[Tuple[int, int, int]] = []

    # 1. 2-radius aura is already bounds-checked inside get_aura_squares
    for sq in get_aura_squares(center, radius=2):
        victim = board.cache_manager.occupancy.get(sq)
        if victim is None or victim.color == current_color:
            continue
        if victim.ptype is PieceType.KING and _any_priest_alive(board, victim.color):
            continue
        victims.append(sq)

    return victims

# ------------------------------------------------------------------
# Dispatcher registration
# ------------------------------------------------------------------
@register(PieceType.BOMB)
def bomb_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_bomb_moves(state, x, y, z)

# ------------------------------------------------------------------
# Backward compatibility exports
# ------------------------------------------------------------------
__all__ = ["generate_bomb_moves", "detonate"]

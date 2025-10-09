# game3d/movement/movetypes/bomb.py
"""
Unified Bomb movement generator + self-detonation logic.

Bomb pieces:
  - move exactly like a King (one step in any of the 26 directions);
  - may **effect-move** to their own square to detonate immediately;
  - automatically detonate when **captured** (handled in turnmove.py);
  - kill every enemy piece inside a 2-radius sphere **except** kings
    whose side still has at least one priest alive.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move, MOVE_FLAGS, convert_legacy_move_args
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_jump_generator
from game3d.attacks.check import _any_priest_alive  # reused for king-protection rule

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ------------------------------------------------------------------
# 26 one-step directions (same as King)
# ------------------------------------------------------------------
KING_DIRECTIONS_3D = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# ------------------------------------------------------------------
# Public generator – used by dispatcher and AI
# ------------------------------------------------------------------
def generate_bomb_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """All legal Bomb moves from (x,y,z) including self-detonation."""
    pos = (x, y, z)
    engine = get_jump_generator()

    # 1.  Normal king-step moves (off-board, friendly, walls already filtered)
    raw_moves = engine.generate_moves(
        piece_type='bomb',
        pos=pos,
        board_occupancy=state.cache.occupancy.mask,
        color=state.color.value,
        max_distance=1,
        directions=KING_DIRECTIONS_3D
    )

    # 2.  Add the **self-detonation** pseudo-move
    self_det = convert_legacy_move_args(
        pos, pos,
        flags=MOVE_FLAGS['SELF_DETONATE']
    )
    self_det.metadata['self_detonate'] = True
    raw_moves.append(self_det)

    # 3.  Final legality filter – kings with priests are immune to the blast
    moves: List[Move] = []
    for m in raw_moves:
        # Normal king-step – keep as-is
        if m.to_coord != pos or not m.metadata.get('self_detonate'):
            moves.append(m)
            continue

        # Self-detonation – validate that **at least one** enemy piece
        # would actually die (otherwise the move is pointless)
        victims = _victims_if_detonate(state, pos)
        if victims:  # non-empty kill list → legal
            moves.append(m)

    return moves

# ------------------------------------------------------------------
# Helper – compute enemy pieces that would die in a 2-sphere detonation
# ------------------------------------------------------------------
def _victims_if_detonate(state: GameState, trigger: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """Return list of enemy squares that would be cleared by a bomb at *trigger*."""
    from game3d.effects.auras.aura import sphere_centre  # local import to avoid cycles

    victims = []
    for sq in sphere_centre(state.board, trigger, radius=2):
        victim = state.cache.top_piece(sq)
        if victim is None or victim.color == state.color:
            continue  # empty or own piece
        if victim.ptype is PieceType.KING and _any_priest_alive(state.board, victim.color):
            continue  # king protected by priest
        victims.append(sq)
    return victims

# ------------------------------------------------------------------
# Dispatcher registration (old name kept)
# ------------------------------------------------------------------
@register(PieceType.BOMB)
def bomb_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_bomb_moves(state, x, y, z)

# ------------------------------------------------------------------
# Backward compatibility exports
# ------------------------------------------------------------------
__all__ = ["generate_bomb_moves"]

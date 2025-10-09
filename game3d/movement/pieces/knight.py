# game3d/movement/movetypes/knight.py
"""
Unified Knight movement generator + share-square logic.

Knights:
  - move with the usual 24 L-shaped offsets;
  - may **end** their move on a square that already contains
    one (or many) friendly pieces;
  - may **capture** enemy pieces normally;
  - the share-square rule is handled **inside this generator**,
    so the rest of the engine sees perfectly ordinary Move objects.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ------------------------------------------------------------------
# 24 knight offsets (3-D)
# ------------------------------------------------------------------
KNIGHT_OFFSETS = np.array([
    (1, 2, 0), (1, -2, 0), (-1, 2, 0), (-1, -2, 0),
    (2, 1, 0), (2, -1, 0), (-2, 1, 0), (-2, -1, 0),
    (1, 0, 2), (1, 0, -2), (-1, 0, 2), (-1, 0, -2),
    (2, 0, 1), (2, 0, -1), (-2, 0, 1), (-2, 0, -1),
    (0, 1, 2), (0, 1, -2), (0, -1, 2), (0, -1, -2),
    (0, 2, 1), (0, 2, -1), (0, -2, 1), (0, -2, -1),
], dtype=np.int8)

# ------------------------------------------------------------------
# Public generator – used by dispatcher and AI
# ------------------------------------------------------------------
def generate_knight_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """All legal Knight moves from (x,y,z) including share-square landings."""
    pos = (x, y, z)
    gen = get_integrated_jump_movement_generator(state.cache)

    # 1.  Let the jump engine discard off-board, enemy-king-with-priest, walls, etc.
    raw_moves = gen.generate_jump_moves(
        color=state.color,
        pos=pos,
        directions=KNIGHT_OFFSETS,
    )

    # 2.  Post-process for share-square
    moves: List[Move] = []
    for m in raw_moves:
        occupants = state.cache.pieces_at(m.to_coord)   # List[Piece] (empty → [])
        enemy_here = any(p.color != state.color for p in occupants)

        if enemy_here:
            # Normal capture – keep the move exactly as returned
            moves.append(m)
        else:
            # Empty or **only** friendly pieces → share-square
            # Force the capture flag off (it is **not** a capture)
            moves.append(
                Move(
                    from_coord=m.from_coord,
                    to_coord=m.to_coord,
                    flags=0,               # not a capture
                    metadata=m.metadata,   # preserve anything else
                )
            )
    return moves

# ------------------------------------------------------------------
# Dispatcher registration (old name kept)
# ------------------------------------------------------------------
@register(PieceType.KNIGHT)
def knight_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_knight_moves(state, x, y, z)

# ------------------------------------------------------------------
# Backward compatibility exports
# ------------------------------------------------------------------
__all__ = ["generate_knight_moves"]

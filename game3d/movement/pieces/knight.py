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
from game3d.movement.movepiece import Move, convert_legacy_move_args
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
    pos = (x, y, z)
    # FIXED: Pass cache_manager to jump generator
    gen = get_integrated_jump_movement_generator(state.cache)

    raw_moves = gen.generate_jump_moves(
        color=state.color,
        pos=pos,
        directions=KNIGHT_OFFSETS,
    )

    moves: List[Move] = []
    for m in raw_moves:
        # FIXED: Use cache_manager's method
        occupants = state.cache.pieces_at(m.to_coord)
        enemy_here = any(p.color != state.color for p in occupants)

        if enemy_here:
            moves.append(m)
        else:
            moves.append(convert_legacy_move_args(
                m.from_coord, m.to_coord, is_capture=False
            ))
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

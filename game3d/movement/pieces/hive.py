# game3d/movement/movetypes/hive.py
"""
Unified Hive / King movement generator.

Hive pieces are treated exactly like Kings (single-step to any of the
26 neighbour cells), **but** a player may perform **several** such
moves per turn – one after another – provided every moved piece is a
Hive of his own colour.  The turn ends only when the player explicitly
signals so (or when he has no Hive left that still has a legal move).

Public API
----------
generate_hive_moves(state, x, y, z) -> List[Move]
    Return every **one-step** Hive move from the given coordinate.
    (Same signature the old hivemoves.py exported.)

generate_king_moves(cache, colour, x, y, z) -> List[Move]
    Old king-move entry-point kept for compatibility.
"""

from __future__ import annotations
from typing import List, TYPE_CHECKING

import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import convert_legacy_move_args
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# ---------------------------------------------------------------------------
# 26 one-step directions (identical to King)
# ---------------------------------------------------------------------------
KING_DIRECTIONS_3D = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# ---------------------------------------------------------------------------
# Single-ply generator (used by both Hive and King)
# ---------------------------------------------------------------------------
def generate_king_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """
    Generate all **one-step** moves for a King (or Hive) located at (x,y,z).
    """
    pos = (x, y, z)
    engine = get_integrated_jump_movement_generator(cache)   # << supply cache
    return engine.generate_jump_moves(
        color=color,
        pos=pos,
        directions=KING_DIRECTIONS_3D,
        allow_capture=True,
        piece_name='king'
    )
# ---------------------------------------------------------------------------
# Hive dispatcher – simply aliases the king generator
# ---------------------------------------------------------------------------
generate_hive_moves = generate_king_moves  # exact same logic

@register(PieceType.HIVE)
def hive_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Entry point registered in the movement dispatcher table."""
    return generate_hive_moves(state.cache, state.color, x, y, z)

# ---------------------------------------------------------------------------
# Multi-Hive turn helpers
# ---------------------------------------------------------------------------
def get_movable_hives(state: GameState, color: Color) -> List[Move]:
    """
    Return **all** single-step Hive moves still available for *color*
    in the current position.  Used by the UI to highlight possible
    next pieces after the first Hive has been moved.
    """
    from game3d.movement.pseudo_legal import generate_pseudo_legal_moves_for_piece

    hives: List[Move] = []
    for coord, piece in state.cache.occupancy.iter_color(color):
        if piece.ptype is PieceType.HIVE:
            hives.extend(generate_pseudo_legal_moves_for_piece(state, coord))
    return hives

def apply_multi_hive_move(state: GameState, move: Move) -> State:
    """
    Apply *one* Hive move and return the new state **without** flipping
    the side-to-move flag.  This allows the same player to continue
    moving his remaining Hives.

    The caller is responsible for:
        - checking that the move is legal;
        - eventually calling `state.pass_turn()` (or whatever your
          UI uses) to end the multi-move sequence.
    """
    # Re-use the normal make_move infrastructure but suppress colour flip
    new_state = state.make_move(move)
    # Undo the colour flip that make_move performed
    object.__setattr__(new_state, "color", state.color)
    # Clear the legal-move cache so the next pick is recomputed
    new_state._clear_caches()
    return new_state

# ---------------------------------------------------------------------------
# Re-export old names for 100 % backward compatibility
# ---------------------------------------------------------------------------
__all__ = [
    "generate_hive_moves",
    "generate_king_moves",
    "get_movable_hives",
    "apply_multi_hive_move",
]

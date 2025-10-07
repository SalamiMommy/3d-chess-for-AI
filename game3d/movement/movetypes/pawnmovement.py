"""Pawn moves — zero-redundancy version using occupancy codes (no piece objects)."""

from __future__ import annotations

from typing import List
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.cache.manager import CacheManager
from game3d.common.common import in_bounds
from game3d.movement.movepiece import MOVE_FLAGS
# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _is_on_start_rank(z: int, color: Color) -> bool:
    return (color == Color.WHITE and z == 1) or (color == Color.BLACK and z == 7)

# ------------------------------------------------------------------
# main generator
# ------------------------------------------------------------------
def generate_pawn_moves(
    cache: CacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate all legal pawn moves from (x, y, z) using occupancy codes."""
    pos = (x, y, z)

    # export occupancy & piece arrays once
    occ, piece_arr = cache.piece_cache.export_arrays()
    own_code = 1 if color == Color.WHITE else 2
    self_code = PieceType.PAWN.value | (own_code << 3)

    if occ[x, y, z] != self_code:          # quick occupancy check
        return []

    dz = 1 if color == Color.WHITE else -1
    moves: List[Move] = []

    # ---------- single push ----------
    fwd = (x, y, z + dz)
    if in_bounds(fwd) and occ[fwd] == 0:
        moves.append(_make_pawn_move(pos, fwd, color))
        # double push from start rank
        if _is_on_start_rank(z, color):
            dfwd = (x, y, z + 2 * dz)
            if in_bounds(dfwd) and occ[dfwd] == 0:
                moves.append(_make_pawn_move(pos, dfwd, color))

    # ---------- captures ----------
    enemy_code = 3 - own_code
    ARMOUR_CODE = PieceType.ARMOUR.value | (enemy_code << 3)

    for dx in (-1, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            tgt = (x + dx, y + dy, z + dz)
            if not in_bounds(tgt):
                continue
            tgt_code = occ[tgt]
            if tgt_code == 0:
                continue                      # empty
            if tgt_code == ARMOUR_CODE:
                continue                      # armour block
            if (tgt_code & 0b111) == enemy_code:  # enemy piece
                moves.append(_make_pawn_move(pos, tgt, color, flags=MOVE_FLAGS['CAPTURE']))

    # (en-passant can be added here later by consulting cache.ep_square)
    return moves

# ------------------------------------------------------------------
# tiny helper — builds Move with promotion logic
# ------------------------------------------------------------------
def _make_pawn_move(
    from_pos: tuple[int, int, int],
    to_pos: tuple[int, int, int],
    color: Color,
    flags: int = 0  # Changed from is_capture to flags
) -> Move:
    _, _, tz = to_pos
    is_promotion = (color == Color.WHITE and tz == 8) or (color == Color.BLACK and tz == 0)
    return Move(
        from_coord=from_pos,
        to_coord=to_coord,
        flags=flags,  # Use flags directly
        is_promotion=is_promotion,
        is_en_passant=False,
        captured_piece=None
    )

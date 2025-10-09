# pawnmoves.py — 3-D pawn (push + trigonal capture) + armour immunity
from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING
from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# --------------------------------------------------------------------------- #
#  Direction tables (small, fast to build)                                    #
# --------------------------------------------------------------------------- #
def _push_dirs(color: Color) -> np.ndarray:
    dz = 1 if color == Color.WHITE else -1
    return np.array([(0, 0, dz)], dtype=np.int8)

def _capture_dirs(color: Color) -> np.ndarray:
    dz = 1 if color == Color.WHITE else -1
    return np.array([(dx, dy, dz) for dx in (1, -1) for dy in (1, -1)], dtype=np.int8)

# --------------------------------------------------------------------------- #
#  Armour immunity helper (works with both flags)                             #
# --------------------------------------------------------------------------- #
def _is_armoured(piece) -> bool:
    return piece is not None and (
        (hasattr(piece, "armoured") and piece.armoured) or piece.ptype is PieceType.ARMOUR
    )

# --------------------------------------------------------------------------- #
#  One-shot generator                                                         #
# --------------------------------------------------------------------------- #
def generate_pawn_moves(cache, color: Color, x: int, y: int, z: int) -> List[Move]:
    pos = (x, y, z)
    # FIXED: Use cache_manager's occupancy
    occ, piece_arr = cache.occupancy.export_arrays()
    own_code = 1 if color == Color.WHITE else 2

    if occ[z, y, x] != own_code or piece_arr[z, y, x] != PieceType.PAWN.value:
        return []

    moves: List[Move] = []
    dz = 1 if color == Color.WHITE else -1
    # FIXED: Pass cache_manager
    jump = get_integrated_jump_movement_generator(cache)

    # Push moves
    push_moves = jump.generate_jump_moves(
        color=color, pos=pos,
        directions=_push_dirs(color),
        allow_capture=False,
    )
    for mv in push_moves:
        tz = mv.to_coord[2]
        flags = MOVE_FLAGS['PROMOTION'] if _is_promotion_rank(tz, color) else 0
        moves.append(Move(pos, mv.to_coord, flags=flags))

        if _is_on_start_rank(z, color):
            db = (x, y, z + 2 * dz)
            if in_bounds(db) and occ[db[2], db[1], db[0]] == 0:
                flags = MOVE_FLAGS['PROMOTION'] if _is_promotion_rank(db[2], color) else 0
                moves.append(Move(pos, db, flags=flags))

    # Capture moves
    cap_moves = jump.generate_jump_moves(
        color=color, pos=pos,
        directions=_capture_dirs(color),
        allow_capture=True,
    )
    for mv in cap_moves:
        tx, ty, tz = mv.to_coord
        # FIXED: Use cache_manager's occupancy
        victim = cache.occupancy.get(mv.to_coord)
        if victim is None or victim.color == color or _is_armoured(victim):
            continue
        flags = MOVE_FLAGS['CAPTURE']
        if _is_promotion_rank(tz, color):
            flags |= MOVE_FLAGS['PROMOTION']
        moves.append(Move(pos, mv.to_coord, flags=flags, captured_piece=victim.ptype.value))

    return moves
# --------------------------------------------------------------------------- #
#  Tiny helpers                                                               #
# --------------------------------------------------------------------------- #
def _is_on_start_rank(z: int, color: Color) -> bool:
    return (color == Color.WHITE and z == 2) or (color == Color.BLACK and z == 6)

def _is_promotion_rank(z: int, color: Color) -> bool:
    return (color == Color.WHITE and z == 8) or (color == Color.BLACK and z == 0)

# --------------------------------------------------------------------------- #
#  Dispatcher registration (kept for compatibility)                           #
# --------------------------------------------------------------------------- #
@register(PieceType.PAWN)
def pawn_move_dispatcher(state: State, x: int, y: int, z: int) -> List[Move]:
    return generate_pawn_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_pawn_moves"]

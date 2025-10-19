# pawn.py  – 3-D pawn (push + trigonal capture) + armour immunity
#            Re-architected to mirror knight.py
from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.enums import Color, PieceType
from game3d.common.common import in_bounds
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move, MOVE_FLAGS

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ------------------------------------------------------------------
# 1. Direction tables – tiny, fast to build
# ------------------------------------------------------------------
def _push_dirs(colour: Color) -> np.ndarray:
    dz = 1 if colour == Color.WHITE else -1
    return np.array([(0, 0, dz)], dtype=np.int8)

def _capture_dirs(colour: Color) -> np.ndarray:
    dz = 1 if colour == Color.WHITE else -1
    return np.array([(dx, dy, dz) for dx in (1, -1) for dy in (1, -1)], dtype=np.int8)

# ------------------------------------------------------------------
# 2. Armour immunity helper
# ------------------------------------------------------------------
def _is_armoured(piece) -> bool:
    return piece is not None and (
        (hasattr(piece, "armoured") and piece.armoured) or piece.ptype is PieceType.ARMOUR
    )

# ------------------------------------------------------------------
# 3. Rank helpers
# ------------------------------------------------------------------
def _is_on_start_rank(z: int, colour: Color) -> bool:
    return (colour == Color.WHITE and z == 2) or (colour == Color.BLACK and z == 6)

def _is_promotion_rank(z: int, colour: Color) -> bool:
    return (colour == Color.WHITE and z == 8) or (colour == Color.BLACK and z == 0)

# ------------------------------------------------------------------
# 4. Public generator – same signature as knight.py
# ------------------------------------------------------------------
def generate_pawn_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Generate every legal pawn move for the piece located at (x,y,z)
    on the supplied GameState.  Mirrors the knight architecture.
    """
    cache   = state.cache
    colour  = state.color
    start   = (x, y, z)

    # --- quick reject ------------------------------------------------
    occ_piece = cache.occupancy.get(start)
    if occ_piece is None or occ_piece.color != colour or occ_piece.ptype != PieceType.PAWN:
        return []

    # --- obtain the common jump generator ----------------------------
    jump = get_integrated_jump_movement_generator(cache)

    moves: List[Move] = []

    # ---------- 4a. PUSH moves ---------------------------------------
    push_moves = jump.generate_jump_moves(
        color=colour,
        pos=start,
        directions=_push_dirs(colour),
        allow_capture=False,
        piece_name="pawn",
    )
    for mv in push_moves:
        tz = mv.to_coord[2]
        flags = MOVE_FLAGS["PROMOTION"] if _is_promotion_rank(tz, colour) else 0
        moves.append(Move(start, mv.to_coord, flags=flags))

    # ----- two-step push from start rank (manually) ------------------
    if _is_on_start_rank(z, colour):
        dz  = 1 if colour == Color.WHITE else -1
        db  = (x, y, z + 2 * dz)
        if in_bounds(db) and not cache.occupancy.is_occupied(*db):
            flags = MOVE_FLAGS["PROMOTION"] if _is_promotion_rank(db[2], colour) else 0
            moves.append(Move(start, db, flags=flags))

    # ---------- 4b. CAPTURE moves ------------------------------------
    cap_moves = jump.generate_jump_moves(
        color=colour,
        pos=start,
        directions=_capture_dirs(colour),
        allow_capture=True,
        piece_name="pawn",
    )
    for mv in cap_moves:
        victim = cache.occupancy.get(mv.to_coord)
        if victim is None or victim.color == colour or _is_armoured(victim):
            continue
        flags = MOVE_FLAGS["CAPTURE"]
        if _is_promotion_rank(mv.to_coord[2], colour):
            flags |= MOVE_FLAGS["PROMOTION"]
        moves.append(
            Move(start, mv.to_coord, flags=flags, captured_piece=victim.ptype.value)
        )

    return moves

# ------------------------------------------------------------------
# 5. Dispatcher registration (kept for compatibility)
# ------------------------------------------------------------------
@register(PieceType.PAWN)
def pawn_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_pawn_moves(state, x, y, z)

__all__ = ["generate_pawn_moves"]

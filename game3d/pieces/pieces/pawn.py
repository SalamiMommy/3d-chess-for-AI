# pawn.py - FIXED
from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.enums import Color, PieceType
from game3d.common.coord_utils import in_bounds
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Direction tables
def _push_dirs(colour: Color) -> np.ndarray:
    dz = 1 if colour == Color.WHITE else -1
    return np.array([(0, 0, dz)], dtype=np.int8)

def _capture_dirs(colour: Color) -> np.ndarray:
    dz = 1 if colour == Color.WHITE else -1
    return np.array([(dx, dy, dz) for dx in (1, -1) for dy in (1, -1)], dtype=np.int8)

# Armour immunity helper
def _is_armoured(piece) -> bool:
    return piece is not None and (
        (hasattr(piece, "armoured") and piece.armoured) or piece.ptype is PieceType.ARMOUR
    )

# Rank helpers
def _is_on_start_rank(z: int, colour: Color) -> bool:
    return (colour == Color.WHITE and z == 2) or (colour == Color.BLACK and z == 6)

def _is_promotion_rank(z: int, colour: Color) -> bool:
    return (colour == Color.WHITE and z == 8) or (colour == Color.BLACK and z == 0)

# Public generator - STANDARDIZED signature
def generate_pawn_moves(
    cache_manager: 'OptimizedCacheManager',  # STANDARDIZED: First parameter
    color: Color,                           # STANDARDIZED: Second parameter
    x: int, y: int, z: int
) -> List[Move]:
    """Generate pawn moves using single cache manager."""
    x, y, z = ensure_int_coords(x, y, z)
    start = (x, y, z)

    # Use standardized cache access
    occ_piece = cache_manager.get_piece(start)
    if occ_piece is None or occ_piece.color != color or occ_piece.ptype != PieceType.PAWN:
        return []

    # Get jump generator from cache manager
    jump_gen = get_integrated_jump_movement_generator(cache_manager)

    moves: List[Move] = []

    # PUSH moves
    push_moves = jump_gen.generate_jump_moves(
        color=color,
        pos=start,
        directions=_push_dirs(color),
        allow_capture=False,
    )
    for mv in push_moves:
        tz = mv.to_coord[2]
        flags = MOVE_FLAGS["PROMOTION"] if _is_promotion_rank(tz, color) else 0
        moves.append(Move(start, mv.to_coord, flags=flags))

    # Two-step push from start rank
    if _is_on_start_rank(z, color):
        dz = 1 if color == Color.WHITE else -1
        db = (x, y, z + 2 * dz)
        if in_bounds(db) and cache_manager.get_piece(db) is None:
            flags = MOVE_FLAGS["PROMOTION"] if _is_promotion_rank(db[2], color) else 0
            moves.append(Move(start, db, flags=flags))

    # CAPTURE moves
    cap_moves = jump_gen.generate_jump_moves(
        color=color,
        pos=start,
        directions=_capture_dirs(color),
        allow_capture=True,
    )
    for mv in cap_moves:
        victim = cache_manager.get_piece(mv.to_coord)  # Use standardized access
        if victim is None or victim.color == color or _is_armoured(victim):
            continue
        flags = MOVE_FLAGS["CAPTURE"]
        if _is_promotion_rank(mv.to_coord[2], color):
            flags |= MOVE_FLAGS["PROMOTION"]
        moves.append(
            Move(start, mv.to_coord, flags=flags, captured_piece=victim.ptype.value)
        )

    return moves

@register(PieceType.PAWN)
def pawn_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    # STANDARDIZED: Use cache_manager property and new signature
    return generate_pawn_moves(state.cache_manager, state.color, x, y, z)

__all__ = ["generate_pawn_moves"]

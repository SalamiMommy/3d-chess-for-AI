"""Pawn movement — Manhattan-distance-1 forward push & trigonal captures, 3D chess."""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager as CacheManager
# ------------------------------------------------------------------
#  Direction tables
# ------------------------------------------------------------------
def get_pawn_push_dirs(color: Color) -> np.ndarray:
    dz = 1 if color == Color.WHITE else -1
    return np.array([(0, 0, dz)], dtype=np.int8)

def get_pawn_capture_dirs(color: Color) -> np.ndarray:
    dz = 1 if color == Color.WHITE else -1
    return np.array([(dx, dy, dz) for dx in (1, -1) for dy in (1, -1)], dtype=np.int8)

# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------
def _is_on_start_rank(z: int, color: Color) -> bool:
    return (color == Color.WHITE and z == 2) or (color == Color.BLACK and z == 6)

def _is_promotion_rank(z: int, color: Color) -> bool:
    return (color == Color.WHITE and z == 8) or (color == Color.BLACK and z == 0)

# ------------------------------------------------------------------
#  Main generator
# ------------------------------------------------------------------
def generate_pawn_moves(
    cache: CacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    pos = (x, y, z)
    occ, piece_arr = cache.piece_cache.export_arrays()
    own_code = 1 if color == Color.WHITE else 2

    # Indexing order: [z, y, x]
    if occ[z, y, x] != own_code or piece_arr[z, y, x] != PieceType.PAWN.value:
        return []

    moves: List[Move] = []
    dz = 1 if color == Color.WHITE else -1

    # ---------- single push ----------
    fwd = (x, y, z + dz)
    if in_bounds(fwd) and occ[fwd[2], fwd[1], fwd[0]] == 0:
        flags = MOVE_FLAGS['PROMOTION'] if _is_promotion_rank(fwd[2], color) else 0
        moves.append(Move(from_coord=pos, to_coord=fwd, flags=flags))

        # double push from start rank
        if _is_on_start_rank(z, color):
            dfwd = (x, y, z + 2 * dz)
            if in_bounds(dfwd) and occ[dfwd[2], dfwd[1], dfwd[0]] == 0:
                flags = MOVE_FLAGS['PROMOTION'] if _is_promotion_rank(dfwd[2], color) else 0
                moves.append(Move(from_coord=pos, to_coord=dfwd, flags=flags))

    # ---------- captures (trigonal) ----------
    jump_gen = get_integrated_jump_movement_generator(cache)
    capture_moves = jump_gen.generate_jump_moves(
        color=color,
        pos=pos,
        directions=get_pawn_capture_dirs(color),
        allow_capture=True,
    )

    enemy_code = 2 if color == Color.WHITE else 1
    ARMOUR_CODE = PieceType.ARMOUR.value

    for mv in capture_moves:
        tx, ty, tz = mv.to_coord
        target_type = PieceType(piece_arr[tz, ty, tx])   # 0-15 fits into uint8
        if target_type == PieceType.ARMOUR and occ[tz, ty, tx] == enemy_code:
            continue

        flags = MOVE_FLAGS['CAPTURE']
        if _is_promotion_rank(tz, color):
            flags |= MOVE_FLAGS['PROMOTION']

        moves.append(Move(
            from_coord=pos,
            to_coord=(tx, ty, tz),
            flags=flags,
            captured_piece=piece_arr[tz, ty, tx],
        ))

    # (en-passant can be added here later by consulting cache.ep_square)
    return moves

# ------------------------------------------------------------------
#  Convenience export
# ------------------------------------------------------------------
def get_pawn_directions(color: Color) -> Tuple[np.ndarray, np.ndarray]:
    return get_pawn_push_dirs(color), get_pawn_capture_dirs(color)

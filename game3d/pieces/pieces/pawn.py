# pawn.py - OPTIMIZED WITH BATCHING
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
    cache_manager: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate pawn moves using single cache manager - OPTIMIZED WITH BATCHING."""
    x, y, z = ensure_int_coords(x, y, z)
    start = (x, y, z)

    occ_piece = cache_manager.get_piece(start)
    if occ_piece is None or occ_piece.color != color or occ_piece.ptype != PieceType.PAWN:
        return []

    jump_gen = get_integrated_jump_movement_generator(cache_manager)

    # OPTIMIZED: Collect all moves for batch processing where possible
    to_coords_list = []
    capture_flags = []
    promotion_moves = []  # Separate list for moves requiring special handling

    # PUSH moves
    push_moves = jump_gen.generate_jump_moves(
        color=color,
        pos=start,
        directions=_push_dirs(color),
        allow_capture=False,
    )
    for mv in push_moves:
        tz = mv.to_coord[2]
        if _is_promotion_rank(tz, color):
            # Promotion moves need special handling
            promotion_moves.append((start, mv.to_coord, False, True))
        else:
            to_coords_list.append(mv.to_coord)
            capture_flags.append(False)

    # Two-step push from start rank
    if _is_on_start_rank(z, color):
        dz = 1 if color == Color.WHITE else -1
        db = (x, y, z + 2 * dz)
        if in_bounds(db) and cache_manager.get_piece(db) is None:
            if _is_promotion_rank(db[2], color):
                promotion_moves.append((start, db, False, True))
            else:
                to_coords_list.append(db)
                capture_flags.append(False)

    # CAPTURE moves
    cap_moves = jump_gen.generate_jump_moves(
        color=color,
        pos=start,
        directions=_capture_dirs(color),
        allow_capture=True,
    )
    for mv in cap_moves:
        victim = cache_manager.get_piece(mv.to_coord)
        if victim is None or victim.color == color or _is_armoured(victim):
            continue

        if _is_promotion_rank(mv.to_coord[2], color):
            # Promotion capture moves need special handling
            promotion_moves.append((start, mv.to_coord, True, True))
        else:
            to_coords_list.append(mv.to_coord)
            capture_flags.append(True)

    # Create batch moves for non-promotion moves
    moves = []
    if to_coords_list:
        to_coords_array = np.array(to_coords_list, dtype=np.int8)
        captures_array = np.array(capture_flags, dtype=bool)
        moves.extend(Move.create_batch(start, to_coords_array, captures_array))

    # Handle promotion moves individually (they need special flags)
    for from_coord, to_coord, is_capture, is_promotion in promotion_moves:
        # For now, use create_simple for promotion moves as they need special handling
        # In a more advanced optimization, we could extend Move.create_batch to handle promotions
        move = Move.create_simple(from_coord, to_coord, is_capture=is_capture)
        if is_promotion:
            move.metadata['is_promotion'] = True
        moves.append(move)

    return moves

@register(PieceType.PAWN)
def pawn_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_pawn_moves(state.cache_manager, state.color, x, y, z)

__all__ = ["generate_pawn_moves"]

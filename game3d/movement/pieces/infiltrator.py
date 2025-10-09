# game3d/movement/piecemoves/pawnfrontteleportmoves.py
"""Infiltrator == pawn-front-teleport ∪ one-step King moves – self-contained."""

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

# ---------- 1.  Teleport targets (Manhattan-1 in front of ANY enemy pawn) ----------
def _build_teleport_directions(cache, color, x, y, z) -> np.ndarray:
    occ, _ = cache.piece_cache.export_arrays()
    enemy_code = 2 if color == Color.WHITE else 1
    pawn_code = PieceType.PAWN.value | (enemy_code << 3)
    dirs = []
    for pos, _ in cache.board.list_occupied():
        px, py, pz = pos
        if occ[px, py, pz] != pawn_code:
            continue
        front = (px, py, pz + 1) if enemy_code == 2 else (px, py, pz - 1)
        if in_bounds(*front) and occ[front] == 0:
            dirs.append(np.array(front) - np.array((x, y, z)))
    return np.array(dirs, dtype=np.int8)

# ---------- 2.  Classic 26 King one-step vectors ----------
_KING_DIRS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# ---------- 3.  Generator ----------
def generate_infiltrator_moves(cache, color, x, y, z) -> List[Move]:
    jump_gen = get_integrated_jump_movement_generator(cache)

    # Use cache.occupancy instead of cache.piece_cache
    occ, _ = cache.occupancy.export_arrays()
    enemy_code = 2 if color == Color.WHITE else 1
    pawn_code = PieceType.PAWN.value | (enemy_code << 3)
    dirs = []

    # Use cache.occupancy.list_occupied() instead of cache.board.list_occupied()
    for pos, piece in cache.occupancy.list_occupied():
        px, py, pz = pos
        if piece.color.value == enemy_code and piece.ptype == PieceType.PAWN:
            front = (px, py, pz + 1) if enemy_code == 2 else (px, py, pz - 1)
            if in_bounds(*front) and occ[front] == 0:
                dirs.append(np.array(front) - np.array((x, y, z)))

    teleport_dirs = np.array(dirs, dtype=np.int8)
    moves = jump_gen.generate_jump_moves(color=color, pos=(x, y, z), directions=teleport_dirs)
    moves += jump_gen.generate_jump_moves(color=color, pos=(x, y, z), directions=_KING_DIRS)

    return moves

# ---------- 4.  Dispatcher ----------
@register(PieceType.INFILTRATOR)
def infiltrator_move_dispatcher(state: State, x: int, y: int, z: int) -> List[Move]:
    return generate_infiltrator_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_infiltrator_moves"]

"""Pawn-Front Teleporter — teleports to any empty square directly in front of an enemy pawn.
   Pawns advance along Z: White +Z, Black -Z.
"""

from typing import List, Set
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.common.common import in_bounds
from game3d.cache.manager import OptimizedCacheManager

def generate_pawn_front_teleport_moves(
    cache: OptimizedCacheManager,  # ← Already correct
    color: Color,
    x: int,
    y: int,
    z: int
) -> List['Move']:
    """
    Generate teleport moves to any EMPTY square directly in front of an enemy pawn.
    Pawns move in Z-direction:
      - White pawns: front = (x, y, z+1)
      - Black pawns: front = (x, y, z-1)
    """
    self_pos = (x, y, z)

    # Validate piece at start (optional, but consistent)
    piece = cache.piece_cache.get(self_pos)
    if piece is None or piece.color != color:
        return []

    candidate_targets: Set[tuple] = set()

    # Iterate over occupied squares to find enemy pawns
    for pos, other_piece in cache.board.list_occupied():  # ← cache.board, not board
        if other_piece.ptype != PieceType.PAWN:
            continue
        if other_piece.color == color:
            continue  # must be enemy pawn

        # Determine front square based on enemy pawn's color
        px, py, pz = pos
        if other_piece.color == Color.WHITE:
            front = (px, py, pz + 1)
        else:
            front = (px, py, pz - 1)

        if not in_bounds(front):
            continue
        if cache.piece_cache.get(front) is not None:
            continue  # must be empty

        candidate_targets.add(front)

    return [
        Move(from_coord=self_pos, to_coord=target, is_capture=False)
        for target in candidate_targets
    ]

"""3D Knight move generation logic — handles share-square rules."""

from __future__ import annotations
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.common.common import in_bounds, add_coords
from game3d.movement.pathvalidation import validate_piece_at

# Precomputed knight offsets for 3D (24 directions)
KNIGHT_OFFSETS = [
    (1, 2, 0), (1, -2, 0), (-1, 2, 0), (-1, -2, 0),
    (2, 1, 0), (2, -1, 0), (-2, 1, 0), (-2, -1, 0),
    (1, 0, 2), (1, 0, -2), (-1, 0, 2), (-1, 0, -2),
    (2, 0, 1), (2, 0, -1), (-2, 0, 1), (-2, 0, -1),
    (0, 1, 2), (0, 1, -2), (0, -1, 2), (0, -1, -2),
    (0, 2, 1), (0, 2, -1), (0, -2, 1), (0, -2, -1),
]

def generate_knight_moves(state: 'GameState', x: int, y: int, z: int) -> List['Move']:
    """Generate all legal knight moves from (x, y, z) – Share-Square aware."""
    pos = (x, y, z)

    # Validate that the piece at pos is a KNIGHT of the current side
    if not validate_piece_at(state.cache, state.color, pos, PieceType.KNIGHT):
        return []

    cache = state.cache
    color = state.color
    moves: List['Move'] = []

    for offset in KNIGHT_OFFSETS:
        target = add_coords(pos, offset)
        if not in_bounds(target):
            continue

        occupants = cache.pieces_at(target)

        if not occupants:
            # Empty square → normal move
            moves.append(Move(from_coord=pos, to_coord=target, is_capture=False))
        else:
            # Filter out knights (they can share)
            non_knights = [p for p in occupants if p.ptype != PieceType.KNIGHT]

            if non_knights:
                friendly_non_knights = [p for p in non_knights if p.color == color]
                if friendly_non_knights:
                    continue  # blocked by friendly non-knight

                enemy_non_knights = [p for p in non_knights if p.color != color]
                if enemy_non_knights:
                    moves.append(Move(from_coord=pos, to_coord=target, is_capture=True))
            else:
                # Only knights occupy the square → sharing allowed
                moves.append(Move(from_coord=pos, to_coord=target, is_capture=False))

    return moves

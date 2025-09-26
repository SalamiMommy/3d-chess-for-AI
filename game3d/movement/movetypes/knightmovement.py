"""3D Knight move generation logic — Share-Square aware."""

from typing import List, Optional
from game3d.pieces.enums import PieceType, Color
from game3d.game.gamestate import GameState
from game3d.movement.pathvalidation import validate_piece_at, in_bounds, add_coords
from game3d.cache.manager import get_cache_manager
from game3d.movement.movepiece import Move
KNIGHT_OFFSETS = [
    (1, 2, 0), (1, -2, 0), (-1, 2, 0), (-1, -2, 0),
    (2, 1, 0), (2, -1, 0), (-2, 1, 0), (-2, -1, 0),
    (1, 0, 2), (1, 0, -2), (-1, 0, 2), (-1, 0, -2),
    (2, 0, 1), (2, 0, -1), (-2, 0, 1), (-2, 0, -1),
    (0, 1, 2), (0, 1, -2), (0, -1, 2), (0, -1, -2),
    (0, 2, 1), (0, 2, -1), (0, -2, 1), (0, -2, -1),
]


def generate_knight_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Generate all legal knight moves from (x, y, z) – Share-Square aware."""
    start = (x, y, z)

    if not validate_piece_at(state, start, PieceType.KNIGHT):
        return []

    mgr = state.cache
    moves: List[Move] = []

    for dx, dy, dz in KNIGHT_OFFSETS:
        target = add_coords(start, (dx, dy, dz))
        if not in_bounds(target):
            continue

        # Multi-occupancy aware
        occupants = mgr.pieces_at(target)  # [] or [Piece, ...]
        top = mgr.top_piece(target)        # None or top Piece

        # 1. Empty square → normal move
        if not occupants:
            moves.append(Move(start, target, is_capture=False))
            continue

        # 2. At least one knight already there → **knights may share**
        #    Capture only if landing on **enemy non-knight**
        enemy_non_knight = [
            p for p in occupants
            if p.color != state.color and p.ptype != PieceType.KNIGHT
        ]
        if enemy_non_knight:
            # capture the top non-knight enemy (cache handles removal order)
            moves.append(Move(start, target, is_capture=True))
        else:
            # all occupants are friendly or enemy knights → **share / stack**
            moves.append(Move(start, target, is_capture=False))

    return moves

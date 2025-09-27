# game3d/movement/movetypes/knightmovement.py
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.pathvalidation import validate_piece_at  # ← may also need update
from game3d.common.common import in_bounds, add_coords

KNIGHT_OFFSETS = [
    (1, 2, 0), (1, -2, 0), (-1, 2, 0), (-1, -2, 0),
    (2, 1, 0), (2, -1, 0), (-2, 1, 0), (-2, -1, 0),
    (1, 0, 2), (1, 0, -2), (-1, 0, 2), (-1, 0, -2),
    (2, 0, 1), (2, 0, -1), (-2, 0, 1), (-2, 0, -1),
    (0, 1, 2), (0, 1, -2), (0, -1, 2), (0, -1, -2),
    (0, 2, 1), (0, 2, -1), (0, -2, 1), (0, -2, -1),
]

def generate_knight_moves(
    board,
    cache,
    color: Color,
    x: int, y: int, z: int
) -> List['Move']:
    """Generate all legal knight moves from (x, y, z) – Share-Square aware."""
    start = (x, y, z)

    # Inline validation instead of calling validate_piece_at(state, ...)
    piece = board.piece_at(start)
    if piece is None or piece.ptype != PieceType.KNIGHT or piece.color != color:
        return []

    moves: List['Move'] = []

    for offset in KNIGHT_OFFSETS:
        target = add_coords(start, offset)
        if not in_bounds(target):
            continue

        occupants = cache.pieces_at(target)

        if not occupants:
            moves.append(Move(from_coord=start, to_coord=target, is_capture=False))
        else:
            non_knights = [p for p in occupants if p.ptype != PieceType.KNIGHT]

            if non_knights:
                enemy_non_knights = [p for p in non_knights if p.color != color]
                friendly_non_knights = [p for p in non_knights if p.color == color]

                if friendly_non_knights:
                    continue  # blocked
                elif enemy_non_knights:
                    moves.append(Move(from_coord=start, to_coord=target, is_capture=True))
            else:
                # Only knights → sharing allowed
                moves.append(Move(from_coord=start, to_coord=target, is_capture=False))

    return moves

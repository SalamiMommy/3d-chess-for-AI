"""Mirror Teleport Move — teleports piece to (8-x, 8-y, 8-z)."""

from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.pathvalidation import validate_piece_at

def generate_mirror_teleport_move(
    board,
    color: Color,
    x: int, y: int, z: int
) -> List['Move']:
    """
    Generate a single teleport move to mirrored coordinates: (8-x, 8-y, 8-z).

    Rules:
    - Cannot teleport to own square (e.g., center (4,4,4))
    - Can land on empty squares or capture enemy pieces
    - Cannot land on friendly pieces
    """
    start = (x, y, z)

    if not validate_piece_at(board, color, start, PieceType.MIRROR):
        return []

    target = (8 - x, 8 - y, 8 - z)

    # Prevent no-op teleport
    if start == target:
        return []

    target_piece = board.piece_at(target)

    # Cannot land on friendly pieces
    if target_piece is not None and target_piece.color == color:
        return []

    # Capture if enemy piece
    is_capture = target_piece is not None and target_piece.color != color

    return [
        Move(
            from_coord=start,
            to_coord=target,
            is_capture=is_capture
        )
    ]

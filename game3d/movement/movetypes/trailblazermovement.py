"""Trailblazer — rook moves capped at 3 squares per ray."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game.move import Move
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at


ROOK_DIRECTIONS_3D = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]


def generate_trailblazer_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Generate all 3-step rook slides from (x, y, z)."""
    start = (x, y, z)

    if not validate_piece_at(state, start, expected_type=PieceType.TRAILBLAZER):
        return []

    return slide_along_directions(
        state=state,
        start=start,
        directions=ROOK_DIRECTIONS_3D,
        allow_capture=True,
        allow_self_block=False,
        max_steps=3,          # ≤ 3 squares
        edge_only=False
    )

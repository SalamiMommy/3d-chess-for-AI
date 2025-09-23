# game3d/movement/movetypes/knightmovement.py

"""3D Knight move generation logic — pure movement rules, no registration."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.pathvalidation import jump_to_targets, validate_piece_at

# Define knight offsets once — could also go in directions.py later
KNIGHT_OFFSETS = [
    # XY plane (dz=0)
    (1, 2, 0), (1, -2, 0), (-1, 2, 0), (-1, -2, 0),
    (2, 1, 0), (2, -1, 0), (-2, 1, 0), (-2, -1, 0),

    # XZ plane (dy=0)
    (1, 0, 2), (1, 0, -2), (-1, 0, 2), (-1, 0, -2),
    (2, 0, 1), (2, 0, -1), (-2, 0, 1), (-2, 0, -1),

    # YZ plane (dx=0)
    (0, 1, 2), (0, 1, -2), (0, -1, 2), (0, -1, -2),
    (0, 2, 1), (0, 2, -1), (0, -2, 1), (0, -2, -1)
]


def generate_knight_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Generate all legal knight moves from (x, y, z).
    Uses centralized jump logic from pathvalidation.py.
    Returns empty list if no valid knight is at start position.
    """
    pos = (x, y, z)

    # Validate piece exists and is correct type/color
    if not validate_piece_at(state, pos, PieceType.KNIGHT):
        return []

    # Delegate to shared jumping logic
    return jump_to_targets(
        state,
        start=pos,
        offsets=KNIGHT_OFFSETS,
        allow_capture=True,      # Knights can capture
        allow_self_block=False   # Knights cannot land on friendly pieces
    )

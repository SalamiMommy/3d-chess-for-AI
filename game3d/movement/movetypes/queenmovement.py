# game3d/movement/movetypes/queenmovement.py

"""3D Queen move generation logic — pure movement rules, no registration."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at


# Define queen directions — combines rook + bishop
# We break it into named groups for clarity and maintainability

ORTHOGONAL_DIRECTIONS = [
    (1, 0, 0), (-1, 0, 0),  # X
    (0, 1, 0), (0, -1, 0),  # Y
    (0, 0, 1), (0, 0, -1)   # Z
]

# Face diagonals (2 axes change, 3rd fixed)
FACE_DIAGONALS = [
    *( (dx, dy, 0) for dx in (-1, 1) for dy in (-1, 1) ),   # XY plane
    *( (dx, 0, dz) for dx in (-1, 1) for dz in (-1, 1) ),   # XZ plane
    *( (0, dy, dz) for dy in (-1, 1) for dz in (-1, 1) ),   # YZ plane
]

# SPACE DIAGONALS — "Equal 3-axis movement" (all 3 change, ±1 each)
# These are the true 3D diagonals — e.g., (1,1,1), (-1,1,-1), etc.
SPACE_DIAGONALS = [
    (dx, dy, dz)
    for dx in (-1, 1)
    for dy in (-1, 1)
    for dz in (-1, 1)
]

# Combine all for queen
QUEEN_DIRECTIONS_3D = ORTHOGONAL_DIRECTIONS + FACE_DIAGONALS + SPACE_DIAGONALS


def generate_queen_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Generate all legal queen moves from (x, y, z).
    Queen = Rook (orthogonal) + Bishop (diagonal) → 6 + 12 + 8 = 26 directions.

    Includes:
      - Orthogonal movement (1 axis)
      - Face diagonals (2 axes)
      - Space diagonals (3 axes — "equal 3-axis movement")
    """
    pos = (x, y, z)

    # Validate piece exists and is correct type/color
    if not validate_piece_at(state, pos, PieceType.QUEEN):
        return []

    # Delegate to shared sliding logic
    return slide_along_directions(
        state,
        start=pos,
        directions=QUEEN_DIRECTIONS_3D,
        allow_capture=True,      # Queens can capture
        allow_self_block=False   # Queens cannot move through or onto friendly pieces
    )

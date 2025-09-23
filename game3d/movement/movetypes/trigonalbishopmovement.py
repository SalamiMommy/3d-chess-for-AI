# game3d/movement/movetypes/trigonalbishopmovement.py

"""3D Trigonal Bishop move generation logic — moves only along 3-axis-equal diagonals (±1,±1,±1)."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at


# Define ONLY the 8 trigonal (space diagonal) directions
# Where dx, dy, dz are all ±1 — equal movement in all 3 axes
TRIGONAL_BISHOP_DIRECTIONS = [
    (dx, dy, dz)
    for dx in (-1, 1)
    for dy in (-1, 1)
    for dz in (-1, 1)
]
# Total: 8 directions — e.g., (1,1,1), (1,-1,1), (-1,1,-1), etc.


def generate_trigonal_bishop_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Generate all legal trigonal bishop moves from (x, y, z).
    Moves ONLY along 3D space diagonals — all axes change equally (±1 per axis per step).

    Example valid directions: (1,1,1), (-1,1,-1), (1,-1,-1), etc.
    Does NOT move in orthogonal or face-diagonal directions.
    """
    pos = (x, y, z)

    # Validate piece exists and is correct type/color
    # ⚠️ You’ll need to define PieceType.TRIGONAL_BISHOP in your enums
    if not validate_piece_at(state, pos, PieceType.TRIGONAL_BISHOP):
        return []

    # Delegate to shared sliding logic — full sliding, like bishop
    return slide_along_directions(
        state,
        start=pos,
        directions=TRIGONAL_BISHOP_DIRECTIONS,
        allow_capture=True,      # Can capture enemy pieces
        allow_self_block=False   # Cannot land on or pass through friendly pieces
    )

# game3d/movement/movetypes/vectorslidermovement.py

"""Universal 3D Vector Slider — moves continuously along any integer direction vector.
Pure movement logic — no registration.
"""

from typing import List, Tuple
from pieces.enums import PieceType, PIECE_SLIDER_PROFILES
from game.state import GameState
from game.move import Move
from game3d.movement.pathvalidation import (
    slide_along_directions,
    validate_piece_at
)


def generate_vector_slider_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Generate all legal slider moves from (x, y, z) along predefined integer direction vectors.
    Uses centralized sliding logic — continues until blocked or out of bounds.
    """
    start = (x, y, z)

    # Validate piece exists and belongs to current player
    if not validate_piece_at(state, start):
        return []

    piece = state.board.piece_at(start)
    if not piece:
        return []

    # Get movement profile for this piece type
    profile = PIECE_SLIDER_PROFILES.get(piece.ptype)
    if not profile:
        return []  # No movement defined

    directions: List[Tuple[int, int, int]] = profile.get("vectors", [])

    # Use centralized slider — handles blocking, capture, bounds, etc.
    moves = slide_along_directions(
        state=state,
        start=start,
        directions=directions,
        allow_capture=True,      # Standard capture rules
        allow_self_block=False,  # Can't slide through allies
        max_steps=None,          # Unlimited unless profile specifies otherwise
        edge_only=False          # Not restricted to edges (unless profile says so)
    )

    return moves

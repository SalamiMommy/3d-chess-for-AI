"""Hive – you may move **every** friendly Hive piece once this turn, nothing else."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game.move import Move
from game3d.movement.pathvalidation import generate_king_moves, validate_piece_at


def generate_hive_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Generate **one** move for **this** Hive piece – caller will aggregate."""
    start = (x, y, z)

    if not validate_piece_at(state, start, PieceType.HIVE):
        return []

    # Hive pieces move exactly like kings (1-step any direction)
    return generate_king_moves(state, x, y, z)

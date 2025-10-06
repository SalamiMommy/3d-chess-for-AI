# game3d/movement/piecemoves/queenmoves.py
"""Exports queen move generator and registers it with the dispatcher.

The 3-D queen is the union of
  – rook (orthogonal)
  – bishop (2-axis diagonal)
  – trigonal-bishop (3-axis diagonal)
moves.
"""

from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.registry import register
from game3d.movement.movepiece import Move

# pull in the three orthogonal / diagonal generators
from game3d.movement.movetypes.rookmovement import generate_rook_moves
from game3d.movement.movetypes.bishopmovement import generate_bishop_moves
from game3d.movement.movetypes.trigonalbishopmovement import generate_trigonal_bishop_moves


def generate_queen_moves(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int,
    y: int,
    z: int
) -> List[Move]:
    """Return every legal queen move by concatenating the three sub-sliders."""
    return (
        generate_rook_moves(cache, color, x, y, z)
        + generate_bishop_moves(cache, color, x, y, z)
        + generate_trigonal_bishop_moves(cache, color, x, y, z)
    )


# keep the old public name for importers
__all__ = ['generate_queen_moves']


@register(PieceType.QUEEN)
def queen_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_queen_moves(state.cache, state.color, x, y, z)
